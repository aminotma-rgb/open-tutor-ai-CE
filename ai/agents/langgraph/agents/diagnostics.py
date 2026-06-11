"""DiagnosticsAgent — LLM level assessment + self-critique + interrupt P1."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langgraph.types import interrupt

log = logging.getLogger(__name__)


def diagnostics_node(state: Dict[str, Any]) -> Dict[str, Any]:
    support = state.get("support", "")
    user_name = state.get("user_name", "")
    first_name = user_name.split()[0] if user_name else "apprenant"
    current_level = state.get("current_level", "beginner")
    weak_concepts = list(state.get("weak_concepts") or [])
    memory_context = list(state.get("memory_context") or [])
    objectives = list(state.get("learning_objectives") or [])
    interactions = state.get("user_message", "")
    human_feedback = state.get("human_feedback", "")
    agent_trace = list(state.get("agent_trace") or [])
    agent_reasoning = dict(state.get("agent_reasoning") or {})

    from ai.agents.helpers import (
        assess_current_level,
        detect_difficulties,
        extract_memory_signals,
        _self_critique,
    )
    from ai.llm.service import call_llm

    # Deterministic baseline
    adj_level = assess_current_level(current_level, interactions, human_feedback)
    difficulties = detect_difficulties(
        support, interactions, human_feedback, objectives
    )
    difficulties += extract_memory_signals(support, memory_context)
    for wc in weak_concepts[:3]:
        hint = f"Concept faible détecté : {wc}"
        if hint not in difficulties:
            difficulties.append(hint)
    difficulties = list(dict.fromkeys(difficulties))[:5]

    # LLM assessment
    prompt = (
        f"Tu évalues le niveau de {first_name} en {support}.\n"
        f"Niveau déclaré : {current_level}\n"
        f"Concepts faibles (KG) : {weak_concepts}\n"
        f"Message de {first_name} : {interactions[:300]}\n"
        f"Mémoires récentes : {[m.get('content', '')[:80] for m in memory_context[:3]]}\n\n"
        f"Évalue le niveau réel (beginner/intermediate/advanced) et liste 3-5 difficultés.\n"
        f'Réponds en JSON : {{"level": "...", "difficulties": [...], "reasoning": "..."}}'
    )
    llm_text = call_llm(prompt, max_tokens=300)
    if llm_text:
        try:
            start = llm_text.index("{")
            end = llm_text.rindex("}") + 1
            data = json.loads(llm_text[start:end])
            llm_level = data.get("level", "")
            if llm_level in ("beginner", "intermediate", "advanced"):
                adj_level = llm_level
            llm_diff = data.get("difficulties") or []
            if llm_diff:
                difficulties = llm_diff[:5]
            agent_reasoning["diagnostics"] = (
                f"[LLM] level={adj_level} — {data.get('reasoning','')[:80]}"
            )
        except Exception:
            agent_reasoning["diagnostics"] = f"[fallback] level={adj_level}"
    else:
        agent_reasoning["diagnostics"] = f"[fallback] level={adj_level}"

    # Self-critique
    _self_critique(
        "diagnostics", f"level={adj_level}, difficulties={difficulties}", state
    )

    # Human-in-the-Loop P1 — ask learner to confirm level assessment
    if not human_feedback:
        human_feedback = interrupt(
            {
                "checkpoint": "P1",
                "message": (
                    f"{first_name}, ton niveau a été évalué : {adj_level}. "
                    f"Concepts faibles : {weak_concepts[:3]}. "
                    f"Continuer ? (oui / correction libre)"
                ),
                "adjusted_level": adj_level,
                "difficulties": difficulties,
            }
        )
        if not isinstance(human_feedback, str):
            human_feedback = ""

    agent_trace.append(
        f"diagnostics → level={adj_level}, {len(difficulties)} difficulties [P1 HITL]"
    )

    return {
        "adjusted_level": adj_level,
        "difficulties": difficulties,
        "human_feedback": human_feedback,
        "agent_trace": agent_trace,
        "agent_reasoning": agent_reasoning,
    }
