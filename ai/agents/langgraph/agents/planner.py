"""PlannerAgent — LLM strategy + reads human_feedback + verification_feedback."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

log = logging.getLogger(__name__)


def planner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    support = state.get("support", "")
    user_name = state.get("user_name", "")
    first_name = user_name.split()[0] if user_name else "apprenant"
    adj_level = state.get("adjusted_level") or state.get("current_level", "beginner")
    difficulties = list(state.get("difficulties") or [])
    weak_concepts = list(state.get("weak_concepts") or [])
    blocked_concepts = list(state.get("blocked_concepts") or [])
    rag_docs = list(state.get("rag_docs") or [])
    memory_context = list(state.get("memory_context") or [])
    human_feedback = state.get("human_feedback", "")
    verification_feedback = list(state.get("verification_feedback") or [])
    objectives = list(state.get("learning_objectives") or [])
    n_retries = dict(state.get("n_retries") or {})
    agent_trace = list(state.get("agent_trace") or [])
    agent_reasoning = dict(state.get("agent_reasoning") or {})

    from ai.agents.helpers import plan_learning_strategy, _self_critique
    from ai.llm.service import call_llm

    # Human feedback as pedagogical constraint (free text = learner preference)
    if human_feedback and human_feedback.lower().strip() not in (
        "oui",
        "yes",
        "y",
        "o",
        "",
    ):
        difficulties = list(difficulties) + [f"Human feedback : {human_feedback[:100]}"]

    # Deterministic baseline
    decisions = plan_learning_strategy(
        support, adj_level, difficulties, human_feedback, memory_context
    )

    # Build context for LLM
    rag_summary = " ".join(d.get("content", "")[:100] for d in rag_docs[:3])
    verification_note = (
        f"\nFeedback vérificateur (retry #{n_retries.get('planner', 0)}) : {verification_feedback}"
        if verification_feedback
        else ""
    )

    # Blocked concepts section — injected before the LLM prompt
    blocked_names = [bc["concept"] for bc in blocked_concepts]
    blocked_note = ""
    if blocked_concepts:
        details = "; ".join(
            f"{bc['concept']} ({bc['attempts']} tentatives"
            + (f", erreur : {bc['last_error']}" if bc.get("last_error") else "")
            + ")"
            for bc in blocked_concepts[:3]
        )
        blocked_note = (
            f"\nBLOCAGES CHRONIQUES : {details}\n"
            f"RÈGLE ABSOLUE : Ces concepts doivent être débloqués AVANT d'introduire "
            f"tout nouveau concept. Propose une approche radicalement différente "
            f"(analogie, décomposition en micro-étapes, exemple du quotidien).\n"
        )

    prompt = (
        f"Génère un plan pédagogique personnalisé pour {first_name} : support={support}, niveau={adj_level}.\n"
        f"Difficultés de {first_name} : {difficulties}\n"
        f"Concepts faibles : {weak_concepts}\n"
        + (f"Concepts bloqués (priorité maximale) : {blocked_names}\n" if blocked_names else "")
        + f"Objectifs : {objectives}\n"
        f"Sources RAG : {rag_summary[:300]}\n"
        f"{blocked_note}"
        f"{verification_note}\n\n"
        f"Génère 3-5 étapes d'apprentissage prioritaires, adaptées au profil de {first_name}.\n"
        f'Réponds en JSON : {{"decisions": [{{"id": 1, "action": "...", "rationale": "...", "priority": 1}}], "reasoning": "..."}}'
    )
    llm_text = call_llm(prompt, max_tokens=400)
    strategy = "; ".join(d.get("action", "") for d in decisions)

    if llm_text:
        try:
            start = llm_text.index("{")
            end = llm_text.rindex("}") + 1
            data = json.loads(llm_text[start:end])
            llm_decisions = data.get("decisions") or []
            if llm_decisions:
                decisions = llm_decisions
                strategy = "; ".join(d.get("action", "") for d in decisions)
            agent_reasoning["planner"] = f"[LLM] {data.get('reasoning', '')[:80]}"
        except Exception:
            agent_reasoning["planner"] = "[fallback] deterministic strategy"
    else:
        agent_reasoning["planner"] = "[fallback] deterministic strategy"

    # Self-critique
    _self_critique("planner", f"strategy: {strategy[:200]}", state)

    n_retries["planner"] = n_retries.get("planner", 0) + 1
    agent_trace.append(
        f"planner → {len(decisions)} decisions (retry #{n_retries['planner']})"
    )

    return {
        "strategy": strategy,
        "strategy_decisions": decisions,
        "n_retries": n_retries,
        "agent_trace": agent_trace,
        "agent_reasoning": agent_reasoning,
    }
