"""DiagnosticsAgent — LLM level assessment + error pattern detection + interrupt P1."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langgraph.types import interrupt

log = logging.getLogger(__name__)


def _detect_error_patterns(
    blocked_concepts: List[Dict[str, Any]],
    answer_history: List[Dict[str, Any]],
) -> Dict[str, Dict]:
    """Detect systematic error patterns for blocked concepts using LLM analysis.

    Only triggered when ≥ 3 wrong answers exist in answer_history.
    Returns {concept: {pattern, hypothesis, severity}} or {}.
    """
    from ai.llm.service import call_llm

    if not blocked_concepts or not answer_history:
        return {}

    wrong_answers = [v for v in answer_history if v.get("correct") is False]
    if len(wrong_answers) < 3:
        return {}

    blocked_names = [bc["concept"] for bc in blocked_concepts]

    error_lines = []
    for v in wrong_answers[-8:]:
        learner = v.get("learner_answer", "")
        expected = v.get("correct_answer", "")
        expl = v.get("explanation", "")
        msg = v.get("user_message", "")
        if learner and expected:
            error_lines.append(
                f"  - Réponse : {learner}  →  attendu : {expected}"
                + (f"  ({expl[:80]})" if expl else "")
            )
        elif msg:
            error_lines.append(f"  - Message : {msg[:100]}")

    if not error_lines:
        return {}

    prompt = (
        f"Voici les erreurs récentes d'un apprenant sur les concepts : {blocked_names}.\n\n"
        f"Erreurs observées :\n" + "\n".join(error_lines) + "\n\n"
        f"Identifie le ou les patterns d'erreur récurrents par concept. "
        f"Y a-t-il une confusion systématique ou une mauvaise représentation précise ?\n"
        f"Réponds en JSON avec un objet par concept concerné :\n"
        f'{{"nom_du_concept": {{"pattern": "...", "hypothesis": "...", "severity": "mild|moderate|severe"}}}}'
    )

    result = call_llm(prompt, max_tokens=300)
    if not result:
        return {}

    try:
        start = result.find("{")
        end = result.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(result[start:end])
            patterns = {}
            for concept, info in data.items():
                if isinstance(info, dict) and "pattern" in info:
                    patterns[concept] = {
                        "pattern": info.get("pattern", ""),
                        "hypothesis": info.get("hypothesis", ""),
                        "severity": info.get("severity", "moderate"),
                    }
            return patterns
    except Exception as exc:
        log.debug("Error pattern detection parse failed: %s", exc)

    return {}


def diagnostics_node(state: Dict[str, Any]) -> Dict[str, Any]:
    support = state.get("support", "")
    user_name = state.get("user_name", "")
    first_name = user_name.split()[0] if user_name else "apprenant"
    current_level = state.get("current_level", "beginner")
    weak_concepts = list(state.get("weak_concepts") or [])
    blocked_concepts = list(state.get("blocked_concepts") or [])
    prerequisite_gaps = list(state.get("prerequisite_gaps") or [])
    memory_context = list(state.get("memory_context") or [])
    objectives = list(state.get("learning_objectives") or [])
    interactions = state.get("user_message", "")
    human_feedback = state.get("human_feedback", "")
    answer_history = list(state.get("answer_history") or [])
    agent_trace = list(state.get("agent_trace") or [])
    agent_reasoning = dict(state.get("agent_reasoning") or {})

    session_summary = state.get("session_summary") or ""

    messages = list(state.get("messages") or [])
    recent_exchanges = []
    for m in messages[-8:]:
        role = m.get("role", "")
        if role in ("user", "assistant"):
            recent_exchanges.append(f"[{role}] {m.get('content', '')[:150]}")
    recent_context = "\n".join(recent_exchanges[-6:])

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
        support, interactions, human_feedback, objectives, session_summary=session_summary
    )
    difficulties += extract_memory_signals(support, memory_context)

    # ── Priority 1 : blocked concepts ────────────────────────────────────────
    for bc in blocked_concepts[:3]:
        label = (
            f"Blocage persistant : {bc['concept']} "
            f"({bc['attempts']} tentatives sans progrès"
            + (f", dernière erreur : {bc['last_error']}" if bc.get("last_error") else "")
            + ")"
        )
        if label not in difficulties:
            difficulties.insert(0, label)

    # ── Priority 2 : error patterns (LLM analysis of answer_history) ─────────
    error_patterns = _detect_error_patterns(blocked_concepts, answer_history)
    for concept, info in error_patterns.items():
        label = (
            f"Pattern d'erreur sur '{concept}' : {info['pattern']}"
            + (f" — hypothèse : {info['hypothesis']}" if info.get("hypothesis") else "")
        )
        if label not in difficulties:
            difficulties.insert(1, label)

    # ── Priority 3 : prerequisite gaps ───────────────────────────────────────
    for gap in prerequisite_gaps[:3]:
        label = f"Prérequis manquant : {gap}"
        if label not in difficulties:
            difficulties.append(label)

    # ── Priority 4 : regular weak concepts ───────────────────────────────────
    for wc in weak_concepts[:3]:
        hint = f"Concept faible détecté : {wc}"
        if hint not in difficulties:
            difficulties.append(hint)

    difficulties = list(dict.fromkeys(difficulties))[:7]

    blocked_names = [bc["concept"] for bc in blocked_concepts]

    prompt = (
        f"Tu évalues le niveau de {first_name} en '{support}'.\n"
        f"Niveau heuristique : {adj_level} (historique mémorisé : {current_level})\n"
        f"IMPORTANT : si {first_name} se déclare explicitement débutant/novice dans son message, prioritise cette déclaration sur l'historique.\n"
        f"Concepts faibles (KG) : {weak_concepts}\n"
        + (f"BLOCAGES CHRONIQUES (≥5 tentatives sans progrès) : {blocked_names}\n" if blocked_names else "")
        + (f"PATTERNS D'ERREUR DÉTECTÉS : {json.dumps(error_patterns, ensure_ascii=False)[:200]}\n" if error_patterns else "")
        + (f"PRÉREQUIS MANQUANTS : {prerequisite_gaps[:3]}\n" if prerequisite_gaps else "")
        + f"Dernier message de {first_name} : {interactions[:300]}\n"
        + (f"Échanges récents :\n{recent_context}\n" if recent_context else "")
        + f"Résumé sessions précédentes : {session_summary[:400]}\n"
        f"Mémoires récentes : {[m.get('content', '')[:80] for m in memory_context[:3]]}\n\n"
        f"Identifie le niveau réel (beginner/intermediate/advanced) et liste 3-5 difficultés spécifiques.\n"
        f'Réponds UNIQUEMENT en JSON : {{"level": "...", "difficulties": ["...", "..."], "reasoning": "..."}}'
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
                difficulties = llm_diff[:6]
            agent_reasoning["diagnostics"] = (
                f"[LLM] level={adj_level} — {data.get('reasoning','')[:80]}"
                + (f" | patterns: {list(error_patterns.keys())}" if error_patterns else "")
            )
        except Exception:
            agent_reasoning["diagnostics"] = f"[fallback] level={adj_level}"
    else:
        agent_reasoning["diagnostics"] = f"[fallback] level={adj_level}"

    _self_critique("diagnostics", f"level={adj_level}, difficulties={difficulties}", state)

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
        f"diagnostics → level={adj_level}, {len(difficulties)} difficulties"
        + (f" | {len(error_patterns)} patterns" if error_patterns else "")
        + " [P1 HITL]"
    )

    return {
        "adjusted_level": adj_level,
        "difficulties": difficulties,
        "error_patterns": error_patterns,
        "human_feedback": human_feedback,
        "agent_trace": agent_trace,
        "agent_reasoning": agent_reasoning,
    }
