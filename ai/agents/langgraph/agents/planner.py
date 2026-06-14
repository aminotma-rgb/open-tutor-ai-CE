"""PlannerAgent — LLM strategy + strategy rotation to avoid repeating failed approaches."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

log = logging.getLogger(__name__)

TEACHING_STRATEGIES = [
    "exemple concret du quotidien",
    "analogie visuelle",
    "décomposition en micro-étapes",
    "contre-exemple",
    "reformulation par l'apprenant",
    "exercice guidé pas à pas",
    "exercice autonome",
]


def _forbidden_strategies(
    concept: str,
    teaching_strategies_used: Dict[str, List[str]],
    memory_context: List[Dict[str, Any]],
) -> List[str]:
    """Return strategies already tried on this concept (intra-session + inter-sessions)."""
    forbidden = list(teaching_strategies_used.get(concept, []))
    for mem in memory_context:
        meta = mem.get("memory_metadata") or mem.get("metadata") or {}
        if (
            meta.get("type") == "strategy_outcome"
            and meta.get("concept") == concept
            and meta.get("outcome") in ("failed", "partial")
        ):
            strat = meta.get("strategy", "")
            if strat and strat not in forbidden:
                forbidden.append(strat)
    return forbidden


def _pick_strategy(concept: str, forbidden: List[str]) -> str:
    """Return the first catalog strategy not yet tried for this concept."""
    for strat in TEACHING_STRATEGIES:
        if strat not in forbidden:
            return strat
    # All strategies exhausted — cycle back to first
    return TEACHING_STRATEGIES[0]


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
    teaching_strategies_used = dict(state.get("teaching_strategies_used") or {})

    from ai.agents.helpers import plan_learning_strategy, _self_critique
    from ai.llm.service import call_llm

    # Human feedback as pedagogical constraint
    if human_feedback and human_feedback.lower().strip() not in ("oui", "yes", "y", "o", ""):
        difficulties = list(difficulties) + [f"Human feedback : {human_feedback[:100]}"]

    # Deterministic baseline
    decisions = plan_learning_strategy(
        support, adj_level, difficulties, human_feedback, memory_context
    )

    # ── Strategy rotation for blocked concepts ────────────────────────────────
    blocked_names = [bc["concept"] for bc in blocked_concepts]
    strategy_directives: List[str] = []
    updated_strategies = dict(teaching_strategies_used)

    for concept in blocked_names[:3]:
        forbidden = _forbidden_strategies(concept, teaching_strategies_used, memory_context)
        chosen = _pick_strategy(concept, forbidden)
        strategy_directives.append(
            f"Pour '{concept}' : utilise UNIQUEMENT '{chosen}' "
            f"(approches déjà tentées et échouées : {forbidden or 'aucune'})"
        )
        concept_list = list(updated_strategies.get(concept, []))
        if chosen not in concept_list:
            concept_list.append(chosen)
        updated_strategies[concept] = concept_list

    # ── Build LLM prompt ──────────────────────────────────────────────────────
    rag_summary = " ".join(d.get("content", "")[:100] for d in rag_docs[:3])
    verification_note = (
        f"\nFeedback vérificateur (retry #{n_retries.get('planner', 0)}) : {verification_feedback}"
        if verification_feedback else ""
    )
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
            f"tout nouveau concept.\n"
        )
    strategy_rotation_note = (
        "\nSTRATÉGIES IMPOSÉES PAR ROTATION :\n" + "\n".join(f"  - {d}" for d in strategy_directives)
        if strategy_directives else ""
    )

    prompt = (
        f"Génère un plan pédagogique personnalisé pour {first_name} : support={support}, niveau={adj_level}.\n"
        f"Difficultés de {first_name} : {difficulties}\n"
        f"Concepts faibles : {weak_concepts}\n"
        + (f"Concepts bloqués (priorité maximale) : {blocked_names}\n" if blocked_names else "")
        + f"Objectifs : {objectives}\n"
        f"Sources RAG : {rag_summary[:300]}\n"
        f"{blocked_note}"
        f"{strategy_rotation_note}"
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
            agent_reasoning["planner"] = (
                f"[LLM] {data.get('reasoning', '')[:80]}"
                + (f" | rotation: {strategy_directives}" if strategy_directives else "")
            )
        except Exception:
            agent_reasoning["planner"] = "[fallback] deterministic strategy"
    else:
        agent_reasoning["planner"] = "[fallback] deterministic strategy"

    _self_critique("planner", f"strategy: {strategy[:200]}", state)

    n_retries["planner"] = n_retries.get("planner", 0) + 1
    agent_trace.append(
        f"planner → {len(decisions)} decisions (retry #{n_retries['planner']})"
        + (f" | rotation: {list(updated_strategies.keys())}" if strategy_directives else "")
    )

    return {
        "strategy": strategy,
        "strategy_decisions": decisions,
        "teaching_strategies_used": updated_strategies,
        "n_retries": n_retries,
        "agent_trace": agent_trace,
        "agent_reasoning": agent_reasoning,
    }
