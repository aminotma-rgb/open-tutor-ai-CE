"""ExerciseAgent — create_react_agent with 6 tools, deterministic fallback."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

log = logging.getLogger(__name__)


def exercise_node(state: Dict[str, Any]) -> Dict[str, Any]:
    support = state.get("support", "")
    adj_level = state.get("adjusted_level") or state.get("current_level", "beginner")
    weak_concepts = list(state.get("weak_concepts") or [])
    blocked_concepts = list(state.get("blocked_concepts") or [])
    objectives = list(state.get("learning_objectives") or [])
    strategy_decisions = list(state.get("strategy_decisions") or [])
    proposed_exercises_history = list(state.get("proposed_exercises_history") or [])
    agent_trace = list(state.get("agent_trace") or [])
    agent_reasoning = dict(state.get("agent_reasoning") or {})
    tool_selection_log = list(state.get("tool_selection_log") or [])

    blocked_names = [bc["concept"] for bc in blocked_concepts]
    already_seen = set(proposed_exercises_history)

    from ai.agents.helpers import generate_exercises, _self_critique

    exercises: List[Dict[str, Any]] = []
    used_react = False

    try:
        exercises, log_entries = _react_exercises(
            support, adj_level, weak_concepts, objectives, strategy_decisions,
            blocked_names=blocked_names,
            already_seen=already_seen,
        )
        tool_selection_log.extend(log_entries)
        used_react = bool(exercises)
    except Exception as exc:
        log.warning("ExerciseAgent ReAct failed, using fallback: %s", exc)

    if not exercises:
        # Prioritise blocked concepts in fallback — put them first
        effective_obj = blocked_names[:2] + [
            c for c in weak_concepts[:3] if c not in blocked_names
        ] + [o for o in objectives if o not in weak_concepts and o not in blocked_names]
        exercises = generate_exercises(support, adj_level, effective_obj or objectives)

    # Filter out already-proposed skill_targets if there are enough alternatives
    novel = [e for e in exercises if e.get("skill_target", "") not in already_seen]
    if novel:
        exercises = novel

    _self_critique("exercise", f"{len(exercises)} exercises generated", state)

    # Accumulate proposed skill_targets — LangGraph persists this across session turns
    new_history = list(proposed_exercises_history)
    for ex in exercises:
        target = ex.get("skill_target", "")
        if target and target not in new_history:
            new_history.append(target)

    tag = "[LLM ReAct]" if used_react else "[fallback]"
    agent_reasoning["exercise"] = (
        f"{tag} {len(exercises)} exercises for level={adj_level}"
        + (f", {len(blocked_names)} blocked" if blocked_names else "")
    )
    agent_trace.append(
        f"exercise → {len(exercises)} exercises {tag}"
        + (f" | blocked: {blocked_names[:2]}" if blocked_names else "")
    )

    return {
        "exercises": exercises,
        "proposed_exercises_history": new_history[-30:],
        "tool_selection_log": tool_selection_log,
        "agent_trace": agent_trace,
        "agent_reasoning": agent_reasoning,
    }


def _react_exercises(
    support: str,
    level: str,
    weak_concepts: list,
    objectives: list,
    decisions: list,
    blocked_names: list = None,
    already_seen: set = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Run create_react_agent with 6 tools. Raises on any failure."""
    from langgraph.prebuilt import create_react_agent
    from ai.llm.service import get_langchain_llm
    from ai.tools.live_code_evaluation import live_code_evaluation
    from ai.tools.sql_evaluator import sql_evaluator
    from ai.tools.math_evaluator import math_evaluator
    from ai.tools.generate_chart import generate_chart
    from ai.tools.grammar_checker import grammar_checker
    from ai.tools.search_web import search_web

    blocked_names = blocked_names or []
    already_seen = already_seen or set()

    llm = get_langchain_llm()
    tools = [
        live_code_evaluation,
        sql_evaluator,
        math_evaluator,
        generate_chart,
        grammar_checker,
        search_web,
    ]

    agent = create_react_agent(llm, tools)
    prompt = (
        f"Génère 3 exercices pédagogiques pour : support={support}, niveau={level}.\n"
        f"Concepts faibles : {weak_concepts}\n"
        + (
            f"CONCEPTS BLOQUÉS (approche radicalement différente obligatoire — analogie, micro-étapes, "
            f"ne pas répéter les mêmes types d'exercices) : {blocked_names}\n"
            if blocked_names else ""
        )
        + (
            f"Skill-targets déjà proposés cette session (à éviter) : {list(already_seen)[-10:]}\n"
            if already_seen else ""
        )
        + f"Objectifs : {objectives}\n"
        f"Stratégie : {[d.get('action', '') for d in decisions[:3]]}\n"
        f"Utilise les outils appropriés selon le type d'exercice.\n"
        f"Retourne une liste JSON :\n"
        f'[{{"id":1,"type":"...","question":"...","hint":"...","answer":"...","skill_target":"..."}}]'
    )

    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    messages = result.get("messages", [])

    for msg in reversed(messages):
        content = getattr(msg, "content", "") if hasattr(msg, "content") else str(msg)
        if "[" in content and "]" in content:
            try:
                start = content.index("[")
                end = content.rindex("]") + 1
                exercises = json.loads(content[start:end])
                if isinstance(exercises, list) and exercises:
                    log_entries = [
                        {
                            "agent": "exercise",
                            "tool": "react_agent",
                            "rationale": "LLM selected tools autonomously",
                            "result": f"{len(exercises)} exercises",
                        }
                    ]
                    return exercises, log_entries
            except Exception:
                pass

    return [], []
