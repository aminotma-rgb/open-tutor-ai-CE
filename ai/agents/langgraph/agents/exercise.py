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
    objectives = list(state.get("learning_objectives") or [])
    strategy_decisions = list(state.get("strategy_decisions") or [])
    agent_trace = list(state.get("agent_trace") or [])
    agent_reasoning = dict(state.get("agent_reasoning") or {})
    tool_selection_log = list(state.get("tool_selection_log") or [])

    from ai.agents.helpers import generate_exercises, _self_critique

    exercises: List[Dict[str, Any]] = []
    used_react = False

    try:
        exercises, log_entries = _react_exercises(
            support, adj_level, weak_concepts, objectives, strategy_decisions
        )
        tool_selection_log.extend(log_entries)
        used_react = bool(exercises)
    except Exception as exc:
        log.warning("ExerciseAgent ReAct failed, using fallback: %s", exc)

    if not exercises:
        effective_obj = weak_concepts[:3] + [
            o for o in objectives if o not in weak_concepts
        ]
        exercises = generate_exercises(support, adj_level, effective_obj or objectives)

    _self_critique("exercise", f"{len(exercises)} exercises generated", state)

    tag = "[LLM ReAct]" if used_react else "[fallback]"
    agent_reasoning["exercise"] = (
        f"{tag} {len(exercises)} exercises for level={adj_level}"
    )
    agent_trace.append(f"exercise → {len(exercises)} exercises {tag}")

    return {
        "exercises": exercises,
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
        f"Objectifs : {objectives}\n"
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
