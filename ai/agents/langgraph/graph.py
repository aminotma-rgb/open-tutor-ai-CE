"""Build and expose the TutorGraph singleton.

build_graph(use_checkpointer=True) compiles the StateGraph with SqliteSaver.
_fallback_response() provides a degraded-mode path when LangGraph is unavailable.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_CHECKPOINT_DB = os.getenv(
    "LANGGRAPH_CHECKPOINT_DB",
    os.path.join(_PROJECT_ROOT, "data", "langgraph_checkpoints.sqlite"),
)


def build_graph(use_checkpointer: bool = True):
    """Compile the 8-node TutorGraph StateGraph."""
    from langgraph.graph import StateGraph, END

    from ai.agents.langgraph.state import TutorGraphState
    from ai.agents.langgraph.orchestrator import (
        orchestrator_node,
        route_after_orchestrator,
    )
    from ai.agents.langgraph.agents.memory import memory_node
    from ai.agents.langgraph.agents.knowledge import knowledge_node
    from ai.agents.langgraph.agents.diagnostics import diagnostics_node
    from ai.agents.langgraph.agents.planner import planner_node
    from ai.agents.langgraph.agents.exercise import exercise_node
    from ai.agents.langgraph.agents.verifier import verifier_node
    from ai.agents.langgraph.agents.feedback import feedback_node

    builder = StateGraph(TutorGraphState)

    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("memory", memory_node)
    builder.add_node("knowledge", knowledge_node)
    builder.add_node("diagnostics", diagnostics_node)
    builder.add_node("planner", planner_node)
    builder.add_node("exercise", exercise_node)
    builder.add_node("verifier", verifier_node)
    builder.add_node("feedback", feedback_node)

    builder.set_entry_point("orchestrator")

    for agent in [
        "memory",
        "knowledge",
        "diagnostics",
        "planner",
        "exercise",
        "verifier",
        "feedback",
    ]:
        builder.add_edge(agent, "orchestrator")

    builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "memory": "memory",
            "knowledge": "knowledge",
            "diagnostics": "diagnostics",
            "planner": "planner",
            "exercise": "exercise",
            "verifier": "verifier",
            "feedback": "feedback",
            "END": END,
        },
    )

    checkpointer = None
    if use_checkpointer:
        try:
            import sqlite3
            from langgraph.checkpoint.sqlite import SqliteSaver

            # from_conn_string() returns a context manager — use a raw connection instead
            _conn = sqlite3.connect(_CHECKPOINT_DB, check_same_thread=False)
            checkpointer = SqliteSaver(_conn)
            log.debug("SqliteSaver checkpointer at %s", _CHECKPOINT_DB)
        except Exception as exc:
            log.warning("SqliteSaver unavailable, falling back to MemorySaver: %s", exc)
            from langgraph.checkpoint.memory import MemorySaver

            checkpointer = MemorySaver()

    return builder.compile(checkpointer=checkpointer)


def _fallback_response(state: dict) -> dict:
    """Degraded mode — run helpers directly, no LangGraph, no LLM."""
    from ai.agents.helpers import (
        assess_current_level,
        detect_difficulties,
        generate_exercises,
        plan_learning_strategy,
    )

    support = state.get("support", "")
    level = state.get("current_level", "beginner")
    objectives = state.get("learning_objectives") or []
    interactions = state.get("user_message", "")
    feedback = state.get("human_feedback", "")

    adj_level = assess_current_level(level, interactions, feedback)
    difficulties = detect_difficulties(support, interactions, feedback, objectives)
    decisions = plan_learning_strategy(support, adj_level, difficulties, feedback, [])
    exercises = generate_exercises(support, adj_level, objectives)

    return {
        **state,
        "adjusted_level": adj_level,
        "difficulties": difficulties,
        "strategy_decisions": decisions,
        "strategy": "; ".join(d.get("action", "") for d in decisions),
        "exercises": exercises,
        "verification": {"verdict": "skipped", "reason": "fallback mode"},
        "agent_trace": ["[FALLBACK] LangGraph unavailable"],
    }


# ── Singleton ─────────────────────────────────────────────────────────────────

try:
    tutor_graph = build_graph(use_checkpointer=True)
except Exception as _exc:
    log.error("Failed to build tutor_graph singleton: %s", _exc)
    tutor_graph = None
