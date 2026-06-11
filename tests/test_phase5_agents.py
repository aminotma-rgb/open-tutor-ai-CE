"""Phase 5 — Adaptive Tutor Engine tests (29 tests).

All tests are in-memory: no LLM calls, no DB, no ChromaDB.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from ai.agents.helpers import (
    assess_current_level,
    detect_difficulties,
    extract_memory_signals,
    generate_exercises,
    is_text_supported,
    plan_learning_strategy,
)
from ai.agents.langgraph.orchestrator import _route, orchestrator_node
from ai.agents.langgraph.graph import _fallback_response, build_graph
from ai.agents.langgraph.state import TutorGraphState


# ── helpers — assess_current_level ───────────────────────────────────────────


def test_assess_level_upgrade():
    assert (
        assess_current_level("beginner", "excellent perfect work", "") == "intermediate"
    )


def test_assess_level_downgrade():
    assert (
        assess_current_level("advanced", "confused lost don't understand", "")
        == "intermediate"
    )


def test_assess_level_stable():
    assert (
        assess_current_level("intermediate", "okay normal session", "")
        == "intermediate"
    )


def test_assess_level_unknown_returns_same():
    assert assess_current_level("unknown", "", "") == "unknown"


# ── helpers — detect_difficulties ────────────────────────────────────────────


def test_detect_difficulties_error_signal():
    result = detect_difficulties("python", "there was an error", "", [])
    assert any("error" in d.lower() or "Error" in d for d in result)


def test_detect_difficulties_objective_gap():
    result = detect_difficulties("python", "", "", ["loops", "functions"])
    assert any("loops" in d or "functions" in d for d in result)


def test_detect_difficulties_max_5():
    result = detect_difficulties(
        "python", "error confused don't understand why", "", ["a", "b", "c", "d"]
    )
    assert len(result) <= 5


def test_detect_difficulties_no_interaction():
    result = detect_difficulties("sql", "", "", [])
    assert any("No prior" in d for d in result)


# ── helpers — extract_memory_signals ─────────────────────────────────────────


def test_extract_memory_signals_negative():
    memories = [{"content": "User struggled with JOIN queries"}]
    assert len(extract_memory_signals("sql", memories)) == 1


def test_extract_memory_signals_positive_ignored():
    memories = [{"content": "User mastered loops perfectly"}]
    assert len(extract_memory_signals("python", memories)) == 0


def test_extract_memory_signals_empty():
    assert extract_memory_signals("python", []) == []


# ── helpers — generate_exercises ─────────────────────────────────────────────


def test_generate_exercises_count():
    assert (
        len(generate_exercises("python", "beginner", ["loops", "functions"], count=3))
        == 3
    )


def test_generate_exercises_required_fields():
    for ex in generate_exercises("python", "intermediate", ["decorators"]):
        assert "id" in ex and "question" in ex and "skill_target" in ex


def test_generate_exercises_cycles_objectives():
    assert len(generate_exercises("python", "beginner", ["loops"], count=4)) == 4


def test_generate_exercises_no_objectives():
    exs = generate_exercises("sql", "beginner", [])
    assert len(exs) > 0


# ── helpers — plan_learning_strategy ─────────────────────────────────────────


def test_plan_strategy_returns_list():
    result = plan_learning_strategy("python", "beginner", ["error in loops"], "", [])
    assert isinstance(result, list) and len(result) >= 1


def test_plan_strategy_max_5():
    result = plan_learning_strategy(
        "python", "beginner", ["a", "b", "c", "d", "e"], "", []
    )
    assert len(result) <= 5


# ── helpers — is_text_supported ──────────────────────────────────────────────


def test_is_text_supported_high_overlap():
    assert is_text_supported(
        "python loops list", "python loops list comprehension", threshold=0.5
    )


def test_is_text_supported_low_overlap():
    assert not is_text_supported(
        "quantum physics relativity", "python loops list", threshold=0.5
    )


def test_is_text_supported_empty_corpus():
    assert not is_text_supported("anything", "", threshold=0.5)


# ── TutorGraphState ───────────────────────────────────────────────────────────


def test_state_accepts_required_fields():
    state: TutorGraphState = {
        "user_id": "u1",
        "support": "python",
        "current_level": "beginner",
    }
    assert state["support"] == "python"


# ── Orchestrator — _route ─────────────────────────────────────────────────────


def test_route_empty_goes_to_memory():
    assert _route({}) == "memory"


def test_route_after_memory_goes_to_knowledge():
    assert _route({"memory_context": [{}]}) == "knowledge"


def test_route_after_knowledge_goes_to_diagnostics():
    assert (
        _route({"memory_context": [{}], "knowledge_graph": {"nodes": []}})
        == "diagnostics"
    )


def test_route_full_pipeline_goes_to_feedback():
    state = {
        "memory_context": [{}],
        "knowledge_graph": {"nodes": []},
        "adjusted_level": "beginner",
        "strategy": "study loops",
        "exercises": [{"id": 1}],
        "verification": {"verdict": "supported"},
    }
    assert _route(state) == "feedback"


def test_route_needs_review_retries_planner():
    state = {
        "memory_context": [{}],
        "knowledge_graph": {"nodes": []},
        "adjusted_level": "beginner",
        "strategy": "study loops",
        "exercises": [{"id": 1}],
        "verification": {"verdict": "needs_review"},
        "n_retries": {"planner": 0},
    }
    assert _route(state) == "planner"


# ── Orchestrator — orchestrator_node ─────────────────────────────────────────


def test_orchestrator_increments_iteration():
    state = {
        "iteration": 2,
        "agent_trace": [],
        "agent_reasoning": {},
        "support": "python",
    }
    with patch("ai.agents.langgraph.orchestrator._llm_route", return_value=None):
        result = orchestrator_node(state)
    assert result["iteration"] == 3


def test_orchestrator_stops_at_max_iterations():
    state = {
        "iteration": 16,
        "agent_trace": [],
        "agent_reasoning": {},
        "support": "python",
    }
    result = orchestrator_node(state)
    assert result["next_agent"] == "END"


# ── Graph ─────────────────────────────────────────────────────────────────────


def test_build_graph_no_checkpointer():
    graph = build_graph(use_checkpointer=False)
    assert graph is not None


def test_fallback_response_has_exercises():
    state = {
        "user_id": "u1",
        "support": "python",
        "current_level": "beginner",
        "learning_objectives": ["loops"],
        "user_message": "",
        "human_feedback": "",
    }
    result = _fallback_response(state)
    assert len(result["exercises"]) > 0
    assert result["verification"]["verdict"] == "skipped"
    assert any("[FALLBACK]" in t for t in result["agent_trace"])
