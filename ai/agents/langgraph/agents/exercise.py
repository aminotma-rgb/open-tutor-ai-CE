"""ExerciseAgent — deterministic exercise generation."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

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

    blocked_names = [bc["concept"] for bc in blocked_concepts]
    already_seen = set(proposed_exercises_history)

    from ai.agents.helpers import generate_exercises, _self_critique

    # Prioritise blocked concepts — put them first in objectives
    effective_obj = blocked_names[:2] + [
        c for c in weak_concepts[:3] if c not in blocked_names
    ] + [o for o in objectives if o not in weak_concepts and o not in blocked_names]

    exercises: List[Dict[str, Any]] = generate_exercises(
        support, adj_level, effective_obj or objectives
    )

    # Filter out already-proposed skill_targets if there are enough alternatives
    novel = [e for e in exercises if e.get("skill_target", "") not in already_seen]
    if novel:
        exercises = novel

    _self_critique("exercise", f"{len(exercises)} exercises generated", state)

    # Accumulate proposed skill_targets
    new_history = list(proposed_exercises_history)
    for ex in exercises:
        target = ex.get("skill_target", "")
        if target and target not in new_history:
            new_history.append(target)

    agent_reasoning["exercise"] = (
        f"[deterministic] {len(exercises)} exercises for level={adj_level}"
        + (f", {len(blocked_names)} blocked" if blocked_names else "")
    )
    agent_trace.append(
        f"exercise → {len(exercises)} exercises"
        + (f" | blocked: {blocked_names[:2]}" if blocked_names else "")
    )

    return {
        "exercises": exercises,
        "proposed_exercises_history": new_history[-30:],
        "agent_trace": agent_trace,
        "agent_reasoning": agent_reasoning,
    }
