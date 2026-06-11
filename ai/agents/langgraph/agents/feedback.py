"""FeedbackAgent — LLM decides what to memorise + KG mastery + interrupt P3."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List

from langgraph.types import interrupt

log = logging.getLogger(__name__)


def feedback_node(state: Dict[str, Any]) -> Dict[str, Any]:
    user_id = state.get("user_id", "")
    user_name = state.get("user_name", "")
    first_name = user_name.split()[0] if user_name else "apprenant"
    support = state.get("support", "")
    exercises = list(state.get("exercises") or [])
    weak_concepts = list(state.get("weak_concepts") or [])
    adj_level = state.get("adjusted_level", "beginner")
    strategy = state.get("strategy", "")
    human_feedback = state.get("human_feedback", "")
    agent_trace = list(state.get("agent_trace") or [])
    agent_reasoning = dict(state.get("agent_reasoning") or {})

    from ai.agents.helpers import _self_critique

    # LLM decides what to memorise
    memories_to_save = _llm_decide_memories(
        support, exercises, adj_level, strategy, first_name
    )

    # Self-critique
    _self_critique("feedback", f"saving {len(memories_to_save)} memories", state)

    # Human-in-the-Loop P3 — ask learner to validate memory persistence
    if not human_feedback:
        summary_text = (
            f"Niveau : {adj_level}. Exercices : {len(exercises)}. {strategy[:100]}"
        )
        human_feedback = interrupt(
            {
                "checkpoint": "P3",
                "message": f"{first_name}, voici ton résumé de session : {summary_text}. Sauvegarder en mémoire ? (oui/non)",
                "memories_preview": [
                    m.get("content", "")[:80] for m in memories_to_save[:3]
                ],
            }
        )
        if not isinstance(human_feedback, str):
            human_feedback = ""

    # Persist only if learner approves (empty = implicit yes)
    should_save = not human_feedback or human_feedback.lower().strip() in (
        "oui",
        "yes",
        "y",
        "o",
        "",
    )
    if should_save and user_id:
        _persist_memories(user_id, support, memories_to_save)
        _update_kg_mastery(user_id, support, weak_concepts, exercises)

    agent_reasoning["feedback"] = (
        f"[LLM] {len(memories_to_save)} memories, saved={should_save}"
    )
    agent_trace.append(
        f"feedback → {len(memories_to_save)} memories saved={should_save} [P3 HITL]"
    )

    return {
        "next_agent": "END",
        "human_feedback": human_feedback,
        "agent_trace": agent_trace,
        "agent_reasoning": agent_reasoning,
    }


def _llm_decide_memories(
    support: str,
    exercises: list,
    level: str,
    strategy: str,
    first_name: str = "apprenant",
) -> List[Dict[str, Any]]:
    from ai.llm.service import call_llm

    prompt = (
        f"Session d'apprentissage terminée pour {first_name} : support={support}, niveau={level}\n"
        f"Stratégie suivie par {first_name} : {strategy[:200]}\n"
        f"Exercices générés : {len(exercises)}\n\n"
        f"Décide quelles informations mémoriser sur {first_name} (2-4 entrées, importance : high/medium).\n"
        f"Réponds en JSON :\n"
        f'[{{"type": "behavioral|episodic|procedural", "content": "...", "importance": "high|medium"}}]'
    )
    text = call_llm(prompt, max_tokens=400)
    if text and "[" in text:
        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            memories = json.loads(text[start:end])
            return [m for m in memories if m.get("importance") in ("high", "medium")]
        except Exception:
            pass

    # Fallback template
    return [
        {
            "type": "procedural",
            "content": f"Session {support} niveau {level} : {strategy[:100]}",
            "importance": "medium",
        }
    ]


def _persist_memories(
    user_id: str, support: str, memories: List[Dict[str, Any]]
) -> None:
    try:
        from data.database import get_db
        from data.models.memory import Memory

        db = next(get_db())
        try:
            for m in memories:
                db.add(
                    Memory(
                        id=str(uuid.uuid4()),
                        user_id=user_id,
                        memory_type=m.get("type", "procedural"),
                        content=m.get("content", ""),
                        memory_metadata={
                            "support_id": support,
                            "importance": m.get("importance", "medium"),
                        },
                    )
                )
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        log.warning("Memory persistence failed: %s", exc)


def _update_kg_mastery(
    user_id: str, support: str, weak_concepts: list, exercises: list
) -> None:
    if not weak_concepts:
        return
    try:
        from data.database import get_db
        from ai.knowledge_graph.service import KnowledgeGraphService

        exercised = {
            e.get("skill_target", "") for e in exercises if e.get("skill_target")
        }
        db = next(get_db())
        try:
            kg_svc = KnowledgeGraphService()
            for concept in weak_concepts:
                delta = 0.05 if concept in exercised else 0.02
                kg_svc.update_mastery(db, user_id, concept, support, delta)
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        log.warning("KG mastery update failed: %s", exc)
