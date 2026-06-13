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
    answer_history = list(state.get("answer_history") or [])
    agent_trace = list(state.get("agent_trace") or [])
    agent_reasoning = dict(state.get("agent_reasoning") or {})

    from ai.agents.helpers import _self_critique

    # LLM decides what to memorise — enriched with session performance stats
    memories_to_save = _llm_decide_memories(
        support, exercises, adj_level, strategy, first_name,
        answer_history=answer_history,
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
        # Use full session answer history for aggregated mastery update
        _update_kg_mastery(user_id, support, weak_concepts, exercises, answer_history=answer_history)

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
    answer_history: list = None,
) -> List[Dict[str, Any]]:
    from ai.llm.service import call_llm

    history = answer_history or []
    correct_count = sum(1 for v in history if v.get("correct") is True)
    wrong_count = sum(1 for v in history if v.get("correct") is False)
    session_perf = (
        f"{correct_count} bonne(s) réponse(s), {wrong_count} erreur(s) sur {len(history)} échanges vérifiés"
        if history else "aucun échange vérifié cette session"
    )

    prompt = (
        f"Session d'apprentissage terminée pour {first_name} : support={support}, niveau={level}\n"
        f"Performance de la session : {session_perf}\n"
        f"Stratégie suivie par {first_name} : {strategy[:200]}\n"
        f"Exercices générés : {len(exercises)}\n\n"
        f"Décide quelles informations mémoriser sur {first_name} (2-4 entrées, importance : high/medium).\n"
        f"Inclus la performance ({session_perf}) dans une entrée episodic si notable.\n"
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
                content = m.get("content", "")
                mem_type = m.get("type", "procedural")
                # Skip exact duplicates (same user + type + content)
                exists = (
                    db.query(Memory)
                    .filter(
                        Memory.user_id == user_id,
                        Memory.memory_type == mem_type,
                        Memory.content == content,
                    )
                    .first()
                )
                if exists:
                    continue
                db.add(
                    Memory(
                        id=str(uuid.uuid4()),
                        user_id=user_id,
                        memory_type=mem_type,
                        content=content,
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
    user_id: str,
    support: str,
    weak_concepts: list,
    exercises: list,
    answer_history: List[Dict[str, Any]] | None = None,
) -> None:
    """Update KG mastery using the full session answer history.

    All verified exchanges in the session contribute:
      correct → +0.10, wrong → -0.05, uncertain → +0.02 per exchange.
    The aggregated delta is applied once to each exercised concept.
    Concepts present but not exercised receive a small exposure bump (+0.02).
    """
    exercised = {
        e.get("skill_target", "") for e in exercises if e.get("skill_target")
    }
    all_concepts = list(dict.fromkeys(list(weak_concepts) + list(exercised)))
    if not all_concepts:
        return

    history = answer_history or []

    # Aggregate deltas across all session verdicts
    session_delta = 0.0
    last_error: str | None = None
    for v in history:
        v_correct = v.get("correct")
        if v_correct is True:
            session_delta += 0.10
        elif v_correct is False:
            session_delta -= 0.05
            # Keep the most recent wrong answer as last_error
            if v.get("learner_answer") and v.get("correct_answer"):
                last_error = (
                    f"{v['learner_answer']} → attendu : {v['correct_answer']}"
                )
        else:
            session_delta += 0.02

    # No verdicts this session → neutral exposure bump
    if not history:
        session_delta = 0.05

    try:
        from data.database import get_db
        from ai.knowledge_graph.service import KnowledgeGraphService

        db = next(get_db())
        try:
            kg_svc = KnowledgeGraphService()
            for concept in all_concepts:
                delta = session_delta if concept in exercised else 0.02
                kg_svc.update_mastery(
                    db, user_id, concept, support, delta,
                    last_error=last_error if concept in exercised else None,
                )
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        log.warning("KG mastery update failed: %s", exc)
