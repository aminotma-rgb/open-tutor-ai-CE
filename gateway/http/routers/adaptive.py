"""Adaptive tutor router — /api/v1/adaptive/*."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from data.database import get_db
from data.models import User
from gateway.http.dependencies import get_current_user
from ai.providers.service import ProvidersService

log = logging.getLogger(__name__)

try:
    from ai.agents.langgraph.graph import tutor_graph, _fallback_response
    from ai.context_manager.service import ContextManager
except Exception as _import_exc:
    log.warning("Phase 6 adaptive imports unavailable: %s", _import_exc)
    tutor_graph = None  # type: ignore[assignment]
    _fallback_response = None  # type: ignore[assignment]
    ContextManager = None  # type: ignore[assignment]

router = APIRouter(prefix="/adaptive", tags=["adaptive"])


# ── Request / Response models ─────────────────────────────────────────────────


class AdaptivePlanRequest(BaseModel):
    support: str
    current_level: str = "intermediate"
    language: str = "en"
    user_message: str = ""
    recent_interactions: List[Any] = Field(default_factory=list)
    feedback_comments: List[str] = Field(default_factory=list)
    learning_objectives: List[str] = Field(default_factory=list)
    preferred_exercise_types: List[str] = Field(default_factory=list)
    session_id: Optional[str] = None


class AdaptivePlanResponse(BaseModel):
    session_id: str
    support: str
    adjusted_level: str
    exercises: List[Dict[str, Any]]
    strategy: Any
    verification: Dict[str, Any]
    agent_trace: List[str]


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post(
    "/plan", response_model=AdaptivePlanResponse, status_code=status.HTTP_200_OK
)
def run_adaptive_plan(
    body: AdaptivePlanRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Run a full adaptive tutoring session through the LangGraph pipeline."""
    session_id = body.session_id or str(uuid.uuid4())

    # Pre-load context (failures are caught silently)
    rag_docs: List[Dict[str, Any]] = []
    session_summary = ""
    try:
        ctx = ContextManager().build_agent_context(
            user_id=current_user.id,
            support=body.support,
            query=body.user_message or body.support,
            db=db,
        )
        rag_docs = ctx.rag_docs
        session_summary = ctx.session_summary
    except Exception as exc:
        log.warning("Context pre-load failed, continuing with empty context: %s", exc)

    initial_state: Dict[str, Any] = {
        "user_id": current_user.id,
        "user_name": getattr(current_user, "name", "") or "",
        "support": body.support,
        "current_level": body.current_level,
        "language": body.language,
        "learning_objectives": body.learning_objectives,
        "user_message": body.user_message,
        "rag_docs": rag_docs,
        "session_summary": session_summary,
        "human_feedback": (
            "; ".join(body.feedback_comments) if body.feedback_comments else ""
        ),
        "iteration": 0,
        "n_retries": {},
        "agent_trace": [],
        "agent_reasoning": {},
    }

    try:
        if tutor_graph is None:
            final_state = _fallback_response(initial_state)
        else:
            # recursion_limit must comfortably exceed orchestrator.MAX_ITERATIONS * 2
            # (orchestrator + agent per iteration) — LangGraph's default of 25 is
            # lower than that worst case (15 * 2 = 30) and raises GraphRecursionError
            # before the orchestrator's own ceiling ever gets a chance to fire.
            config = {"configurable": {"thread_id": session_id}, "recursion_limit": 50}
            final_state = tutor_graph.invoke(initial_state, config=config)
    except Exception as exc:
        log.error("tutor_graph.invoke failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Adaptive pipeline error: {exc}",
        )

    return AdaptivePlanResponse(
        session_id=session_id,
        support=body.support,
        adjusted_level=final_state.get("adjusted_level") or body.current_level,
        exercises=final_state.get("exercises") or [],
        strategy=final_state.get("strategy_decisions")
        or final_state.get("strategy")
        or [],
        verification=final_state.get("verification") or {},
        agent_trace=final_state.get("agent_trace") or [],
    )


@router.get("/session/{session_id}")
def get_adaptive_session(
    session_id: str,
    current_user: User = Depends(get_current_user),
):
    """Retrieve the last saved checkpoint for a session."""
    try:
        if tutor_graph is None or not hasattr(tutor_graph, "checkpointer"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Checkpointer unavailable",
            )

        config = {"configurable": {"thread_id": session_id}}
        checkpoint = tutor_graph.get_state(config)

        if checkpoint is None or checkpoint.values is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )

        state = checkpoint.values
        if state.get("user_id") and state["user_id"] != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found",
            )

        return {
            "session_id": session_id,
            "support": state.get("support"),
            "adjusted_level": state.get("adjusted_level"),
            "exercises": state.get("exercises") or [],
            "agent_trace": state.get("agent_trace") or [],
            "verification": state.get("verification") or {},
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {exc}",
        )


# ── /adaptive/chat — agentic tutor as default chat endpoint ──────────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class AdaptiveChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "tutor"
    stream: bool = True
    support: Optional[str] = None
    current_level: str = "intermediate"
    language: str = "fr"
    session_id: Optional[str] = None
    learning_objectives: List[str] = Field(default_factory=list)
    # System prompt built from support details in the frontend (topic, objectives, level…)
    system_prompt: Optional[str] = None
    # Full name of the learner for personalised tutor exchanges
    user_name: Optional[str] = None


def _extract_support(messages: List[ChatMessage], fallback: Optional[str]) -> str:
    if fallback:
        return fallback
    all_text = " ".join(m.content for m in messages).lower()
    for kw in ("python", "sql", "javascript", "java", "math", "physics", "history"):
        if kw in all_text:
            return kw
    return "general"


def _persist_adjusted_level(
    user_id: str,
    support: str,
    adjusted_level: str,
    difficulties: List[str],
) -> None:
    """Upsert a behavioral memory that stores the evaluated level for user+support.

    Overwrite the previous entry (same user + support + memory_type=behavioral
    with metadata.adjusted_level) so only the latest level is kept — avoids
    duplicating the rows Lina's session already had.
    """
    try:
        from data.database import get_db
        from data.models.memory import Memory

        db = next(get_db())
        try:
            # SQLite JSON path operators are unreliable — filter Python-side
            all_behavioral = (
                db.query(Memory)
                .filter(
                    Memory.user_id == user_id,
                    Memory.memory_type == "behavioral",
                )
                .all()
            )
            matches = [
                m for m in all_behavioral
                if m.memory_metadata.get("support") == support
            ]
            content = f"Niveau évalué : {adjusted_level} pour '{support}'"
            if matches:
                # Preserve previously identified difficulties if the current session found none
                prev_difficulties = matches[0].memory_metadata.get("difficulties") or []
                merged_difficulties = difficulties[:5] if difficulties else prev_difficulties
            else:
                merged_difficulties = difficulties[:5]
            metadata = {
                "support": support,
                "adjusted_level": adjusted_level,
                "difficulties": merged_difficulties,
            }
            if matches:
                # Update the first match, delete any duplicates
                existing = matches[0]
                existing.content = content
                existing.memory_metadata = metadata
                for dup in matches[1:]:
                    db.delete(dup)
            else:
                db.add(
                    Memory(
                        id=str(uuid.uuid4()),
                        user_id=user_id,
                        memory_type="behavioral",
                        content=content,
                        memory_metadata=metadata,
                    )
                )
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        log.warning("Failed to persist adjusted_level: %s", exc)


# ── Validation phrases that falsely confirm a wrong answer ───────────────────
_VALIDATION_PHRASES = (
    "c'est correct", "c'est exact", "c'est juste", "exactement", "bravo",
    "très bien", "parfait", "excellent", "bonne réponse", "bien joué",
    "that's correct", "that's right", "exactly", "well done", "great job",
    "good answer", "correct", "right answer",
)


def _check_tutor_response(content: str, state: dict) -> Optional[Dict[str, Any]]:
    """Detect rule violations in the tutor's generated response.

    Returns a violation dict {rule, detail} or None if the response is clean.
    Checks three rules in priority order:
      1. False validation — tutor praises a wrong answer
      2. New concept while blocked — tutor ignores ABSOLUTE RULE
      3. Off-topic — tutor ignores all known difficulties
    """
    if not content:
        return None

    lower = content.lower()

    # Rule 1 — false validation of a wrong answer
    verdict = state.get("learner_answer_verdict") or {}
    if verdict.get("correct") is False:
        for phrase in _VALIDATION_PHRASES:
            if phrase in lower:
                return {
                    "rule": "false_validation",
                    "detail": (
                        f"La réponse de l'apprenant était fausse "
                        f"(attendu : {verdict.get('correct_answer', '?')}) "
                        f"mais le tuteur a validé avec '{phrase}'."
                    ),
                    "correct_answer": verdict.get("correct_answer", ""),
                    "learner_answer": verdict.get("learner_answer", ""),
                }

    # Rule 2 — new concept introduced while blocked concepts exist
    blocked_concepts = state.get("blocked_concepts") or []
    if blocked_concepts:
        blocked_names = {bc["concept"].lower() for bc in blocked_concepts}
        known_concepts = (
            set(state.get("weak_concepts") or [])
            | {bc["concept"] for bc in blocked_concepts}
        )
        # A simple heuristic: if the response is very long and doesn't mention
        # any blocked concept, flag it as potentially off-topic
        if len(content) > 200:
            mentions_blocked = any(name in lower for name in blocked_names)
            if not mentions_blocked:
                return {
                    "rule": "ignores_blocked",
                    "detail": (
                        f"Le tuteur n'a mentionné aucun des concepts bloqués : "
                        f"{[bc['concept'] for bc in blocked_concepts[:3]]}"
                    ),
                    "blocked": [bc["concept"] for bc in blocked_concepts[:2]],
                }

    # Rule 3 — response completely ignores known difficulties
    difficulties = state.get("difficulties") or []
    if difficulties and len(content) > 300:
        diff_keywords = []
        for d in difficulties[:3]:
            # Extract the concept name from the label
            part = d.split(":")[-1].strip().lower() if ":" in d else d.lower()
            diff_keywords.extend(part.split()[:3])
        diff_keywords = [w for w in diff_keywords if len(w) > 3]
        if diff_keywords and not any(kw in lower for kw in diff_keywords):
            return {
                "rule": "off_topic",
                "detail": f"Réponse hors sujet — difficultés ignorées : {difficulties[:2]}",
                "difficulties": difficulties[:2],
            }

    # Rule 4 — premature topic change while weak concepts / difficulties remain
    weak_concepts = state.get("weak_concepts") or []
    if weak_concepts or difficulties:
        topic_change_phrases = (
            "passer à une autre",
            "autre notion",
            "autre opération",
            "autre chapitre",
            "nouvelle notion",
            "nouveau sujet",
            "move on to",
            "another topic",
            "something else",
            "voulez-tu continuer avec autre",
            "veux-tu continuer avec autre",
        )
        if any(phrase in lower for phrase in topic_change_phrases):
            pending = (
                [bc["concept"] for bc in (state.get("blocked_concepts") or [])]
                + [c for c in weak_concepts if c not in {bc["concept"] for bc in (state.get("blocked_concepts") or [])}]
            )
            return {
                "rule": "premature_topic_change",
                "detail": (
                    f"Le tuteur a proposé de changer de notion alors que des concepts "
                    f"ne sont pas encore maîtrisés : {pending[:3]}"
                ),
                "pending_concepts": pending[:3],
            }

    return None


def _fix_tutor_response(
    content: str,
    violation: Dict[str, Any],
    state: dict,
    enriched_system: str,
    messages: List[Any],
    base_url: str,
    api_key: str,
    llm_path: str,
    model: str,
) -> str:
    """Attempt to fix a rule violation by asking the LLM to regenerate.

    Adds the violation context to the system prompt and retries once.
    Returns the corrected content, or the original if regeneration fails.
    """
    rule = violation.get("rule", "")

    if rule == "false_validation":
        correction_directive = (
            f"\n\nCORRECTION OBLIGATOIRE : Ta réponse précédente contenait une validation "
            f"incorrecte. La réponse de l'apprenant '{violation.get('learner_answer')}' est FAUSSE "
            f"— la bonne réponse est '{violation.get('correct_answer')}'. "
            f"Tu DOIS signaler l'erreur clairement, expliquer pourquoi, donner la bonne réponse, "
            f"et NE PAS utiliser de formule de félicitation."
        )
    elif rule == "ignores_blocked":
        blocked = violation.get("blocked", [])
        correction_directive = (
            f"\n\nCORRECTION OBLIGATOIRE : Tu as ignoré les concepts bloqués {blocked}. "
            f"Ta réponse DOIT se concentrer exclusivement sur ces concepts "
            f"et proposer une nouvelle approche pédagogique."
        )
    elif rule == "premature_topic_change":
        pending = violation.get("pending_concepts", [])
        correction_directive = (
            f"\n\nCORRECTION OBLIGATOIRE : Tu as proposé de passer à un autre sujet alors que "
            f"l'apprenant n'a pas encore maîtrisé : {pending}. "
            f"INTERDIT de suggérer un changement de notion. "
            f"Reste sur le concept en cours, propose un nouvel exercice ou une nouvelle approche "
            f"pédagogique sur ce même concept."
        )
    else:  # off_topic
        diffs = violation.get("difficulties", [])
        correction_directive = (
            f"\n\nCORRECTION OBLIGATOIRE : Ta réponse ignorait les difficultés connues {diffs}. "
            f"Recentre ta réponse sur ces difficultés."
        )

    corrected_system = enriched_system + correction_directive
    try:
        corrected = _llm_conversational_response(
            corrected_system, messages, base_url, api_key, llm_path, model
        )
        if corrected:
            log.info("Tutor response corrected (rule=%s)", rule)
            return corrected
    except Exception as exc:
        log.warning("Tutor response fix failed: %s", exc)

    return content


def _build_enriched_system_prompt(state: dict, is_first_message: bool = False) -> str:
    """Enrich the support system prompt with the adaptive context produced by agents.

    The base system_prompt (topic, objectives, education level, learner name) comes
    from the frontend. Agents add the real evaluated level, difficulties, memories
    and teaching strategy so the LLM adapts its response accordingly.
    """
    base = state.get("system_prompt") or ""

    parts: List[str] = []

    user_name = state.get("user_name", "")
    if user_name:
        first_name = user_name.split()[0]
        parts.append(f"LEARNER: {user_name} (address them as {first_name})")

    adjusted_level = state.get("adjusted_level", "")
    if adjusted_level:
        parts.append(f"EVALUATED LEVEL: {adjusted_level}")

    concept_levels: Dict[str, str] = state.get("concept_levels") or {}
    if concept_levels:
        profile = ", ".join(
            f"{concept} ({level})"
            for concept, level in concept_levels.items()
        )
        parts.append(f"CONCEPT LEVEL PROFILE: {profile}")

    # ── Blocked concepts (highest priority — chronic failure) ────────────────────
    blocked_concepts = state.get("blocked_concepts") or []
    if blocked_concepts:
        blocked_details = "; ".join(
            f"{bc['concept']} ({bc['attempts']} tentatives"
            + (f", dernière erreur : {bc['last_error']}" if bc.get("last_error") else "")
            + ")"
            for bc in blocked_concepts[:3]
        )
        parts.append(f"BLOCKED CONCEPTS (chronic failure): {blocked_details}")
        parts.append(
            "ABSOLUTE RULE — BLOCKED CONCEPTS: The learner has repeatedly failed on the concepts "
            "listed above across multiple sessions. You MUST NOT introduce any new concept until "
            "each blocked concept is resolved. Use a completely different teaching approach from "
            "previous sessions: try a concrete real-life analogy, break the concept into the "
            "smallest possible sub-steps, or ask the learner to explain their reasoning so you "
            "can identify exactly where the misunderstanding lies. "
            "Do NOT repeat the same exercise type that already failed."
        )
        if is_first_message:
            bc0 = blocked_concepts[0]
            last_err = f" La dernière fois, tu as répondu '{bc0['last_error']}'." if bc0.get("last_error") else ""
            parts.append(
                f"FIRST MESSAGE DIRECTIVE (BLOCKED): Open with warm acknowledgement that this concept "
                f"has been difficult across {bc0['attempts']} sessions.{last_err} "
                f"Immediately try a brand-new approach on '{bc0['concept']}' — "
                f"NOT the same exercise type as before."
            )

    # ── Regular difficulties ──────────────────────────────────────────────────
    difficulties = state.get("difficulties") or []
    # Filter out blocked signals already shown above to avoid duplication
    blocked_names = {bc["concept"] for bc in blocked_concepts}
    non_blocked_difficulties = [
        d for d in difficulties
        if not any(name in d for name in blocked_names)
    ]
    if non_blocked_difficulties:
        diff_labels = "; ".join(str(d) for d in non_blocked_difficulties[:5])
        parts.append(f"IDENTIFIED DIFFICULTIES: {diff_labels}")
        parts.append(
            "PEDAGOGICAL RULE: The learner has unresolved difficulties listed above. "
            "You MUST address and resolve each one before introducing any new concept or moving forward in the course. "
            "If the learner tries to skip or ignore a difficulty, gently but firmly bring them back to it. "
            "Pedagogical continuity is non-negotiable."
        )
        if is_first_message and not blocked_concepts:
            concepts = [
                d.split(":")[-1].strip() if ":" in d else str(d)
                for d in non_blocked_difficulties[:2]
            ]
            parts.append(
                f"FIRST MESSAGE DIRECTIVE: This is the opening of the session. "
                f"The learner has known unresolved difficulties with: {', '.join(concepts)}. "
                f"Start your response by briefly and warmly acknowledging that you remember where they left off, "
                f"then immediately present a targeted exercise on '{concepts[0]}' to pick up exactly where they struggled. "
                f"Do NOT start with a general introduction or easy basics — go directly to the known difficulty."
            )

    weak_concepts = state.get("weak_concepts") or []
    non_blocked_weak = [c for c in weak_concepts if c not in blocked_names]
    if non_blocked_weak:
        parts.append(f"CONCEPTS TO REINFORCE: {', '.join(str(c) for c in non_blocked_weak[:5])}")
        parts.append(
            "MASTERY GATE — TOPIC CHANGE FORBIDDEN: The learner has NOT yet demonstrated "
            "sufficient mastery of the concepts listed above. "
            "You MUST NOT suggest moving to another topic, operation, or notion. "
            "Do NOT offer options like 'passer à une autre notion', 'autre opération', "
            "'voulez-tu continuer avec autre chose', or any equivalent phrasing. "
            "Stay exclusively on the current concept until the learner demonstrates clear understanding."
        )

    memory_summary = state.get("memory_summary") or state.get("session_summary") or ""
    if memory_summary:
        parts.append(f"KNOWN FACTS ABOUT THE LEARNER (use these to answer questions about their past): {memory_summary[:1200]}")

    decisions = state.get("strategy_decisions") or []
    if decisions:
        actions = "; ".join(
            d.get("action", str(d)) if isinstance(d, dict) else str(d)
            for d in decisions[:3]
        )
        parts.append(f"TEACHING STRATEGY: {actions}")

    # ── Exercises generated by ExerciseAgent ─────────────────────────────────
    exercises = state.get("exercises") or []
    if exercises:
        ex_lines = []
        for i, ex in enumerate(exercises[:3]):
            q = ex.get("question", "")
            a = ex.get("answer", "")
            h = ex.get("hint", "")
            sk = ex.get("skill_target", "")
            line = f"  [{i+1}] {q}"
            if h:
                line += f" (indice : {h})"
            if a:
                line += f" → réponse : {a}"
            if sk:
                line += f" [concept : {sk}]"
            ex_lines.append(line)
        parts.append(
            "PREPARED EXERCISES (use these — do not invent new ones):\n"
            + "\n".join(ex_lines)
        )

    # Learner answer verdict from VerifierAgent
    verdict = state.get("learner_answer_verdict") or {}
    if verdict:
        correct = verdict.get("correct")
        learner_ans = verdict.get("learner_answer", "")
        correct_ans = verdict.get("correct_answer", "")
        explanation = verdict.get("explanation", "")
        method = verdict.get("method", "")

        if correct is True:
            parts.append(
                f"ANSWER VERIFICATION [{method}]: The learner's answer '{learner_ans}' is CORRECT. "
                f"{explanation} — Congratulate them warmly and move on to the next step."
            )
        elif correct is False:
            parts.append(
                f"ANSWER VERIFICATION [{method}]: The learner's answer '{learner_ans}' is WRONG. "
                f"Correct answer: {correct_ans}. {explanation}. "
                f"CRITICAL: Do NOT say 'correct', 'exactement', 'bravo' or any validating phrase. "
                f"Gently tell the learner their answer is incorrect, explain why, "
                f"give the correct answer ({correct_ans}), and invite them to try again."
            )
        elif correct is None and explanation:
            parts.append(
                f"ANSWER VERIFICATION [web]: Context found: {explanation} "
                f"— Use your judgment to evaluate the learner's answer."
            )

    if not parts:
        return base

    adaptive_block = "\n\n[ADAPTIVE CONTEXT — do not display this block, use it to guide your response]\n"
    adaptive_block += "\n".join(parts)
    adaptive_block += "\n[END ADAPTIVE CONTEXT]"

    return base + adaptive_block


def _llm_conversational_response(
    enriched_system: str,
    messages: List[ChatMessage],
    base_url: str,
    api_key: str,
    path: str,
    model: str,
) -> str:
    """Call the configured LLM provider with the enriched system prompt and full
    conversation history. Uses the same provider resolution as /api/chat/completions
    so the model selected in the UI is respected.

    Returns the assistant's conversational reply as plain text.
    """
    import httpx

    llm_messages: List[Dict[str, str]] = [{"role": "system", "content": enriched_system}]
    for m in messages:
        if m.role in ("user", "assistant"):
            llm_messages.append({"role": m.role, "content": m.content})

    url = f"{base_url.rstrip('/')}/{path.lstrip('/')}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    try:
        resp = httpx.post(
            url,
            headers=headers,
            json={"model": model, "messages": llm_messages, "stream": False, "max_tokens": 1200},
            timeout=120.0,
        )
        resp.raise_for_status()
        text = (
            resp.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return text or "Je suis prêt à vous aider. Que souhaitez-vous apprendre ?"
    except Exception as exc:
        log.warning("LLM conversational call failed (%s): %s", url, exc)
        return "Une erreur s'est produite. Veuillez réessayer."


async def _sse_stream(content: str, session_id: str) -> AsyncGenerator[str, None]:
    """Stream a pre-generated string as OpenAI-compatible SSE chunks.

    Word-by-word chunking preserves readability while respecting the SSE protocol
    expected by the frontend's createOpenAITextStream parser.
    """
    cid = f"chatcmpl-{session_id[:8]}"
    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'model': 'tutor', 'choices': [{'delta': {'role': 'assistant'}, 'index': 0, 'finish_reason': None}]})}\n\n"

    # Chunk word by word so the UI renders progressively
    words = content.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'model': 'tutor', 'choices': [{'delta': {'content': chunk}, 'index': 0, 'finish_reason': None}]})}\n\n"
        await asyncio.sleep(0)

    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'model': 'tutor', 'choices': [{'delta': {}, 'index': 0, 'finish_reason': 'stop'}]})}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/chat")
async def adaptive_chat(
    body: AdaptiveChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Agentic tutor chat — wraps tutor_graph in an OpenAI-compatible SSE stream."""
    session_id = body.session_id or str(uuid.uuid4())
    support = _extract_support(body.messages, body.support)
    user_message = next(
        (m.content for m in reversed(body.messages) if m.role == "user"), ""
    )

    rag_docs: List[Dict[str, Any]] = []
    session_summary = ""
    try:
        ctx = ContextManager().build_agent_context(
            user_id=current_user.id,
            support=support,
            query=user_message or support,
            db=db,
        )
        rag_docs = ctx.rag_docs
        session_summary = ctx.session_summary
    except Exception as exc:
        log.warning("Context pre-load failed: %s", exc)

    # user_name: prefer value sent by frontend (already known), fall back to DB field
    user_name = body.user_name or getattr(current_user, "name", "") or ""

    initial_state: Dict[str, Any] = {
        "user_id": current_user.id,
        "user_name": user_name,
        "support": support,
        "current_level": body.current_level,
        "language": body.language,
        "learning_objectives": body.learning_objectives,
        "user_message": user_message,
        # Full conversation history — agents use this to see the exchange, not just the last message
        "messages": [{"role": m.role, "content": m.content} for m in body.messages],
        "rag_docs": rag_docs,
        "session_summary": session_summary,
        # system_prompt from frontend contains support details (topic, objectives, level…)
        "system_prompt": body.system_prompt or "",
        "human_feedback": "yes",  # auto-pass all HITL checkpoints in chat mode
        "iteration": 0,
        "n_retries": {},
        "agent_trace": [],
        "agent_reasoning": {},
        # answer_history and proposed_exercises_history are NOT reset here —
        # LangGraph carries them from the previous checkpoint for the same session_id
    }

    try:
        if tutor_graph is None:
            final_state = _fallback_response(initial_state)
        else:
            # recursion_limit must comfortably exceed orchestrator.MAX_ITERATIONS * 2
            # (orchestrator + agent per iteration) — LangGraph's default of 25 is
            # lower than that worst case (15 * 2 = 30) and raises GraphRecursionError
            # before the orchestrator's own ceiling ever gets a chance to fire.
            config = {"configurable": {"thread_id": session_id}, "recursion_limit": 50}
            final_state = await asyncio.to_thread(
                tutor_graph.invoke, initial_state, config
            )
    except Exception as exc:
        log.error("tutor_graph chat failed: %s", exc)
        try:
            final_state = _fallback_response(initial_state)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Tuteur agentique indisponible : {exc}",
            )

    # Persist adjusted_level so the next message starts from the real evaluated level
    _persist_adjusted_level(
        user_id=current_user.id,
        support=support,
        adjusted_level=final_state.get("adjusted_level") or body.current_level,
        difficulties=final_state.get("difficulties") or [],
    )

    # First message = no previous assistant turn in history
    is_first_message = not any(m.role == "assistant" for m in body.messages)

    # Build enriched system prompt from agent output + support context
    enriched_system = _build_enriched_system_prompt(final_state, is_first_message=is_first_message)

    # Resolve the LLM provider configured in the UI (same as /api/chat/completions)
    try:
        base_url, api_key, llm_path = await ProvidersService(db).resolve_provider(body.model)
    except Exception as exc:
        log.warning("Provider resolution failed for model '%s': %s", body.model, exc)
        # Fallback: try first available Ollama URL from config
        try:
            _svc = ProvidersService(db)
            ol_cfg = _svc.config.get_ollama()
            urls = ol_cfg.get("OLLAMA_BASE_URLS") or []
            if urls:
                base_url, api_key, llm_path = urls[0].rstrip("/"), "", "v1/chat/completions"
            else:
                raise RuntimeError("No Ollama URLs configured")
        except Exception as fallback_exc:
            log.error("LLM provider fallback also failed: %s", fallback_exc)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"LLM provider unavailable: {exc}",
            )

    # Generate conversational LLM response using enriched context + full history
    content = await asyncio.to_thread(
        _llm_conversational_response,
        enriched_system,
        body.messages,
        base_url,
        api_key,
        llm_path,
        body.model,
    )

    # Post-generation verification — fix rule violations before sending to learner
    violation = _check_tutor_response(content, final_state)
    if violation:
        log.warning("Tutor response violation detected: rule=%s — %s", violation["rule"], violation["detail"])
        content = await asyncio.to_thread(
            _fix_tutor_response,
            content, violation, final_state,
            enriched_system, body.messages,
            base_url, api_key, llm_path, body.model,
        )

    if not body.stream:
        return {
            "id": f"chatcmpl-{session_id[:8]}",
            "object": "chat.completion",
            "model": "tutor",
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
        }

    return StreamingResponse(
        _sse_stream(content, session_id),
        media_type="text/event-stream",
        headers={"X-Adaptive-Session-Id": session_id},
    )
