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
        "tool_selection_log": [],
    }

    try:
        if tutor_graph is None:
            final_state = _fallback_response(initial_state)
        else:
            config = {"configurable": {"thread_id": session_id}}
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
            metadata = {
                "support": support,
                "adjusted_level": adjusted_level,
                "difficulties": difficulties[:5],
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

    difficulties = state.get("difficulties") or []
    if difficulties:
        diff_labels = "; ".join(str(d) for d in difficulties[:5])
        parts.append(f"IDENTIFIED DIFFICULTIES: {diff_labels}")
        parts.append(
            "PEDAGOGICAL RULE: The learner has unresolved difficulties listed above. "
            "You MUST address and resolve each one before introducing any new concept or moving forward in the course. "
            "If the learner tries to skip or ignore a difficulty, gently but firmly bring them back to it. "
            "Pedagogical continuity is non-negotiable."
        )
        if is_first_message:
            # Extract readable concept names (strip "Difficulté identifiée : " prefix if present)
            concepts = [
                d.split(":")[-1].strip() if ":" in d else str(d)
                for d in difficulties[:2]
            ]
            parts.append(
                f"FIRST MESSAGE DIRECTIVE: This is the opening of the session. "
                f"The learner has known unresolved difficulties with: {', '.join(concepts)}. "
                f"Start your response by briefly and warmly acknowledging that you remember where they left off, "
                f"then immediately present a targeted exercise on '{concepts[0]}' to pick up exactly where they struggled. "
                f"Do NOT start with a general introduction or easy basics — go directly to the known difficulty."
            )

    weak_concepts = state.get("weak_concepts") or []
    if weak_concepts:
        parts.append(f"CONCEPTS TO REINFORCE: {', '.join(str(c) for c in weak_concepts[:5])}")

    memory_summary = state.get("memory_summary") or state.get("session_summary") or ""
    if memory_summary:
        parts.append(f"PREVIOUS SESSION SUMMARY: {memory_summary[:400]}")

    decisions = state.get("strategy_decisions") or []
    if decisions:
        actions = "; ".join(
            d.get("action", str(d)) if isinstance(d, dict) else str(d)
            for d in decisions[:3]
        )
        parts.append(f"TEACHING STRATEGY: {actions}")

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
        "language": "fr",
        "learning_objectives": body.learning_objectives,
        "user_message": user_message,
        "rag_docs": rag_docs,
        "session_summary": session_summary,
        # system_prompt from frontend contains support details (topic, objectives, level…)
        "system_prompt": body.system_prompt or "",
        "human_feedback": "yes",  # auto-pass all HITL checkpoints in chat mode
        "iteration": 0,
        "n_retries": {},
        "agent_trace": [],
        "agent_reasoning": {},
        "tool_selection_log": [],
    }

    try:
        if tutor_graph is None:
            final_state = _fallback_response(initial_state)
        else:
            config = {"configurable": {"thread_id": session_id}}
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
