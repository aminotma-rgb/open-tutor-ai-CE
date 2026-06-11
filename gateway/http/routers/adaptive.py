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
        ctx = ContextManager(db).build_agent_context(
            user_id=current_user.id,
            support=body.support,
            query=body.user_message or body.support,
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


def _extract_support(messages: List[ChatMessage], fallback: Optional[str]) -> str:
    if fallback:
        return fallback
    all_text = " ".join(m.content for m in messages).lower()
    for kw in ("python", "sql", "javascript", "java", "math", "physics", "history"):
        if kw in all_text:
            return kw
    return "general"


def _format_response(state: dict, support: str) -> str:
    first_name = ""
    uname = state.get("user_name", "")
    if uname:
        first_name = uname.split()[0] + ", "

    level = state.get("adjusted_level", "intermédiaire")
    decisions = state.get("strategy_decisions") or []
    exercises = state.get("exercises") or []
    trace = state.get("agent_trace") or []
    verification = state.get("verification") or {}
    session_id = state.get("_session_id", "")

    lines = [f"## {first_name}Session d'apprentissage — {support}\n"]
    lines.append(f"**Niveau évalué :** {level}\n")

    if decisions:
        lines.append("### Plan d'apprentissage")
        for i, d in enumerate(decisions[:5], 1):
            action = d.get("action", "") if isinstance(d, dict) else str(d)
            lines.append(f"{i}. {action}")
        lines.append("")

    if exercises:
        lines.append("### Exercices proposés\n")
        for ex in exercises:
            ex_type = ex.get("type", "explain")
            diff = ex.get("difficulty", "medium")
            skill = ex.get("skill_target", "")
            question = ex.get("question", "")
            hint = ex.get("hint", "")
            lines.append(f"**Exercice {ex.get('id', '')} — {ex_type}** ({diff})")
            if skill:
                lines.append(f"*Objectif :* {skill}")
            lines.append(question)
            if hint:
                lines.append(f"> 💡 {hint}")
            lines.append("")

    verdict = verification.get("verdict", "")
    if verdict and verdict not in ("skipped",):
        lines.append(f"*Vérification :* {verdict}")

    if session_id:
        lines.append(f"\n---\n*session_id : `{session_id}`*")

    return "\n".join(lines)


async def _sse_stream(content: str, session_id: str) -> AsyncGenerator[str, None]:
    cid = f"chatcmpl-{session_id[:8]}"
    header = json.dumps(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "model": "tutor",
            "choices": [
                {"delta": {"role": "assistant"}, "index": 0, "finish_reason": None}
            ],
        }
    )
    yield f"data: {header}\n\n"

    chunk_size = 40
    for i in range(0, len(content), chunk_size):
        chunk = content[i : i + chunk_size]
        payload = json.dumps(
            {
                "id": cid,
                "object": "chat.completion.chunk",
                "model": "tutor",
                "choices": [
                    {"delta": {"content": chunk}, "index": 0, "finish_reason": None}
                ],
            }
        )
        yield f"data: {payload}\n\n"
        await asyncio.sleep(0)

    stop = json.dumps(
        {
            "id": cid,
            "object": "chat.completion.chunk",
            "model": "tutor",
            "choices": [{"delta": {}, "index": 0, "finish_reason": "stop"}],
        }
    )
    yield f"data: {stop}\n\n"
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
        ctx = ContextManager(db).build_agent_context(
            user_id=current_user.id,
            support=support,
            query=user_message or support,
        )
        rag_docs = ctx.rag_docs
        session_summary = ctx.session_summary
    except Exception as exc:
        log.warning("Context pre-load failed: %s", exc)

    initial_state: Dict[str, Any] = {
        "user_id": current_user.id,
        "user_name": getattr(current_user, "name", "") or "",
        "support": support,
        "current_level": body.current_level,
        "language": "fr",
        "learning_objectives": body.learning_objectives,
        "user_message": user_message,
        "rag_docs": rag_docs,
        "session_summary": session_summary,
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

    final_state["_session_id"] = session_id
    content = _format_response(final_state, support)

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
