"""Adaptive tutor router — /api/v1/adaptive/*."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
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
