"""TutorGraphState — full state shared across all agent nodes."""

from __future__ import annotations

from typing import Any, Dict, List

from typing_extensions import TypedDict


class TutorGraphState(TypedDict, total=False):
    # ── Input (set by caller before graph.invoke) ─────────────────────────────
    user_id: str
    user_name: str  # full name from User.name — agents extract first name
    support: str  # cours / topic (renamed from topic per project convention)
    current_level: str  # beginner | intermediate | advanced
    language: str  # fr | en | ar
    learning_objectives: List[str]
    user_message: str

    # ── From ContextManager (Phase 4) ────────────────────────────────────────
    rag_docs: List[Dict[str, Any]]
    session_summary: str
    # System prompt built from support details in the frontend — enriched by agents before LLM call
    system_prompt: str

    # ── Agent outputs ─────────────────────────────────────────────────────────
    memory_context: List[Dict[str, Any]]
    memory_summary: str
    knowledge_graph: Dict[str, Any]
    weak_concepts: List[str]
    adjusted_level: str
    difficulties: List[str]
    strategy: str
    strategy_decisions: List[Dict[str, Any]]
    exercises: List[Dict[str, Any]]
    verification: Dict[str, Any]

    # ── Control flow ──────────────────────────────────────────────────────────
    next_agent: str
    iteration: int
    agent_trace: List[str]
    n_retries: Dict[str, int]

    # ── OTAI-agentique enrichments ────────────────────────────────────────────
    agent_reasoning: Dict[str, str]  # agent_name → reasoning text
    tool_selection_log: List[Dict[str, Any]]  # {agent, tool, rationale, result}
    verification_feedback: List[str]  # Verifier → Planner targeted feedback
    human_feedback: str  # injected via Command(resume=...) at HITL points
