"""Dynamic Context Manager.

Assembles the context injected into each agent call:
  - RAG documents from ChromaDB (Phase 2)
  - Session summary from opentutorai_memory (Phase 3)
  - Memory filtering by support + recency (for MemoryAgent)

Token budget is enforced by dropping lowest-scoring RAG docs first.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

_DEFAULT_MAX_CONTEXT_TOKENS = 3000
_DEFAULT_MIN_SCORE = 0.4
_DEFAULT_MAX_AGE_DAYS = 14


def _get_context_cfg() -> Dict[str, Any]:
    try:
        from config import settings

        return getattr(settings, "CONTEXT_RETRIEVAL_CONFIG", {})
    except Exception:
        return {}


# ── AgentContext ──────────────────────────────────────────────────────────────


@dataclass
class AgentContext:
    """Typed container for pre-assembled agent context."""

    user_id: str
    user_name: str = ""
    support: str = ""
    language: str = "fr"
    learning_objectives: List[str] = field(default_factory=list)
    user_message: str = ""
    rag_docs: List[Dict[str, Any]] = field(default_factory=list)
    session_summary: str = ""
    token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "user_name": self.user_name,
            "support": self.support,
            "language": self.language,
            "learning_objectives": self.learning_objectives,
            "user_message": self.user_message,
            "rag_docs": self.rag_docs,
            "session_summary": self.session_summary,
            "token_count": self.token_count,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _count_tokens(text: str) -> int:
    """tiktoken cl100k_base with word-count fallback."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text.split())


def _trim_to_budget(
    docs: List[Dict[str, Any]],
    max_tokens: int,
    summary: str,
) -> List[Dict[str, Any]]:
    """Drop lowest-scoring docs until total token count fits the budget."""
    summary_tokens = _count_tokens(summary)
    remaining = max_tokens - summary_tokens
    if remaining <= 0:
        return []

    # Sort descending by score to keep the best ones
    sorted_docs = sorted(docs, key=lambda d: d.get("score", 0.0), reverse=True)
    kept: List[Dict[str, Any]] = []
    used = 0
    for doc in sorted_docs:
        tokens = _count_tokens(doc.get("content", ""))
        if used + tokens <= remaining:
            kept.append(doc)
            used += tokens
        # Continue — a shorter doc later may still fit
    return kept


def _fetch_rag_docs(
    user_id: str,
    query: str,
    support: str,
    min_score: float,
    top_k: int,
    collection_name: str,
) -> List[Dict[str, Any]]:
    """Call ContextRetrievalService and filter by min_score."""
    try:
        from ai.retrieval.context_retrieval import ContextRetrievalService

        svc = ContextRetrievalService()
        docs = svc.retrieve_pedagogical_documents(
            user_id=user_id,
            query=query,
            top_k=top_k,
            collection_name=collection_name,
        )
        return [d for d in docs if d.get("score", 0.0) >= min_score]
    except Exception as exc:
        log.warning("RAG fetch failed: %s", exc)
        return []


# ── ContextManager ────────────────────────────────────────────────────────────


class ContextManager:
    """Assemble agent context from RAG + summarization services."""

    def build_agent_context(
        self,
        user_id: str,
        support: str,
        query: str,
        db: Session,
        user_name: str = "",
        language: str = "fr",
        learning_objectives: Optional[List[str]] = None,
        collection_name: str = "pedagogical_docs",
        session_summary: Optional[str] = None,
    ) -> AgentContext:
        """Fetch RAG docs + session summary and trim to token budget.

        Memories are excluded — MemoryAgent loads them fresh each graph run
        and calls filter_memories() on the live list.
        """
        cfg = _get_context_cfg()
        filtering = cfg.get("filtering", {})
        rag_cfg = cfg.get("rag", {})

        max_tokens: int = filtering.get(
            "max_context_tokens", _DEFAULT_MAX_CONTEXT_TOKENS
        )
        min_score: float = filtering.get("relevance_threshold", _DEFAULT_MIN_SCORE)
        top_k: int = rag_cfg.get("top_k_documents", 5)

        effective_query = query.strip() or support

        # Session summary (use passed value or load from cache)
        if session_summary is None:
            try:
                from ai.summarization.service import SummarizationService

                summary_svc = SummarizationService()
                session_summary = (
                    summary_svc.get_cached_summary(user_id, support, db) or ""
                )
            except Exception as exc:
                log.warning("Summary cache fetch failed: %s", exc)
                session_summary = ""

        # RAG docs
        rag_docs: List[Dict[str, Any]] = []
        if rag_cfg.get("enabled", True) and effective_query:
            try:
                rag_docs = _fetch_rag_docs(
                    user_id=user_id,
                    query=effective_query,
                    support=support,
                    min_score=min_score,
                    top_k=top_k,
                    collection_name=collection_name,
                )
            except Exception as exc:
                log.warning("RAG fetch failed in build_agent_context: %s", exc)

        # Trim to token budget
        rag_docs = _trim_to_budget(rag_docs, max_tokens, session_summary)

        # Count total tokens
        all_text = session_summary + " ".join(d.get("content", "") for d in rag_docs)
        token_count = _count_tokens(all_text)

        return AgentContext(
            user_id=user_id,
            user_name=user_name,
            support=support,
            language=language,
            learning_objectives=learning_objectives or [],
            user_message=query,
            rag_docs=rag_docs,
            session_summary=session_summary,
            token_count=token_count,
        )

    def filter_memories(
        self,
        memories: List[Dict[str, Any]],
        support: str,
        max_age_days: int = _DEFAULT_MAX_AGE_DAYS,
    ) -> List[Dict[str, Any]]:
        """Filter a memory list by support match and recency.

        Called by MemoryAgent on a freshly loaded list — not during
        build_agent_context() which only touches RAG + summaries.
        """
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        result = []
        for mem in memories:
            # Support filter — only pedagogical memories (behavioral/procedural)
            # are scoped to a course topic; episodic memories are personal facts
            # about the learner and must stay recallable regardless of which
            # support is currently active. support_id lives in memory_metadata
            # (see _persist_memories in feedback.py), not at the top level.
            if mem.get("memory_type") != "episodic":
                metadata = mem.get("memory_metadata") or {}
                mem_support = metadata.get("support_id") or metadata.get("support") or ""
                if mem_support and mem_support != support:
                    continue

            # Recency filter
            created_raw = mem.get("created_at")
            if created_raw:
                try:
                    if isinstance(created_raw, str):
                        created_at = datetime.fromisoformat(created_raw)
                    elif isinstance(created_raw, datetime):
                        created_at = created_raw
                    else:
                        result.append(mem)
                        continue
                    if created_at < cutoff:
                        continue
                except Exception:
                    pass  # invalid date → keep

            result.append(mem)
        return result
