"""Phase 4 — Dynamic Context Manager tests (16 tests)."""

import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from data.database import Base
from data.models.memory import Memory
from ai.context_manager.service import (
    AgentContext,
    ContextManager,
    _count_tokens,
    _trim_to_budget,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def mgr():
    return ContextManager()


def _doc(content: str, score: float = 0.8) -> dict:
    return {"id": str(uuid.uuid4()), "content": content, "score": score, "metadata": {}}


# ── filter_memories ───────────────────────────────────────────────────────────


def test_filter_memories_same_support(mgr):
    memories = [{"support": "python", "created_at": datetime.utcnow().isoformat()}]
    result = mgr.filter_memories(memories, support="python")
    assert len(result) == 1


def test_filter_memories_no_support_kept(mgr):
    memories = [{"support": "", "created_at": datetime.utcnow().isoformat()}]
    result = mgr.filter_memories(memories, support="python")
    assert len(result) == 1


def test_filter_memories_wrong_support_excluded(mgr):
    memories = [{"support": "java", "created_at": datetime.utcnow().isoformat()}]
    result = mgr.filter_memories(memories, support="python")
    assert len(result) == 0


def test_filter_memories_recent_kept(mgr):
    memories = [
        {
            "support": "",
            "created_at": (datetime.utcnow() - timedelta(days=5)).isoformat(),
        }
    ]
    result = mgr.filter_memories(memories, support="python", max_age_days=14)
    assert len(result) == 1


def test_filter_memories_old_excluded(mgr):
    memories = [
        {
            "support": "",
            "created_at": (datetime.utcnow() - timedelta(days=30)).isoformat(),
        }
    ]
    result = mgr.filter_memories(memories, support="python", max_age_days=14)
    assert len(result) == 0


def test_filter_memories_invalid_date_kept(mgr):
    memories = [{"support": "", "created_at": "not-a-date"}]
    result = mgr.filter_memories(memories, support="python")
    assert len(result) == 1


def test_filter_memories_empty_list(mgr):
    assert mgr.filter_memories([], support="python") == []


# ── _count_tokens ─────────────────────────────────────────────────────────────


def test_count_tokens_nonzero():
    assert _count_tokens("hello world this is a test") > 0


def test_count_tokens_empty():
    assert _count_tokens("") == 0


# ── _trim_to_budget ───────────────────────────────────────────────────────────


def test_trim_keeps_high_score_docs():
    docs = [_doc("short", score=0.9), _doc("also short", score=0.5)]
    result = _trim_to_budget(docs, max_tokens=5000, summary="")
    assert any(d["score"] == 0.9 for d in result)


def test_trim_respects_budget():
    big_content = "word " * 1000  # ~1000 tokens
    docs = [_doc(big_content, score=0.9)] * 5
    result = _trim_to_budget(docs, max_tokens=500, summary="")
    total = sum(_count_tokens(d["content"]) for d in result)
    assert total <= 500


# ── build_agent_context ───────────────────────────────────────────────────────


def test_build_agent_context_returns_agent_context(mgr, db):
    with patch("ai.context_manager.service._fetch_rag_docs", return_value=[]):
        with patch(
            "ai.summarization.service.SummarizationService.get_cached_summary",
            return_value="",
        ):
            ctx = mgr.build_agent_context(
                user_id="u1", support="python", query="What is a list?", db=db
            )
    assert isinstance(ctx, AgentContext)
    assert ctx.user_id == "u1"
    assert ctx.support == "python"


def test_build_agent_context_empty_query_falls_back(mgr, db):
    with patch(
        "ai.context_manager.service._fetch_rag_docs", return_value=[]
    ) as mock_fetch:
        with patch(
            "ai.summarization.service.SummarizationService.get_cached_summary",
            return_value="",
        ):
            mgr.build_agent_context(user_id="u1", support="python", query="  ", db=db)
    # Empty query → falls back to support as query
    called_query = mock_fetch.call_args[1]["query"]
    assert called_query == "python"


def test_build_agent_context_no_rag_graceful(mgr, db):
    with patch(
        "ai.context_manager.service._fetch_rag_docs",
        side_effect=Exception("chroma down"),
    ):
        with patch(
            "ai.summarization.service.SummarizationService.get_cached_summary",
            return_value="",
        ):
            ctx = mgr.build_agent_context(
                user_id="u1", support="math", query="derivatives", db=db
            )
    assert ctx.rag_docs == []


def test_build_agent_context_token_count_within_budget(mgr, db):
    big_doc = _doc("word " * 2000, score=0.9)
    with patch("ai.context_manager.service._fetch_rag_docs", return_value=[big_doc]):
        with patch(
            "ai.summarization.service.SummarizationService.get_cached_summary",
            return_value="",
        ):
            ctx = mgr.build_agent_context(
                user_id="u1",
                support="math",
                query="integrals",
                db=db,
                session_summary="",
            )
    assert ctx.token_count <= 3000


def test_build_agent_context_carries_session_summary(mgr, db):
    with patch("ai.context_manager.service._fetch_rag_docs", return_value=[]):
        ctx = mgr.build_agent_context(
            user_id="u1",
            support="physics",
            query="gravity",
            db=db,
            session_summary="L'apprenant a bien compris les forces.",
        )
    assert ctx.session_summary == "L'apprenant a bien compris les forces."
