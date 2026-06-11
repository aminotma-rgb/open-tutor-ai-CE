"""Phase 3 — Session Summarization Service tests (17 tests)."""

import uuid
import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from data.database import Base
from data.models.memory import Memory
from ai.summarization.service import SummarizationService, extract_exchanges


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
def svc():
    return SummarizationService()


def _make_exchanges(n: int) -> list:
    result = []
    for i in range(n):
        result.append({"role": "user", "content": f"Question {i}"})
        result.append({"role": "assistant", "content": f"Réponse {i}"})
    return result


# ── extract_exchanges ─────────────────────────────────────────────────────────


def test_extract_exchanges_valid_dict():
    messages = [{"role": "user", "content": "bonjour"}]
    assert extract_exchanges({"messages": messages}) == messages


def test_extract_exchanges_missing_key():
    assert extract_exchanges({"other_key": []}) == []


def test_extract_exchanges_not_dict():
    assert extract_exchanges(None) == []
    assert extract_exchanges("raw string") == []
    assert extract_exchanges([]) == []


def test_extract_exchanges_empty_messages():
    assert extract_exchanges({"messages": []}) == []


# ── count_tokens ──────────────────────────────────────────────────────────────


def test_count_tokens_returns_int(svc):
    exchanges = [{"role": "user", "content": "hello world"}]
    result = svc.count_tokens(exchanges)
    assert isinstance(result, int)
    assert result > 0


def test_count_tokens_empty(svc):
    assert svc.count_tokens([]) == 0


def test_count_tokens_scales_with_length(svc):
    short = [{"role": "user", "content": "hi"}]
    long = [{"role": "user", "content": "hi " * 100}]
    assert svc.count_tokens(long) > svc.count_tokens(short)


# ── should_summarize ──────────────────────────────────────────────────────────


def test_should_summarize_count_trigger(svc):
    exchanges = _make_exchanges(3)  # 6 messages = 3 exchanges pairs
    cfg = {"exchanges_per_summary": 5, "auto_summarize_token_limit": 9999}
    assert svc.should_summarize(exchanges, cfg) is True


def test_should_summarize_below_threshold(svc):
    exchanges = _make_exchanges(1)  # 2 messages
    cfg = {"exchanges_per_summary": 10, "auto_summarize_token_limit": 9999}
    assert svc.should_summarize(exchanges, cfg) is False


def test_should_summarize_token_trigger(svc):
    # Create exchanges with very long content to trigger token threshold
    long_content = "mot " * 400  # ~400 tokens
    exchanges = [{"role": "user", "content": long_content}] * 4
    cfg = {"exchanges_per_summary": 999, "auto_summarize_token_limit": 100}
    assert svc.should_summarize(exchanges, cfg) is True


def test_should_summarize_empty(svc):
    cfg = {"exchanges_per_summary": 5, "auto_summarize_token_limit": 1200}
    assert svc.should_summarize([], cfg) is False


# ── cache (get/set/invalidate) ────────────────────────────────────────────────


def test_cache_summary_and_retrieve(svc, db):
    user_id = str(uuid.uuid4())
    support_id = "support-abc"
    svc.cache_summary(user_id, support_id, "Résumé test", db)
    result = svc.get_cached_summary(user_id, support_id, db)
    assert result == "Résumé test"


def test_get_cached_summary_miss(svc, db):
    result = svc.get_cached_summary("unknown-user", "unknown-support", db)
    assert result is None


def test_get_cached_summary_wrong_support(svc, db):
    user_id = str(uuid.uuid4())
    svc.cache_summary(user_id, "support-A", "Résumé A", db)
    result = svc.get_cached_summary(user_id, "support-B", db)
    assert result is None


def test_invalidate_cache_removes_entries(svc, db):
    user_id = str(uuid.uuid4())
    support_id = "support-xyz"
    svc.cache_summary(user_id, support_id, "Résumé 1", db)
    svc.cache_summary(user_id, support_id, "Résumé 2", db)
    svc.invalidate_cache(user_id, support_id, db)
    result = svc.get_cached_summary(user_id, support_id, db)
    assert result is None


def test_invalidate_cache_does_not_remove_other_supports(svc, db):
    user_id = str(uuid.uuid4())
    svc.cache_summary(user_id, "support-keep", "Gardé", db)
    svc.cache_summary(user_id, "support-del", "Supprimé", db)
    svc.invalidate_cache(user_id, "support-del", db)
    assert svc.get_cached_summary(user_id, "support-keep", db) == "Gardé"
    assert svc.get_cached_summary(user_id, "support-del", db) is None


# ── summarize_session fallback ────────────────────────────────────────────────


def test_summarize_session_below_threshold_returns_cached(svc, db):
    user_id = str(uuid.uuid4())
    support_id = "support-test"
    svc.cache_summary(user_id, support_id, "Résumé existant", db)
    exchanges = _make_exchanges(1)
    cfg = {"exchanges_per_summary": 10, "auto_summarize_token_limit": 9999}
    result = svc.summarize_session(
        user_id=user_id,
        support_id=support_id,
        exchanges=exchanges,
        db=db,
    )
    # Below threshold → cached summary returned without LLM call
    assert result == "Résumé existant"


def test_summarize_session_llm_failure_caches_fallback(svc, db, monkeypatch):
    user_id = str(uuid.uuid4())
    support_id = "support-llm-fail"
    exchanges = _make_exchanges(3)  # triggers count threshold (6 msgs >= 5)

    def _raise(*a, **kw):
        raise RuntimeError("LLM unavailable")

    monkeypatch.setattr(svc, "_call_llm", _raise)

    result = svc.summarize_session(
        user_id=user_id,
        support_id=support_id,
        exchanges=exchanges,
        db=db,
    )
    # Fallback string cached and returned
    assert "6" in result or "échanges" in result.lower() or "session" in result.lower()
    cached = svc.get_cached_summary(user_id, support_id, db)
    assert cached is not None
