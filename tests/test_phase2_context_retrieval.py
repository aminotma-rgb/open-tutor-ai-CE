"""Phase 2 — Context Retrieval Engine tests (15 tests)."""

import os
import tempfile
import pytest

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from data.database import Base
from data.models.memory import Memory
from ai.retrieval.context_retrieval import ContextRetrievalService


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
    return ContextRetrievalService()


@pytest.fixture
def txt_file():
    """Create a temporary text file for indexing tests."""
    content = " ".join([f"word{i}" for i in range(600)])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(content)
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def chroma_collection(svc, tmp_path, monkeypatch):
    """Isolated ChromaDB collection using a temp directory."""
    import chromadb
    from ai.retrieval import context_retrieval as cr_module

    client = chromadb.EphemeralClient()
    monkeypatch.setattr(cr_module, "_CHROMA_CLIENT", client)
    return svc.get_or_create_collection("test_collection")


# ── _chunk_text tests ─────────────────────────────────────────────────────────


def test_chunk_text_basic(svc):
    """Text longer than chunk_size is split into multiple chunks."""
    words = ["word"] * 600
    text = " ".join(words)
    chunks = svc._chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.split()) <= 500


def test_chunk_text_short(svc):
    """Text shorter than chunk_size returns a single chunk."""
    text = "short text with few words"
    chunks = svc._chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_empty(svc):
    """Empty text returns empty list."""
    assert svc._chunk_text("") == []


# ── _extract_text tests ───────────────────────────────────────────────────────


def test_extract_text_from_txt(svc, txt_file):
    """Text file content is extracted correctly."""
    text = svc._extract_text(txt_file)
    assert len(text) > 0
    assert "word0" in text


def test_extract_text_missing_file(svc):
    """Missing file returns empty string without raising."""
    result = svc._extract_text("/nonexistent/path/file.txt")
    assert result == ""


# ── retrieve_pedagogical_documents tests ─────────────────────────────────────


def test_retrieve_returns_list_on_empty_collection(svc, chroma_collection):
    """Querying an empty collection returns an empty list."""
    results = svc.retrieve_pedagogical_documents(
        user_id="u1", query="test", top_k=5, collection_name="test_collection"
    )
    assert isinstance(results, list)
    assert len(results) == 0


def test_retrieve_returns_empty_on_blank_query(svc, chroma_collection):
    """Blank query returns empty list immediately."""
    results = svc.retrieve_pedagogical_documents(
        user_id="u1", query="  ", top_k=5, collection_name="test_collection"
    )
    assert results == []


def test_retrieve_with_documents(svc, chroma_collection):
    """Documents indexed then retrieved by semantic query."""
    chroma_collection.upsert(
        documents=["Python is a programming language used for data science"],
        ids=["doc_1"],
        metadatas=[{"user_id": "u1"}],
    )
    results = svc.retrieve_pedagogical_documents(
        user_id="u1",
        query="programming language",
        top_k=1,
        collection_name="test_collection",
    )
    assert len(results) == 1
    assert "id" in results[0]
    assert "content" in results[0]
    assert "score" in results[0]


def test_retrieve_score_between_0_and_1(svc, chroma_collection):
    """Score is always in [0, 1]."""
    chroma_collection.upsert(
        documents=["Machine learning is a subset of artificial intelligence"],
        ids=["doc_2"],
        metadatas=[{"user_id": "u1"}],
    )
    results = svc.retrieve_pedagogical_documents(
        user_id="u1",
        query="AI machine learning",
        top_k=1,
        collection_name="test_collection",
    )
    assert len(results) == 1
    score = results[0]["score"]
    assert 0.0 <= score <= 1.0


# ── index_document tests ──────────────────────────────────────────────────────


def test_index_document_txt(svc, txt_file, monkeypatch):
    """Indexing a text file returns a positive chunk count."""
    import chromadb
    from ai.retrieval import context_retrieval as cr_module

    monkeypatch.setattr(cr_module, "_CHROMA_CLIENT", chromadb.EphemeralClient())
    count = svc.index_document(
        file_path=txt_file,
        metadata={"filename": "test.txt"},
        user_id="u1",
        collection_name="test_index",
    )
    assert count > 0


def test_index_document_empty_file(svc, monkeypatch):
    """Indexing an empty file returns 0 chunks."""
    import chromadb
    from ai.retrieval import context_retrieval as cr_module

    monkeypatch.setattr(cr_module, "_CHROMA_CLIENT", chromadb.EphemeralClient())
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("")
        path = f.name
    try:
        count = svc.index_document(
            file_path=path, user_id="u1", collection_name="test_empty"
        )
        assert count == 0
    finally:
        os.unlink(path)


# ── retrieve_internal_memory tests ───────────────────────────────────────────


def test_retrieve_internal_memory_empty(svc, db):
    """No memories → empty list."""
    results = svc.retrieve_internal_memory(user_id="u1", query="anything", db=db)
    assert results == []


def test_retrieve_internal_memory_finds_match(svc, db):
    """Memory whose content matches the query is returned."""
    db.add(
        Memory(
            user_id="u1",
            memory_type="episodic",
            content="User struggled with JOIN queries",
        )
    )
    db.commit()

    results = svc.retrieve_internal_memory(user_id="u1", query="JOIN", db=db)
    assert len(results) == 1
    assert "JOIN" in results[0]["content"]


def test_retrieve_internal_memory_type_filter(svc, db):
    """memory_types filter restricts results to matching types."""
    db.add(
        Memory(
            user_id="u1", memory_type="episodic", content="episodic content about SQL"
        )
    )
    db.add(
        Memory(
            user_id="u1", memory_type="behavioral", content="behavioral note about SQL"
        )
    )
    db.commit()

    results = svc.retrieve_internal_memory(
        user_id="u1", query="SQL", memory_types=["episodic"], db=db
    )
    assert all(r["memory_type"] == "episodic" for r in results)


def test_retrieve_internal_memory_limit(svc, db):
    """limit parameter caps the number of results."""
    for i in range(5):
        db.add(Memory(user_id="u1", memory_type="episodic", content=f"memory note {i}"))
    db.commit()

    results = svc.retrieve_internal_memory(user_id="u1", query="memory", limit=3, db=db)
    assert len(results) <= 3
