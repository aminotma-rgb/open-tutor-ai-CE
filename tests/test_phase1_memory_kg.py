"""Phase 1 — Hybrid Memory System tests (12 tests)."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from data.database import Base
from data.models.memory import Memory, KGConcept, KGRelation, KGUserMastery
from ai.knowledge_graph.service import KnowledgeGraphService


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
def kg(db):
    return KnowledgeGraphService()


# ── Model import tests ────────────────────────────────────────────────────────


def test_models_importable():
    """All 4 ORM models can be imported without error."""
    assert Memory is not None
    assert KGConcept is not None
    assert KGRelation is not None
    assert KGUserMastery is not None


# ── KGConcept tests ───────────────────────────────────────────────────────────


def test_upsert_concept_creates(db, kg):
    """upsert_concept creates a new concept when it does not exist."""
    concept = kg.upsert_concept(db, name="SELECT", support="SQL")
    assert concept.id is not None
    assert concept.name == "SELECT"
    assert concept.support == "SQL"
    assert concept.difficulty == 0.5


def test_upsert_concept_idempotent(db, kg):
    """upsert_concept returns the same row on repeated calls."""
    c1 = kg.upsert_concept(db, name="JOIN", support="SQL")
    c2 = kg.upsert_concept(db, name="JOIN", support="SQL")
    assert c1.id == c2.id
    assert db.query(KGConcept).filter_by(name="JOIN", support="SQL").count() == 1


# ── KGRelation tests ──────────────────────────────────────────────────────────


def test_add_relation(db, kg):
    """add_relation creates a directed edge between two concepts."""
    edge = kg.add_relation(
        db, source="SELECT", target="JOIN", relation="requires", support="SQL"
    )
    assert edge.id is not None
    assert edge.relation == "requires"
    src = db.query(KGConcept).filter_by(name="SELECT", support="SQL").first()
    tgt = db.query(KGConcept).filter_by(name="JOIN", support="SQL").first()
    assert edge.source_id == src.id
    assert edge.target_id == tgt.id


def test_add_relation_idempotent(db, kg):
    """add_relation returns existing edge and does not create duplicates."""
    e1 = kg.add_relation(
        db, source="WHERE", target="SELECT", relation="requires", support="SQL"
    )
    e2 = kg.add_relation(
        db, source="WHERE", target="SELECT", relation="requires", support="SQL"
    )
    assert e1.id == e2.id
    assert db.query(KGRelation).count() == 1


# ── KGUserMastery tests ───────────────────────────────────────────────────────


def test_update_mastery_creates_row(db, kg):
    """update_mastery creates a mastery row when none exists."""
    row = kg.update_mastery(
        db, user_id="u1", concept="SELECT", support="SQL", delta=0.2
    )
    assert row.user_id == "u1"
    assert abs(row.mastery - 0.2) < 1e-9
    assert row.attempts == 1


def test_update_mastery_increments(db, kg):
    """update_mastery increments mastery and attempt count on subsequent calls."""
    kg.update_mastery(db, user_id="u1", concept="SELECT", support="SQL", delta=0.2)
    row = kg.update_mastery(
        db, user_id="u1", concept="SELECT", support="SQL", delta=0.3
    )
    assert abs(row.mastery - 0.5) < 1e-9
    assert row.attempts == 2


def test_mastery_capped_at_1(db, kg):
    """update_mastery never exceeds 1.0."""
    kg.update_mastery(db, user_id="u1", concept="SELECT", support="SQL", delta=0.8)
    row = kg.update_mastery(
        db, user_id="u1", concept="SELECT", support="SQL", delta=0.8
    )
    assert row.mastery == 1.0


# ── Graph query tests ─────────────────────────────────────────────────────────


def test_get_weak_concepts(db, kg):
    """get_weak_concepts returns concepts below the mastery threshold."""
    kg.upsert_concept(db, "SELECT", "SQL")
    kg.upsert_concept(db, "JOIN", "SQL")
    kg.update_mastery(db, "u1", "SELECT", "SQL", delta=0.8)
    # JOIN has no mastery row → score 0.0

    weak = kg.get_weak_concepts("u1", "SQL", db, threshold=0.4)
    assert "JOIN" in weak
    assert "SELECT" not in weak


def test_build_graph_nodes(db, kg):
    """build_graph returns a DiGraph with all support concepts as nodes."""
    kg.upsert_concept(db, "SELECT", "SQL")
    kg.upsert_concept(db, "JOIN", "SQL")
    kg.add_relation(db, "SELECT", "JOIN", "requires", "SQL")
    kg.update_mastery(db, "u1", "SELECT", "SQL", delta=0.6)

    G = kg.build_graph("u1", "SQL", db)
    assert "SELECT" in G.nodes
    assert "JOIN" in G.nodes
    assert G.nodes["SELECT"]["mastery"] == pytest.approx(0.6)
    assert G.nodes["JOIN"]["mastery"] == pytest.approx(0.0)
    assert G.has_edge("SELECT", "JOIN")


def test_find_prerequisites(db, kg):
    """find_prerequisites returns concepts that 'require' the given concept."""
    kg.add_relation(
        db, source="WHERE", target="SELECT", relation="requires", support="SQL"
    )
    kg.add_relation(
        db, source="JOIN", target="SELECT", relation="requires", support="SQL"
    )

    prereqs = kg.find_prerequisites("SELECT", "SQL", db)
    assert set(prereqs) == {"WHERE", "JOIN"}


# ── Memory model test ─────────────────────────────────────────────────────────


def test_memory_create(db):
    """Memory rows can be created and retrieved."""
    mem = Memory(
        user_id="u1",
        memory_type="episodic",
        content="User struggled with GROUP BY",
        memory_metadata={"topic": "SQL"},
    )
    db.add(mem)
    db.commit()

    fetched = db.query(Memory).filter_by(user_id="u1").first()
    assert fetched is not None
    assert fetched.content == "User struggled with GROUP BY"
    assert fetched.memory_type == "episodic"
    assert fetched.memory_metadata["topic"] == "SQL"
