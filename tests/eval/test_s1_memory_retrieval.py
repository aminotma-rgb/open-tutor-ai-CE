"""S1 — Vérification de la mémoire (automatisé).

Objectif : MRA = M_correctes / M_attendues = 100 %
Domaine : HTML (distinct des données de développement Python)
Précondition : base de données vide (fixture db)
"""

import uuid
import pytest

from data.models.memory import Memory
from ai.summarization.service import SummarizationService

# ── Données de test (domaine HTML — distinct du dev Python) ──────────────────

SESSION_1 = {
    "support": "html-balises-debutant",
    "concept": "balises HTML",
    "niveau": "débutant",
    "notions_vues": ["<h1>", "<p>", "<a>", "<img>"],
}

EXPECTED_FIELDS = ["support", "concept", "niveau", "notions_vues"]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _store_session_memory(user_id: str, session: dict, db) -> None:
    """Persiste le profil de session en mémoire épisodique."""
    row = Memory(
        id=str(uuid.uuid4()),
        user_id=user_id,
        memory_type="episodic",
        content=session["concept"],
        memory_metadata={
            "support": session["support"],
            "niveau": session["niveau"],
            "notions_vues": session["notions_vues"],
        },
    )
    db.add(row)
    db.commit()


def _retrieve_session_memory(user_id: str, support: str, db) -> dict:
    """Récupère la dernière mémoire épisodique pour user + support."""
    row = (
        db.query(Memory)
        .filter(
            Memory.user_id == user_id,
            Memory.memory_type == "episodic",
        )
        .order_by(Memory.created_at.desc())
        .first()
    )
    if not row:
        return {}
    meta = row.memory_metadata or {}
    return {
        "support": meta.get("support"),
        "concept": row.content,
        "niveau": meta.get("niveau"),
        "notions_vues": meta.get("notions_vues"),
    }


def _compute_mra(retrieved: dict, expected: dict, fields: list) -> float:
    """MRA = M_correctes / M_attendues."""
    correct = sum(1 for f in fields if retrieved.get(f) == expected.get(f))
    return correct / len(fields)


# ── Tests ────────────────────────────────────────────────────────────────────

def test_s1_memory_stored_and_retrieved(db, user_id):
    """Le système restitue exactement les informations de la session précédente."""
    _store_session_memory(user_id, SESSION_1, db)
    retrieved = _retrieve_session_memory(user_id, SESSION_1["support"], db)

    assert retrieved, "Aucune mémoire retrouvée — la mémoire n'a pas été persistée."

    mra = _compute_mra(retrieved, SESSION_1, EXPECTED_FIELDS)
    assert mra == 1.0, (
        f"MRA = {mra:.0%} — champs incorrects : "
        + str([f for f in EXPECTED_FIELDS if retrieved.get(f) != SESSION_1.get(f)])
    )


def test_s1_mra_field_by_field(db, user_id):
    """Assertion champ par champ pour identifier précisément toute perte."""
    _store_session_memory(user_id, SESSION_1, db)
    retrieved = _retrieve_session_memory(user_id, SESSION_1["support"], db)

    assert retrieved.get("support") == SESSION_1["support"], "Champ 'support' incorrect"
    assert retrieved.get("concept") == SESSION_1["concept"], "Champ 'concept' incorrect"
    assert retrieved.get("niveau") == SESSION_1["niveau"], "Champ 'niveau' incorrect"
    assert retrieved.get("notions_vues") == SESSION_1["notions_vues"], "Champ 'notions_vues' incorrect"


def test_s1_new_session_isolation(db, user_id):
    """La mémoire d'un utilisateur n'est pas accessible à un autre."""
    other_user = str(uuid.uuid4())
    _store_session_memory(user_id, SESSION_1, db)

    retrieved = _retrieve_session_memory(other_user, SESSION_1["support"], db)
    assert not retrieved, "Fuite de mémoire entre utilisateurs."


def test_s1_no_memory_returns_empty(db, user_id):
    """Sans session précédente, le système ne restitue rien (pas d'hallucination)."""
    retrieved = _retrieve_session_memory(user_id, SESSION_1["support"], db)
    assert retrieved == {}, "Le système restitue des données alors qu'aucune session n'a été enregistrée."
