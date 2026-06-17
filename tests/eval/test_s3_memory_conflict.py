"""S3 — Conflit mémoire (automatisé).

Objectif : détecter une contradiction de profil, mettre à jour le niveau
et conserver l'historique versionné avec horodatage.

MRA = 100 % + présence des deux entrées horodatées en base.
Domaine : SQL (distinct des données de développement Python)
Dépendance : S1 doit être exécuté avant (profil débutant déjà en base).
"""

import uuid
from datetime import datetime, timedelta

import pytest

from data.models.memory import Memory


# ── Helpers ──────────────────────────────────────────────────────────────────

def _store_profile(user_id: str, support: str, niveau: str, db, ts: datetime = None) -> Memory:
    """Persiste un profil apprenant avec un horodatage contrôlé."""
    row = Memory(
        id=str(uuid.uuid4()),
        user_id=user_id,
        memory_type="episodic",
        content=f"Profil {niveau}",
        memory_metadata={"support": support, "niveau": niveau},
        created_at=ts or datetime.utcnow(),
        updated_at=ts or datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    return row


def _update_profile(user_id: str, support: str, new_niveau: str, db) -> Memory:
    """Met à jour le niveau courant — ne supprime PAS l'ancienne entrée."""
    new_row = Memory(
        id=str(uuid.uuid4()),
        user_id=user_id,
        memory_type="episodic",
        content=f"Profil {new_niveau}",
        memory_metadata={"support": support, "niveau": new_niveau},
    )
    db.add(new_row)
    db.commit()
    return new_row


def _get_current_profile(user_id: str, support: str, db) -> dict:
    """Retourne le profil le plus récent (valeur courante)."""
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
    return {"niveau": meta.get("niveau"), "created_at": row.created_at}


def _get_history(user_id: str, db) -> list:
    """Retourne l'historique complet des profils, du plus ancien au plus récent."""
    rows = (
        db.query(Memory)
        .filter(
            Memory.user_id == user_id,
            Memory.memory_type == "episodic",
        )
        .order_by(Memory.created_at.asc())
        .all()
    )
    return [
        {
            "niveau": (row.memory_metadata or {}).get("niveau"),
            "created_at": row.created_at,
        }
        for row in rows
    ]


# ── Tests ────────────────────────────────────────────────────────────────────

def test_s3_current_profile_updated_to_avance(db, user_id):
    """Après le conflit, le profil courant indique 'avancé'."""
    t0 = datetime.utcnow() - timedelta(days=3)
    _store_profile(user_id, "sql-requetes", "débutant", db, ts=t0)
    _update_profile(user_id, "sql-requetes", "avancé", db)

    current = _get_current_profile(user_id, "sql-requetes", db)
    assert current.get("niveau") == "avancé", (
        f"Niveau courant attendu : 'avancé' — obtenu : '{current.get('niveau')}'"
    )


def test_s3_historical_profile_preserved(db, user_id):
    """L'entrée 'débutant' est conservée dans l'historique (non écrasée)."""
    t0 = datetime.utcnow() - timedelta(days=3)
    _store_profile(user_id, "sql-requetes", "débutant", db, ts=t0)
    _update_profile(user_id, "sql-requetes", "avancé", db)

    history = _get_history(user_id, db)
    niveaux = [h["niveau"] for h in history]

    assert "débutant" in niveaux, "L'entrée 'débutant' a été supprimée — versionnement absent."
    assert "avancé" in niveaux, "L'entrée 'avancé' est absente de l'historique."


def test_s3_history_is_ordered_chronologically(db, user_id):
    """L'entrée 'débutant' est antérieure à l'entrée 'avancé'."""
    t0 = datetime.utcnow() - timedelta(days=3)
    _store_profile(user_id, "sql-requetes", "débutant", db, ts=t0)
    _update_profile(user_id, "sql-requetes", "avancé", db)

    history = _get_history(user_id, db)
    assert len(history) >= 2

    debutant_ts = next(h["created_at"] for h in history if h["niveau"] == "débutant")
    avance_ts = next(h["created_at"] for h in history if h["niveau"] == "avancé")

    assert debutant_ts < avance_ts, (
        "Horodatage incohérent : 'débutant' n'est pas antérieur à 'avancé'."
    )


def test_s3_mra_after_conflict(db, user_id):
    """MRA = 100 % : les deux champs attendus (niveau courant + historique) sont présents."""
    t0 = datetime.utcnow() - timedelta(days=3)
    _store_profile(user_id, "sql-requetes", "débutant", db, ts=t0)
    _update_profile(user_id, "sql-requetes", "avancé", db)

    current = _get_current_profile(user_id, "sql-requetes", db)
    history = _get_history(user_id, db)

    expected = {
        "niveau_courant": "avancé",
        "historique_debutant_present": True,
        "historique_horodaté": True,
    }
    retrieved = {
        "niveau_courant": current.get("niveau"),
        "historique_debutant_present": any(h["niveau"] == "débutant" for h in history),
        "historique_horodaté": all(h["created_at"] is not None for h in history),
    }

    correct = sum(1 for k in expected if retrieved[k] == expected[k])
    mra = correct / len(expected)

    assert mra == 1.0, (
        f"MRA = {mra:.0%} — champs incorrects : "
        + str([k for k in expected if retrieved[k] != expected[k]])
    )


def test_s3_no_cross_user_contamination(db, user_id):
    """Le conflit d'un utilisateur ne modifie pas le profil d'un autre."""
    other = str(uuid.uuid4())
    t0 = datetime.utcnow() - timedelta(days=3)
    _store_profile(user_id, "sql-requetes", "débutant", db, ts=t0)
    _store_profile(other, "sql-requetes", "intermédiaire", db, ts=t0)
    _update_profile(user_id, "sql-requetes", "avancé", db)

    other_current = _get_current_profile(other, "sql-requetes", db)
    assert other_current.get("niveau") == "intermédiaire", (
        "Le profil d'un autre utilisateur a été modifié."
    )
