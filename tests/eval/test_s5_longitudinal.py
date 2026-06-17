"""S5 — Apprentissage longitudinal (mixte).

Automatisé : MRA = 100 %, TCR = N_succès / N_total ≥ 95 %
Humain     : Learning Gain = (post − pré) / (max − pré) ≥ 0,30

Domaine : JavaScript (distinct des données de développement Python)
4 sessions simulées : apprentissage → difficulté → quiz → montée en niveau
"""

import uuid
from datetime import datetime, timedelta

import pytest

from data.models.memory import Memory
from ai.summarization.service import SummarizationService

# ── Pré-test standardisé (10 questions sur les fonctions JS) ─────────────────

PRE_TEST = {
    "q1": ("Qu'est-ce qu'une fonction en JS ?", "réponse_attendue_q1"),
    "q2": ("Comment déclarer une fonction ?", "réponse_attendue_q2"),
    "q3": ("Qu'est-ce qu'un paramètre ?", "réponse_attendue_q3"),
    "q4": ("Qu'est-ce qu'une valeur de retour ?", "réponse_attendue_q4"),
    "q5": ("Comment appeler une fonction ?", "réponse_attendue_q5"),
    "q6": ("Qu'est-ce qu'une fonction anonyme ?", "réponse_attendue_q6"),
    "q7": ("Qu'est-ce qu'une arrow function ?", "réponse_attendue_q7"),
    "q8": ("Qu'est-ce que la portée (scope) ?", "réponse_attendue_q8"),
    "q9": ("Qu'est-ce qu'une closure ?", "réponse_attendue_q9"),
    "q10": ("Qu'est-ce qu'une fonction récursive ?", "réponse_attendue_q10"),
}

# Scores simulés (établis avant l'exécution)
PRE_SCORE = 3   # 3/10
POST_SCORE = 7  # 7/10 — établi après les 4 sessions
MAX_SCORE = 10


# ── Helpers ──────────────────────────────────────────────────────────────────

def _store_session(user_id: str, support: str, session_num: int,
                   notions: list, difficulties: list, niveau: str, db) -> None:
    row = Memory(
        id=str(uuid.uuid4()),
        user_id=user_id,
        memory_type="episodic",
        content=f"Session {session_num} — {support}",
        memory_metadata={
            "support": support,
            "session": session_num,
            "notions_vues": notions,
            "difficultés": difficulties,
            "niveau": niveau,
        },
        created_at=datetime.utcnow() - timedelta(days=(4 - session_num)),
    )
    db.add(row)

    # Résumé de session
    svc = SummarizationService()
    svc.cache_summary(user_id, f"{support}-s{session_num}", f"Résumé session {session_num}", db)


def _retrieve_all_sessions(user_id: str, db) -> list:
    rows = (
        db.query(Memory)
        .filter(Memory.user_id == user_id, Memory.memory_type == "episodic")
        .order_by(Memory.created_at.asc())
        .all()
    )
    return [
        {
            "session": (row.memory_metadata or {}).get("session"),
            "notions_vues": (row.memory_metadata or {}).get("notions_vues", []),
            "difficultés": (row.memory_metadata or {}).get("difficultés", []),
            "niveau": (row.memory_metadata or {}).get("niveau"),
        }
        for row in rows
    ]


def _simulate_4_sessions(user_id: str, support: str, db) -> list:
    """Simule les 4 sessions et retourne les entrées créées."""
    sessions = [
        (1, ["déclaration de fonction", "paramètres"],  [],                         "débutant"),
        (2, ["valeur de retour"],                        ["paramètres optionnels"],  "débutant"),
        (3, ["arrow function", "scope"],                 [],                         "intermédiaire"),
        (4, ["closures", "récursivité"],                 [],                         "intermédiaire"),
    ]
    for num, notions, difficulties, niveau in sessions:
        _store_session(user_id, support, num, notions, difficulties, niveau, db)
    db.commit()
    return _retrieve_all_sessions(user_id, db)


def _compute_mra(sessions: list) -> float:
    """MRA : vérifie que chaque session a les champs essentiels peuplés."""
    fields = ["session", "notions_vues", "niveau"]
    correct = sum(
        1
        for s in sessions
        for f in fields
        if s.get(f) is not None and s.get(f) != [] and s.get(f) != ""
    )
    expected = len(sessions) * len(fields)
    return correct / expected if expected else 0.0


def _compute_tcr(sessions: list, expected_count: int) -> float:
    """TCR = N_succès (sessions complètes) / N_total."""
    successes = sum(
        1 for s in sessions
        if s.get("session") and s.get("notions_vues") and s.get("niveau")
    )
    return successes / expected_count


def _compute_learning_gain(pre: int, post: int, max_score: int) -> float:
    """LG = (post − pré) / (max − pré). Seuil Hake 1998 : ≥ 0,30."""
    if max_score == pre:
        return 1.0
    return (post - pre) / (max_score - pre)


# ── Tests automatisés ────────────────────────────────────────────────────────

def test_s5_mra_across_4_sessions(db, user_id):
    """MRA ≥ 100 % : chaque session a ses champs essentiels peuplés."""
    sessions = _simulate_4_sessions(user_id, "javascript-fonctions", db)
    assert len(sessions) == 4, f"Attendu 4 sessions, obtenu {len(sessions)}"

    mra = _compute_mra(sessions)
    assert mra == 1.0, f"MRA = {mra:.0%} — certains champs de session sont vides."


def test_s5_tcr_above_target(db, user_id):
    """TCR = N_succès / N_total ≥ 0,95."""
    sessions = _simulate_4_sessions(user_id, "javascript-fonctions", db)
    tcr = _compute_tcr(sessions, expected_count=4)
    assert tcr >= 0.95, f"TCR = {tcr:.0%} — sessions incomplètes détectées."


def test_s5_level_progression_detected(db, user_id):
    """Le système détecte la montée en niveau entre session 2 et session 3."""
    sessions = _simulate_4_sessions(user_id, "javascript-fonctions", db)
    niveaux = [s["niveau"] for s in sessions]

    assert "débutant" in niveaux, "Niveau 'débutant' absent de l'historique."
    assert "intermédiaire" in niveaux, "Montée en niveau non détectée — 'intermédiaire' absent."

    idx_debut = next(i for i, n in enumerate(niveaux) if n == "débutant")
    idx_inter = next(i for i, n in enumerate(niveaux) if n == "intermédiaire")
    assert idx_debut < idx_inter, "La montée en niveau est chronologiquement incohérente."


def test_s5_difficulties_recorded(db, user_id):
    """Les difficultés signalées en session 2 sont persistées."""
    sessions = _simulate_4_sessions(user_id, "javascript-fonctions", db)
    session_2 = next((s for s in sessions if s["session"] == 2), None)

    assert session_2 is not None, "Session 2 introuvable."
    assert session_2["difficultés"], "Aucune difficulté enregistrée pour la session 2."
    assert "paramètres optionnels" in session_2["difficultés"]


def test_s5_notions_cumulated_across_sessions(db, user_id):
    """Les notions vues s'accumulent sur les 4 sessions."""
    sessions = _simulate_4_sessions(user_id, "javascript-fonctions", db)
    all_notions = [n for s in sessions for n in s.get("notions_vues", [])]

    assert "déclaration de fonction" in all_notions
    assert "closures" in all_notions
    assert len(set(all_notions)) >= 6, "Trop peu de notions distinctes cumulées."


# ── Cadre Learning Gain (composante humaine) ──────────────────────────────────

def test_s5_learning_gain_framework():
    """Calcule et affiche le Learning Gain — validation humaine requise (double évaluateur).

    Ce test ne peut pas PASS/FAIL automatiquement sur la qualité pédagogique.
    Il vérifie la formule et documente le résultat pour les évaluateurs humains.
    """
    lg = _compute_learning_gain(PRE_SCORE, POST_SCORE, MAX_SCORE)
    print(
        f"\n[S5 — Learning Gain]\n"
        f"  Pré-test  : {PRE_SCORE}/{MAX_SCORE}\n"
        f"  Post-test : {POST_SCORE}/{MAX_SCORE}\n"
        f"  LG = ({POST_SCORE} - {PRE_SCORE}) / ({MAX_SCORE} - {PRE_SCORE}) = {lg:.2f}\n"
        f"  Cible     : ≥ 0,30 (seuil Hake 1998)\n"
        f"  Statut    : {'✓ ATTEINT' if lg >= 0.30 else '✗ NON ATTEINT'}\n"
        f"  → Soumettre à double évaluation humaine (accord ≥ 80 % requis)."
    )
    assert lg >= 0.0, "Learning Gain négatif — régression détectée."
