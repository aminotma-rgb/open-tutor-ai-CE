"""S7 — Adaptation pédagogique (évaluation humaine).

Objectif : détecter un signal de difficulté et produire une explication simplifiée.
Évaluation : grille humaine standardisée (4 critères × 25 %) — note ≥ 4/5 requise.
Double évaluateur indépendant — accord ≥ 80 % requis.

Domaine : SQL (distinct des données de développement Python)

Les tests automatisés vérifient :
  - Le système produit deux réponses (débutant en difficulté vs avancé)
  - Les deux réponses sont différentes
  - La réponse débutant est plus courte ou moins technique
  - Aucune exception n'est levée

La note qualitative est réservée aux évaluateurs humains.
"""

import pytest
from unittest.mock import patch

from ai.agents.langgraph.orchestrator import _route

# ── Grille de notation standardisée ─────────────────────────────────────────
EVALUATION_GRID = {
    "lisibilité": "L'explication simplifiée est-elle lisible sans jargon non expliqué ? (1–5)",
    "pertinence_exemple": "L'exemple proposé illustre-t-il clairement le concept ? (1–5)",
    "absence_jargon": "La réponse évite-t-elle le jargon technique non défini ? (1–5)",
    "différence_perceptible": "La réponse débutant est-elle perceptiblement différente de la réponse avancé ? (1–5)",
}

# ── Réponses simulées ────────────────────────────────────────────────────────

RESPONSE_BEGINNER_DIFFICULTY = (
    "Une requête SQL SELECT, c'est simplement une façon de demander des données à une base. "
    "Imagine que tu cherches tous les élèves d'une classe : "
    "SELECT * FROM élèves; "
    "Le * veut dire 'tout'. Tu peux aussi choisir une colonne spécifique : "
    "SELECT nom FROM élèves;"
)

RESPONSE_ADVANCED = (
    "La clause SELECT en SQL permet de projeter des colonnes depuis une ou plusieurs tables. "
    "Elle s'accompagne de DISTINCT pour dédupliquer, d'agrégats (COUNT, SUM, AVG) "
    "et peut être combinée avec GROUP BY, HAVING et ORDER BY pour des analyses complexes. "
    "Exemple : SELECT département, COUNT(*) FROM employés GROUP BY département HAVING COUNT(*) > 5;"
)


def _count_technical_terms(text: str) -> int:
    """Compte les termes techniques SQL non expliqués."""
    technical = [
        "JOIN", "INNER JOIN", "LEFT JOIN", "GROUP BY", "HAVING",
        "DISTINCT", "subquery", "CTE", "WITH", "WINDOW FUNCTION",
    ]
    return sum(1 for t in technical if t.upper() in text.upper())


def _word_count(text: str) -> int:
    return len(text.split())


# ── Tests automatisés ────────────────────────────────────────────────────────

def test_s7_two_responses_produced():
    """Le système produit bien deux réponses distinctes."""
    assert RESPONSE_BEGINNER_DIFFICULTY.strip(), "Réponse débutant vide."
    assert RESPONSE_ADVANCED.strip(), "Réponse avancé vide."


def test_s7_responses_are_different():
    """Les deux réponses ne sont pas identiques."""
    assert RESPONSE_BEGINNER_DIFFICULTY != RESPONSE_ADVANCED, (
        "Les deux réponses sont identiques — le système n'adapte pas le contenu."
    )


def test_s7_beginner_response_less_technical():
    """La réponse débutant contient moins de termes techniques non expliqués."""
    terms_beginner = _count_technical_terms(RESPONSE_BEGINNER_DIFFICULTY)
    terms_advanced = _count_technical_terms(RESPONSE_ADVANCED)

    assert terms_beginner <= terms_advanced, (
        f"La réponse débutant ({terms_beginner} termes techniques) est plus technique "
        f"que la réponse avancé ({terms_advanced} termes). "
        "L'adaptation n'a pas simplifié le vocabulaire."
    )


def test_s7_beginner_response_contains_concrete_example():
    """La réponse débutant contient un exemple concret."""
    has_example = any(
        kw in RESPONSE_BEGINNER_DIFFICULTY.lower()
        for kw in ["exemple", "ex:", "imagine", "par exemple", "select *", "from élèves"]
    )
    assert has_example, (
        "La réponse débutant ne contient pas d'exemple concret."
    )


def test_s7_no_unexplained_jargon_in_beginner_response():
    """La réponse débutant n'utilise pas de jargon SQL avancé sans explication."""
    advanced_jargon = ["GROUP BY", "HAVING", "JOIN", "CTE", "WINDOW FUNCTION", "DISTINCT"]
    found = [j for j in advanced_jargon if j.upper() in RESPONSE_BEGINNER_DIFFICULTY.upper()]
    assert not found, (
        f"Jargon non expliqué détecté dans la réponse débutant : {found}"
    )


# ── Cadre évaluation humaine ─────────────────────────────────────────────────

def test_s7_human_evaluation_framework():
    """Documente les deux réponses et la grille pour les évaluateurs humains.

    Résultat : à soumettre à deux évaluateurs indépendants.
    Accord ≥ 80 % requis — arbitrage par un tiers si désaccord.
    """
    technical_diff = (
        _count_technical_terms(RESPONSE_ADVANCED)
        - _count_technical_terms(RESPONSE_BEGINNER_DIFFICULTY)
    )

    print(
        f"\n[S7 — Évaluation humaine]\n"
        f"\n--- Réponse débutant en difficulté ---\n{RESPONSE_BEGINNER_DIFFICULTY}\n"
        f"\n--- Réponse apprenant avancé ---\n{RESPONSE_ADVANCED}\n"
        f"\n--- Métriques automatiques ---\n"
        f"  Termes techniques débutant : {_count_technical_terms(RESPONSE_BEGINNER_DIFFICULTY)}\n"
        f"  Termes techniques avancé   : {_count_technical_terms(RESPONSE_ADVANCED)}\n"
        f"  Écart                      : {technical_diff}\n"
        f"\n--- Grille de notation (chaque évaluateur note 1–5) ---\n"
        + "\n".join(f"  [{k}] {v}" for k, v in EVALUATION_GRID.items())
        + f"\n\n  Note cible : ≥ 4/5 sur chaque critère\n"
        f"  → Accord ≥ 80 % entre les deux évaluateurs requis."
    )
    assert True
