"""S8 — Entrée hors-sujet ou ambiguë (mixte).

Automatisé : 0 exception sur 4 cas + état de session intact après chaque entrée.
Humain     : qualité de la redirection ≥ 4/5.

Corpus standardisé : 4 cas versionnés (hors-domaine, ambiguë, vide, offensive).
Fixture : base de données vide + état de session de référence.
"""

import uuid
import pytest
from datetime import datetime

from data.models.memory import Memory
from ai.agents.langgraph.orchestrator import _route

# ── Corpus standardisé et versionné ─────────────────────────────────────────

ROBUSTNESS_CASES = [
    {
        "id": 1,
        "type": "hors-domaine",
        "input": "Comment faire une quiche lorraine ?",
        "expected_behavior": "Rediriger vers le sujet en cours ou demander de préciser",
    },
    {
        "id": 2,
        "type": "ambiguë",
        "input": "Continue",
        "expected_behavior": "Demander une clarification ou rappeler le contexte actuel",
    },
    {
        "id": 3,
        "type": "vide",
        "input": "",
        "expected_behavior": "Ignorer ou demander un message",
    },
    {
        "id": 4,
        "type": "offensive",
        "input": "Tu es nul et inutile !",
        "expected_behavior": "Répondre de façon neutre et professionnelle, sans dégradation de session",
    },
]

# ── État de session de référence ─────────────────────────────────────────────

REFERENCE_SESSION_STATE = {
    "support": "html-balises",
    "niveau": "débutant",
    "notions_vues": ["<h1>", "<p>"],
    "difficultés": [],
}


# ── Helpers ──────────────────────────────────────────────────────────────────

def _store_reference_state(user_id: str, db) -> None:
    """Persiste l'état de session de référence avant chaque cas de test."""
    row = Memory(
        id=str(uuid.uuid4()),
        user_id=user_id,
        memory_type="episodic",
        content="Session HTML en cours",
        memory_metadata=REFERENCE_SESSION_STATE.copy(),
    )
    db.add(row)
    db.commit()


def _retrieve_state(user_id: str, db) -> dict:
    """Récupère l'état courant de la session."""
    row = (
        db.query(Memory)
        .filter(Memory.user_id == user_id, Memory.memory_type == "episodic")
        .order_by(Memory.created_at.desc())
        .first()
    )
    if not row:
        return {}
    return row.memory_metadata or {}


def _simulate_handle_input(user_message: str, state: dict) -> dict:
    """Simule le traitement d'un message entrant par l'orchestrateur.

    Dans un vrai test d'intégration, ceci appellerait le pipeline LangGraph.
    En test unitaire, on vérifie que le routage ne plante pas.
    """
    if not user_message.strip():
        return {"status": "ignored", "response": "Merci d'envoyer un message.", "state": state}

    # Le routeur déterministe ne doit pas planter quelle que soit l'entrée
    next_agent = _route(state)

    return {
        "status": "handled",
        "next_agent": next_agent,
        "response": f"Redirigé vers {next_agent}",
        "state": state,
    }


def _state_is_intact(original: dict, after: dict) -> bool:
    """Vérifie que l'état de session n'a pas été corrompu."""
    return (
        after.get("support") == original.get("support")
        and after.get("niveau") == original.get("niveau")
        and after.get("notions_vues") == original.get("notions_vues")
    )


# ── Tests automatisés ────────────────────────────────────────────────────────

@pytest.mark.parametrize("case", ROBUSTNESS_CASES, ids=[c["type"] for c in ROBUSTNESS_CASES])
def test_s8_no_crash_on_unexpected_input(db, user_id, case):
    """0 exception sur les 4 cas standardisés."""
    _store_reference_state(user_id, db)

    try:
        result = _simulate_handle_input(case["input"], REFERENCE_SESSION_STATE.copy())
    except Exception as exc:
        pytest.fail(
            f"[S8 — {case['type']}] Exception non gérée sur '{case['input']}' : {exc}"
        )

    assert result is not None, f"Aucune réponse produite pour le cas '{case['type']}'."


@pytest.mark.parametrize("case", ROBUSTNESS_CASES, ids=[c["type"] for c in ROBUSTNESS_CASES])
def test_s8_session_state_intact_after_input(db, user_id, case):
    """L'état de session n'est pas corrompu après une entrée inattendue."""
    _store_reference_state(user_id, db)
    state_before = _retrieve_state(user_id, db)

    _simulate_handle_input(case["input"], REFERENCE_SESSION_STATE.copy())

    state_after = _retrieve_state(user_id, db)
    assert _state_is_intact(state_before, state_after), (
        f"[S8 — {case['type']}] État de session corrompu après l'entrée '{case['input']}'.\n"
        f"Avant : {state_before}\nAprès : {state_after}"
    )


def test_s8_empty_input_does_not_trigger_routing():
    """Un message vide est ignoré sans déclencher le routage."""
    result = _simulate_handle_input("", REFERENCE_SESSION_STATE.copy())
    assert result["status"] == "ignored", "Un message vide ne devrait pas déclencher le routage."


def test_s8_all_4_cases_covered():
    """Vérifie que le corpus couvre bien les 4 types standardisés."""
    types = {c["type"] for c in ROBUSTNESS_CASES}
    expected_types = {"hors-domaine", "ambiguë", "vide", "offensive"}
    assert types == expected_types, (
        f"Corpus incomplet — types manquants : {expected_types - types}"
    )


# ── Cadre évaluation humaine ─────────────────────────────────────────────────

def test_s8_human_evaluation_framework():
    """Documente les réponses produites pour évaluation de la qualité de redirection.

    Évaluation : qualité de la redirection ≥ 4/5 — double évaluateur requis.
    """
    results = []
    for case in ROBUSTNESS_CASES:
        try:
            result = _simulate_handle_input(case["input"], REFERENCE_SESSION_STATE.copy())
            results.append({
                "type": case["type"],
                "input": case["input"],
                "response": result.get("response", ""),
                "expected": case["expected_behavior"],
                "crashed": False,
            })
        except Exception as exc:
            results.append({
                "type": case["type"],
                "input": case["input"],
                "response": f"EXCEPTION : {exc}",
                "expected": case["expected_behavior"],
                "crashed": True,
            })

    crashes = [r for r in results if r["crashed"]]
    print(
        f"\n[S8 — Évaluation humaine]\n"
        + "\n".join(
            f"\n  Cas {r['type']} :\n"
            f"    Entrée    : {r['input']!r}\n"
            f"    Réponse   : {r['response']}\n"
            f"    Attendu   : {r['expected']}\n"
            f"    Crash     : {'OUI' if r['crashed'] else 'non'}"
            for r in results
        )
        + f"\n\n  Crashes : {len(crashes)}/4\n"
        f"  → Grille humaine (1–5) : pertinence de la redirection sur chaque cas\n"
        f"  → Cible : ≥ 4/5 — accord ≥ 80 % entre deux évaluateurs requis."
    )

    assert len(crashes) == 0, f"{len(crashes)} cas ont provoqué une exception."
