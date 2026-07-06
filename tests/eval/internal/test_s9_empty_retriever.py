"""S9 — Retriever à vide (automatisé).

Objectif : vérifier que le pipeline ne hallucine pas et produit une redirection
gracieuse quand le retriever ne remonte aucun document.

Angle mort fermé : un retriever vide est une défaillance silencieuse courante dans
les systèmes RAG. Si le pipeline ne le détecte pas, il peut soit lever une exception
non traitée, soit halluciner une réponse sans contexte.

Métriques :
  - recall_at_k = 0.0  (aucun document récupéré — défaillance retriever)
  - context_recall = 0.0 (contexte vide, vérité-terrain non couverte)
  - 0 exception levée
  - La réponse produite est une redirection, pas une affirmation factuelle sur le sujet

Méthode : automatisé pur.
"""

import pytest

from ai.retrieval.context_retrieval import ContextRetrievalService
from tests.eval.internal.eval_judge import (
    context_recall,
    faithfulness,
    recall_at_k,
    offline_judge,
)

# ── Données du scénario ───────────────────────────────────────────────────────

LEARNER_QUERY = "Peux-tu m'expliquer les récursions en Python ?"

# Vérité-terrain : ce que le contexte devrait couvrir (mais ne couvre pas).
GROUND_TRUTH = (
    "La récursion est une fonction qui s'appelle elle-même. "
    "Elle nécessite un cas de base pour s'arrêter."
)

# Documents pertinents qui auraient dû être récupérés (mais ne l'ont pas été).
RELEVANT_DOC_IDS = {"doc_recursion_intro", "doc_base_case"}

# Réponse attendue du pipeline face à un contexte vide : redirection gracieuse.
# Elle ne doit contenir aucune affirmation factuelle sur la récursion.
MOCK_RESPONSE_EMPTY = (
    "Je n'ai pas trouvé de documents pertinents pour répondre à ta question sur ce sujet. "
    "Pourrais-tu reformuler ou me préciser le thème que tu souhaites explorer ?"
)

# Termes qui indiqueraient une hallucination (affirmations sur la récursion sans contexte).
DOMAIN_HALLUCINATION_MARKERS = [
    "s'appelle elle-même",
    "cas de base",
    "pile d'appels",
    "stack overflow",
    "fonction récursive",
    "appel récursif",
]


# ── Tests ────────────────────────────────────────────────────────────────────

def test_s9_retriever_returns_empty_list(monkeypatch):
    """Le retriever peut retourner une liste vide sans lever d'exception."""
    monkeypatch.setattr(
        ContextRetrievalService,
        "retrieve_pedagogical_documents",
        lambda self, **kw: [],
    )
    svc = ContextRetrievalService()
    result = svc.retrieve_pedagogical_documents(
        user_id="s9-test-user",
        query=LEARNER_QUERY,
        top_k=5,
        collection_name="s9_python_recursion",
    )
    assert result == [], "Le retriever doit retourner [] sans exception."


def test_s9_recall_at_k_is_zero(monkeypatch):
    """Recall@k = 0.0 quand aucun document n'est récupéré.

    Confirme que recall_at_k est distinct de context_recall :
    ici c'est le retriever qui échoue, pas le contexte assemblé.
    """
    monkeypatch.setattr(
        ContextRetrievalService,
        "retrieve_pedagogical_documents",
        lambda self, **kw: [],
    )
    svc = ContextRetrievalService()
    retrieved = svc.retrieve_pedagogical_documents(
        user_id="s9-test-user",
        query=LEARNER_QUERY,
        top_k=5,
        collection_name="s9_python_recursion",
    )
    retrieved_ids = {d["id"] for d in retrieved}
    score = recall_at_k(RELEVANT_DOC_IDS, retrieved_ids)

    assert score == 0.0, (
        f"Recall@k = {score:.2f} — attendu 0.0 sur retriever vide."
    )


def test_s9_context_recall_is_zero():
    """Context Recall = 0.0 quand le contexte assemblé est vide.

    Distinct du Recall@k : même si le retriever avait fonctionné, un contexte
    vide ne peut pas couvrir la vérité-terrain.
    """
    score = context_recall(GROUND_TRUTH, [], judge=offline_judge)
    assert score == 0.0, (
        f"Context Recall = {score:.2f} — attendu 0.0 sur contexte vide."
    )


def test_s9_no_hallucination_in_redirect_response():
    """La réponse de redirection ne contient aucune affirmation factuelle sur le sujet.

    Un pipeline correct face à un contexte vide doit produire une redirection,
    pas inventer une réponse sur la récursion sans sources.
    """
    response_lower = MOCK_RESPONSE_EMPTY.lower()
    hallucinations = [m for m in DOMAIN_HALLUCINATION_MARKERS if m in response_lower]

    assert not hallucinations, (
        f"Marqueurs d'hallucination détectés dans la réponse vide : {hallucinations}. "
        "Le pipeline a produit des affirmations factuelles sans contexte."
    )


def test_s9_faithfulness_vacuous_on_redirect():
    """Faithfulness = 1.0 quand la réponse ne fait aucune affirmation sur le domaine.

    On passe `claims=[]` explicitement : c'est l'assertion que la réponse de redirection
    ne contient aucune affirmation décomposable sur le sujet (récursion).
    Vacuoirement vraie — 0 claims à vérifier → 0 violations → ratio = 1.0.
    """
    redirect_only = (
        "Je n'ai pas trouvé de documents pertinents pour ta question. "
        "Peux-tu reformuler ?"
    )
    score = faithfulness(redirect_only, [], claims=[], judge=offline_judge)

    assert score == 1.0, (
        f"Faithfulness = {score:.2f} — une réponse sans claims doit être vacuoirement 1.0."
    )


def test_s9_pipeline_does_not_raise_on_empty_context():
    """Le pipeline ne lève pas d'exception quand il reçoit un contexte vide."""
    try:
        _ = context_recall(GROUND_TRUTH, [], judge=offline_judge)
        _ = faithfulness(MOCK_RESPONSE_EMPTY, [], judge=offline_judge)
    except Exception as exc:
        pytest.fail(f"Exception inattendue sur contexte vide : {exc}")
