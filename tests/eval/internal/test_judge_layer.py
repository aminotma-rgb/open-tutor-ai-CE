"""Tests de démonstration de la couche d'évaluation par juge LLM.

Exécutés en mode `offline_judge` (déterministe, sans LLM) → reproductibles en CI.
Pour une mesure réelle, remplacer `offline_judge` par `llm_judge`.

    pytest tests/eval/test_judge_layer.py -v -s
"""

from tests.eval.internal.eval_judge import (
    faithfulness,
    answer_relevancy,
    context_recall,
    offline_judge,
)

# ── Données d'exemple (domaine Python — boucles for, cohérent avec S0) ────────

CONTEXTS = [
    "Une boucle for en Python répète un bloc de code pour chaque élément d'une séquence.",
    "La fonction range(n) génère les entiers de 0 à n-1.",
    "On écrit : for i in range(5): print(i) pour afficher 0 à 4.",
]
QUESTION = "Comment fonctionne une boucle for en Python ?"
ANSWER_FAITHFUL = (
    "Une boucle for répète un bloc pour chaque élément d'une séquence. "
    "La fonction range(n) produit les entiers de 0 à n-1."
)
ANSWER_HALLUCINATED = (
    "Une boucle for en Python compile le code en C avant l'exécution "
    "et utilise un GPU pour accélérer les itérations."
)
GROUND_TRUTH = "Une boucle for itère sur une séquence ; range(n) donne 0 à n-1."


# ── Dimension 2 — RAGAS-style ────────────────────────────────────────────────

def test_d2_faithfulness_detecte_reponse_ancree():
    score = faithfulness(ANSWER_FAITHFUL, CONTEXTS, judge=offline_judge)
    print(f"\n[D2] Faithfulness (réponse ancrée)   = {score:.2f}")
    assert score >= 0.50, "Une réponse ancrée devrait obtenir un Faithfulness élevé."


def test_d2_faithfulness_detecte_hallucination():
    score = faithfulness(ANSWER_HALLUCINATED, CONTEXTS, judge=offline_judge)
    print(f"[D2] Faithfulness (hallucination)    = {score:.2f}")
    assert score < 0.50, "Une hallucination devrait faire chuter le Faithfulness."


def test_d2_answer_relevancy():
    score = answer_relevancy(QUESTION, ANSWER_FAITHFUL, judge=offline_judge)
    print(f"[D2] Answer Relevancy                = {score:.2f}")
    assert 0.0 <= score <= 1.0


def test_d2_context_recall():
    score = context_recall(GROUND_TRUTH, CONTEXTS, judge=offline_judge)
    print(f"[D2] Context Recall                  = {score:.2f}")
    assert score >= 0.50, "Le contexte couvre la vérité-terrain → recall élevé."

