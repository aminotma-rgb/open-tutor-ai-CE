"""Tests externes LongMemEval-S — marqués @pytest.mark.external (exclus de la CI rapide).

Exécuter manuellement avec un LLM réel configuré :
    pytest tests/eval/external/test_external_longmemeval.py -v -m external
"""

import json
import os
import warnings
import pytest
from collections import defaultdict

DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets", "longmemeval_s_sample.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "longmemeval_results.json")

_RUN_EXTERNAL = os.getenv("OTAI_RUN_EXTERNAL", "").lower() in ("1", "true", "yes")


@pytest.mark.external
def test_longmemeval_accuracy_by_category():
    """Mesure la précision par catégorie LongMemEval-S et produit les résultats.

    Cible de 0,50 indicative, pas bloquante (2026-07-11) — par cohérence avec
    MRBench et LoCoMo : à N=4 (portée réduite à knowledge-update), chaque
    instance pèse 25 points, un score bas émet un warning plutôt qu'un échec
    de test. Ça ne réintroduit pas le risque de "faux succès silencieux" déjà
    rencontré en Phase 3 (score 0,00 invisible faute de LLM_MODEL chargé) :
    un warning reste visible dans le résumé pytest, seul le blocage du run
    disparaît.
    """
    if not _RUN_EXTERNAL:
        pytest.skip("Mettre OTAI_RUN_EXTERNAL=1 pour activer les tests avec LLM réel.")
    if not os.path.exists(DATASET_PATH):
        pytest.skip("Dataset LongMemEval-S absent — exécuter scripts/sample_longmemeval.py d'abord.")

    from tests.eval.external.longmemeval_runner import run_longmemeval
    from tests.eval.internal.eval_judge import llm_judge

    results = run_longmemeval(judge=llm_judge)

    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["category"]].append(r["correct"])

    for cat, vals in by_cat.items():
        acc = sum(vals) / len(vals)
        print(f"  [{cat}] accuracy = {acc:.2f}  ({sum(vals)}/{len(vals)})")
        if acc < 0.50:
            warnings.warn(f"Score sous la cible indicative (0,50) pour {cat} : {acc:.2f}")

    overall = sum(r["correct"] for r in results) / len(results)
    print(f"  Overall accuracy = {overall:.2f}")
    if overall < 0.50:
        warnings.warn(f"Précision globale sous la cible indicative (0,50) : {overall:.2f}")


@pytest.mark.external
def test_longmemeval_results_persisted():
    """Le fichier de résultats JSON est écrit après l'exécution."""
    if not _RUN_EXTERNAL:
        pytest.skip("Mettre OTAI_RUN_EXTERNAL=1 pour activer les tests avec LLM réel.")
    if not os.path.exists(RESULTS_PATH):
        pytest.skip("Résultats absents — exécuter test_longmemeval_accuracy_by_category d'abord.")

    with open(RESULTS_PATH) as f:
        data = json.load(f)
    assert len(data) > 0
    assert all("question_id" in r and "correct" in r for r in data)
