"""Tests externes LongMemEval-S — marqués @pytest.mark.external (exclus de la CI rapide).

Exécuter manuellement avec un LLM réel configuré :
    pytest tests/eval/external/test_external_longmemeval.py -v -m external
"""

import json
import os
import pytest
from collections import defaultdict

DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets", "longmemeval_s_sample.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "longmemeval_results.json")

_RUN_EXTERNAL = os.getenv("OTAI_RUN_EXTERNAL", "").lower() in ("1", "true", "yes")


@pytest.mark.external
def test_longmemeval_accuracy_by_category():
    """Précision globale ≥ 0,50 et résultats disponibles par catégorie."""
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

    overall = sum(r["correct"] for r in results) / len(results)
    print(f"  Overall accuracy = {overall:.2f}")
    assert overall >= 0.50, f"Précision globale trop basse : {overall:.2f}"


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
