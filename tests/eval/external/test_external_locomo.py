"""Tests externes LoCoMo — marqués @pytest.mark.external (exclus de la CI rapide).

Séquentiel comme les autres runners externes via
tests/eval/external/conftest.py (aucune configuration supplémentaire requise
ici — le marker xdist_group s'applique à tout le package eval/external).

Exécuter manuellement avec un LLM réel configuré :
    pytest tests/eval/external/test_external_locomo.py -v -m external
"""

import json
import os
import warnings
from collections import defaultdict

import pytest

DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets", "locomo_sample.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "locomo_results.json")

_RUN_EXTERNAL = os.getenv("OTAI_RUN_EXTERNAL", "").lower() in ("1", "true", "yes")


@pytest.mark.external
def test_locomo_accuracy_by_category():
    """Mesure la précision par catégorie LoCoMo — cible 0,50 indicative, pas
    bloquante (voir warnings.warn ci-dessous). Échantillon volontairement
    petit (3 conversations, ~28 questions, 5 catégories dont certaines à
    n=2) — même logique que pour MRBench : produire une mesure de
    performance, pas garantir un seuil de qualité en CI à cet ordre de
    grandeur, statistiquement peu discriminant.
    """
    if not _RUN_EXTERNAL:
        pytest.skip("Mettre OTAI_RUN_EXTERNAL=1 pour activer les tests avec LLM réel.")
    if not os.path.exists(DATASET_PATH):
        pytest.skip("Dataset LoCoMo absent — exécuter scripts/sample_locomo.py d'abord.")

    from tests.eval.external.locomo_runner import run_locomo
    from tests.eval.internal.eval_judge import llm_judge

    results = run_locomo(judge=llm_judge)

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


@pytest.mark.external
def test_locomo_results_persisted():
    """Le fichier de résultats JSON est écrit après l'exécution."""
    if not _RUN_EXTERNAL:
        pytest.skip("Mettre OTAI_RUN_EXTERNAL=1 pour activer les tests avec LLM réel.")
    if not os.path.exists(RESULTS_PATH):
        pytest.skip("Résultats absents — exécuter test_locomo_accuracy_by_category d'abord.")

    with open(RESULTS_PATH) as f:
        data = json.load(f)
    assert len(data) > 0
    assert all("sample_id" in r and "category" in r and "correct" in r for r in data)
