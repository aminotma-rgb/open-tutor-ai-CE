"""Tests externes MRBench — marqués @pytest.mark.external (exclus de la CI rapide).

Exécuter manuellement avec un LLM réel configuré :
    pytest tests/eval/external/test_external_mrbench.py -v -m external
"""

import json
import os
import warnings

import pytest

DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets", "mrbench_sample.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results", "mrbench_results.json")
AXES = [
    "mistake_identification",
    "mistake_location",
    "guidance",
    "actionability",
    "revealing_of_answer",
]

_RUN_EXTERNAL = os.getenv("OTAI_RUN_EXTERNAL", "").lower() in ("1", "true", "yes")


@pytest.mark.external
def test_mrbench_scores_by_axis():
    """Mesure le score DAMR par axe MRBench et génère les résultats.

    La cible de 0,50 par axe est indicative, pas bloquante : un score sous la
    cible émet un warning (visible dans le résumé pytest) plutôt que de faire
    échouer le test. Objectif de ce test = produire une mesure de performance
    du système, pas valider un seuil de qualité — voir DESCRIPTION dans
    docs/evaluation.md.
    """
    if not _RUN_EXTERNAL:
        pytest.skip("Mettre OTAI_RUN_EXTERNAL=1 pour activer les tests avec LLM réel.")
    if not os.path.exists(DATASET_PATH):
        pytest.skip("Dataset MRBench absent — exécuter scripts/sample_mrbench.py d'abord.")

    from tests.eval.external.mrbench_runner import run_mrbench
    from tests.eval.internal.eval_judge import llm_judge

    results = run_mrbench(judge=llm_judge)

    for axis in AXES:
        avg = sum(r[axis] for r in results) / len(results)
        print(f"  [{axis}] = {avg:.2f}")
        if avg < 0.50:
            warnings.warn(f"Score sous la cible indicative (0,50) pour {axis} : {avg:.2f}")


@pytest.mark.external
def test_mrbench_results_persisted():
    """Le fichier de résultats JSON est écrit après l'exécution."""
    if not _RUN_EXTERNAL:
        pytest.skip("Mettre OTAI_RUN_EXTERNAL=1 pour activer les tests avec LLM réel.")
    if not os.path.exists(RESULTS_PATH):
        pytest.skip("Résultats absents — exécuter test_mrbench_scores_by_axis d'abord.")

    with open(RESULTS_PATH) as f:
        data = json.load(f)
    assert len(data) > 0
    assert all("id" in r and "source" in r for r in data)
    for axis in AXES:
        assert all(axis in r for r in data), f"Axe '{axis}' manquant dans les résultats."
