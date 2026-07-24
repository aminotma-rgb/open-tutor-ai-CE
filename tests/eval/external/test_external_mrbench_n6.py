"""Test externe MRBench allégé — revealing_of_answer seul, N=6.

Vérifie rapidement l'effet du 2e correctif (suppression de la réponse dans
le bloc PREPARED EXERCISES) avant d'investir dans un run plus large.
Séquentiel via tests/eval/external/conftest.py (xdist_group sur tout le
package).

Exécuter manuellement avec un LLM réel configuré :
    OTAI_RUN_EXTERNAL=1 pytest tests/eval/external/test_external_mrbench_n6.py -v -m external
"""

import json
import os
import warnings

import pytest

DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets", "mrbench_sample_n6.json")
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "mrbench_results_n6_revealing.json"
)

_RUN_EXTERNAL = os.getenv("OTAI_RUN_EXTERNAL", "").lower() in ("1", "true", "yes")


@pytest.mark.external
def test_mrbench_n6_revealing_of_answer():
    """Mesure DAMR de revealing_of_answer sur 6 instances (3 Bridge + 3 MathDial).

    Cible indicative (warning), pas bloquante — échantillon délibérément petit,
    pensé pour un signal directionnel rapide, pas une mesure précise.
    """
    if not _RUN_EXTERNAL:
        pytest.skip("Mettre OTAI_RUN_EXTERNAL=1 pour activer les tests avec LLM réel.")
    if not os.path.exists(DATASET_PATH):
        pytest.skip("Dataset mrbench_sample_n6.json absent.")

    from tests.eval.external.mrbench_runner_n6 import run_mrbench_n6, AXIS
    from tests.eval.internal.eval_judge import llm_judge

    results = run_mrbench_n6(judge=llm_judge)

    avg = sum(r[AXIS] for r in results) / len(results)
    print(f"  [{AXIS}] (N=6) = {avg:.2f}")
    if avg < 0.50:
        warnings.warn(f"Score sous la cible indicative (0,50) pour {AXIS} (N=6) : {avg:.2f}")


@pytest.mark.external
def test_mrbench_n6_results_persisted():
    """Le fichier de résultats JSON est écrit après l'exécution."""
    if not _RUN_EXTERNAL:
        pytest.skip("Mettre OTAI_RUN_EXTERNAL=1 pour activer les tests avec LLM réel.")
    if not os.path.exists(RESULTS_PATH):
        pytest.skip("Résultats absents — exécuter test_mrbench_n6_revealing_of_answer d'abord.")

    with open(RESULTS_PATH) as f:
        data = json.load(f)
    assert len(data) > 0
    assert all("id" in r and "source" in r and "revealing_of_answer" in r for r in data)
