"""Harness MRBench allégé — mesure rapide de revealing_of_answer seul (N=10).

Objectif : mesurer l'effet du correctif de prompt appliqué à
gateway/http/routers/adaptive.py (branche "correct is False" de
_build_enriched_system_prompt, qui ordonnait explicitement de révéler la
réponse) sans attendre un run complet sur les 5 axes / 40 instances.

Le temps d'un run MRBench est dominé par la génération (faire tourner le
pipeline LangGraph complet par dialogue, ~28 min/dialogue observé), pas par
le jugement — juger moins d'axes ne fait presque rien gagner. Le seul levier
qui réduit vraiment le temps est de réduire N, d'où ce harness séparé plutôt
qu'une simple option sur mrbench_runner.py.

Échantillon : les 10 premières instances (5 Bridge + 5 MathDial) du même
tests/eval/external/datasets/mrbench_sample.json déjà utilisé pour le run à
N=40 — sous-ensemble déterministe, pas un nouveau tirage aléatoire, pour
rester comparable au run précédent sur ces mêmes dialogues.

Réutilise generate_response()/score_axis() de mrbench_runner.py — aucune
logique dupliquée, seule la boucle d'orchestration diffère (1 axe au lieu
de 5, dataset réduit, fichier de résultats séparé).
"""

from __future__ import annotations

import json
import os

from tests.eval.external.mrbench_runner import generate_response, score_axis

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "datasets", "mrbench_sample_n10.json"
)
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "mrbench_results_n10_revealing.json"
)

AXIS = "revealing_of_answer"


def run_mrbench_n10(judge=None):
    if judge is None:
        from tests.eval.internal.eval_judge import llm_judge
        judge = llm_judge

    import tempfile
    import ai.agents.langgraph.graph as _graph_module
    from ai.agents.langgraph.graph import build_graph

    with open(DATASET_PATH) as f:
        sample = json.load(f)

    tmp_db = tempfile.mktemp(suffix=".sqlite", prefix="otai_eval_mrbench_n10_")
    _original_db = _graph_module._CHECKPOINT_DB
    _graph_module._CHECKPOINT_DB = tmp_db
    try:
        graph = build_graph(use_checkpointer=True)
        results = []
        for dialogue in sample:
            response = generate_response(dialogue, graph)
            context_str = str(dialogue["context"])
            score = score_axis(
                AXIS, context_str, dialogue["student_error_turn"], response, judge
            )
            results.append({"id": dialogue["id"], "source": dialogue["source"], AXIS: score})
    finally:
        _graph_module._CHECKPOINT_DB = _original_db
        try:
            os.unlink(tmp_db)
        except FileNotFoundError:
            pass

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results
