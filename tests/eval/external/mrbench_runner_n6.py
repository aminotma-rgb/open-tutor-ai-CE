

from __future__ import annotations

import json
import os

from tests.eval.external.mrbench_runner import generate_response, score_axis

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "datasets", "mrbench_sample_n6.json"
)
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "mrbench_results_n6_revealing.json"
)

AXIS = "revealing_of_answer"


def run_mrbench_n6(judge=None):
    if judge is None:
        from tests.eval.internal.eval_judge import llm_judge
        judge = llm_judge

    import tempfile
    import ai.agents.langgraph.graph as _graph_module
    from ai.agents.langgraph.graph import build_graph

    with open(DATASET_PATH) as f:
        sample = json.load(f)

    tmp_db = tempfile.mktemp(suffix=".sqlite", prefix="otai_eval_mrbench_n6_")
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
