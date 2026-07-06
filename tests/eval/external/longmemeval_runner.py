"""Harness LongMemEval-S — évalue la mémoire persistante (D1).

Rejoue les sessions du haystack via build_graph(), puis pose la question finale
et juge la réponse avec llm_judge (binaire correct/incorrect).

Pré-requis : tests/eval/external/datasets/longmemeval_s_sample.json doit exister
(généré par scripts/sample_longmemeval.py — Phase 2).
"""

from __future__ import annotations

import json
import os

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "datasets", "longmemeval_s_sample.json"
)
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "longmemeval_results.json"
)


def replay_haystack(instance: dict, graph) -> str:
    """Replay the haystack sessions to build up persisted memory, then answer.

    The graph itself never writes to state["messages"] — it only produces
    structured fields (memory_context, exercises, verification...). The
    actual conversational reply is synthesised downstream, exactly as
    production does in gateway/http/routers/adaptive.py::adaptive_chat():
    build an enriched system prompt from the graph's structured output
    (which includes retrieved memory), then call the LLM with the new
    session's messages.
    """
    from ai.llm.service import call_llm_with_messages
    from gateway.http.routers.adaptive import _build_enriched_system_prompt

    thread_id = f"longmemeval-{instance['question_id']}"
    # recursion_limit must comfortably exceed orchestrator.MAX_ITERATIONS * 2 — see
    # gateway/http/routers/adaptive.py for the same fix and its full rationale.
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    for session in instance["haystack_sessions"]:
        # Replay the whole session in one invoke (all turns, both roles) instead
        # of one invoke per individual user turn. state["messages"] never
        # accumulates across invokes (no reducer — see generate_response() in
        # mrbench_runner.py), so per-turn replay meant feedback_node only ever
        # saw one isolated line, never the surrounding dialogue. This starved
        # _llm_decide_memories() of the context it needs to extract personal/
        # contextual facts (assistant turns were also silently skipped
        # entirely). Matches production's actual semantics: adaptive_chat()
        # passes the full current-session history in "messages" per request,
        # not a single message.
        last_user_turn = next(
            (t["content"] for t in reversed(session) if t["role"] == "user"), ""
        )
        graph.invoke(
            {
                "messages": session,
                "user_message": last_user_turn,
                # feedback_node only persists memories when user_id is
                # truthy ("if should_save and user_id:") — left unset,
                # it silently no-ops and nothing ever reaches the DB.
                "user_id": thread_id,
                # Auto-pass HITL checkpoints (P2/P3), exactly as production
                # does in adaptive_chat() — without this, feedback_node's
                # interrupt() pauses before _persist_memories() ever runs,
                # so nothing from the haystack ever reaches the DB.
                "human_feedback": "yes",
            },
            config=config,
        )

    question_turn = [{"role": "user", "content": instance["question"]}]
    final_state = graph.invoke(
        {
            "messages": question_turn,
            "user_message": instance["question"],
            "user_id": thread_id,
            "human_feedback": "yes",
        },
        config=config,
    )

    enriched_system = _build_enriched_system_prompt(final_state, is_first_message=False)
    llm_messages = [{"role": "system", "content": enriched_system}] + question_turn
    return call_llm_with_messages(llm_messages, max_tokens=600) or ""


def score_correctness(question: str, gold: str, generated: str, judge) -> bool:
    prompt = (
        f"QUESTION: {question}\nRÉPONSE_ATTENDUE: {gold}\n"
        f"RÉPONSE_GÉNÉRÉE: {generated}\n\n"
        "La réponse générée est-elle correcte par rapport à la réponse attendue "
        '(même si formulée différemment) ? Réponds UNIQUEMENT en JSON : '
        '{"verdict": "yes"|"no"}'
    )
    result = judge(prompt)
    return bool(result and "yes" in result.lower())


def run_longmemeval(judge=None):
    if judge is None:
        from tests.eval.internal.eval_judge import llm_judge
        judge = llm_judge

    import tempfile
    import ai.agents.langgraph.graph as _graph_module
    from ai.agents.langgraph.graph import build_graph

    with open(DATASET_PATH) as f:
        sample = json.load(f)

    # SQLite isolé pour ce run — évite les conflits avec la DB de production
    # et les erreurs "database is locked" si un autre runner tourne en parallèle.
    tmp_db = tempfile.mktemp(suffix=".sqlite", prefix="otai_eval_longmemeval_")
    _original_db = _graph_module._CHECKPOINT_DB
    _graph_module._CHECKPOINT_DB = tmp_db
    try:
        graph = build_graph(use_checkpointer=True)
        results = []
        for instance in sample:
            generated = replay_haystack(instance, graph)
            correct = score_correctness(
                instance["question"], instance["answer"], generated, judge
            )
            results.append({
                "question_id": instance["question_id"],
                "category": instance["question_type"],
                "correct": correct,
            })
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
