

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
                # Reset per-turn orchestrator state exactly as adaptive_chat()
                # does for every HTTP request (see gateway/http/routers/adaptive.py).
                # Without this, "iteration" is read back from the checkpointer and
                # keeps climbing across the whole haystack replay (same thread_id
                # for all sessions) instead of per turn — it trips orchestrator's
                # MAX_ITERATIONS ceiling within the first session or two, after
                # which every subsequent invoke (including the final question)
                # silently no-ops: memory is never refreshed, feedback never
                # persists, for the rest of the haystack.
                "iteration": 0,
                "n_retries": {},
                "agent_trace": [],
                "agent_reasoning": {},
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
            "iteration": 0,
            "n_retries": {},
            "agent_trace": [],
            "agent_reasoning": {},
        },
        config=config,
    )

    enriched_system = _build_enriched_system_prompt(final_state, is_first_message=False)
    # Papier Figure 13 ("Current Date: {question_date}") : sans cette ancre, le LLM
    # ne peut pas résoudre "le mois dernier"/"cette semaine" mentionnés dans la
    # mémoire persistée par rapport à la date réelle de la question.
    enriched_system = f"{enriched_system}\n\nCurrent Date: {instance['question_date']}"
    llm_messages = [{"role": "system", "content": enriched_system}] + question_turn
    return call_llm_with_messages(llm_messages, max_tokens=600) or ""


# Instructions de jugement par catégorie — traduction fidèle des 4 prompts de la
# Figure 10 (Appendix A.4, Wu et al. 2025). Le papier ne juge jamais toutes les
# catégories avec la même consigne ; en particulier temporal-reasoning tolère les
# erreurs de ±1 jour, knowledge-update accepte l'ancienne info à côté de la mise à
# jour, et single-session-preference évalue un rubric plutôt qu'une réponse fixe.
_JUDGE_INSTRUCTIONS = {
    "temporal-reasoning": (
        'Réponds "yes" si la réponse générée contient la réponse correcte, "no" '
        "sinon. Si la réponse est équivalente à la réponse correcte ou contient "
        "toutes les étapes intermédiaires pour l'obtenir, réponds aussi \"yes\". Si "
        "elle ne contient qu'une partie de l'information requise, réponds \"no\". Ne "
        "pénalise PAS les erreurs de ±1 jour sur un nombre de jours/semaines/mois "
        "(ex. prédire 19 jours quand la réponse attendue est 18 reste correct)."
    ),
    "knowledge-update": (
        'Réponds "yes" si la réponse générée contient la réponse correcte, "no" '
        "sinon. Si la réponse contient l'ancienne information en plus de la valeur "
        "mise à jour, elle doit quand même être jugée correcte tant que la valeur "
        "mise à jour attendue y figure."
    ),
    "single-session-preference": (
        "RÉPONSE_ATTENDUE est un rubric décrivant la réponse personnalisée "
        'souhaitée, pas une réponse factuelle fixe. Réponds "yes" si la réponse '
        'générée satisfait ce rubric, "no" sinon. Le modèle n\'a pas besoin de '
        "refléter tous les points du rubric — la réponse est correcte tant qu'elle "
        "rappelle et utilise correctement l'information personnelle de l'utilisateur."
    ),
}
_DEFAULT_JUDGE_INSTRUCTION = (
    'Réponds "yes" si la réponse générée contient la réponse correcte, "no" sinon. '
    "Si la réponse est équivalente à la réponse correcte ou contient toutes les "
    "étapes intermédiaires pour l'obtenir, réponds aussi \"yes\". Si elle ne "
    "contient qu'une partie de l'information requise, réponds \"no\"."
)


def score_correctness(
    question: str, gold: str, generated: str, judge, question_type: str
) -> bool:
    """Juge correct/incorrect avec le prompt propre à la catégorie de la question
    (voir _JUDGE_INSTRUCTIONS) — fidèle à la Figure 10 du papier plutôt qu'à un
    prompt générique unique pour les 6 catégories.
    """
    instruction = _JUDGE_INSTRUCTIONS.get(question_type, _DEFAULT_JUDGE_INSTRUCTION)
    gold_label = (
        "RUBRIC_ATTENDU" if question_type == "single-session-preference" else "RÉPONSE_ATTENDUE"
    )
    prompt = (
        f"QUESTION: {question}\n{gold_label}: {gold}\n"
        f"RÉPONSE_GÉNÉRÉE: {generated}\n\n"
        f"{instruction}\n"
        'Réponds UNIQUEMENT en JSON : {"verdict": "yes"|"no"}'
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
                instance["question"],
                instance["answer"],
                generated,
                judge,
                instance["question_type"],
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
