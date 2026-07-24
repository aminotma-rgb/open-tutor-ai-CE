
from __future__ import annotations

import json
import os

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "datasets", "locomo_sample.json"
)
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "locomo_results.json"
)


def replay_sessions(instance: dict, graph) -> str:
    """Rejoue toutes les sessions d'une conversation pour construire la mémoire
    persistée, une seule fois par conversation. Retourne le thread_id à
    réutiliser par answer_question() pour toutes les questions de cette
    conversation — reproduit la persistance inter-sessions de S1/S3/S5, comme
    longmemeval_runner.py::replay_haystack().
    """
    thread_id = f"locomo-{instance['sample_id']}"
    # recursion_limit must comfortably exceed orchestrator.MAX_ITERATIONS * 2 — see
    # gateway/http/routers/adaptive.py for the same fix and its full rationale.
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    for session in instance["sessions"]:
        last_user_turn = next(
            (t["content"] for t in reversed(session) if t["role"] == "user"), ""
        )
        graph.invoke(
            {
                "messages": session,
                "user_message": last_user_turn,
                "user_id": thread_id,
                "human_feedback": "yes",
                "iteration": 0,
                "n_retries": {},
                "agent_trace": [],
                "agent_reasoning": {},
            },
            config=config,
        )
    return thread_id


def answer_question(question: str, thread_id: str, current_date: str, graph) -> str:
    """Pose une question contre la mémoire déjà construite pour ce thread_id.

    Synthétise la réponse exactement comme replay_haystack() dans
    longmemeval_runner.py : le graphe ne produit que des champs structurés,
    la réponse conversationnelle est assemblée en aval via
    _build_enriched_system_prompt + call_llm_with_messages, avec la date
    courante injectée pour ancrer les expressions temporelles relatives.
    """
    from ai.llm.service import call_llm_with_messages
    from gateway.http.routers.adaptive import _build_enriched_system_prompt

    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}
    question_turn = [{"role": "user", "content": question}]
    final_state = graph.invoke(
        {
            "messages": question_turn,
            "user_message": question,
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
    if current_date:
        enriched_system = f"{enriched_system}\n\nCurrent Date: {current_date}"
    llm_messages = [{"role": "system", "content": enriched_system}] + question_turn
    return call_llm_with_messages(llm_messages, max_tokens=600) or ""


_TEMPORAL_INSTRUCTION = (
    'Réponds "yes" si la réponse générée contient la réponse correcte, "no" '
    "sinon. Si la réponse est équivalente à la réponse correcte ou contient "
    "toutes les étapes intermédiaires pour l'obtenir, réponds aussi \"yes\". Si "
    "elle ne contient qu'une partie de l'information requise, réponds \"no\". Ne "
    "pénalise PAS les erreurs de ±1 jour sur un nombre de jours/semaines/mois "
    "(ex. prédire 19 jours quand la réponse attendue est 18 reste correct) — "
    "même tolérance que pour temporal-reasoning dans longmemeval_runner.py."
)
_DEFAULT_INSTRUCTION = (
    'Réponds "yes" si la réponse générée contient la réponse correcte, "no" sinon. '
    "Si la réponse est équivalente à la réponse correcte ou contient toutes les "
    "étapes intermédiaires pour l'obtenir, réponds aussi \"yes\". Si elle ne "
    "contient qu'une partie de l'information requise, réponds \"no\"."
)


def score_correctness(question: str, gold: str, generated: str, judge, category: str) -> bool:
    """Juge binaire correct/incorrect — tolérance ±1 jour pour temporal-reasoning,
    prompt générique tolérant à la paraphrase pour les autres catégories (voir
    docstring du module sur le choix LLM-judge plutôt que F1)."""
    instruction = _TEMPORAL_INSTRUCTION if category == "temporal-reasoning" else _DEFAULT_INSTRUCTION
    prompt = (
        f"QUESTION: {question}\nRÉPONSE_ATTENDUE: {gold}\n"
        f"RÉPONSE_GÉNÉRÉE: {generated}\n\n"
        f"{instruction}\n"
        'Réponds UNIQUEMENT en JSON : {"verdict": "yes"|"no"}'
    )
    result = judge(prompt)
    return bool(result and "yes" in result.lower())


def run_locomo(judge=None):
    if judge is None:
        from tests.eval.internal.eval_judge import llm_judge
        judge = llm_judge

    import tempfile
    import ai.agents.langgraph.graph as _graph_module
    from ai.agents.langgraph.graph import build_graph

    with open(DATASET_PATH) as f:
        sample = json.load(f)

    # SQLite isolé pour ce run — même précaution que longmemeval_runner.py et
    # mrbench_runner.py (évite "database is locked" en cas d'exécution
    # simultanée, même si ces runners doivent de toute façon tourner en
    # séquentiel — voir tests/eval/external/conftest.py).
    tmp_db = tempfile.mktemp(suffix=".sqlite", prefix="otai_eval_locomo_")
    _original_db = _graph_module._CHECKPOINT_DB
    _graph_module._CHECKPOINT_DB = tmp_db
    try:
        graph = build_graph(use_checkpointer=True)
        results = []
        for instance in sample:
            thread_id = replay_sessions(instance, graph)
            current_date = instance["session_dates"][-1] if instance["session_dates"] else ""
            for qa in instance["qa"]:
                generated = answer_question(qa["question"], thread_id, current_date, graph)
                correct = score_correctness(
                    qa["question"], qa["answer"], generated, judge, qa["category"]
                )
                results.append(
                    {
                        "sample_id": instance["sample_id"],
                        "question": qa["question"],
                        "category": qa["category"],
                        "correct": correct,
                    }
                )
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
