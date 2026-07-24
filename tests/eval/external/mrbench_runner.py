"""Harness MRBench — évalue la qualité pédagogique de correction d'erreurs (D3).

Évalue la réponse du système sur 5 axes issus du benchmark MRBench V2
(Bridge + MathDial) :
    mistake_identification, mistake_location, guidance, actionability,
    revealing_of_answer

Métrique fidèle au papier (Maurya et al. 2025, §5.4) : DAMR (Desired
Annotation Match Rate) — pourcentage de réponses dont le verdict du juge
correspond exactement au label désiré (Table 2/4). Pas de pondération
partielle sur "to some extent" (voir score_axis()).

Le dataset est pré-normalisé par scripts/sample_mrbench.py. Format attendu :
    id, source, context (liste de {role, content}), student_error_turn

Pré-requis : tests/eval/external/datasets/mrbench_sample.json doit exister
(généré par scripts/sample_mrbench.py — Phase 2).
"""

from __future__ import annotations

import json
import os

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "datasets", "mrbench_sample.json"
)
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "mrbench_results.json"
)

AXES = [
    "mistake_identification",
    "mistake_location",
    "guidance",
    "actionability",
    "revealing_of_answer",
]

_AXIS_PROMPTS = {
    "mistake_identification": "Le tuteur identifie-t-il clairement qu'une erreur a été commise par l'étudiant ?",
    "mistake_location": "Le tuteur localise-t-il précisément où se trouve l'erreur dans le raisonnement de l'étudiant ?",
    "guidance": "Le tuteur fournit-il une guidance pédagogique utile pour corriger l'erreur ?",
    "actionability": "La réponse du tuteur permet-elle à l'étudiant de savoir concrètement quoi faire ensuite ?",
    "revealing_of_answer": "Le tuteur révèle-t-il la réponse finale à l'étudiant (que celle-ci soit correcte ou incorrecte), au lieu de le guider vers sa propre correction ?",
}

# Label désiré par axe (Table 2/4, Maurya et al. 2025). Les 4 premiers axes ont
# "Yes" comme Desiderata ; revealing_of_answer est le seul où le comportement
# souhaité est l'absence de révélation ("No") — un tuteur qui donne la réponse
# court-circuite l'apprentissage actif (cf. principe 1, §4 du papier).
_DESIRED_VERDICT = {
    "mistake_identification": "yes",
    "mistake_location": "yes",
    "guidance": "yes",
    "actionability": "yes",
    "revealing_of_answer": "no",
}


def generate_response(dialogue: dict, graph) -> str:
    """Replay the dialogue through the graph, then synthesise the tutor's reply.

    The graph itself never writes to state["messages"] — it only produces
    structured fields (exercises, strategy, verification...). The actual
    conversational reply is synthesised downstream, exactly as production
    does in gateway/http/routers/adaptive.py::adaptive_chat(): build an
    enriched system prompt from the graph's structured output, then call the
    LLM with the full conversation history.
    """
    from ai.llm.service import call_llm_with_messages
    from gateway.http.routers.adaptive import _build_enriched_system_prompt

    thread_id = f"mrbench-{dialogue['id']}"
    # recursion_limit must comfortably exceed orchestrator.MAX_ITERATIONS * 2 — see
    # gateway/http/routers/adaptive.py for the same fix and its full rationale.
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 50}

    history = [
        {
            "role": "assistant" if turn["role"] == "tutor" else "user",
            "content": turn["content"],
        }
        for turn in dialogue["context"]
    ]
    history.append({"role": "user", "content": dialogue["student_error_turn"]})

    final_state = graph.invoke(
        {
            "messages": history,
            "user_message": dialogue["student_error_turn"],
            # feedback_node only persists memories when user_id is truthy
            # ("if should_save and user_id:") — left unset, it silently no-ops.
            "user_id": thread_id,
            # Auto-pass HITL checkpoints (P2/P3), exactly as production does in
            # adaptive_chat() — without this, feedback_node's interrupt() pauses
            # before _persist_memories() ever runs.
            "human_feedback": "yes",
        },
        config=config,
    )

    enriched_system = _build_enriched_system_prompt(final_state, is_first_message=False)
    llm_messages = [{"role": "system", "content": enriched_system}] + history
    return call_llm_with_messages(llm_messages, max_tokens=600) or ""


def score_axis(
    axis: str, context: str, student_turn: str, response: str, judge
) -> float:
    """Desired Annotation Match Rate (DAMR) pour un axe — Maurya et al. 2025, §5.4.

    Le papier ne pondère pas les verdicts intermédiaires : DAMR est le
    pourcentage de réponses dont le verdict correspond exactement au label
    désiré (Table 2/4). "to some extent" ne vaut donc pas un demi-point — c'est
    un non-match au même titre que "no", sans quoi le score cesserait de
    représenter un taux de correspondance et deviendrait une moyenne pondérée.
    """
    criterion = _AXIS_PROMPTS[axis]
    # Cadrage anti-complaisance (2026-07-10) : la calibration F1 (docs/F1_judge.md)
    # a mesuré que ce juge (mistral:7b-instruct-q4_K_M) sur-prédit systématiquement
    # "yes" (F1≈0,32) et écrase presque toujours "to some extent" vers "yes". Le
    # raisonnement explicite avant verdict + la consigne sceptique visent
    # directement ce biais diagnostiqué, sans changer de modèle ni d'infra.
    prompt = (
        f"CONTEXTE_DIALOGUE: {context}\n"
        f"TOUR_ÉTUDIANT_À_ÉVALUER: {student_turn}\n"
        f"RÉPONSE_TUTEUR: {response}\n\n"
        f"Critère d'évaluation : {criterion}\n\n"
        'Sois strict et sceptique : ne réponds "yes" que si le critère est '
        "indiscutablement et entièrement rempli. Avant de conclure, identifie "
        "explicitement ce qui manquerait pour que ce ne soit PAS un \"yes\" "
        "(ambiguïté, information partielle, formulation vague). Utilise "
        '"to some extent" chaque fois que le critère n\'est que partiellement '
        "ou implicitement rempli — ne l'écrase pas artificiellement vers "
        '"yes" par complaisance.\n\n'
        "Réponds en JSON, raisonnement avant verdict : "
        '{"raisonnement": "<ce qui justifie ou limite le verdict, 1-2 phrases>", '
        '"verdict": "yes"|"to some extent"|"no"}'
    )
    result = judge(prompt)
    if not result:
        return 0.0
    try:
        start, end = result.find("{"), result.rfind("}") + 1
        verdict = json.loads(result[start:end]).get("verdict", "no").lower()
    except Exception:
        return 0.0
    return 1.0 if verdict == _DESIRED_VERDICT[axis] else 0.0


def run_mrbench(judge=None):
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
    tmp_db = tempfile.mktemp(suffix=".sqlite", prefix="otai_eval_mrbench_")
    _original_db = _graph_module._CHECKPOINT_DB
    _graph_module._CHECKPOINT_DB = tmp_db
    try:
        graph = build_graph(use_checkpointer=True)
        results = []
        for dialogue in sample:
            response = generate_response(dialogue, graph)
            context_str = str(dialogue["context"])
            scores = {
                axis: score_axis(
                    axis, context_str, dialogue["student_error_turn"], response, judge
                )
                for axis in AXES
            }
            results.append({"id": dialogue["id"], "source": dialogue["source"], **scores})
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
