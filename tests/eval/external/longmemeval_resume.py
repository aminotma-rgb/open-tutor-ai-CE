"""Reprise du run LongMemEval-S interrompu le 2026-07-10.

Contexte : le run à N=4 (knowledge-update) a été arrêté après 30 sessions
complètes sur les 52 de la 1ère instance (4d6b87c8) — confirmées terminées
proprement via [END] feedback signalled session completion. La 31e session
avait été coupée en plein milieu (0 itération, trace vide) : refaite depuis
le début plutôt que résumée à mi-chemin.

Reprend la base de checkpoints sauvegardée (copie de travail, l'original
n'est pas touché) et ne rejoue que ce qui n'a pas encore été fait :
- 4d6b87c8 : sessions[30:] (22 sessions restantes) + question finale
- 1cea1afa, 18bc8abd, 852ce960 : replay complet, jamais démarrées

Réutilise replay_haystack()/score_correctness() de longmemeval_runner.py
sans les modifier — un dict d'instance avec haystack_sessions tronqué suffit,
replay_haystack() ne lit pas haystack_dates directement et le thread_id
(f"longmemeval-{question_id}") est inchangé, donc le checkpointer retrouve
l'état déjà persisté pour 4d6b87c8 et continue dessus.

Usage :
    python3 -m tests.eval.external.longmemeval_resume
"""

from __future__ import annotations

import json
import os
import shutil

from tests.eval.external.longmemeval_runner import (
    DATASET_PATH,
    RESULTS_PATH,
    replay_haystack,
    score_correctness,
)

CHECKPOINT_BASE = (
    "/tmp/claude-1001/-home-hadoop-Bureau-OTAI-open-tutor-ai-CE/"
    "8454e684-0748-479c-8d28-3c3aaba67b72/scratchpad/longmemeval_resume_base.sqlite"
)
SESSIONS_ALREADY_DONE = {"4d6b87c8": 30}


def run_resume(judge=None):
    if judge is None:
        from tests.eval.internal.eval_judge import llm_judge
        judge = llm_judge

    import ai.agents.langgraph.graph as _graph_module
    from ai.agents.langgraph.graph import build_graph

    with open(DATASET_PATH) as f:
        sample = json.load(f)

    # Copie de travail dédiée à cette reprise — ne touche pas la base sauvegardée
    # d'origine ni /tmp/otai_eval_longmemeval_yxph0740.sqlite.
    working_db = "/tmp/otai_eval_longmemeval_resume_working.sqlite"
    shutil.copy(CHECKPOINT_BASE, working_db)

    _original_db = _graph_module._CHECKPOINT_DB
    _graph_module._CHECKPOINT_DB = working_db
    try:
        graph = build_graph(use_checkpointer=True)
        results = []
        for instance in sample:
            qid = instance["question_id"]
            n_done = SESSIONS_ALREADY_DONE.get(qid, 0)
            if n_done:
                print(f"  [{qid}] reprise après {n_done} sessions déjà traitées "
                      f"({len(instance['haystack_sessions']) - n_done} restantes)")
                instance = {**instance, "haystack_sessions": instance["haystack_sessions"][n_done:]}
            else:
                print(f"  [{qid}] replay complet ({len(instance['haystack_sessions'])} sessions)")

            generated = replay_haystack(instance, graph)
            correct = score_correctness(
                instance["question"],
                instance["answer"],
                generated,
                judge,
                instance["question_type"],
            )
            results.append({
                "question_id": qid,
                "category": instance["question_type"],
                "correct": correct,
            })
            print(f"  [{qid}] -> correct={correct}")
    finally:
        _graph_module._CHECKPOINT_DB = _original_db
        try:
            os.unlink(working_db)
        except FileNotFoundError:
            pass

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    return results


if __name__ == "__main__":
    print(run_resume())
