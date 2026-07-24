"""Calibration F1 du juge MRBench — 5 axes, avec revealing_of_answer (2026-07-10).

Reproduit la méthodologie de docs/F1_judge.md (run du 2026-07-07, 4 axes,
F1 macro moyen ≈ 0,32) avec le juge actuel — qui inclut depuis le
prompt anti-complaisance ajouté à score_axis() (raisonnement avant verdict,
cadrage sceptique) — plus un 5e axe désormais possible : revealing_of_answer.

Pourquoi une source différente du run original : `mrbench_v3_devset.json`
(utilisé le 2026-07-07) n'annote que 4 axes (Mistake_Identification,
Mistake_Location, Providing_Guidance, Actionability) — pas de gold label
pour revealing_of_answer. `MRBench_V2.json` (200 dialogues, déjà utilisé par
scripts/sample_mrbench.py pour générer nos propres réponses) annote les 8
dimensions du papier, y compris Revealing_of_the_Answer — c'est la source
utilisée ici.

Échantillon : n=40 (20 Bridge + 20 MathDial, seed=42), un bot tiré au hasard
par dialogue (seed=42) parmi les 8 disponibles (Sonnet, Llama318B,
Llama31405B, GPT4, Mistral, Expert, Gemini, Phi3) — même logique que le run
du 2026-07-07 (éviter le biais d'un bot toujours identique).

Normalisation des labels gold :
- 4 axes standards : "Yes" / "To some extent" / "No" → déjà alignés au
  format de verdict du juge (yes / to some extent / no).
- revealing_of_answer : gold ∈ {"No", "Yes (and the answer is correct)",
  "Yes (but the answer is incorrect)"} → normalisé en {no, yes} (les deux
  variantes "Yes" fusionnées — notre juge ne distingue pas si la réponse
  révélée est correcte, seulement si l'axe critère "révèle-t-il la
  réponse" est rempli).

Réutilise _AXIS_PROMPTS de mrbench_runner.py (même prompt exact, y compris
le cadrage anti-complaisance) mais extrait le verdict brut (yes / to some
extent / no) au lieu du score DAMR binaire, pour calculer un F1 multi-classe
par axe (sklearn), comparable au F1≈0,32 déjà mesuré.

Usage :
    python3 -m tests.eval.external.mrbench_f1_calibration_5axes
"""

from __future__ import annotations

import json
import os
import random

SRC = "/tmp/mrbench/MRBench/MRBench_V2.json"
RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "results", "mrbench_f1_judge_validation_5axes.json"
)
SUMMARY_PATH = os.path.join(
    os.path.dirname(__file__), "results", "mrbench_f1_judge_validation_5axes_summary.json"
)

SEED = 42
N_PER_SOURCE = 20

GOLD_FIELDS = {
    "mistake_identification": "Mistake_Identification",
    "mistake_location": "Mistake_Location",
    "guidance": "Providing_Guidance",
    "actionability": "Actionability",
    "revealing_of_answer": "Revealing_of_the_Answer",
}


def _normalize_gold(axis: str, raw: str) -> str:
    raw = (raw or "").strip().lower()
    if axis == "revealing_of_answer":
        return "no" if raw == "no" else "yes"
    return raw


def _judge_verdict(axis: str, context: str, student_turn: str, response: str, judge) -> str:
    """Variante de score_axis() qui retourne le verdict brut, pas le score DAMR
    binaire — nécessaire pour calculer un F1 multi-classe plutôt qu'un match
    au label désiré. Prompt identique à mrbench_runner.py::score_axis()."""
    from tests.eval.external.mrbench_runner import _AXIS_PROMPTS

    criterion = _AXIS_PROMPTS[axis]
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
        return "no"
    try:
        start, end = result.find("{"), result.rfind("}") + 1
        return json.loads(result[start:end]).get("verdict", "no").strip().lower()
    except Exception:
        return "no"


def run_calibration(judge=None):
    import config.settings as _settings  # noqa: F401 — triggers load_dotenv() outside pytest

    assert os.getenv("OLLAMA_BASE_URL"), (
        "OLLAMA_BASE_URL not set — .env failed to load, judge calls would silently return None"
    )

    if judge is None:
        from tests.eval.internal.eval_judge import llm_judge
        judge = llm_judge

    with open(SRC) as f:
        data = json.load(f)

    by_source: dict[str, list] = {}
    for d in data:
        by_source.setdefault(d["Data"], []).append(d)

    random.seed(SEED)
    sample = []
    for src in sorted(by_source):
        picked = random.sample(by_source[src], min(N_PER_SOURCE, len(by_source[src])))
        sample.extend(picked)

    axes = list(GOLD_FIELDS.keys())
    detail = []
    for d in sample:
        bots_with_gold = [
            b for b, r in d["anno_llm_responses"].items()
            if all(GOLD_FIELDS[a] in r.get("annotation", {}) for a in axes)
        ]
        if not bots_with_gold:
            continue
        bot = random.choice(bots_with_gold)
        resp = d["anno_llm_responses"][bot]
        context = d["conversation_history"]
        # Dernier tour "Student:" comme tour à évaluer — cohérent avec
        # student_error_turn de scripts/sample_mrbench.py.
        lines = [ln.strip() for ln in context.split("\n") if ln.strip()]
        student_lines = [ln for ln in lines if ln.startswith("Student:")]
        student_turn = student_lines[-1] if student_lines else ""

        instance_result = {"conversation_id": d["conversation_id"], "source": d["Data"], "bot": bot, "axes": {}}
        for axis in axes:
            pred = _judge_verdict(axis, context, student_turn, resp["response"], judge)
            gold_raw = resp["annotation"][GOLD_FIELDS[axis]]
            gold = _normalize_gold(axis, gold_raw)
            instance_result["axes"][axis] = {"pred": pred, "gold": gold}
        detail.append(instance_result)

        os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
        with open(RESULTS_PATH, "w") as f:
            json.dump(detail, f, ensure_ascii=False, indent=2)

    from sklearn.metrics import f1_score

    summary = {}
    for axis in axes:
        preds = [r["axes"][axis]["pred"] for r in detail]
        golds = [r["axes"][axis]["gold"] for r in detail]
        f1 = f1_score(golds, preds, average="macro", zero_division=0)
        summary[axis] = round(f1, 3)

    summary["moyenne_4_axes_originaux"] = round(
        sum(summary[a] for a in ["mistake_identification", "mistake_location", "guidance", "actionability"]) / 4, 3
    )
    summary["moyenne_5_axes"] = round(sum(summary[a] for a in axes) / len(axes), 3)

    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


if __name__ == "__main__":
    print(run_calibration())
