"""Échantillonnage stratifié de MRBench V2 (Bridge + MathDial, seed=42).

Source : https://github.com/kaushal0494/UnifyingAITutorEvaluation
Fichier source attendu : /tmp/mrbench/MRBench/MRBench_V2.json

Champs réels du dataset :
    conversation_id, conversation_history, Data (source), anno_llm_responses
    → on extrait context_turns et student_error_turn depuis conversation_history

Usage :
    python3 scripts/sample_mrbench.py
"""

import json
import random
import re
from pathlib import Path

SRC = Path("/tmp/mrbench/MRBench/MRBench_V2.json")
DST = Path("tests/eval/external/datasets/mrbench_sample.json")
SEED = 42
PER_SOURCE = 20  # Bridge(55) + MathDial(145) → ~20+20 = 40 instances


def parse_conversation(history: str) -> tuple[list[dict], str]:
    """Extrait les tours de dialogue et le dernier tour étudiant (l'erreur à corriger)."""
    raw = [ln.strip("\xa0").strip() for ln in history.split("\n") if ln.strip("\xa0").strip()]

    turns = []
    cur_role: str | None = None
    cur_lines: list[str] = []

    for line in raw:
        if line.startswith("Tutor:"):
            if cur_role and cur_lines:
                turns.append({"role": cur_role, "content": "\n".join(cur_lines)})
            cur_role = "tutor"
            cur_lines = [line[len("Tutor:"):].strip()]
        elif line.startswith("Student:"):
            if cur_role and cur_lines:
                turns.append({"role": cur_role, "content": "\n".join(cur_lines)})
            cur_role = "student"
            cur_lines = [line[len("Student:"):].strip()]
        else:
            cur_lines.append(line)

    if cur_role and cur_lines:
        turns.append({"role": cur_role, "content": "\n".join(cur_lines)})

    if turns and turns[-1]["role"] == "student":
        return turns[:-1], turns[-1]["content"]
    return turns, ""


with open(SRC) as f:
    data = json.load(f)

by_source: dict[str, list] = {}
for item in data:
    by_source.setdefault(item["Data"], []).append(item)

print("Sources et tailles :")
for src, items in sorted(by_source.items()):
    print(f"  {src}: {len(items)}")

random.seed(SEED)
sample = []
for src, items in sorted(by_source.items()):
    picked = random.sample(items, min(PER_SOURCE, len(items)))
    sample.extend(picked)
    print(f"  → {len(picked)} instances retenues pour '{src}'")

# Normaliser vers le format attendu par mrbench_runner.py
normalized = []
for d in sample:
    context_turns, error_turn = parse_conversation(d["conversation_history"])
    normalized.append({
        "id": d["conversation_id"],
        "source": d["Data"],
        "context": context_turns,
        "student_error_turn": error_turn,
        "conversation_history_raw": d["conversation_history"],
    })

DST.parent.mkdir(parents=True, exist_ok=True)
with open(DST, "w") as f:
    json.dump(normalized, f, ensure_ascii=False, indent=2)

print(f"\nÉchantillon : {len(normalized)} instances → {DST}")
# Vérification rapide du parsing
ok = sum(1 for d in normalized if d["student_error_turn"])
print(f"Instances avec student_error_turn non-vide : {ok}/{len(normalized)}")
