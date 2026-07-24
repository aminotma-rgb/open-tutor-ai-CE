"""Échantillonnage de LongMemEval-S — catégorie knowledge-update uniquement.

Source : https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
Fichier source attendu : /tmp/longmemeval/data/longmemeval_s_cleaned.json

Champs retenus dans l'échantillon (noms réels du dataset) :
    question_id, question_type, question, answer, haystack_sessions,
    haystack_dates, question_date

Le fichier source brut contient aussi answer_session_ids/haystack_session_ids
(identifiants des sessions évidence, non utilisés ici) et haystack_dates/
question_date (horodatages, format "YYYY/MM/DD (Day) HH:MM") — nécessaires
pour que le tuteur puisse répondre aux questions temporal-reasoning/
knowledge-update (cf. "Current Date" dans le prompt de lecture du papier,
Figure 13), et donc conservés dans l'échantillon.

Portée réduite à knowledge-update (2026-07-08) : LongMemEval-S ne couvre plus
que cette seule catégorie dans le protocole d'OpenTutorAI. Les 5 autres
capacités mémoire (single-session-user/assistant/preference, multi-session,
temporal-reasoning) sont désormais couvertes par LoCoMo (voir
scripts/sample_locomo.py, tests/eval/external/locomo_runner.py), beaucoup
plus léger en contexte (~16,6k tokens/instance contre ~115k) et donc adapté
au CPU sans GPU utilisé ici. knowledge-update est la seule capacité que
LoCoMo ne teste pas (confirmé Table 1 du papier LongMemEval) — c'est aussi
la catégorie où un bug de runner a déjà été diagnostiqué (docs/evaluation.md
— "Diagnostic échec knowledge-update"), d'où l'intérêt de la garder
surveillée spécifiquement via LongMemEval-S plutôt que de l'abandonner.

Note : la catégorie "abstention" (7e type de question du papier, §3.2) est
absente de `longmemeval_s_cleaned.json` — vérifié sur les 500 instances
brutes (seules 6 catégories existent, dont knowledge-update). Ce n'est pas
un artefact de l'échantillonnage : cette release "cleaned" ne couvre que 6
des 7 types de questions du papier.

N=4 : reprend exactement les 4 instances knowledge-update qui étaient déjà
allouées dans l'ancien échantillon proportionnel à N=24 (seed=42) — aucun
changement de composition, seule la portée (catégories couvertes) change.

Usage :
    python3 scripts/sample_longmemeval.py
"""

import json
import random
from pathlib import Path

SRC = Path("/tmp/longmemeval/data/longmemeval_s_cleaned.json")
DST = Path("tests/eval/external/datasets/longmemeval_s_sample.json")
SEED = 42
CATEGORY = "knowledge-update"
# Poids réel de knowledge-update dans les 500 instances (15.6%) × ancien
# TOTAL_SAMPLE_SIZE=24 → 4, via la même méthode du plus grand reste que
# l'ancien échantillonnage proportionnel (reproduit ici pour ne sélectionner
# que les mêmes 4 instances, sans changer la composition).
N_SAMPLE = 4

with open(SRC) as f:
    data = json.load(f)

by_category: dict[str, list] = {}
for item in data:
    by_category.setdefault(item["question_type"], []).append(item)

print("Catégories et tailles réelles (poids dans les 500 instances) :")
for cat, items in sorted(by_category.items()):
    print(f"  {cat}: {len(items)} ({len(items) / len(data):.1%})")

# "knowledge-update" est alphabétiquement la première des 6 catégories —
# l'ancien script itérait sur sorted(by_category.items()) avec un seul
# random.seed(42) global, donc son tout premier tirage était déjà exactement
# random.sample(by_category["knowledge-update"], 4). Reproduit ici à
# l'identique (mêmes 4 instances, vérifié : 18bc8abd, 1cea1afa, 4d6b87c8,
# 852ce960).
random.seed(SEED)
sample = random.sample(by_category[CATEGORY], N_SAMPLE)

print(f"\n→ {len(sample)} instances retenues pour '{CATEGORY}'")

# Garder uniquement les champs utiles (haystack_sessions peut être lourd) —
# haystack_dates et question_date sont indispensables au raisonnement
# temporel du tuteur (voir longmemeval_runner.py::replay_haystack).
slim = [
    {
        "question_id": d["question_id"],
        "question_type": d["question_type"],
        "question": d["question"],
        "answer": d["answer"],
        "question_date": d["question_date"],
        "haystack_sessions": d["haystack_sessions"],
        "haystack_dates": d["haystack_dates"],
    }
    for d in sample
]

DST.parent.mkdir(parents=True, exist_ok=True)
with open(DST, "w") as f:
    json.dump(slim, f, ensure_ascii=False, indent=2)

print(f"\nÉchantillon : {len(slim)} instances → {DST}")
