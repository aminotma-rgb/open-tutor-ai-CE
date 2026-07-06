"""Échantillonnage stratifié proportionnel de LongMemEval-S (seed=42, N=24).

Source : https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
Fichier source attendu : /tmp/longmemeval/data/longmemeval_s_cleaned.json

Champs retenus dans l'échantillon (noms réels du dataset) :
    question_id, question_type, question, answer, haystack_sessions

Répartition proportionnelle (méthode du plus grand reste) : chaque catégorie
reçoit un nombre d'instances proportionnel à son poids réel dans les 500
instances du dataset, plutôt qu'un nombre fixe identique pour toutes les
catégories. Voir docs/evaluation.md (Phase 2 — Représentativité de
l'échantillon) pour la justification et le calcul détaillé.

N=24 est le plus petit total pour lequel l'allocation proportionnelle
(Hare-Niemeyer) donne encore ≥2 instances à chaque catégorie, y compris la
plus petite (`single-session-preference`, 6% du dataset) — en dessous de 24,
au moins une catégorie retombe à 1 instance, donc à un résultat binaire
(0%/100%) non significatif.

Usage :
    python3 scripts/sample_longmemeval.py
"""

import json
import random
from pathlib import Path

SRC = Path("/tmp/longmemeval/data/longmemeval_s_cleaned.json")
DST = Path("tests/eval/external/datasets/longmemeval_s_sample.json")
SEED = 42
TOTAL_SAMPLE_SIZE = 24  # plus petit N représentatif (≥2 instances/catégorie) — voir docs/evaluation.md


def _proportional_allocation(category_sizes: dict[str, int], total: int) -> dict[str, int]:
    """Répartit `total` instances entre catégories proportionnellement à leur
    poids réel, via la méthode du plus grand reste (Hare-Niemeyer) — garantit
    que la somme des allocations vaut exactement `total`, contrairement à un
    simple arrondi catégorie par catégorie."""
    dataset_size = sum(category_sizes.values())
    exact_shares = {
        cat: total * size / dataset_size for cat, size in category_sizes.items()
    }
    allocation = {cat: int(share) for cat, share in exact_shares.items()}
    remainder = total - sum(allocation.values())

    remainders = sorted(
        category_sizes, key=lambda c: exact_shares[c] - allocation[c], reverse=True
    )
    for cat in remainders[:remainder]:
        allocation[cat] += 1

    return allocation


with open(SRC) as f:
    data = json.load(f)

by_category: dict[str, list] = {}
for item in data:
    by_category.setdefault(item["question_type"], []).append(item)

print("Catégories et tailles réelles (poids dans les 500 instances) :")
for cat, items in sorted(by_category.items()):
    print(f"  {cat}: {len(items)} ({len(items) / len(data):.1%})")

category_sizes = {cat: len(items) for cat, items in by_category.items()}
allocation = _proportional_allocation(category_sizes, TOTAL_SAMPLE_SIZE)

random.seed(SEED)
sample = []
print(f"\nAllocation proportionnelle pour N={TOTAL_SAMPLE_SIZE} :")
for cat, items in sorted(by_category.items()):
    n_picked = min(allocation[cat], len(items))
    picked = random.sample(items, n_picked)
    sample.extend(picked)
    print(f"  → {n_picked} instances retenues pour '{cat}'")

# Garder uniquement les champs utiles (haystack_sessions peut être lourd)
slim = [
    {
        "question_id": d["question_id"],
        "question_type": d["question_type"],
        "question": d["question"],
        "answer": d["answer"],
        "haystack_sessions": d["haystack_sessions"],
    }
    for d in sample
]

DST.parent.mkdir(parents=True, exist_ok=True)
with open(DST, "w") as f:
    json.dump(slim, f, ensure_ascii=False, indent=2)

print(f"\nÉchantillon : {len(slim)} instances → {DST}")
