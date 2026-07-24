"""Échantillonnage de LoCoMo (Maharana et al., ACL 2024) — complément léger à
LongMemEval-S pour D1 (mémoire persistante).

Source : https://github.com/snap-research/locomo (data/locomo10.json)
Fichier source attendu : /tmp/locomo/data/locomo10.json

Pourquoi LoCoMo en plus de LongMemEval-S : LongMemEval-S a été réduit à la
seule catégorie knowledge-update (voir scripts/sample_longmemeval.py) car ses
instances sont très coûteuses à rejouer sur CPU (~115k tokens, ~50 sessions
par instance). LoCoMo teste les 5 autres capacités mémoire pertinentes
(single-hop, multi-hop, temporal-reasoning, open-domain, adversarial) avec un
volume de texte ~5-10x plus léger par conversation (10-22k tokens mesurés
sur les 3 premières conversations, contre ~115k pour LongMemEval-S).

Nuance importante : le nombre de *sessions* par conversation LoCoMo (19-32)
est comparable à LongMemEval-S (~50), donc le nombre d'appels au graphe
LangGraph par instance ne baisse pas dans les mêmes proportions que le
volume de tokens — le gain de temps réel vient surtout de la longueur plus
courte de chaque tour de dialogue, pas d'un nombre d'invocations réduit.

Écart au format d'origine — conversation humain-humain, pas humain-tuteur :
LoCoMo enregistre des dialogues entre deux personnes (ex. Caroline/Melanie),
sans rôle "assistant". Pour rejouer ces sessions dans notre pipeline
(qui attend des tours user/assistant), `speaker_a` est arbitrairement mappé
sur "user" et `speaker_b` sur "assistant" — une adaptation, pas une
correspondance naturelle. C'est l'approche standard pour adapter LoCoMo à un
système de mémoire conversationnelle (cf. littérature citée dans
docs/evaluation.md).

Catégories de questions (champ "category", entier) — mapping vérifié par
comptage sur les 1 986 questions du dataset et recoupé avec la distribution
publiée par les auteurs (841 single-hop, 282 multi-hop, 321 temporal-
reasoning, 96 open-domain, reste = adversarial) :
    1 = multi-hop reasoning
    2 = temporal-reasoning
    3 = open-domain knowledge
    4 = single-hop retrieval
    5 = adversarial (pas de champ "answer", mais "adversarial_answer" —
        vérifié sur plusieurs exemples : ce sont de vraies réponses
        factuelles ancrées dans le dialogue, comme les autres catégories,
        pas des questions à prémisse fausse. "Adversarial" fait référence à
        la difficulté de récupération — questions formulées pour être
        difficiles à relier à l'évidence par une recherche naïve — et non à
        une capacité d'abstention. Contrairement à ce qui a été supposé
        initialement, cette catégorie ne couvre donc PAS la même capacité
        que l'abstention de LongMemEval-S ; jugée avec le même prompt
        générique que les autres catégories factuelles.)

Échantillon : N_CONVERSATIONS conversations tirées au hasard (seed=42), puis
N_QA_PER_CATEGORY questions par catégorie par conversation retenue —
rejouer une conversation une seule fois pour en tirer plusieurs questions
est beaucoup plus économe que LongMemEval-S, où chaque instance a son propre
haystack à rejouer intégralement.

Usage :
    python3 scripts/sample_locomo.py
"""

import json
import random
from pathlib import Path

SRC = Path("/tmp/locomo/data/locomo10.json")
DST = Path("tests/eval/external/datasets/locomo_sample.json")
SEED = 42
N_CONVERSATIONS = 3
N_QA_PER_CATEGORY = 2

CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal-reasoning",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}


def _ordered_sessions(conv: dict) -> tuple[list[list[dict]], list[str]]:
    """Extrait les sessions dans l'ordre chronologique (session_1, session_2, ...)
    et convertit chaque tour {speaker, dia_id, text} vers {role, content},
    speaker_a → "user", speaker_b → "assistant"."""
    speaker_a, speaker_b = conv["speaker_a"], conv["speaker_b"]
    session_keys = sorted(
        (k for k in conv if k.startswith("session_") and not k.endswith("_date_time")),
        key=lambda k: int(k.split("_")[1]),
    )
    sessions, dates = [], []
    for key in session_keys:
        turns = [
            {
                "role": "user" if turn["speaker"] == speaker_a else "assistant",
                "content": turn["text"],
            }
            for turn in conv[key]
        ]
        sessions.append(turns)
        dates.append(conv.get(f"{key}_date_time", ""))
    return sessions, dates


with open(SRC) as f:
    data = json.load(f)

random.seed(SEED)
picked_conversations = random.sample(data, N_CONVERSATIONS)

sample = []
for conv_entry in picked_conversations:
    sessions, dates = _ordered_sessions(conv_entry["conversation"])

    by_category: dict[int, list] = {}
    for qa in conv_entry["qa"]:
        by_category.setdefault(qa["category"], []).append(qa)

    picked_qa = []
    for cat, items in sorted(by_category.items()):
        n = min(N_QA_PER_CATEGORY, len(items))
        for qa in random.sample(items, n):
            gold = qa["adversarial_answer"] if cat == 5 else qa["answer"]
            picked_qa.append(
                {
                    "question": qa["question"],
                    "answer": gold,
                    "category": CATEGORY_NAMES[cat],
                }
            )

    sample.append(
        {
            "sample_id": conv_entry["sample_id"],
            "sessions": sessions,
            "session_dates": dates,
            "qa": picked_qa,
        }
    )

DST.parent.mkdir(parents=True, exist_ok=True)
with open(DST, "w") as f:
    json.dump(sample, f, ensure_ascii=False, indent=2)

n_qa = sum(len(c["qa"]) for c in sample)
print(f"Échantillon : {len(sample)} conversations, {n_qa} questions → {DST}")
for c in sample:
    cats = sorted({qa["category"] for qa in c["qa"]})
    print(f"  {c['sample_id']}: {len(c['sessions'])} sessions, {len(c['qa'])} questions ({', '.join(cats)})")
