"""Couche d'évaluation par juge LLM (RAGAS-style + G-Eval) pour OpenTutorAI CE.

Ce module remplace les `judge_fn` stubbés (S0) par de vraies métriques calculées
par un juge LLM, en réutilisant le point d'entrée LLM du dépôt :
    ai.llm.service.call_llm(prompt, model=None, max_tokens=...)

Aucune clé API externe n'est requise : le module passe par le routing multi-provider
déjà configuré (Mistral local par défaut). Compatible avec les restrictions réseau.

Trois familles de métriques :
  Dimension 2 (Qualité du contenu — style RAGAS) :
    - recall_at_k(relevant_ids, retrieved_ids) : métrique retriever — documents pertinents dans top-k
    - faithfulness(answer, contexts)           : A_ancrées / A_total          (cible ≥ 0,90)
    - answer_relevancy(question, answer)        : pertinence réponse↔question   (cible ≥ 0,80)
    - context_recall(ground_truth, contexts)   : couverture du contexte assemblé (cible ≥ 0,80)
  Dimension 3 (Qualité pédagogique — style G-Eval) :
    - geval(criteria, **inputs)                : score 1–5 sur critère libre

  NB: recall_at_k et context_recall mesurent deux étapes distinctes du pipeline RAG.
      recall_at_k = qualité du retriever (documents récupérés).
      context_recall = qualité du contexte assemblé (information couverte).

Mode déterministe pour les tests :
  Passer `judge=offline_judge` (heuristique sans LLM) pour des tests reproductibles
  en CI. Passer `judge=llm_judge` pour une mesure réelle.

Usage minimal :
    from tests.eval.eval_judge import faithfulness, llm_judge, offline_judge

    score = faithfulness(answer, contexts, judge=llm_judge)      # mesure réelle
    score = faithfulness(answer, contexts, judge=offline_judge)  # CI déterministe

Formule composite (section 5 du modèle d'évaluation) :
    E(système) = WEIGHT_D1 * D1 + WEIGHT_D2 * D2 + WEIGHT_D3 * D3

    WEIGHT_D1 = 0.25  — D1 (intégrité technique) est une condition nécessaire, pas
                         la finalité. Un système qui "tourne" ne garantit ni contenu
                         fiable ni apprentissage (distinction vérification / validation,
                         ingénierie des systèmes).
    WEIGHT_D2 = 0.30  — D2 (qualité contenu) conditionne la confiance pédagogique.
                         Un tuteur qui hallucine est inutilisable (Es et al., RAGAS 2023 :
                         la Faithfulness est le facteur le plus critique dans l'adoption
                         des systèmes RAG par les enseignants).
    WEIGHT_D3 = 0.45  — D3 (efficacité pédagogique) est la finalité du tuteur. γ > 0.5
                         serait souhaitable mais fragiliserait D2 ; γ = 0.45 préserve
                         l'équilibre fiabilité / pédagogie tout en affirmant la priorité
                         pédagogique (Kirkpatrick niveau 3 : résultats d'apprentissage).
                         α + β + γ = 1.0 ✓
"""

from __future__ import annotations

import json
import re
from typing import Callable, List, Optional

# Signature d'un juge : (prompt: str) -> Optional[str]  (texte de réponse, ou None)
Judge = Callable[[str], Optional[str]]

# Pondérations de la formule composite (justification dans le docstring du module).
WEIGHT_D1: float = 0.25
WEIGHT_D2: float = 0.30
WEIGHT_D3: float = 0.45


# ──────────────────────────────────────────────────────────────────────────────
# Juges
# ──────────────────────────────────────────────────────────────────────────────

def llm_judge(prompt: str) -> Optional[str]:
    """Juge réel — appelle le LLM du dépôt via le routing multi-provider."""
    try:
        from ai.llm.service import call_llm
        return call_llm(prompt, max_tokens=300)
    except Exception:
        return None


def offline_judge(prompt: str) -> Optional[str]:
    """Juge déterministe pour CI — heuristique lexicale, aucun LLM.

    Trois chemins selon la structure du prompt :
    - faithfulness   : AFFIRMATION présente → verdict yes/no par recouvrement contenu.
    - context_recall : RÉFÉRENCE présente   → score [0-1] par recouvrement sémantique.
    - G-Eval         : CRITÈRE + REPONSE   → score [1-5] discriminant par critère pédagogique.
    """
    # Chemin faithfulness : AFFIRMATION présente (contexte peut être vide → overlap 0).
    claim = _extract_field(prompt, "AFFIRMATION")
    if claim:
        ctx = _extract_field(prompt, "CONTEXTE")
        overlap = _content_overlap(claim, ctx) if ctx else 0.0
        verdict = "yes" if overlap >= 0.55 else "no"
        return json.dumps({"verdict": verdict, "score": round(overlap, 2)})

    # Chemin context_recall : RÉFÉRENCE présente — contexte vide → score 0.0 explicite.
    ref = _extract_field(prompt, "RÉFÉRENCE")
    if ref:
        sources = _extract_field(prompt, "CONTEXTE")
        if not sources:
            return json.dumps({"score": 0.0})
        overlap = _content_overlap(ref, sources)
        return json.dumps({"score": round(overlap, 2)})

    # Chemin G-Eval : CRITÈRE + REPONSE → score 1-5 discriminant (pas de score fixe).
    criteria = _extract_field(prompt, "CRITÈRE")
    reponse = _extract_field(prompt, "REPONSE") or _extract_field(prompt, "RÉPONSE")
    if criteria and reponse:
        score = _geval_heuristic(criteria, reponse)
        return json.dumps({"score": score, "raison": "offline heuristic"})

    # Fallback : score neutre 3 — seul _parse_score_1_5 l'utilise (→ 0.5 normalisé).
    return json.dumps({"score": 3, "raison": "offline heuristic — champs non reconnus"})


# ──────────────────────────────────────────────────────────────────────────────
# Dimension 2 — Métriques RAGAS-style
# ──────────────────────────────────────────────────────────────────────────────

def recall_at_k(
    relevant_ids: set,
    retrieved_ids: set,
) -> float:
    """Recall@k = |D_pertinents ∩ D_top_k| / |D_pertinents|.

    Métrique retriever : les documents pertinents apparaissent-ils dans le top-k ?
    Distinct de context_recall (qui mesure si la vérité-terrain est couverte par
    le contexte assemblé — étape suivante du pipeline). Cible ≥ 0,80.
    """
    if not relevant_ids:
        return 0.0
    return len(relevant_ids & retrieved_ids) / len(relevant_ids)


def faithfulness(
    answer: str,
    contexts: List[str],
    *,
    claims: Optional[List[str]] = None,
    judge: Judge = offline_judge,
) -> float:
    """Faithfulness = (# affirmations ancrées dans le contexte) / (# affirmations).

    Décompose la réponse en affirmations atomiques puis demande au juge, pour
    chacune, si elle est supportée par le contexte récupéré. Mesure l'hallucination.
    Cible OpenTutorAI : ≥ 0,90.
    """
    if claims is None:
        claims = _split_claims(answer)
    if not claims:
        return 1.0
    sources = "\n".join(contexts)
    anchored = 0
    for claim in claims:
        prompt = (
            "Tu es un vérificateur factuel. Indique si l'AFFIRMATION est entièrement "
            "supportée par le CONTEXTE.\n\n"
            f"AFFIRMATION: {claim}\n\n"
            f"CONTEXTE: {sources}\n\n"
            'Réponds UNIQUEMENT en JSON : {"verdict": "yes|no"}'
        )
        verdict = _parse_verdict(judge(prompt))
        anchored += 1 if verdict else 0
    return anchored / len(claims)


def answer_relevancy(
    question: str,
    answer: str,
    *,
    judge: Judge = offline_judge,
) -> float:
    """Answer Relevancy — la réponse adresse-t-elle réellement la question.

    Le juge note la pertinence sur 1–5, normalisée en [0,1]. Cible ≥ 0,80.
    """
    prompt = (
        "Évalue dans quelle mesure la RÉPONSE adresse directement la QUESTION, "
        "sans digression ni information hors-sujet.\n\n"
        f"QUESTION: {question}\n\n"
        f"RÉPONSE: {answer}\n\n"
        'Réponds UNIQUEMENT en JSON : {"score": <1-5>}'
    )
    return _parse_score_1_5(judge(prompt), question=question, answer=answer)


def context_recall(
    ground_truth: str,
    contexts: List[str],
    *,
    judge: Judge = offline_judge,
) -> float:
    """Context Recall — l'information de la vérité-terrain est-elle dans le contexte assemblé.

    Mesure la couverture sémantique du contexte final passé au LLM.
    Distinct de recall_at_k (qui opère sur les IDs de documents récupérés par le retriever).
    Cible ≥ 0,80.
    """
    sources = "\n".join(contexts)
    prompt = (
        "Quelle proportion de l'information de la RÉFÉRENCE est présente dans le "
        "CONTEXTE récupéré ?\n\n"
        f"RÉFÉRENCE: {ground_truth}\n\n"
        f"CONTEXTE: {sources}\n\n"
        'Réponds UNIQUEMENT en JSON : {"score": <0.0-1.0>}'
    )
    return _parse_score_0_1(judge(prompt), ref=ground_truth, ctx=sources)


# ──────────────────────────────────────────────────────────────────────────────
# Dimension 3 — G-Eval (critères pédagogiques libres)
# ──────────────────────────────────────────────────────────────────────────────

# Critères correspondant à la grille humaine S7 (4 × 25 %).
GEVAL_PEDAGOGY = {
    "lisibilite": "L'explication est lisible et compréhensible sans jargon non expliqué.",
    "pertinence_exemple": "L'exemple proposé illustre clairement le concept enseigné.",
    "absence_jargon": "La réponse évite le jargon technique non défini pour le niveau.",
    "adaptation_niveau": "Le ton et la complexité sont adaptés au niveau de l'apprenant.",
}


def geval(
    criteria: str,
    *,
    judge: Judge = offline_judge,
    **inputs: str,
) -> float:
    """G-Eval — note une sortie sur un critère décrit en langage naturel (1–5 → [0,1]).

    Exemple :
        geval(GEVAL_PEDAGOGY["lisibilite"], judge=llm_judge,
              question=q, reponse=r)

    Le juge reçoit le critère + les entrées nommées et produit un score motivé.
    """
    fields = "\n".join(f"{k.upper()}: {v}" for k, v in inputs.items())
    prompt = (
        "Tu es un évaluateur pédagogique rigoureux. Évalue la sortie selon le "
        "CRITÈRE ci-dessous, étape par étape, puis attribue une note.\n\n"
        f"CRITÈRE: {criteria}\n\n"
        f"{fields}\n\n"
        'Réponds UNIQUEMENT en JSON : {"score": <1-5>, "raison": "<courte>"}'
    )
    return _parse_score_1_5(
        judge(prompt),
        question=inputs.get("question", ""),
        answer=inputs.get("reponse", inputs.get("answer", "")),
    )


def geval_pedagogy_full(
    *,
    judge: Judge = offline_judge,
    **inputs: str,
) -> dict:
    """Applique les 4 critères de la grille S7 et renvoie scores + moyenne.

    Renvoie {"lisibilite": .., ..., "moyenne": ..} — directement comparable à la
    note humaine ≥ 4/5 (soit ≥ 0,80 normalisé).
    """
    scores = {
        name: geval(desc, judge=judge, **inputs)
        for name, desc in GEVAL_PEDAGOGY.items()
    }
    scores["moyenne"] = sum(scores.values()) / len(GEVAL_PEDAGOGY)
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de parsing / heuristiques
# ──────────────────────────────────────────────────────────────────────────────

def _geval_heuristic(criteria: str, reponse: str) -> int:
    """Score 1–5 discriminant par critère pédagogique — sans LLM.

    Quatre critères reconnus (GEVAL_PEDAGOGY) → heuristiques spécifiques.
    Fallback neutre 3 si le critère est inconnu.
    """
    c = criteria.lower()
    r = reponse.lower()
    words = re.findall(r"\w+", r)
    if not words:
        return 1

    # Lisibilité : phrases courtes + peu de mots très longs (≥ 12 lettres).
    if "lisib" in c or "compréhensib" in c:
        sentences = [s for s in re.split(r"[.!?]+", reponse.strip()) if s.strip()]
        avg_len = sum(len(re.findall(r"\w+", s)) for s in sentences) / max(len(sentences), 1)
        long_ratio = sum(1 for w in words if len(w) >= 12) / len(words)
        if avg_len <= 20 and long_ratio < 0.05:
            return 5
        if avg_len <= 30 and long_ratio < 0.10:
            return 4
        if avg_len <= 40:
            return 3
        return 2

    # Pertinence d'exemple : présence d'un exemple concret (code ou marqueur explicite).
    if "exemple" in c or "illustre" in c:
        has_marker = any(kw in r for kw in ["exemple", "par exemple", "voici"])
        has_code = bool(re.search(r"\d|`|for |range|print|def |=", r))
        if has_marker and has_code:
            return 5
        if has_marker or has_code:
            return 4
        return 2

    # Absence de jargon : faible ratio de mots techniques longs (≥ 9 lettres).
    if "jargon" in c or "technique" in c:
        tech_ratio = sum(1 for w in words if len(w) >= 9) / len(words)
        if tech_ratio < 0.05:
            return 5
        if tech_ratio < 0.10:
            return 4
        if tech_ratio < 0.15:
            return 3
        return 2

    # Adaptation au niveau : marqueurs pédagogiques de simplification.
    if "adapt" in c or "niveau" in c or "complexit" in c or "ton" in c:
        markers = [
            "simplement", "c'est-à-dire", "veut dire", "autrement dit",
            "en d'autres termes", "c'est ", "facile", "basique",
        ]
        found = sum(1 for m in markers if m in r)
        if found >= 3:
            return 5
        if found >= 2:
            return 4
        if found >= 1:
            return 3
        return 2

    return 3  # critère non reconnu → score neutre


def _split_claims(text: str) -> List[str]:
    """Découpe une réponse en affirmations atomiques (phrases non vides)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


def _extract_field(prompt: str, label: str) -> str:
    m = re.search(rf"{label}:\s*(.+?)(?:\n\n|$)", prompt, re.S)
    return m.group(1).strip() if m else ""


def _lexical_overlap(a: str, b: str) -> float:
    """Jaccard mot-à-mot (fallback offline déterministe)."""
    wa = set(re.findall(r"\w+", a.lower()))
    wb = set(re.findall(r"\w+", b.lower()))
    if not wa:
        return 0.0
    return len(wa & wb) / len(wa)


_STOPWORDS = {
    "une", "un", "le", "la", "les", "des", "de", "du", "et", "ou", "que",
    "qui", "pour", "dans", "sur", "avec", "par", "est", "sont", "ce", "cette",
    "son", "ses", "aux", "au", "en", "il", "elle", "on", "se", "sa", "à",
    "python", "code", "fonction", "bloc",  # vocabulaire de domaine générique
}


def _content_overlap(a: str, b: str) -> float:
    """Recouvrement sur les mots de contenu (≥ 4 lettres, hors stopwords).

    Plus discriminant que le Jaccard brut : une affirmation hallucinée partage
    quelques mots de surface mais peu de mots de contenu réellement ancrés.
    """
    def content_words(s: str) -> set:
        return {
            w for w in re.findall(r"\w+", s.lower())
            if len(w) >= 4 and w not in _STOPWORDS
        }
    wa, wb = content_words(a), content_words(b)
    if not wa:
        return 0.0
    return len(wa & wb) / len(wa)


def _parse_json(text: Optional[str]) -> dict:
    if not text:
        return {}
    start, end = text.find("{"), text.rfind("}") + 1
    if start < 0 or end <= start:
        return {}
    try:
        return json.loads(text[start:end])
    except Exception:
        return {}


def _parse_verdict(text: Optional[str]) -> bool:
    data = _parse_json(text)
    return str(data.get("verdict", "")).lower() in ("yes", "oui", "true")


def _parse_score_1_5(
    text: Optional[str],
    *,
    question: str = "",
    answer: str = "",
) -> float:
    data = _parse_json(text)
    if "score" in data:
        try:
            return max(0.0, min(1.0, (float(data["score"]) - 1) / 4))
        except Exception:
            pass
    # fallback offline : recouvrement lexical question↔réponse
    return _lexical_overlap(question, answer) if question else 0.5


def _parse_score_0_1(
    text: Optional[str],
    *,
    ref: str = "",
    ctx: str = "",
) -> float:
    data = _parse_json(text)
    if "score" in data:
        try:
            return max(0.0, min(1.0, float(data["score"])))
        except Exception:
            pass
    return _lexical_overlap(ref, ctx) if ref else 0.5
