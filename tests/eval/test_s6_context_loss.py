"""S6 — Résistance à la perte de contexte (mixte).

Automatisé : CR ≤ 0,30 + IRR = I_conservées / I_clés ≥ 0,80
Humain     : cohérence réponse post-compression ≥ 4/5

Dépendance : S2 (corpus compressé réutilisé)
Corpus : fixe et versionné (~12 000 tokens)
LLM : mocké — résumé fixe incluant les informations clés
"""

import uuid
import pytest

from ai.summarization.service import SummarizationService

# ── Informations clés définies A PRIORI (avant exécution) ───────────────────
# I_clés : établie manuellement avant le test — ne pas modifier après l'exécution

KEY_INFORMATION = [
    "balises HTML",          # concept principal abordé
    "débutant",              # niveau de l'apprenant
    "difficultés avec <a>",  # difficulté détectée
    "<h1>",                  # notion vue
    "<p>",                   # notion vue
    "liens hypertexte",      # notion vue (variante)
]

# ── Corpus fixe ~12 000 tokens ───────────────────────────────────────────────

_SENTENCE_A = (
    "L'apprenant étudie les balises HTML : <h1> pour les titres, "
    "<p> pour les paragraphes, <a href='#'> pour les liens hypertexte. "
    "Il est au niveau débutant et progresse bien. "
)
_SENTENCE_B = (
    "Le tuteur signale des difficultés avec <a> et les attributs href. "
    "L'apprenant a correctement utilisé <h1> et <p> lors des exercices. "
)
CORPUS_12K = (_SENTENCE_A * 500) + (_SENTENCE_B * 250)  # ~17 500 tokens


# Résumé mocké — contient TOUTES les informations clés
MOCK_SUMMARY_FULL = (
    "L'apprenant de niveau débutant a étudié les balises HTML fondamentales : "
    "<h1>, <p> et les liens hypertexte avec <a>. "
    "Des difficultés avec <a> ont été détectées, notamment sur l'attribut href. "
    "Les balises <h1> et <p> sont maîtrisées."
)

# Résumé mocké dégradé — manque 2 informations clés (pour tester IRR < 1.0)
MOCK_SUMMARY_DEGRADED = (
    "L'apprenant a étudié des balises HTML. "
    "Quelques difficultés ont été signalées. "
    "Le cours s'est bien passé."
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def _compute_cr(tokens_before: int, tokens_after: int) -> float:
    return tokens_after / tokens_before


def _compute_irr(summary: str, key_info: list) -> float:
    """IRR = I_conservées / I_clés — recherche de chaîne insensible à la casse."""
    summary_lower = summary.lower()
    conserved = sum(1 for info in key_info if info.lower() in summary_lower)
    return conserved / len(key_info)


def _make_exchanges(corpus: str) -> list:
    chunks = [corpus[i:i+600] for i in range(0, len(corpus), 600)]
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": chunk}
        for i, chunk in enumerate(chunks[:120])
    ]


# ── Tests automatisés ────────────────────────────────────────────────────────

def test_s6_compression_ratio_within_target(db, user_id, monkeypatch):
    """CR = Tokens_après / Tokens_avant ≤ 0,30 sur un corpus de ~12 000 tokens."""
    svc = SummarizationService()
    monkeypatch.setattr(svc, "_call_llm", lambda *a, **kw: MOCK_SUMMARY_FULL)

    exchanges = _make_exchanges(CORPUS_12K)
    tokens_before = svc.count_tokens(exchanges)

    summary = svc.summarize_session(
        user_id=user_id,
        support_id="html-s6",
        exchanges=exchanges,
        db=db,
        force=True,
    )

    tokens_after = _count_tokens(summary)
    cr = _compute_cr(tokens_before, tokens_after)

    assert tokens_before >= 10000, (
        f"Corpus insuffisant : {tokens_before} tokens — ajuster CORPUS_12K."
    )
    assert cr <= 0.30, (
        f"CR = {cr:.3f} ({tokens_after}/{tokens_before}) — dépasse le seuil de 0,30."
    )


def test_s6_irr_above_target(db, user_id, monkeypatch):
    """IRR = I_conservées / I_clés ≥ 0,80 : les informations pédagogiques essentielles sont conservées."""
    svc = SummarizationService()
    monkeypatch.setattr(svc, "_call_llm", lambda *a, **kw: MOCK_SUMMARY_FULL)

    exchanges = _make_exchanges(CORPUS_12K)
    summary = svc.summarize_session(
        user_id=user_id, support_id="html-s6",
        exchanges=exchanges, db=db, force=True,
    )

    irr = _compute_irr(summary, KEY_INFORMATION)
    missing = [info for info in KEY_INFORMATION if info.lower() not in summary.lower()]

    assert irr >= 0.80, (
        f"IRR = {irr:.0%} ({len(KEY_INFORMATION) - len(missing)}/{len(KEY_INFORMATION)}) "
        f"— informations perdues : {missing}"
    )


def test_s6_cr_and_irr_not_both_degraded(db, user_id, monkeypatch):
    """Trade-off CR ↔ IRR : si CR est très agressif (< 0,10), l'IRR ne doit pas chuter sous 0,80."""
    svc = SummarizationService()
    monkeypatch.setattr(svc, "_call_llm", lambda *a, **kw: MOCK_SUMMARY_FULL)

    exchanges = _make_exchanges(CORPUS_12K)
    tokens_before = svc.count_tokens(exchanges)

    summary = svc.summarize_session(
        user_id=user_id, support_id="html-s6-tradeoff",
        exchanges=exchanges, db=db, force=True,
    )

    cr = _compute_cr(tokens_before, _count_tokens(summary))
    irr = _compute_irr(summary, KEY_INFORMATION)

    if cr < 0.10:
        assert irr >= 0.80, (
            f"Trade-off critique : CR={cr:.3f} trop agressif entraîne IRR={irr:.0%} < 80 %."
        )


def test_s6_degraded_summary_irr_below_target(db, user_id, monkeypatch):
    """Un résumé trop vague fait chuter l'IRR — valide la sensibilité de la métrique."""
    svc = SummarizationService()
    monkeypatch.setattr(svc, "_call_llm", lambda *a, **kw: MOCK_SUMMARY_DEGRADED)

    exchanges = _make_exchanges(CORPUS_12K)
    summary = svc.summarize_session(
        user_id=user_id, support_id="html-s6-degraded",
        exchanges=exchanges, db=db, force=True,
    )

    irr = _compute_irr(summary, KEY_INFORMATION)
    assert irr < 0.80, (
        f"Le résumé dégradé obtient IRR={irr:.0%} ≥ 80 % — "
        "la métrique IRR n'est pas assez discriminante."
    )


# ── Cadre évaluation humaine ─────────────────────────────────────────────────

def test_s6_post_compression_response_framework(db, user_id, monkeypatch):
    """Génère la réponse post-compression et la documente pour évaluation humaine.

    Question simulée : 'rappelle-moi ce qu'on a vu sur les balises HTML'
    Évaluation : cohérence ≥ 4/5 — double évaluateur requis.
    """
    svc = SummarizationService()
    monkeypatch.setattr(svc, "_call_llm", lambda *a, **kw: MOCK_SUMMARY_FULL)

    exchanges = _make_exchanges(CORPUS_12K)
    summary = svc.summarize_session(
        user_id=user_id, support_id="html-s6-human",
        exchanges=exchanges, db=db, force=True,
    )

    irr = _compute_irr(summary, KEY_INFORMATION)

    print(
        f"\n[S6 — Évaluation humaine]\n"
        f"  Question post-compression : 'rappelle-moi ce qu'on a vu sur les balises HTML'\n"
        f"  Résumé produit :\n  {summary}\n\n"
        f"  IRR automatique : {irr:.0%}\n"
        f"  → Grille humaine (1-5) : lisibilité / fidélité / cohérence pédagogique / absence d'hallucination\n"
        f"  → Cible : ≥ 4/5 — accord ≥ 80 % entre deux évaluateurs requis."
    )

    assert summary.strip(), "Résumé vide — aucune réponse post-compression possible."
