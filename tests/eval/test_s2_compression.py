"""S2 — Compression de contexte (automatisé).

Objectif : CR = Tokens_après / Tokens_avant ≤ 0,30
Corpus : texte fixe et versionné de ~10 000 tokens (reproductible)
LLM : mocké — retourne un résumé court fixe
"""

import uuid
import pytest

from ai.summarization.service import SummarizationService

# ── Corpus fixe et reproductible (~10 000 tokens) ───────────────────────────
# ~4 chars / token → 40 000 chars ≈ 10 000 tokens

_SENTENCE = (
    "L'apprenant étudie les balises HTML de base : "
    "<h1> pour les titres, <p> pour les paragraphes, "
    "<a href='#'> pour les liens et <img src='#'> pour les images. "
    "Le tuteur explique chaque balise avec un exemple concret. "
)
CORPUS_10K_TOKENS = _SENTENCE * 600  # ~600 × 22 tokens ≈ 13 200 tokens


# Résumé fixe retourné par le LLM mocké (~80 tokens)
MOCK_SUMMARY = (
    "L'apprenant a étudié les balises HTML fondamentales : "
    "<h1>, <p>, <a> et <img>. "
    "Le tuteur a fourni des exemples pour chacune. "
    "Niveau débutant, aucune difficulté majeure signalée."
)


def _count_tokens(text: str) -> int:
    """Compte les tokens avec tiktoken cl100k_base (même tokenizer que le service)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return len(text) // 4


def _make_exchanges_from_corpus(corpus: str) -> list:
    """Découpe le corpus en échanges user/assistant simulés."""
    chunks = [corpus[i:i+500] for i in range(0, len(corpus), 500)]
    exchanges = []
    for i, chunk in enumerate(chunks[:100]):  # 100 échanges max (~10 000 tokens)
        role = "user" if i % 2 == 0 else "assistant"
        exchanges.append({"role": role, "content": chunk})
    return exchanges


# ── Tests ────────────────────────────────────────────────────────────────────

def test_s2_compression_ratio_within_target(db, user_id, monkeypatch):
    """CR = Tokens_après / Tokens_avant doit être ≤ 0,30."""
    svc = SummarizationService()
    monkeypatch.setattr(svc, "_call_llm", lambda *a, **kw: MOCK_SUMMARY)

    exchanges = _make_exchanges_from_corpus(CORPUS_10K_TOKENS)
    tokens_before = svc.count_tokens(exchanges)

    summary = svc.summarize_session(
        user_id=user_id,
        support_id="html-s2",
        exchanges=exchanges,
        db=db,
        force=True,
    )

    tokens_after = _count_tokens(summary)
    cr = tokens_after / tokens_before

    assert tokens_before >= 8000, (
        f"Corpus trop petit ({tokens_before} tokens) — ajuster CORPUS_10K_TOKENS."
    )
    assert cr <= 0.30, (
        f"CR = {cr:.3f} — compression insuffisante "
        f"({tokens_after} tokens après / {tokens_before} avant). Cible : ≤ 0,30."
    )


def test_s2_summary_is_non_empty(db, user_id, monkeypatch):
    """Le résumé produit n'est pas vide."""
    svc = SummarizationService()
    monkeypatch.setattr(svc, "_call_llm", lambda *a, **kw: MOCK_SUMMARY)

    exchanges = _make_exchanges_from_corpus(CORPUS_10K_TOKENS)
    summary = svc.summarize_session(
        user_id=user_id,
        support_id="html-s2",
        exchanges=exchanges,
        db=db,
        force=True,
    )
    assert summary.strip(), "Le résumé est vide."


def test_s2_summary_cached_after_compression(db, user_id, monkeypatch):
    """Le résumé est bien persisté en base après compression."""
    svc = SummarizationService()
    monkeypatch.setattr(svc, "_call_llm", lambda *a, **kw: MOCK_SUMMARY)

    exchanges = _make_exchanges_from_corpus(CORPUS_10K_TOKENS)
    svc.summarize_session(
        user_id=user_id,
        support_id="html-s2",
        exchanges=exchanges,
        db=db,
        force=True,
    )

    cached = svc.get_cached_summary(user_id, "html-s2", db)
    assert cached is not None, "Résumé non persisté en base après compression."


def test_s2_compression_reproducible(db, user_id, monkeypatch):
    """Deux exécutions avec le même corpus produisent le même CR."""
    svc = SummarizationService()
    monkeypatch.setattr(svc, "_call_llm", lambda *a, **kw: MOCK_SUMMARY)

    exchanges = _make_exchanges_from_corpus(CORPUS_10K_TOKENS)
    tokens_before = svc.count_tokens(exchanges)

    uid2 = str(uuid.uuid4())
    for uid in [user_id, uid2]:
        summary = svc.summarize_session(
            user_id=uid,
            support_id="html-s2",
            exchanges=exchanges,
            db=db,
            force=True,
        )
        cr = _count_tokens(summary) / tokens_before
        assert cr <= 0.30, f"CR non reproductible pour user {uid} : {cr:.3f}"
