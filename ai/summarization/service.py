"""Session Summarization Service.

Dual-trigger strategy: summarize when exchange count >= threshold OR
token count >= threshold. Summaries are cached in opentutorai_memory
with memory_type='session_summary'. LLM call is provider-agnostic —
tries Ollama then OpenAI-compatible based on configured provider URLs.
"""

from __future__ import annotations

import uuid
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

log = logging.getLogger(__name__)

_EXCHANGES_THRESHOLD = 5
_TOKEN_THRESHOLD = 1200


def _get_summarization_cfg() -> Dict[str, Any]:
    try:
        from config import settings

        return getattr(settings, "SUMMARIZATION_CONFIG", {})
    except Exception:
        return {}


def extract_exchanges(chat_json: Any) -> List[Dict[str, Any]]:
    """Defensive extraction of messages from Chat.chat opaque JSON blob."""
    if not isinstance(chat_json, dict):
        return []
    return chat_json.get("messages", [])


class SummarizationService:
    """Session summarization with dual threshold trigger and SQL cache."""

    # ── Token counting ────────────────────────────────────────────────────────

    def count_tokens(self, exchanges: List[Dict[str, Any]]) -> int:
        """Count tokens using tiktoken cl100k_base, falling back to char estimate."""
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            total = 0
            for msg in exchanges:
                content = msg.get("content", "") if isinstance(msg, dict) else ""
                total += len(enc.encode(str(content)))
            return total
        except Exception:
            total_chars = sum(
                len(str(m.get("content", ""))) for m in exchanges if isinstance(m, dict)
            )
            return total_chars // 4

    # ── Trigger ───────────────────────────────────────────────────────────────

    def should_summarize(
        self,
        exchanges: List[Dict[str, Any]],
        cfg: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Return True when exchange count OR token count exceeds threshold."""
        cfg = cfg or _get_summarization_cfg()
        n_thresh = cfg.get("exchanges_per_summary", _EXCHANGES_THRESHOLD)
        tok_thresh = cfg.get("auto_summarize_token_limit", _TOKEN_THRESHOLD)

        if len(exchanges) >= n_thresh:
            return True
        if self.count_tokens(exchanges) >= tok_thresh:
            return True
        return False

    # ── Cache (opentutorai_memory) ────────────────────────────────────────────

    def get_cached_summary(
        self, user_id: str, support_id: str, db: Session
    ) -> Optional[str]:
        """Return the most recent cached session_summary for user+support, or None."""
        from data.models.memory import Memory

        rows = (
            db.query(Memory)
            .filter(
                Memory.user_id == user_id,
                Memory.memory_type == "session_summary",
            )
            .order_by(Memory.created_at.desc())
            .all()
        )
        for row in rows:
            meta = row.memory_metadata or {}
            if meta.get("support_id") == support_id:
                return row.content
        return None

    def cache_summary(
        self, user_id: str, support_id: str, summary: str, db: Session
    ) -> None:
        """Persist summary in opentutorai_memory as session_summary."""
        from data.models.memory import Memory

        row = Memory(
            id=str(uuid.uuid4()),
            user_id=user_id,
            memory_type="session_summary",
            content=summary,
            memory_metadata={"support_id": support_id},
        )
        db.add(row)
        db.commit()

    def invalidate_cache(self, user_id: str, support_id: str, db: Session) -> None:
        """Delete all session_summary entries for this user+support."""
        from data.models.memory import Memory

        rows = (
            db.query(Memory)
            .filter(
                Memory.user_id == user_id,
                Memory.memory_type == "session_summary",
            )
            .all()
        )
        for row in rows:
            meta = row.memory_metadata or {}
            if meta.get("support_id") == support_id:
                db.delete(row)
        db.commit()

    # ── LLM call ─────────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, db: Session, model: Optional[str] = None) -> str:
        """Provider-agnostic LLM call. Tries Ollama, then OpenAI-compatible."""
        import httpx
        from ai.providers.config_service import ProviderConfigService

        config = ProviderConfigService(db)

        # Try Ollama (native /api/generate)
        ollama_cfg = config.get_ollama()
        ollama_urls: list = ollama_cfg.get("OLLAMA_BASE_URLS") or []
        for url in ollama_urls:
            try:
                resp = httpx.post(
                    f"{url.rstrip('/')}/api/generate",
                    json={
                        "model": model or "mistral",
                        "prompt": prompt,
                        "stream": False,
                    },
                    timeout=30.0,
                )
                resp.raise_for_status()
                text = resp.json().get("response", "").strip()
                if text:
                    return text
            except Exception as exc:
                log.debug("Ollama summarization failed (%s): %s", url, exc)

        # Try OpenAI-compatible /chat/completions
        oa_cfg = config.get_openai()
        oa_urls: list = oa_cfg.get("OPENAI_API_BASE_URLS") or []
        oa_keys: list = oa_cfg.get("OPENAI_API_KEYS") or []
        for idx, url in enumerate(oa_urls):
            key = oa_keys[idx] if idx < len(oa_keys) else ""
            try:
                resp = httpx.post(
                    f"{url.rstrip('/')}/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": model or "gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 300,
                    },
                    timeout=30.0,
                )
                resp.raise_for_status()
                text = (
                    resp.json()
                    .get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )
                if text:
                    return text
            except Exception as exc:
                log.debug("OpenAI-compatible summarization failed (%s): %s", url, exc)

        raise RuntimeError("No LLM provider available for summarization")

    # ── Main entry point ──────────────────────────────────────────────────────

    def summarize_session(
        self,
        user_id: str,
        support_id: str,
        exchanges: List[Dict[str, Any]],
        db: Session,
        model: Optional[str] = None,
        force: bool = False,
    ) -> str:
        """Generate, cache and return a session summary.

        If force=False and the trigger threshold is not met, returns
        the cached summary (if any) or an empty string.
        On LLM failure, caches a minimal fallback string.
        """
        cfg = _get_summarization_cfg()
        if not force and not self.should_summarize(exchanges, cfg):
            cached = self.get_cached_summary(user_id, support_id, db)
            return cached or ""

        conversation = "\n".join(
            f"{m.get('role', 'user').upper()}: {m.get('content', '')}"
            for m in exchanges
            if isinstance(m, dict)
        )
        prompt = (
            "Résume en 3-5 phrases la session d'apprentissage suivante.\n"
            "IMPORTANT : distingue clairement les réponses de l'apprenant de celles du tuteur.\n"
            "- Si l'apprenant a donné une mauvaise réponse, mentionne-la explicitement (ex: 'l'apprenant a répondu X au lieu de Y').\n"
            "- Ne présente jamais une correction du tuteur comme une réussite de l'apprenant.\n"
            "Mets en avant : les concepts abordés, les erreurs commises par l'apprenant, "
            "et les progrès réels observés.\n\n"
            f"{conversation}\n\nRésumé:"
        )

        try:
            summary = self._call_llm(prompt, db, model)
        except Exception as exc:
            log.warning("Summarization LLM call failed — using fallback: %s", exc)
            summary = f"Session avec {len(exchanges)} échanges."

        self.invalidate_cache(user_id, support_id, db)
        self.cache_summary(user_id, support_id, summary, db)
        return summary
