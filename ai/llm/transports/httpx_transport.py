"""Sync httpx LLM transport — Ollama then OpenAI-compatible (no Anthropic)."""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import httpx

log = logging.getLogger(__name__)

_DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")


def call_llm_sync(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 500,
    timeout: float = 30.0,
) -> Optional[str]:
    """Provider-agnostic sync LLM call. Returns text or None on any failure."""
    model = model or _DEFAULT_MODEL

    for url in _ollama_urls():
        try:
            resp = httpx.post(
                f"{url.rstrip('/')}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            if text:
                return text
        except Exception as exc:
            log.debug("Ollama call failed (%s): %s", url, exc)

    oa_urls = _openai_urls()
    oa_keys = _openai_keys()
    for idx, url in enumerate(oa_urls):
        key = oa_keys[idx] if idx < len(oa_keys) else ""
        try:
            resp = httpx.post(
                f"{url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                },
                timeout=timeout,
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
            log.debug("OpenAI-compatible call failed (%s): %s", url, exc)

    return None


def call_llm_with_messages(
    messages: List[dict],
    model: Optional[str] = None,
    max_tokens: int = 800,
    timeout: float = 60.0,
) -> Optional[str]:
    """Multi-turn messages call (system + user/assistant roles).

    Ollama: maps system role to the dedicated `system` field of /api/generate.
    OpenAI-compatible: passes the messages array directly to /chat/completions.
    Returns text or None on any failure.
    """
    model = model or _DEFAULT_MODEL

    for url in _ollama_urls():
        try:
            resp = httpx.post(
                f"{url.rstrip('/')}/api/chat",
                json={"model": model, "messages": messages, "stream": False},
                timeout=timeout,
            )
            resp.raise_for_status()
            text = (
                resp.json()
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if text:
                return text
        except Exception as exc:
            log.debug("Ollama messages call failed (%s): %s", url, exc)

    oa_urls = _openai_urls()
    oa_keys = _openai_keys()
    for idx, url in enumerate(oa_urls):
        key = oa_keys[idx] if idx < len(oa_keys) else ""
        try:
            resp = httpx.post(
                f"{url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": messages, "max_tokens": max_tokens},
                timeout=timeout,
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
            log.debug("OpenAI messages call failed (%s): %s", url, exc)

    return None


def get_langchain_llm(model: Optional[str] = None):
    """Return a LangChain ChatOpenAI instance pointed at the configured provider."""
    from langchain_openai import ChatOpenAI

    model = model or _DEFAULT_MODEL

    # Prefer Ollama via its OpenAI-compatible /v1 endpoint
    urls = _ollama_urls()
    if urls:
        return ChatOpenAI(
            base_url=f"{urls[0].rstrip('/')}/v1",
            api_key="ollama",
            model=model,
            temperature=0.7,
        )

    oa_urls = _openai_urls()
    oa_keys = _openai_keys()
    if oa_urls:
        return ChatOpenAI(
            base_url=oa_urls[0],
            api_key=oa_keys[0] if oa_keys else "none",
            model=model,
            temperature=0.7,
        )

    raise RuntimeError(
        "No LLM provider configured (set OLLAMA_BASE_URL or OPENAI_API_BASE_URL)"
    )


def _ollama_urls() -> list:
    multi = os.getenv("OLLAMA_BASE_URLS", "")
    if multi:
        return [u.strip() for u in multi.split(",") if u.strip()]
    single = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    return [single] if single else []


def _openai_urls() -> list:
    multi = os.getenv("OPENAI_API_BASE_URLS", "")
    if multi:
        return [u.strip() for u in multi.split(",") if u.strip()]
    single = os.getenv("OPENAI_API_BASE_URL", "")
    return [single] if single else []


def _openai_keys() -> list:
    multi = os.getenv("OPENAI_API_KEYS", "")
    if multi:
        return [k.strip() for k in multi.split(",") if k.strip()]
    single = os.getenv("OPENAI_API_KEY", "")
    return [single] if single else []
