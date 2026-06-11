"""LLM service for coordinating completions."""

from typing import Optional

from ai.llm.schemas import LLMRequest, LLMResponse
from ai.llm.transports.base import LLMTransport


class LLMService:
    """Service for LLM operations."""

    def __init__(self, transport: LLMTransport):
        self.transport = transport

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Get completion from LLM."""
        return await self.transport.complete(request)


def call_llm(
    prompt: str,
    model: Optional[str] = None,
    max_tokens: int = 500,
) -> Optional[str]:
    """Sync convenience wrapper — returns text or None on failure."""
    from ai.llm.transports.httpx_transport import call_llm_sync

    return call_llm_sync(prompt, model=model, max_tokens=max_tokens)


def get_langchain_llm(model: Optional[str] = None):
    """Return a LangChain-compatible LLM for use with create_react_agent."""
    from ai.llm.transports.httpx_transport import get_langchain_llm as _get

    return _get(model)
