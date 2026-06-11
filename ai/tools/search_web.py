"""Tool — web search via DuckDuckGo Instant Answer API, graceful stub fallback."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def search_web(query: str) -> str:
    """Search the web for educational resources related to a query."""
    try:
        import httpx

        resp = httpx.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            timeout=5.0,
        )
        data = resp.json()
        abstract = data.get("AbstractText", "")
        if abstract:
            return f"DuckDuckGo : {abstract[:500]}"
        related = data.get("RelatedTopics", [])
        texts = [
            t.get("Text", "")
            for t in related[:3]
            if isinstance(t, dict) and t.get("Text")
        ]
        if texts:
            return "\n".join(texts)
    except Exception:
        pass

    return (
        f"Résultats web pour « {query} » : "
        "indisponible en mode hors-ligne. "
        "Consulte la documentation officielle ou MDN/Wikipedia."
    )
