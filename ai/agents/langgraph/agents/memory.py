"""MemoryAgent — LLM selects 4-6 most relevant memories, falls back to top-6."""

from __future__ import annotations

import logging
from typing import Any, Dict

log = logging.getLogger(__name__)


def memory_node(state: Dict[str, Any]) -> Dict[str, Any]:
    user_id = state.get("user_id", "")
    support = state.get("support", "")
    question = state.get("user_message", "")
    agent_trace = list(state.get("agent_trace") or [])
    agent_reasoning = dict(state.get("agent_reasoning") or {})

    memories = []
    memory_summary = ""

    try:
        from data.database import get_db
        from ai.retrieval.context_retrieval import ContextRetrievalService
        from ai.context_manager.service import ContextManager

        db = next(get_db())
        try:
            svc = ContextRetrievalService()
            # No substring query here on purpose: retrieve_internal_memory does
            # a literal ILIKE match, and a real question or topic name almost
            # never appears verbatim inside a stored memory's content — passing
            # either as `query` silently empties the candidate pool. Relevance
            # judgment happens downstream in _llm_select(), which sees the
            # actual question.
            raw = svc.retrieve_internal_memory(
                user_id=user_id,
                query="",
                memory_types=["episodic", "behavioral", "procedural"],
                limit=20,
                db=db,
            )
            mgr = ContextManager()
            filtered = mgr.filter_memories(raw, support=support)

            if filtered:
                selected = _llm_select(filtered, question, support, state)
                memories = selected if selected else filtered[:6]

            memory_summary = "; ".join(m.get("content", "")[:200] for m in memories)
        finally:
            db.close()
    except Exception as exc:
        log.warning("MemoryAgent degraded (empty context): %s", exc)

    agent_trace.append(f"memory → {len(memories)} memories loaded")
    agent_reasoning["memory"] = (
        f"[{'LLM' if memories else 'empty'}] {len(memories)} memories"
    )

    return {
        "memory_context": memories,
        "memory_summary": memory_summary,
        "agent_trace": agent_trace,
        "agent_reasoning": agent_reasoning,
    }


def _llm_select(
    memories: list, question: str, support: str, state: Dict[str, Any]
) -> list:
    from ai.llm.service import call_llm

    summaries = "\n".join(
        f"{i}. [{m.get('memory_type', '?')}] {m.get('content', '')[:100]}"
        for i, m in enumerate(memories)
    )
    prompt = (
        f"Question actuelle de l'apprenant : {question or '(aucune question précise, session générale)'}\n"
        f"Support : {support}\n"
        f"Niveau : {state.get('current_level', '?')}\n\n"
        f"Mémoires disponibles :\n{summaries}\n\n"
        f"Sélectionne les 4-6 indices les plus pertinents pour répondre à la question ci-dessus "
        f"ou, à défaut de question précise, pour personnaliser cette session.\n"
        f"Réponds uniquement avec les indices séparés par des virgules, ex : 0,2,4"
    )
    text = call_llm(prompt, model=state.get("model"), max_tokens=50)
    if text:
        try:
            indices = [
                int(x.strip()) for x in text.strip().split(",") if x.strip().isdigit()
            ]
            return [memories[i] for i in indices if 0 <= i < len(memories)]
        except Exception:
            pass
    return []
