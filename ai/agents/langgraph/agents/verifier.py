"""VerifierAgent — LLM structured judgment + text-overlap fallback + interrupt P2."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from langgraph.types import interrupt

log = logging.getLogger(__name__)


def verifier_node(state: Dict[str, Any]) -> Dict[str, Any]:
    exercises = list(state.get("exercises") or [])
    strategy = state.get("strategy", "")
    rag_docs = list(state.get("rag_docs") or [])
    human_feedback = state.get("human_feedback", "")
    agent_trace = list(state.get("agent_trace") or [])
    agent_reasoning = dict(state.get("agent_reasoning") or {})
    n_retries = dict(state.get("n_retries") or {})
    verification_feedback = list(state.get("verification_feedback") or [])

    from ai.agents.helpers import is_text_supported
    from ai.llm.service import call_llm

    try:
        from config import settings

        threshold = settings.CONTEXT_RETRIEVAL_CONFIG.get("rag", {}).get(
            "verification_threshold", 0.65
        )
    except Exception:
        threshold = 0.65

    corpus = " ".join(d.get("content", "") for d in rag_docs)
    verification: Dict[str, Any]

    if not rag_docs:
        verification = {
            "verdict": "no_sources",
            "score": 0.0,
            "specific_feedback": [],
            "unsupported_items": [],
        }
        agent_reasoning["verifier"] = "[no_sources] No RAG docs available"
    else:
        exercises_summary = json.dumps(exercises[:3], ensure_ascii=False)[:500]
        prompt = (
            f"Vérifie que les exercices et la stratégie sont supportés par les sources RAG.\n\n"
            f"Stratégie : {strategy[:200]}\n"
            f"Exercices : {exercises_summary}\n"
            f"Sources RAG : {corpus[:500]}\n\n"
            f"Réponds en JSON :\n"
            f'{{"verdict": "supported|needs_review|no_sources", "score": 0.0, '
            f'"specific_feedback": ["..."], "unsupported_items": ["..."]}}'
        )
        llm_text = call_llm(prompt, max_tokens=300)

        if llm_text:
            try:
                start = llm_text.index("{")
                end = llm_text.rindex("}") + 1
                verification = json.loads(llm_text[start:end])
                agent_reasoning["verifier"] = (
                    f"[LLM] verdict={verification.get('verdict')} "
                    f"score={verification.get('score', 0):.2f}"
                )
            except Exception:
                verification = _text_overlap_verdict(
                    strategy, exercises, corpus, threshold
                )
                agent_reasoning["verifier"] = "[fallback] text-overlap"
        else:
            verification = _text_overlap_verdict(strategy, exercises, corpus, threshold)
            agent_reasoning["verifier"] = "[fallback] text-overlap"

    # Propagate specific feedback to planner on next retry
    if verification.get("specific_feedback"):
        verification_feedback = list(verification_feedback) + list(
            verification["specific_feedback"]
        )

    agent_trace.append(
        f"verifier → verdict={verification.get('verdict')} "
        f"score={verification.get('score', 0.0):.2f}"
    )

    # Human-in-the-Loop P2 — ask learner when items cannot be verified
    if verification.get("verdict") == "needs_review" and not human_feedback:
        unsupported = verification.get("unsupported_items", [])
        human_feedback = interrupt(
            {
                "checkpoint": "P2",
                "message": (
                    f"Éléments non vérifiés dans les sources RAG : {unsupported}. "
                    f"Continuer quand même ? (oui / non)"
                ),
                "verification": verification,
            }
        )
        if not isinstance(human_feedback, str):
            human_feedback = ""

    return {
        "verification": verification,
        "verification_feedback": verification_feedback,
        "human_feedback": human_feedback,
        "agent_trace": agent_trace,
        "agent_reasoning": agent_reasoning,
        "n_retries": n_retries,
    }


def _text_overlap_verdict(
    strategy: str,
    exercises: list,
    corpus: str,
    threshold: float,
) -> Dict[str, Any]:
    from ai.agents.helpers import is_text_supported

    text = strategy + " ".join(e.get("question", "") for e in exercises)
    supported = is_text_supported(text, corpus, threshold)
    return {
        "verdict": "supported" if supported else "needs_review",
        "score": 0.8 if supported else 0.4,
        "specific_feedback": [],
        "unsupported_items": [],
    }
