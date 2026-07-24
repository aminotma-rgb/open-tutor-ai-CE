"""OrchestratorAgent — LLM-primary routing with deterministic fallback.

Routing decision order:
  1. _llm_route() — structured JSON {next_agent, reasoning, confidence}
  2. _route()     — deterministic fallback based on populated state fields
  3. Hard ceiling — MAX_ITERATIONS stops infinite loops
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

MAX_ITERATIONS = 15
_AGENTS = [
    "memory",
    "knowledge",
    "diagnostics",
    "planner",
    "exercise",
    "verifier",
    "feedback",
]


def _route(state: Dict[str, Any]) -> str:
    """Deterministic fallback — route to first unvisited step.

    Uses key presence, not truthiness: an agent can legitimately return an
    empty list/dict (e.g. no memories found for a new user), and a falsy
    check would send the graph back to that same agent forever.
    """
    if "memory_context" not in state:
        return "memory"
    if "knowledge_graph" not in state:
        return "knowledge"
    if "adjusted_level" not in state:
        return "diagnostics"
    if "strategy" not in state:
        return "planner"
    if "exercises" not in state:
        return "exercise"
    if "verification" not in state:
        return "verifier"
    if state.get("verification", {}).get("verdict") == "needs_review":
        n_retries = state.get("n_retries") or {}
        if n_retries.get("planner", 0) < 3:
            return "planner"
    return "feedback"


def _llm_route(state: Dict[str, Any]) -> Optional[str]:
    """LLM-primary routing — returns agent name or None on failure."""
    from ai.llm.service import call_llm

    prompt = (
        f"Tu es l'orchestrateur d'un tuteur adaptatif.\n\n"
        f"État actuel :\n"
        f"- support : {state.get('support', '')}\n"
        f"- niveau : {state.get('current_level', 'beginner')}\n"
        f"- iteration : {state.get('iteration', 0)}\n"
        f"- agent_trace : {state.get('agent_trace', [])[-5:]}\n"
        f"- weak_concepts : {state.get('weak_concepts', [])}\n"
        f"- verification : {state.get('verification', {})}\n"
        f"- n_retries : {state.get('n_retries', {})}\n"
        f"- human_feedback : {state.get('human_feedback', '')}\n\n"
        f"Agents disponibles : {_AGENTS + ['END']}\n\n"
        f"Règles :\n"
        f"- Suis l'ordre logique : memory → knowledge → diagnostics → planner → exercise → verifier → feedback → END\n"
        f"- Si verification.verdict=needs_review et planner retries < 3 → rappelle planner\n"
        f"- Décide END quand la session est pédagogiquement complète\n\n"
        f"Réponds uniquement en JSON valide :\n"
        f'{{ "next_agent": "<nom>", "reasoning": "<explication courte>", "confidence": 0.0 }}'
    )

    text = call_llm(prompt, model=state.get("model"), max_tokens=150)
    if not text:
        return None
    try:
        # Extract JSON even if surrounded by prose
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
        agent = data.get("next_agent", "")
        if agent in _AGENTS + ["END"]:
            log.debug(
                "LLM route → %s (confidence=%.2f) — %s",
                agent,
                data.get("confidence", 0.0),
                data.get("reasoning", "")[:60],
            )
            return agent
    except (ValueError, json.JSONDecodeError, KeyError):
        pass
    return None


def orchestrator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """Increment iteration, route via LLM then fallback, enforce ceiling."""
    incoming_iteration = state.get("iteration") or 0
    iteration = incoming_iteration + 1
    agent_trace = list(state.get("agent_trace") or [])
    agent_reasoning = dict(state.get("agent_reasoning") or {})

    # feedback_node is the only agent that proactively signals completion
    # (it returns next_agent="END"). Without this check, every orchestrator
    # call unconditionally recomputed a fresh routing decision and silently
    # discarded that signal, so the graph could never terminate on its own —
    # it always ran until MAX_ITERATIONS forced a stop, even right after a
    # session had already finished (confirmed via LoCoMo diagnostic: 3/3
    # sessions hit the hard ceiling despite every structured field already
    # being populated).
    #
    # Guarded by incoming_iteration > 0: "next_agent" has no reducer on this
    # TypedDict state, so it persists in the checkpointed state across
    # invoke() calls. Every caller (production adaptive_chat() and all eval
    # runners) resets "iteration" to 0 on every fresh invoke(), so seeing
    # incoming_iteration == 0 here means any next_agent="END" present is a
    # stale leftover from the END of the *previous* invoke, not a real
    # signal for this one — respecting it unconditionally would short-
    # circuit every subsequent invoke on the same thread without doing any
    # work.
    if incoming_iteration > 0 and state.get("next_agent") == "END":
        agent_trace.append("[END] feedback signalled session completion")
        return {
            "next_agent": "END",
            "iteration": iteration,
            "agent_trace": agent_trace,
            "agent_reasoning": agent_reasoning,
        }

    if iteration > MAX_ITERATIONS:
        agent_trace.append(f"[STOP] Max iterations ({MAX_ITERATIONS}) reached")
        return {
            "next_agent": "END",
            "iteration": iteration,
            "agent_trace": agent_trace,
            "agent_reasoning": agent_reasoning,
        }

    # Attempt LLM routing
    use_llm = True
    try:
        from config import settings

        use_llm = settings.CONTEXT_RETRIEVAL_CONFIG.get("langchain", {}).get(
            "orchestrator_use_llm", True
        )
    except Exception:
        pass

    next_agent: Optional[str] = None
    if use_llm:
        next_agent = _llm_route(state)
        if next_agent == "planner":
            # _llm_route()'s prompt tells it to respect the same n_retries<3
            # cap that _route()'s deterministic fallback enforces below, but
            # a 7B model doesn't reliably follow that instruction — confirmed
            # via a real LoCoMo run reaching "planner → retry #6" before the
            # hard MAX_ITERATIONS ceiling had to intervene. Validate here
            # instead of trusting the prompt: discard the LLM's decision so
            # next_agent falls through to _route(), whose retry cap is
            # enforced in code, not by LLM compliance.
            n_retries = state.get("n_retries") or {}
            if n_retries.get("planner", 0) >= 3:
                next_agent = None
        if next_agent:
            agent_reasoning["orchestrator"] = f"[LLM] → {next_agent}"
            agent_trace.append(f"[LLM] orchestrator → {next_agent}")

    if next_agent is None:
        next_agent = _route(state)
        agent_reasoning["orchestrator"] = f"[fallback] → {next_agent}"
        agent_trace.append(f"[fallback] orchestrator → {next_agent}")

    return {
        "next_agent": next_agent,
        "iteration": iteration,
        "agent_trace": agent_trace,
        "agent_reasoning": agent_reasoning,
    }


def route_after_orchestrator(state: Dict[str, Any]) -> str:
    """Conditional edge function — returns next_agent from state."""
    return state.get("next_agent") or "END"
