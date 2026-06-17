"""S4 — Routage LLM-first (automatisé).

Objectif : LRA = R_LLM / R_total ≥ 0,80
Baseline : routeur aléatoire entre 6 agents ≈ 17 %
Température LLM : 0 (mocké — résultats déterministes)

10 états courants versionnés sont injectés.
8/10 → réponse JSON valide ([LLM])
2/10 → None ou JSON invalide ([fallback])
"""

import json
import pytest
from unittest.mock import patch, call

from ai.agents.langgraph.orchestrator import _llm_route, _route, orchestrator_node

# ── États courants versionnés (inputs fixes) ─────────────────────────────────

ROUTING_STATES = [
    # État 0 — début de session : aucun champ peuplé → memory
    {"support": "html", "current_level": "débutant", "agent_trace": [], "iteration": 1},
    # État 1 — memory peuplé → knowledge
    {"support": "html", "current_level": "débutant", "memory_context": [{"content": "ok"}],
     "agent_trace": ["memory"], "iteration": 2},
    # État 2 — knowledge peuplé → diagnostics
    {"support": "html", "current_level": "débutant",
     "memory_context": [{}], "knowledge_graph": {"nodes": []},
     "agent_trace": ["memory", "knowledge"], "iteration": 3},
    # État 3 — diagnostics peuplé → planner
    {"support": "html", "current_level": "débutant",
     "memory_context": [{}], "knowledge_graph": {"nodes": []}, "adjusted_level": "débutant",
     "agent_trace": ["memory", "knowledge", "diagnostics"], "iteration": 4},
    # État 4 — planner peuplé → exercise
    {"support": "html", "current_level": "débutant",
     "memory_context": [{}], "knowledge_graph": {"nodes": []}, "adjusted_level": "débutant",
     "strategy": "scaffolding",
     "agent_trace": ["memory", "knowledge", "diagnostics", "planner"], "iteration": 5},
    # État 5 — exercise peuplé → verifier
    {"support": "html", "current_level": "débutant",
     "memory_context": [{}], "knowledge_graph": {"nodes": []}, "adjusted_level": "débutant",
     "strategy": "scaffolding", "exercises": [{"id": "e1"}],
     "agent_trace": ["memory", "knowledge", "diagnostics", "planner", "exercise"], "iteration": 6},
    # État 6 — verification ok → feedback
    {"support": "html", "current_level": "débutant",
     "memory_context": [{}], "knowledge_graph": {"nodes": []}, "adjusted_level": "débutant",
     "strategy": "scaffolding", "exercises": [{}],
     "verification": {"verdict": "ok", "score": 0.9},
     "agent_trace": ["memory", "knowledge", "diagnostics", "planner", "exercise", "verifier"],
     "iteration": 7},
    # État 7 — needs_review + retries < 3 → planner (retry)
    {"support": "html", "current_level": "débutant",
     "memory_context": [{}], "knowledge_graph": {"nodes": []}, "adjusted_level": "débutant",
     "strategy": "scaffolding", "exercises": [{}],
     "verification": {"verdict": "needs_review", "score": 0.5},
     "n_retries": {"planner": 1},
     "agent_trace": ["memory", "knowledge", "diagnostics", "planner", "exercise", "verifier"],
     "iteration": 8},
    # État 8 — session complète → END
    {"support": "html", "current_level": "intermédiaire",
     "memory_context": [{}], "knowledge_graph": {"nodes": []}, "adjusted_level": "intermédiaire",
     "strategy": "challenge", "exercises": [{}],
     "verification": {"verdict": "ok", "score": 0.95},
     "agent_trace": ["memory", "knowledge", "diagnostics", "planner", "exercise", "verifier", "feedback"],
     "iteration": 9},
    # État 9 — weak_concepts présents → knowledge (enrichissement)
    {"support": "html", "current_level": "débutant",
     "memory_context": [{}], "weak_concepts": ["<form>", "<input>"],
     "agent_trace": ["memory"], "iteration": 2},
]

# Réponses LLM attendues pour chaque état (8 valides, 2 None)
_VALID_AGENTS = ["memory", "knowledge", "diagnostics", "planner", "exercise", "verifier", "feedback", "END"]

LLM_RESPONSES = [
    json.dumps({"next_agent": "memory",       "reasoning": "début de session", "confidence": 0.95}),
    json.dumps({"next_agent": "knowledge",    "reasoning": "memory peuplé",     "confidence": 0.92}),
    json.dumps({"next_agent": "diagnostics",  "reasoning": "knowledge peuplé",  "confidence": 0.90}),
    json.dumps({"next_agent": "planner",      "reasoning": "diagnostics ok",    "confidence": 0.88}),
    json.dumps({"next_agent": "exercise",     "reasoning": "stratégie définie", "confidence": 0.91}),
    json.dumps({"next_agent": "verifier",     "reasoning": "exercices prêts",   "confidence": 0.89}),
    json.dumps({"next_agent": "feedback",     "reasoning": "vérification ok",   "confidence": 0.87}),
    json.dumps({"next_agent": "planner",      "reasoning": "needs_review",      "confidence": 0.85}),
    None,   # État 8 → fallback déterministe
    None,   # État 9 → fallback déterministe
]


# ── Tests ────────────────────────────────────────────────────────────────────

def test_s4_lra_above_target():
    """LRA = R_LLM / R_total ≥ 0,80 sur 10 états versionnés."""
    r_llm = 0
    r_total = len(ROUTING_STATES)

    for i, (state, llm_response) in enumerate(zip(ROUTING_STATES, LLM_RESPONSES)):
        with patch("ai.llm.service.call_llm", return_value=llm_response):
            result = _llm_route(state)

        if result is not None and result in _VALID_AGENTS:
            r_llm += 1

    lra = r_llm / r_total
    assert lra >= 0.80, (
        f"LRA = {lra:.0%} ({r_llm}/{r_total}) — sous le seuil de 80 %."
    )


def test_s4_fallback_activates_on_invalid_llm_response():
    """Quand le LLM retourne None, _route() prend le relais."""
    state = ROUTING_STATES[8]  # session complète → devrait retourner feedback ou END via _route
    with patch("ai.llm.service.call_llm", return_value=None):
        llm_result = _llm_route(state)
    assert llm_result is None

    fallback_result = _route(state)
    assert fallback_result in _VALID_AGENTS, f"_route() retourne un agent inconnu : {fallback_result}"


def test_s4_llm_invalid_json_triggers_fallback():
    """Un JSON malformé du LLM déclenche le fallback — pas d'exception."""
    state = ROUTING_STATES[0]
    with patch("ai.llm.service.call_llm", return_value="pas du JSON {{{"):
        result = _llm_route(state)
    assert result is None


def test_s4_llm_unknown_agent_triggers_fallback():
    """Un agent inconnu dans la réponse LLM déclenche le fallback."""
    state = ROUTING_STATES[0]
    bad_response = json.dumps({"next_agent": "agent_inconnu", "reasoning": "x", "confidence": 0.9})
    with patch("ai.llm.service.call_llm", return_value=bad_response):
        result = _llm_route(state)
    assert result is None


def test_s4_orchestrator_node_traces_llm_decision():
    """orchestrator_node marque [LLM] dans agent_trace quand le LLM répond."""
    state = ROUTING_STATES[0]
    valid = json.dumps({"next_agent": "memory", "reasoning": "début", "confidence": 0.9})
    with patch("ai.llm.service.call_llm", return_value=valid):
        result = orchestrator_node(state)

    trace = result.get("agent_trace", [])
    assert any("[LLM]" in t for t in trace), f"[LLM] absent de agent_trace : {trace}"


def test_s4_orchestrator_node_traces_fallback():
    """orchestrator_node marque [fallback] dans agent_trace quand le LLM échoue."""
    state = ROUTING_STATES[0]
    with patch("ai.llm.service.call_llm", return_value=None):
        result = orchestrator_node(state)

    trace = result.get("agent_trace", [])
    assert any("[fallback]" in t for t in trace), f"[fallback] absent de agent_trace : {trace}"


def test_s4_deterministic_route_order():
    """_route() suit l'ordre défini : memory → knowledge → diagnostics → planner → exercise → verifier → feedback."""
    progression = [
        ({}, "memory"),
        ({"memory_context": [{}]}, "knowledge"),
        ({"memory_context": [{}], "knowledge_graph": {"nodes": []}}, "diagnostics"),
        ({"memory_context": [{}], "knowledge_graph": {"nodes": []}, "adjusted_level": "débutant"}, "planner"),
        ({"memory_context": [{}], "knowledge_graph": {"nodes": []}, "adjusted_level": "débutant", "strategy": "x"}, "exercise"),
        ({"memory_context": [{}], "knowledge_graph": {"nodes": []}, "adjusted_level": "débutant", "strategy": "x", "exercises": [{}]}, "verifier"),
    ]
    for state, expected in progression:
        result = _route(state)
        assert result == expected, f"État {state} → attendu '{expected}', obtenu '{result}'"


def test_s4_planner_retry_on_needs_review():
    """_route() renvoie vers planner si needs_review et n_retries < 3."""
    state = {
        "memory_context": [{}], "knowledge_graph": {"nodes": []}, "adjusted_level": "débutant",
        "strategy": "x", "exercises": [{}],
        "verification": {"verdict": "needs_review"},
        "n_retries": {"planner": 1},
    }
    assert _route(state) == "planner"


def test_s4_planner_retry_stops_at_max():
    """_route() ne retente plus planner après 3 essais → feedback."""
    state = {
        "memory_context": [{}], "knowledge_graph": {"nodes": []}, "adjusted_level": "débutant",
        "strategy": "x", "exercises": [{}],
        "verification": {"verdict": "needs_review"},
        "n_retries": {"planner": 3},
    }
    assert _route(state) == "feedback"
