"""Pure computation helpers shared by all agent nodes.

No I/O, no LLM calls — every function is deterministic and testable in isolation.
_self_critique() is the only exception: it calls call_llm() but degrades gracefully.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

_UPGRADE_SIGNALS = {
    "excellent",
    "perfect",
    "advanced",
    "brillant",
    "parfait",
    "very good",
    "expert",
}
_DOWNGRADE_SIGNALS = {
    "confused",
    "lost",
    "don't understand",
    "help",
    "error",
    "wrong",
    "difficile",
    "perdu",
    "ne comprends pas",
    "je comprends pas",
    "confusion",
    "débutant",
    "débutante",
    "novice",
    "je suis nul",
    "je suis nulle",
    "je débute",
    "je commence",
    "jamais fait",
    "première fois",
    "first time",
    "beginner",
    "i'm a beginner",
    "i am a beginner",
}
_LEVELS = ("beginner", "intermediate", "advanced")


# ── Level assessment ──────────────────────────────────────────────────────────


def assess_current_level(level: str, interactions: str, feedback: str) -> str:
    """Adjust level up/down based on keyword signals in interactions and feedback."""
    text = (interactions + " " + feedback).lower()
    up = sum(1 for s in _UPGRADE_SIGNALS if s in text)
    down = sum(1 for s in _DOWNGRADE_SIGNALS if s in text)
    idx = _LEVELS.index(level) if level in _LEVELS else 0
    if up > down and idx < len(_LEVELS) - 1:
        return _LEVELS[idx + 1]
    if down > up and idx > 0:
        return _LEVELS[idx - 1]
    return level


# ── Difficulty detection ──────────────────────────────────────────────────────


def detect_difficulties(
    support: str,
    interactions: str,
    feedback: str,
    objectives: List[str],
    session_summary: str = "",
) -> List[str]:
    """Return up to 5 difficulty signals from the last message and past session summaries."""
    import re

    difficulties: List[str] = []
    text = (interactions + " " + feedback).lower()

    # Generic signals in last user message
    if "error" in text or "erreur" in text:
        difficulties.append(f"Erreurs rencontrées en {support}")
    if any(
        kw in text
        for kw in (
            "don't understand",
            "ne comprends pas",
            "je comprends pas",
            "je ne comprends pas",
        )
    ):
        difficulties.append(f"Difficulté de compréhension en {support}")
    if "why" in text or "pourquoi" in text:
        difficulties.append(f"Lacunes conceptuelles en {support}")
    if not interactions.strip():
        difficulties.append(f"Aucune interaction récente en {support}")

    # Concept-specific difficulties extracted from session summary
    # Handles: "difficultés avec la X", "du mal avec X et Y", "struggled with X and Y"
    if session_summary:
        summary_lower = session_summary.lower()
        # Match the full phrase after difficulty keywords, then split on "et"/"and"
        _diff_patterns = [
            r"difficultés?\s+avec\s+(?:la?\s+)?(.+?)(?:\s*[.\n]|$)",
            r"difficultés?\s+(?:dans|en|sur)\s+(?:la?\s+)?(.+?)(?:\s*[.\n]|$)",
            r"du\s+mal\s+avec\s+(?:la?\s+)?(.+?)(?:\s*[.\n]|$)",
            r"struggled?\s+with\s+(.+?)(?:\s*[.,\n]|$)",
            r"a\s+(?:eu|rencontré)\s+des\s+difficultés?\s+(?:avec\s+)?(?:la?\s+)?(.+?)(?:\s*[.\n]|$)",
        ]
        for pattern in _diff_patterns:
            for m in re.finditer(pattern, summary_lower):
                raw = m.group(1).strip()
                # Split on "et" / "and" / "," to get individual concepts
                parts = re.split(r"\s+et\s+|\s+and\s+|,\s*", raw)
                for part in parts:
                    concept = part.strip().strip(".,").strip()
                    # Remove article prefixes (la, le, les, l')
                    concept = re.sub(r"^l[ae]s?\s+|^l'", "", concept).strip()
                    if concept and 2 < len(concept) < 50:
                        hint = f"Difficulté identifiée : {concept}"
                        if hint not in difficulties:
                            difficulties.append(hint)

    for obj in objectives[:3]:
        if obj.lower() not in text:
            difficulties.append(f"Objectif non encore abordé : {obj}")

    return list(dict.fromkeys(difficulties))[:5]


def extract_memory_signals(support: str, memories: List[Dict[str, Any]]) -> List[str]:
    """Extract negative learning signals from past memories."""
    signals: List[str] = []
    negative_kws = (
        "struggled",
        "failed",
        "difficulty",
        "difficile",
        "erreur",
        "confusion",
        "wrong",
    )
    for mem in memories:
        content = mem.get("content", "").lower()
        if any(kw in content for kw in negative_kws):
            signals.append(f"Past difficulty: {mem.get('content', '')[:80]}")
    return signals[:3]


# ── Exercise generation ───────────────────────────────────────────────────────


def generate_exercises(
    support: str,
    level: str,
    objectives: List[str],
    count: int = 3,
) -> List[Dict[str, Any]]:
    """Return count structured exercises, cycling through objectives."""
    if not objectives:
        objectives = [f"{support} fundamentals"]
    difficulty_map = {"beginner": "easy", "intermediate": "medium", "advanced": "hard"}
    difficulty = difficulty_map.get(level, "medium")

    exercises: List[Dict[str, Any]] = []
    for i in range(count):
        obj = objectives[i % len(objectives)]
        ex_type = _detect_exercise_type(support, obj)
        exercises.append(
            {
                "id": i + 1,
                "type": ex_type,
                "difficulty": difficulty,
                "question": f"Explain the concept of '{obj}' in {support}.",
                "hint": f"Think about the definition and a practical example of {obj}.",
                "answer": f"A clear explanation of {obj} in the context of {support}.",
                "skill_target": obj,
            }
        )
    return exercises


_CHART_KEYWORDS = (
    "parabola",
    "courbe",
    "curve",
    "graphe",
    "tracer",
    "trace",
    "plot",
    "visualis",
    "timeline",
    "gantt",
)
_SQL_KEYWORDS = (
    "sql",
    "select",
    "query",
    "join",
    "database",
    "base de données",
    "table",
    "requête",
    "jointure",
)
_MATH_KEYWORDS = (
    "équation",
    "equation",
    "calcul",
    "dérivée",
    "derivative",
    "intégrale",
    "integral",
    "algèbre",
    "algebra",
    "math",
    "expression",
)
_CODE_KEYWORDS = (
    "python",
    "code",
    "program",
    "fonction",
    "boucle",
    "loop",
    "classe",
    "class",
    "algorithm",
    "algorithme",
)


def _detect_exercise_type(support: str, objective: str) -> str:
    text = (support + " " + objective).lower()
    if any(kw in text for kw in _CHART_KEYWORDS):
        return "chart"
    if any(kw in text for kw in _SQL_KEYWORDS):
        return "sql"
    if any(kw in text for kw in _MATH_KEYWORDS):
        return "math"
    if any(kw in text for kw in _CODE_KEYWORDS):
        return "coding"
    return "explain"


# ── Strategy planning ─────────────────────────────────────────────────────────


def plan_learning_strategy(
    support: str,
    level: str,
    difficulties: List[str],
    feedback: str,
    memory: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build a prioritised list of learning decisions."""
    decisions: List[Dict[str, Any]] = []
    priority = 1

    for diff in difficulties[:3]:
        decisions.append(
            {
                "id": priority,
                "action": f"Address: {diff}",
                "rationale": f"Identified difficulty in {support}",
                "priority": priority,
            }
        )
        priority += 1

    if level == "beginner":
        decisions.append(
            {
                "id": priority,
                "action": f"Review fundamentals of {support}",
                "rationale": "Beginner level requires solid foundations",
                "priority": priority,
            }
        )
    elif level == "advanced":
        decisions.append(
            {
                "id": priority,
                "action": f"Explore advanced patterns in {support}",
                "rationale": "Advanced learner ready for complex concepts",
                "priority": priority,
            }
        )

    return decisions[:5]


# ── RAG verification ──────────────────────────────────────────────────────────


def is_text_supported(text: str, corpus: str, threshold: float = 0.65) -> bool:
    """Term-overlap check — True if fraction of text terms found in corpus >= threshold."""
    if not corpus:
        return False
    text_terms = set(text.lower().split())
    corpus_terms = set(corpus.lower().split())
    if not text_terms:
        return False
    overlap = len(text_terms & corpus_terms) / len(text_terms)
    return overlap >= threshold


# ── Self-critique (shared by all agent nodes) ─────────────────────────────────


def _self_critique(
    agent_name: str,
    output_summary: str,
    state: Dict[str, Any],
) -> Optional[str]:
    """Internal mini-LLM self-critique. Returns critique text or None on failure."""
    from ai.llm.service import call_llm

    support = state.get("support", "")
    level = state.get("adjusted_level", state.get("current_level", "beginner"))
    prompt = (
        f"Agent {agent_name} a produit : {output_summary}\n"
        f"Support : {support}, Niveau : {level}\n"
        f"Cette sortie est-elle cohérente avec l'objectif pédagogique et le niveau ? "
        f"Réponds en une phrase : OK ou correction nécessaire."
    )
    critique = call_llm(prompt, model=state.get("model"), max_tokens=80)
    if critique:
        log.debug("Self-critique [%s]: %s", agent_name, critique[:100])
    return critique
