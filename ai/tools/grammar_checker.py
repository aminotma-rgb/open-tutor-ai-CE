"""Tool — basic grammar checker (rule-based, no external dependency)."""

from __future__ import annotations

import re

from langchain_core.tools import tool

_RULES = [
    (r"\bi\b", "Utilise 'I' majuscule pour le pronom de première personne."),
    (r"([.!?])\s+[a-z]", "Une phrase doit commencer par une majuscule."),
    (r"\s{2,}", "Espaces consécutifs détectés."),
    (r",([^\s])", "Espace manquant après une virgule."),
    (
        r"\.{2}(?!\.)",
        "Double point suspect — utilise '...' (ellipse) ou un seul point.",
    ),
]


@tool
def grammar_checker(text: str) -> str:
    """Check basic grammar and style rules in a text and return actionable feedback.

    Use this for language/writing exercises to validate or evaluate a learner's written answer.
    Pass the raw text to analyse — no instructions, no surrounding context.

    Example:
      grammar_checker("il mange une pomme et elle boit du lait.")
      → "Une phrase doit commencer par une majuscule."
    """
    issues = []
    for pattern, message in _RULES:
        if re.search(pattern, text):
            issues.append(message)

    if not issues:
        return "Aucun problème grammatical de base détecté."
    return "Retours grammaticaux :\n" + "\n".join(f"- {i}" for i in issues)
