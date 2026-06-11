"""Tool — generate a textual / ASCII chart from a description."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def generate_chart(description: str) -> str:
    """Generate an ASCII chart or textual description from a chart request."""
    desc = description.lower()

    if any(k in desc for k in ("bar", "histogram", "barre")):
        return (
            "Bar chart (ASCII) :\n"
            "A ████████████ 80 %\n"
            "B ████████     55 %\n"
            "C █████        30 %\n"
            "(Valeurs illustratives — fournis des données réelles pour un rendu précis)"
        )

    if any(k in desc for k in ("timeline", "gantt", "chronologie")):
        return (
            "Timeline :\n"
            "──────────────────────────────────────\n"
            "Étape 1  [████████            ]\n"
            "Étape 2  [    ████████        ]\n"
            "Étape 3  [          ████████  ]\n"
        )

    if any(k in desc for k in ("function", "courbe", "graph", "curve")):
        return (
            "Graphe de fonction (conceptuel) :\n"
            "y ^\n"
            "  │   *\n"
            "  │  * *\n"
            "  │ *   *\n"
            "  │*     *\n"
            "  └──────────► x\n"
            f"({description})"
        )

    return (
        f"Chart request noted : « {description} »\n"
        "(Utilise une librairie frontend — Chart.js, Plotly — pour le rendu interactif)"
    )
