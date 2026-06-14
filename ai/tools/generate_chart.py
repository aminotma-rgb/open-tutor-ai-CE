"""Tool — generate a chart image (matplotlib) from a description or expression."""

from __future__ import annotations

import base64
import io
import re

from langchain_core.tools import tool


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def _parse_expression(expr: str):
    """Try to parse a math expression like 'y=2x', 'y=x^2', 'y=sin(x)'."""
    import numpy as np

    expr = expr.strip().lower()
    # Remove 'y=' prefix
    expr = re.sub(r"^y\s*=\s*", "", expr)
    # Replace ^ with ** for exponentiation
    expr = expr.replace("^", "**")
    # Insert * between coefficient and x (e.g. 2x → 2*x)
    expr = re.sub(r"(\d)(x)", r"\1*\2", expr)

    x = np.linspace(-10, 10, 400)
    safe_env = {
        "x": x, "np": np,
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "sqrt": np.sqrt, "abs": np.abs, "exp": np.exp, "log": np.log,
    }
    y = eval(expr, {"__builtins__": {}}, safe_env)
    return x, y, expr


@tool
def generate_chart(description: str) -> str:
    """Generate a chart image from a description or mathematical expression.

    Pass a descriptive sentence or a math expression like 'y=2x', 'y=x^2',
    'curve of sin(x)', 'bar chart of scores'.

    Returns a markdown image tag with base64-encoded PNG, ready to display in chat.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return "(matplotlib non disponible — installe-le avec `pip install matplotlib`)"

    desc = description.lower()

    # ── Detect math function / curve ─────────────────────────────────────────
    _curve_keywords = ("y=", "sin", "cos", "tan", "x²", "x^2", "2x", "3x",
                       "droite", "parabole", "courbe", "curve", "function",
                       "graphe", "graph", "sqrt", "exp", "log")
    if any(k in desc for k in _curve_keywords):
        # Extract the expression: prefer "y=..." form, then "of/de X", else full description
        y_match = re.search(r"y\s*=\s*([^\s,;]+(?:\s*[\+\-\*\/\^]\s*[^\s,;]+)*)", desc)
        of_match = re.search(r"(?:of|de|du|for)\s+([\w\(\)\+\-\*\/\^\s\.]+)", desc)
        if y_match:
            expr_raw = y_match.group(0)
        elif of_match:
            expr_raw = "y=" + of_match.group(1).strip()
        else:
            expr_raw = description.strip()

        try:
            x, y_vals, label = _parse_expression(expr_raw)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x, y_vals, color="#4F46E5", linewidth=2)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.set_title(f"y = {label}", fontsize=12)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            img_b64 = _fig_to_base64(fig)
            plt.close(fig)
            return f"![chart](data:image/png;base64,{img_b64})"
        except Exception:
            pass  # fall through to generic

    # ── Bar chart ─────────────────────────────────────────────────────────────
    if any(k in desc for k in ("bar", "barre", "histogram", "score")):
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["A", "B", "C", "D"]
        values = [80, 55, 70, 40]
        bars = ax.bar(labels, values, color=["#4F46E5", "#7C3AED", "#A78BFA", "#C4B5FD"])
        ax.set_title(description[:60], fontsize=10)
        ax.set_ylabel("Valeur")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 1, str(val),
                    ha="center", va="bottom", fontsize=9)
        img_b64 = _fig_to_base64(fig)
        plt.close(fig)
        return f"![chart](data:image/png;base64,{img_b64})"

    # ── Timeline ──────────────────────────────────────────────────────────────
    if any(k in desc for k in ("timeline", "chronologie", "gantt")):
        fig, ax = plt.subplots(figsize=(7, 3))
        steps = ["Étape 1", "Étape 2", "Étape 3"]
        starts = [0, 3, 6]
        durations = [3, 3, 3]
        colors = ["#4F46E5", "#7C3AED", "#A78BFA"]
        for i, (s, d, c) in enumerate(zip(starts, durations, colors)):
            ax.barh(i, d, left=s, color=c, height=0.5)
            ax.text(s + d / 2, i, steps[i], ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
        ax.set_yticks([])
        ax.set_xlabel("Temps")
        ax.set_title(description[:60], fontsize=10)
        img_b64 = _fig_to_base64(fig)
        plt.close(fig)
        return f"![chart](data:image/png;base64,{img_b64})"

    # ── Generic fallback ──────────────────────────────────────────────────────
    return (
        f"Graphique demandé : « {description} »\n"
        "Précise une expression mathématique (ex: y=2x) ou un type de graphique "
        "(bar chart, courbe, timeline) pour générer une image."
    )
