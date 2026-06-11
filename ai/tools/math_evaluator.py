"""Tool — evaluate mathematical expressions via sympy with safe eval fallback."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def math_evaluator(expression: str) -> str:
    """Evaluate a mathematical expression using sympy (falls back to safe eval)."""
    try:
        from sympy import latex, simplify, sympify

        expr = sympify(expression)
        simplified = simplify(expr)
        result = f"Result: {simplified}"
        try:
            result += f"\nLaTeX: ${latex(simplified)}$"
        except Exception:
            pass
        return result
    except ImportError:
        pass
    except Exception as exc:
        return f"sympy error: {exc}"

    # Safe eval fallback (no builtins)
    try:
        value = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
        return f"Result: {value}"
    except Exception as exc:
        return f"Evaluation error: {exc}"
