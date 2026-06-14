"""Tool — evaluate mathematical expressions via sympy with safe eval fallback."""

from __future__ import annotations

from langchain_core.tools import tool


@tool
def math_evaluator(expression: str) -> str:
    """Evaluate a mathematical expression and return the numeric result with LaTeX if possible.

    Use this to verify arithmetic, algebra, or to compute the correct answer for a math exercise.
    Pass only the expression as a string — no equals sign, no surrounding text.

    Examples:
      math_evaluator("47 + 38")         → "Result: 85"
      math_evaluator("x**2 + 2*x + 1") → "Result: (x+1)**2  LaTeX: $(x+1)^{2}$"
    """
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
