"""Tool — execute Python code in an isolated subprocess (5 s timeout)."""

from __future__ import annotations

import subprocess
import sys

from langchain_core.tools import tool


@tool
def live_code_evaluation(code: str) -> str:
    """Execute a Python code snippet and return its stdout/stderr output (timeout 5 s).

    Use this to validate a Python exercise answer or to demonstrate a concept live.
    Pass only the code string — no markdown fences, no surrounding text.

    Example:
      live_code_evaluation("print(sum(range(1, 6)))")  → "15"
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = (result.stdout + result.stderr).strip()
        return output[:500] if output else "Code executed with no output."
    except subprocess.TimeoutExpired:
        return "Timeout: code execution exceeded 5 seconds."
    except Exception as exc:
        return f"Execution error: {exc}"
