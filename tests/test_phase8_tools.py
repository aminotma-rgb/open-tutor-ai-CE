"""Phase 8 — OTAI Tools integration tests (14 tests, all in-memory)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from gateway.http.app import create_app
from data.models import User
from ai.agents.helpers import _detect_exercise_type, generate_exercises
from ai.tools.owui_tools import OTAI_TOOLS


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _fake_user():
    u = MagicMock(spec=User)
    u.id = "user-test-1"
    u.role = "admin"
    return u


def _make_client():
    app = create_app()
    from gateway.http.dependencies import get_current_user
    from data.database import get_db

    app.dependency_overrides[get_current_user] = lambda: _fake_user()
    app.dependency_overrides[get_db] = lambda: MagicMock()
    return TestClient(app, raise_server_exceptions=False)


# ── owui_tools constants ──────────────────────────────────────────────────────


def test_otai_tools_has_4_entries():
    assert len(OTAI_TOOLS) == 4


def test_otai_tools_ids():
    ids = {t["id"] for t in OTAI_TOOLS}
    assert ids == {
        "otai_generate_chart",
        "otai_math_evaluator",
        "otai_live_code",
        "otai_search_web",
    }


def test_otai_tools_all_have_content():
    for t in OTAI_TOOLS:
        assert "class Tools" in t["content"]


# ── _detect_exercise_type ─────────────────────────────────────────────────────


def test_detect_chart_parabola():
    assert _detect_exercise_type("math", "tracer une parabole") == "chart"


def test_detect_chart_plot():
    assert _detect_exercise_type("math", "plot y = x**2") == "chart"


def test_detect_sql():
    assert _detect_exercise_type("sql", "write a SELECT query with JOIN") == "sql"


def test_detect_math():
    assert _detect_exercise_type("algebra", "résoudre une équation") == "math"


def test_detect_coding():
    assert _detect_exercise_type("python", "write a function") == "coding"


def test_detect_fallback_explain():
    assert _detect_exercise_type("history", "the french revolution") == "explain"


# ── generate_exercises type detection ────────────────────────────────────────


def test_generate_exercises_chart_type():
    exs = generate_exercises("math", "beginner", ["tracer une parabole"], count=1)
    assert exs[0]["type"] == "chart"


def test_generate_exercises_sql_type():
    exs = generate_exercises("sql", "beginner", ["SELECT with JOIN"], count=1)
    assert exs[0]["type"] == "sql"


# ── /tools/ router ────────────────────────────────────────────────────────────


def test_list_tools_returns_list():
    client = _make_client()
    with patch("gateway.http.routers.tools.ToolsService") as MockSvc:
        tool = MagicMock()
        tool.to_dict.return_value = {"id": "otai_math_evaluator", "name": "Math"}
        MockSvc.return_value.list_all.return_value = [tool]

        resp = client.get("/api/v1/tools/")

    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_create_tool_returns_201():
    client = _make_client()
    with patch("gateway.http.routers.tools.ToolsService") as MockSvc:
        tool = MagicMock()
        tool.to_dict.return_value = {"id": "my-tool", "name": "My Tool"}
        MockSvc.return_value.upsert.return_value = tool

        resp = client.post(
            "/api/v1/tools/create",
            json={"name": "My Tool", "content": "class Tools: pass"},
        )

    assert resp.status_code == 201


def test_get_tool_not_found_returns_404():
    client = _make_client()
    with patch("gateway.http.routers.tools.ToolsService") as MockSvc:
        MockSvc.return_value.get_by_id.return_value = None
        resp = client.get("/api/v1/tools/id/unknown-tool")

    assert resp.status_code == 404
