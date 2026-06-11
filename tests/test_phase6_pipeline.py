"""Phase 6 — Global Pipeline Integration tests (17 tests).

All in-memory: no LLM, no real DB, no ChromaDB.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from gateway.http.app import create_app
from data.models import User


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _fake_user():
    u = MagicMock(spec=User)
    u.id = "user-test-1"
    u.role = "user"
    return u


def _make_client(fake_user=None):
    app = create_app()
    if fake_user is None:
        fake_user = _fake_user()
    from gateway.http.dependencies import get_current_user
    from data.database import get_db

    app.dependency_overrides[get_current_user] = lambda: fake_user
    app.dependency_overrides[get_db] = lambda: MagicMock()
    return TestClient(app, raise_server_exceptions=False)


# ── POST /adaptive/plan ───────────────────────────────────────────────────────


def _adaptive_payload(**kwargs):
    base = {
        "support": "python",
        "current_level": "beginner",
        "language": "fr",
        "user_message": "Explique les listes",
        "learning_objectives": ["listes", "boucles"],
    }
    base.update(kwargs)
    return base


def _fake_final_state():
    return {
        "user_id": "user-test-1",
        "support": "python",
        "adjusted_level": "beginner",
        "exercises": [{"id": 1, "question": "Q1", "skill_target": "listes"}],
        "strategy_decisions": [{"action": "review lists"}],
        "strategy": "review lists",
        "verification": {"verdict": "supported", "support_score": 0.8},
        "agent_trace": ["memory → ok", "exercise → 1 exercise [fallback]"],
    }


def test_adaptive_plan_success():
    client = _make_client()
    with (
        patch("gateway.http.routers.adaptive.ContextManager") as MockCtx,
        patch("gateway.http.routers.adaptive.tutor_graph") as mock_graph,
    ):
        MockCtx.return_value.build_agent_context.return_value = MagicMock(
            rag_docs=[], session_summary=""
        )
        mock_graph.invoke.return_value = _fake_final_state()

        resp = client.post("/api/v1/adaptive/plan", json=_adaptive_payload())

    assert resp.status_code == 200
    data = resp.json()
    assert data["support"] == "python"
    assert data["adjusted_level"] == "beginner"
    assert len(data["exercises"]) == 1
    assert "session_id" in data


def test_adaptive_plan_generates_session_id():
    client = _make_client()
    with (
        patch("gateway.http.routers.adaptive.ContextManager"),
        patch("gateway.http.routers.adaptive.tutor_graph") as mock_graph,
    ):
        mock_graph.invoke.return_value = _fake_final_state()
        resp = client.post("/api/v1/adaptive/plan", json=_adaptive_payload())

    assert resp.status_code == 200
    assert resp.json()["session_id"] != ""


def test_adaptive_plan_uses_provided_session_id():
    client = _make_client()
    with (
        patch("gateway.http.routers.adaptive.ContextManager"),
        patch("gateway.http.routers.adaptive.tutor_graph") as mock_graph,
    ):
        mock_graph.invoke.return_value = _fake_final_state()
        resp = client.post(
            "/api/v1/adaptive/plan",
            json=_adaptive_payload(session_id="my-session-42"),
        )

    assert resp.status_code == 200
    assert resp.json()["session_id"] == "my-session-42"


def test_adaptive_plan_fallback_when_graph_none():
    client = _make_client()
    with (
        patch("gateway.http.routers.adaptive.ContextManager"),
        patch("gateway.http.routers.adaptive.tutor_graph", None),
        patch("gateway.http.routers.adaptive._fallback_response") as mock_fb,
    ):
        mock_fb.return_value = {
            **_fake_final_state(),
            "agent_trace": ["[FALLBACK] LangGraph unavailable"],
        }
        resp = client.post("/api/v1/adaptive/plan", json=_adaptive_payload())

    assert resp.status_code == 200
    assert any("FALLBACK" in t for t in resp.json()["agent_trace"])


def test_adaptive_plan_context_failure_continues():
    client = _make_client()
    with (
        patch(
            "gateway.http.routers.adaptive.ContextManager",
            side_effect=Exception("chroma down"),
        ),
        patch("gateway.http.routers.adaptive.tutor_graph") as mock_graph,
    ):
        mock_graph.invoke.return_value = _fake_final_state()
        resp = client.post("/api/v1/adaptive/plan", json=_adaptive_payload())

    assert resp.status_code == 200


def test_adaptive_plan_graph_error_returns_500():
    client = _make_client()
    with (
        patch("gateway.http.routers.adaptive.ContextManager"),
        patch("gateway.http.routers.adaptive.tutor_graph") as mock_graph,
    ):
        mock_graph.invoke.side_effect = RuntimeError("graph exploded")
        resp = client.post("/api/v1/adaptive/plan", json=_adaptive_payload())

    assert resp.status_code == 500
    assert "Adaptive pipeline error" in resp.json()["detail"]


def test_adaptive_plan_missing_support_returns_422():
    client = _make_client()
    resp = client.post("/api/v1/adaptive/plan", json={"current_level": "beginner"})
    assert resp.status_code == 422


# ── GET /adaptive/session/{id} ────────────────────────────────────────────────


def _fake_checkpoint(user_id="user-test-1"):
    chk = MagicMock()
    chk.values = {
        "user_id": user_id,
        "support": "python",
        "adjusted_level": "beginner",
        "exercises": [{"id": 1}],
        "agent_trace": ["memory → ok"],
        "verification": {"verdict": "supported"},
    }
    return chk


def test_get_session_returns_state():
    client = _make_client()
    with patch("gateway.http.routers.adaptive.tutor_graph") as mock_graph:
        mock_graph.get_state.return_value = _fake_checkpoint()
        resp = client.get("/api/v1/adaptive/session/my-session-42")

    assert resp.status_code == 200
    assert resp.json()["support"] == "python"


def test_get_session_unknown_returns_404():
    client = _make_client()
    with patch("gateway.http.routers.adaptive.tutor_graph") as mock_graph:
        mock_graph.get_state.return_value = MagicMock(values=None)
        resp = client.get("/api/v1/adaptive/session/unknown-id")

    assert resp.status_code == 404


def test_get_session_wrong_user_returns_404():
    client = _make_client()
    with patch("gateway.http.routers.adaptive.tutor_graph") as mock_graph:
        mock_graph.get_state.return_value = _fake_checkpoint(user_id="other-user")
        resp = client.get("/api/v1/adaptive/session/stolen-id")

    assert resp.status_code == 404


# ── GET /knowledge-graph/{support} ───────────────────────────────────────────


def _mock_kg_svc(nodes=None, edges=None, weak=None):
    svc = MagicMock()
    G = MagicMock()
    G.nodes.return_value = [
        (n, {"difficulty": 0.5, "mastery": 0.1}) for n in (nodes or [])
    ]
    G.nodes.data = True
    G.edges.return_value = [
        (e[0], e[1], {"relation": "requires"}) for e in (edges or [])
    ]
    G.edges.data = True
    svc.build_graph.return_value = G
    svc.get_weak_concepts.return_value = weak or []
    return svc


def test_get_kg_returns_structure():
    client = _make_client()
    with patch("gateway.http.routers.knowledge_graph.KnowledgeGraphService") as MockSvc:
        svc = MagicMock()
        G = MagicMock()
        G.nodes.return_value = MagicMock()
        list_nodes = [("recursion", {"difficulty": 0.5, "mastery": 0.2})]
        G.nodes.return_value.__iter__ = lambda s: iter(list_nodes)
        list_edges = [("recursion", "sorting", {"relation": "requires"})]
        G.edges.return_value.__iter__ = lambda s: iter(list_edges)
        svc.build_graph.return_value = G
        svc.get_weak_concepts.return_value = ["recursion"]
        MockSvc.return_value = svc

        resp = client.get("/api/v1/knowledge-graph/python")

    assert resp.status_code == 200
    data = resp.json()
    assert data["support"] == "python"
    assert isinstance(data["nodes"], list)
    assert isinstance(data["edges"], list)
    assert "weak_concepts" in data


# ── POST /knowledge-graph/{support}/concepts ─────────────────────────────────


def test_add_concept_returns_201():
    client = _make_client()
    with patch("gateway.http.routers.knowledge_graph.KnowledgeGraphService") as MockSvc:
        concept = MagicMock()
        concept.id = "c1"
        concept.name = "recursion"
        concept.difficulty = 0.5
        MockSvc.return_value.upsert_concept.return_value = concept

        resp = client.post(
            "/api/v1/knowledge-graph/python/concepts",
            json={"name": "recursion", "difficulty": "intermediate"},
        )

    assert resp.status_code == 201
    assert resp.json()["name"] == "recursion"


def test_add_concept_with_relation():
    client = _make_client()
    with patch("gateway.http.routers.knowledge_graph.KnowledgeGraphService") as MockSvc:
        concept = MagicMock()
        concept.id = "c1"
        concept.name = "sorting"
        concept.difficulty = 0.5
        MockSvc.return_value.upsert_concept.return_value = concept

        resp = client.post(
            "/api/v1/knowledge-graph/python/concepts",
            json={
                "name": "sorting",
                "relation_to": "recursion",
                "relation_type": "requires",
            },
        )

    assert resp.status_code == 201
    MockSvc.return_value.add_relation.assert_called_once()


# ── PUT /knowledge-graph/{support}/mastery ────────────────────────────────────


def test_update_mastery_returns_new_score():
    client = _make_client()
    with patch("gateway.http.routers.knowledge_graph.KnowledgeGraphService") as MockSvc:
        row = MagicMock()
        row.mastery = 0.45
        row.attempts = 4
        MockSvc.return_value.update_mastery.return_value = row

        resp = client.put(
            "/api/v1/knowledge-graph/python/mastery",
            json={"concept_name": "recursion", "delta": 0.1},
        )

    assert resp.status_code == 200
    assert resp.json()["mastery"] == 0.45
    assert resp.json()["attempts"] == 4


# ── DELETE /knowledge-graph/{support}/mastery ─────────────────────────────────


def test_reset_mastery_returns_204():
    client = _make_client()
    with patch("gateway.http.routers.knowledge_graph.KnowledgeGraphService"):
        resp = client.delete("/api/v1/knowledge-graph/python/mastery")

    assert resp.status_code == 204
