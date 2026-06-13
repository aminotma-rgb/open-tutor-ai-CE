"""KnowledgeAgent — deterministic KG read, no LLM."""

from __future__ import annotations

import logging
from typing import Any, Dict

log = logging.getLogger(__name__)


def knowledge_node(state: Dict[str, Any]) -> Dict[str, Any]:
    user_id = state.get("user_id", "")
    support = state.get("support", "")
    agent_trace = list(state.get("agent_trace") or [])
    agent_reasoning = dict(state.get("agent_reasoning") or {})

    knowledge_graph: Dict[str, Any] = {
        "nodes": [],
        "edges": [],
        "weak_concepts": [],
        "node_count": 0,
        "edge_count": 0,
    }
    weak_concepts: list = []
    blocked_concepts: list = []

    try:
        from data.database import get_db
        from ai.knowledge_graph.service import KnowledgeGraphService

        db = next(get_db())
        try:
            kg_svc = KnowledgeGraphService()
            weak = kg_svc.get_weak_concepts(user_id, support, db)
            weak_concepts = [c.name if hasattr(c, "name") else str(c) for c in weak]

            # Blocked = weak AND stuck (attempts >= 5) — require a strategy change
            blocked_concepts = kg_svc.get_blocked_concepts(user_id, support, db)

            try:
                g = kg_svc.build_graph(user_id, support, db)
                nodes = [{"id": n, "data": d} for n, d in g.nodes(data=True)]
                edges = [
                    {"source": u, "target": v, "data": d}
                    for u, v, d in g.edges(data=True)
                ]
                knowledge_graph = {
                    "nodes": nodes,
                    "edges": edges,
                    "weak_concepts": weak_concepts,
                    "node_count": g.number_of_nodes(),
                    "edge_count": g.number_of_edges(),
                }
            except Exception:
                knowledge_graph["weak_concepts"] = weak_concepts
        finally:
            db.close()
    except Exception as exc:
        log.warning("KnowledgeAgent failed: %s", exc)

    agent_trace.append(
        f"knowledge → {len(weak_concepts)} weak, {len(blocked_concepts)} blocked concepts"
    )
    agent_reasoning["knowledge"] = (
        f"[deterministic] {len(weak_concepts)} weak, {len(blocked_concepts)} blocked"
    )

    return {
        "knowledge_graph": knowledge_graph,
        "weak_concepts": weak_concepts,
        "blocked_concepts": blocked_concepts,
        "agent_trace": agent_trace,
        "agent_reasoning": agent_reasoning,
    }
