"""KnowledgeAgent — KG read + prerequisite inference via LLM (once per support)."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Tuple

log = logging.getLogger(__name__)


def _infer_prerequisites_from_rag(
    rag_docs: list, support: str, model: str = None
) -> List[Tuple[str, str]]:
    """Ask the LLM to extract prerequisite pairs from RAG docs or general knowledge.

    Returns a list of (source, target) where source --requires--> target,
    i.e. 'to learn source you must first master target'.
    Falls back to LLM general knowledge when RAG corpus is empty.
    """
    from ai.llm.service import call_llm

    corpus = "\n".join(d.get("content", "") for d in rag_docs[:6])[:2500]

    if corpus.strip():
        prompt = (
            f"Tu es un expert pédagogique. Analyse ce contenu de cours sur '{support}'.\n\n"
            f"Contenu :\n{corpus}\n\n"
            f"Identifie les relations de prérequis entre les concepts du cours. "
            f"Une relation [A, B] signifie 'pour apprendre A il faut d'abord maîtriser B'.\n"
            f'Exemple : [["division", "multiplication"], ["fractions", "division"]]\n\n'
            f"Réponds UNIQUEMENT avec un tableau JSON de paires, sans explication."
        )
    else:
        prompt = (
            f"Tu es un expert pédagogique. "
            f"Liste les relations de prérequis pédagogiques standards pour un cours sur '{support}'.\n"
            f"Une relation [A, B] signifie 'pour apprendre A il faut d'abord maîtriser B'.\n"
            f'Exemple : [["division", "multiplication"], ["fractions", "division"]]\n\n'
            f"Réponds UNIQUEMENT avec un tableau JSON de paires, sans explication."
        )

    result = call_llm(prompt, model=model, max_tokens=400)
    if not result:
        return []

    match = re.search(r"\[.*\]", result, re.DOTALL)
    if not match:
        return []

    try:
        pairs = json.loads(match.group(0))
        return [
            (str(p[0]).strip(), str(p[1]).strip())
            for p in pairs
            if isinstance(p, (list, tuple))
            and len(p) == 2
            and p[0]
            and p[1]
            and p[0] != p[1]
        ]
    except Exception:
        return []


def knowledge_node(state: Dict[str, Any]) -> Dict[str, Any]:
    user_id = state.get("user_id", "")
    support = state.get("support", "")
    rag_docs = list(state.get("rag_docs") or [])
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
    prerequisite_gaps: list = []
    concept_levels: Dict[str, str] = {}

    try:
        from data.database import get_db
        from data.models.memory import KGConcept, KGUserMastery
        from ai.knowledge_graph.service import KnowledgeGraphService

        db = next(get_db())
        try:
            kg_svc = KnowledgeGraphService()

            # ── Step 1: load weak and blocked concepts ────────────────────────
            weak = kg_svc.get_weak_concepts(user_id, support, db)
            weak_concepts = [c.name if hasattr(c, "name") else str(c) for c in weak]
            blocked_concepts = kg_svc.get_blocked_concepts(user_id, support, db)

            # ── Step 2: infer prerequisite relations if none exist yet ────────
            inferred_count = 0
            if not kg_svc.has_requires_relations(support, db) and support:
                pairs = _infer_prerequisites_from_rag(
                    rag_docs, support, model=state.get("model")
                )
                for source, target in pairs:
                    try:
                        kg_svc.add_relation(db, source, target, "requires", support)
                        inferred_count += 1
                    except Exception as exc:
                        log.debug("Skipping relation %s→%s: %s", source, target, exc)
                if inferred_count:
                    log.info(
                        "KnowledgeAgent: inferred %d prerequisite relations for '%s'",
                        inferred_count,
                        support,
                    )

            # ── Step 3: find unmastered prerequisites of blocked/weak concepts ─
            seen: set = set()
            for bc in blocked_concepts:
                for gap in kg_svc.get_unmastered_prerequisites(
                    user_id, bc["concept"], support, db
                ):
                    if gap not in seen:
                        prerequisite_gaps.append(gap)
                        seen.add(gap)
            # Also check top weak concepts (max 3)
            for wc in weak_concepts[:3]:
                for gap in kg_svc.get_unmastered_prerequisites(
                    user_id, wc, support, db
                ):
                    if gap not in seen:
                        prerequisite_gaps.append(gap)
                        seen.add(gap)

            # ── Step 3b: build concept_levels from mastery scores ────────────
            all_concepts = (
                db.query(KGConcept).filter(KGConcept.support == support).all()
            )
            for c in all_concepts:
                row = (
                    db.query(KGUserMastery)
                    .filter(
                        KGUserMastery.user_id == user_id,
                        KGUserMastery.concept_id == c.id,
                    )
                    .first()
                )
                score = row.mastery if row else 0.0
                if score >= 0.7:
                    concept_levels[c.name] = "advanced"
                elif score >= 0.3:
                    concept_levels[c.name] = "intermediate"
                else:
                    concept_levels[c.name] = "beginner"

            # ── Step 4: build full graph for state ───────────────────────────
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
        f"knowledge → {len(weak_concepts)} weak, {len(blocked_concepts)} blocked"
        + (f", {len(prerequisite_gaps)} prereq gaps" if prerequisite_gaps else "")
    )
    agent_reasoning["knowledge"] = (
        f"[deterministic] {len(weak_concepts)} weak, {len(blocked_concepts)} blocked"
        + (f" | prereq gaps: {prerequisite_gaps[:3]}" if prerequisite_gaps else "")
    )

    return {
        "knowledge_graph": knowledge_graph,
        "weak_concepts": weak_concepts,
        "blocked_concepts": blocked_concepts,
        "prerequisite_gaps": prerequisite_gaps,
        "concept_levels": concept_levels,
        "agent_trace": agent_trace,
        "agent_reasoning": agent_reasoning,
    }
