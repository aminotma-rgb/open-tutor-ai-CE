"""Knowledge Graph router — /api/v1/knowledge-graph/*."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from data.database import get_db
from data.models import User
from data.models.memory import KGConcept, KGUserMastery
from gateway.http.dependencies import get_current_user

log = logging.getLogger(__name__)

try:
    from ai.knowledge_graph.service import KnowledgeGraphService
except Exception as _import_exc:
    log.warning("KnowledgeGraphService unavailable: %s", _import_exc)
    KnowledgeGraphService = None  # type: ignore[assignment,misc]

router = APIRouter(prefix="/knowledge-graph", tags=["knowledge-graph"])


# ── Request / Response models ─────────────────────────────────────────────────


class ConceptNode(BaseModel):
    name: str
    difficulty: float
    mastery: float


class KGEdge(BaseModel):
    source: str
    target: str
    relation: str


class KGResponse(BaseModel):
    support: str
    nodes: List[ConceptNode]
    edges: List[KGEdge]
    weak_concepts: List[str]


class ConceptAddRequest(BaseModel):
    name: str
    difficulty: str = "intermediate"
    relation_to: Optional[str] = None
    relation_type: str = "relates_to"


class MasteryUpdateRequest(BaseModel):
    concept_name: str
    delta: float
    last_error: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

_DIFFICULTY_MAP = {"beginner": 0.2, "intermediate": 0.5, "advanced": 0.8}


def _difficulty_float(label: str) -> float:
    return _DIFFICULTY_MAP.get(label.lower(), 0.5)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get("/{support}", response_model=KGResponse)
def get_knowledge_graph(
    support: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the full knowledge graph for the authenticated user and support."""
    svc = KnowledgeGraphService()
    G = svc.build_graph(current_user.id, support, db)
    weak = svc.get_weak_concepts(current_user.id, support, db)

    nodes = [
        ConceptNode(
            name=n,
            difficulty=data.get("difficulty", 0.5),
            mastery=data.get("mastery", 0.0),
        )
        for n, data in G.nodes(data=True)
    ]
    edges = [
        KGEdge(source=u, target=v, relation=data.get("relation", "relates_to"))
        for u, v, data in G.edges(data=True)
    ]

    return KGResponse(support=support, nodes=nodes, edges=edges, weak_concepts=weak)


@router.post(
    "/{support}/concepts",
    response_model=ConceptNode,
    status_code=status.HTTP_201_CREATED,
)
def add_concept(
    support: str,
    body: ConceptAddRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add a concept (and optional relation) to the knowledge graph."""
    svc = KnowledgeGraphService()
    diff = _difficulty_float(body.difficulty)
    concept = svc.upsert_concept(db, body.name, support, difficulty=diff)

    if body.relation_to:
        svc.add_relation(db, body.name, body.relation_to, body.relation_type, support)

    mastery_row = (
        db.query(KGUserMastery)
        .filter(
            KGUserMastery.user_id == current_user.id,
            KGUserMastery.concept_id == concept.id,
        )
        .first()
    )
    return ConceptNode(
        name=concept.name,
        difficulty=concept.difficulty,
        mastery=mastery_row.mastery if mastery_row else 0.0,
    )


@router.put("/{support}/mastery")
def update_mastery(
    support: str,
    body: MasteryUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Increment or decrement mastery for one concept."""
    svc = KnowledgeGraphService()
    row = svc.update_mastery(
        db,
        user_id=current_user.id,
        concept=body.concept_name,
        support=support,
        delta=body.delta,
        last_error=body.last_error,
    )
    return {
        "concept": body.concept_name,
        "mastery": row.mastery,
        "attempts": row.attempts,
    }


@router.delete("/{support}/mastery", status_code=status.HTTP_204_NO_CONTENT)
def reset_mastery(
    support: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Reset all mastery scores for the authenticated user on a support."""
    concepts = db.query(KGConcept).filter(KGConcept.support == support).all()
    concept_ids = {c.id for c in concepts}
    if concept_ids:
        db.query(KGUserMastery).filter(
            KGUserMastery.user_id == current_user.id,
            KGUserMastery.concept_id.in_(concept_ids),
        ).delete(synchronize_session=False)
        db.commit()
