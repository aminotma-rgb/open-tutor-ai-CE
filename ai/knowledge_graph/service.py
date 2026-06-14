"""Knowledge Graph service — read/write concept graph with per-user mastery overlay."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import networkx as nx
from sqlalchemy.orm import Session

from data.models.memory import KGConcept, KGRelation, KGUserMastery


class KnowledgeGraphService:
    """Manages the pedagogical concept graph and per-user mastery scores."""

    # ── Concept operations ────────────────────────────────────────────────────

    def upsert_concept(
        self,
        db: Session,
        name: str,
        support: str,
        difficulty: float = 0.5,
    ) -> KGConcept:
        """Return existing concept or create a new one. Idempotent."""
        concept = (
            db.query(KGConcept)
            .filter(KGConcept.name == name, KGConcept.support == support)
            .first()
        )
        if concept:
            return concept
        concept = KGConcept(name=name, support=support, difficulty=difficulty)
        db.add(concept)
        db.commit()
        db.refresh(concept)
        return concept

    # ── Relation operations ───────────────────────────────────────────────────

    def add_relation(
        self,
        db: Session,
        source: str,
        target: str,
        relation: str,
        support: str,
    ) -> KGRelation:
        """Add a directed edge between two concepts. Idempotent."""
        src = self.upsert_concept(db, source, support)
        tgt = self.upsert_concept(db, target, support)

        existing = (
            db.query(KGRelation)
            .filter(
                KGRelation.source_id == src.id,
                KGRelation.target_id == tgt.id,
                KGRelation.relation == relation,
            )
            .first()
        )
        if existing:
            return existing

        edge = KGRelation(source_id=src.id, target_id=tgt.id, relation=relation)
        db.add(edge)
        db.commit()
        db.refresh(edge)
        return edge

    # ── Mastery operations ────────────────────────────────────────────────────

    def update_mastery(
        self,
        db: Session,
        user_id: str,
        concept: str,
        support: str,
        delta: float,
        last_error: Optional[str] = None,
    ) -> KGUserMastery:
        """Increment mastery score for a user/concept pair (capped at 1.0)."""
        c = self.upsert_concept(db, concept, support)
        row = (
            db.query(KGUserMastery)
            .filter(
                KGUserMastery.user_id == user_id,
                KGUserMastery.concept_id == c.id,
            )
            .first()
        )
        if not row:
            row = KGUserMastery(
                user_id=user_id, concept_id=c.id, mastery=0.0, attempts=0
            )
            db.add(row)

        row.mastery = max(0.0, min(1.0, row.mastery + delta))
        row.attempts += 1
        row.last_seen = datetime.utcnow()
        if last_error is not None:
            row.last_error = last_error

        db.commit()
        db.refresh(row)
        return row

    # ── Graph queries ─────────────────────────────────────────────────────────

    def build_graph(self, user_id: str, support: str, db: Session) -> nx.DiGraph:
        """Build a DiGraph for a support with mastery scores as node attributes."""
        concepts = db.query(KGConcept).filter(KGConcept.support == support).all()
        concept_ids = {c.id for c in concepts}

        G = nx.DiGraph()
        for c in concepts:
            mastery_row = (
                db.query(KGUserMastery)
                .filter(
                    KGUserMastery.user_id == user_id,
                    KGUserMastery.concept_id == c.id,
                )
                .first()
            )
            G.add_node(
                c.name,
                id=c.id,
                difficulty=c.difficulty,
                mastery=mastery_row.mastery if mastery_row else 0.0,
            )

        relations = (
            db.query(KGRelation)
            .filter(
                KGRelation.source_id.in_(concept_ids),
                KGRelation.target_id.in_(concept_ids),
            )
            .all()
        )
        id_to_name = {c.id: c.name for c in concepts}
        for r in relations:
            G.add_edge(
                id_to_name[r.source_id], id_to_name[r.target_id], relation=r.relation
            )

        return G

    def get_weak_concepts(
        self,
        user_id: str,
        support: str,
        db: Session,
        threshold: float = 0.4,
    ) -> List[str]:
        """Return concept names where mastery < threshold for this user/support."""
        concepts = db.query(KGConcept).filter(KGConcept.support == support).all()
        weak = []
        for c in concepts:
            mastery_row = (
                db.query(KGUserMastery)
                .filter(
                    KGUserMastery.user_id == user_id,
                    KGUserMastery.concept_id == c.id,
                )
                .first()
            )
            score = mastery_row.mastery if mastery_row else 0.0
            if score < threshold:
                weak.append(c.name)
        return weak

    def get_blocked_concepts(
        self,
        user_id: str,
        support: str,
        db: Session,
        mastery_threshold: float = 0.4,
        attempts_threshold: int = 5,
    ) -> List[dict]:
        """Return concepts where mastery < threshold AND attempts >= attempts_threshold.

        These are concepts the learner has repeatedly failed — require a strategy
        change, not just another exercise of the same type.
        """
        concepts = db.query(KGConcept).filter(KGConcept.support == support).all()
        blocked = []
        for c in concepts:
            row = (
                db.query(KGUserMastery)
                .filter(
                    KGUserMastery.user_id == user_id,
                    KGUserMastery.concept_id == c.id,
                )
                .first()
            )
            if (
                row
                and row.mastery < mastery_threshold
                and row.attempts >= attempts_threshold
            ):
                blocked.append(
                    {
                        "concept": c.name,
                        "attempts": row.attempts,
                        "mastery": round(row.mastery, 3),
                        "last_error": row.last_error,
                    }
                )
        # Sort by attempts descending — most stuck concept first
        return sorted(blocked, key=lambda x: x["attempts"], reverse=True)

    def has_requires_relations(self, support: str, db: Session) -> bool:
        """Return True if at least one 'requires' edge exists for this support."""
        concepts = db.query(KGConcept).filter(KGConcept.support == support).all()
        if not concepts:
            return False
        concept_ids = {c.id for c in concepts}
        return (
            db.query(KGRelation)
            .filter(
                KGRelation.source_id.in_(concept_ids),
                KGRelation.relation == "requires",
            )
            .first()
            is not None
        )

    def get_prerequisites_of(
        self,
        concept_name: str,
        support: str,
        db: Session,
    ) -> List[str]:
        """Return names of concepts that concept_name requires (its direct prerequisites).

        Edge direction: source --requires--> target means source needs target first.
        So prerequisites of X = targets of edges where source == X.
        """
        concept = (
            db.query(KGConcept)
            .filter(KGConcept.name == concept_name, KGConcept.support == support)
            .first()
        )
        if not concept:
            return []
        relations = (
            db.query(KGRelation)
            .filter(
                KGRelation.source_id == concept.id,
                KGRelation.relation == "requires",
            )
            .all()
        )
        target_ids = [r.target_id for r in relations]
        if not target_ids:
            return []
        prereqs = db.query(KGConcept).filter(KGConcept.id.in_(target_ids)).all()
        return [p.name for p in prereqs]

    def get_unmastered_prerequisites(
        self,
        user_id: str,
        concept_name: str,
        support: str,
        db: Session,
        threshold: float = 0.4,
    ) -> List[str]:
        """Return prerequisites of concept_name that the user hasn't mastered yet."""
        prereqs = self.get_prerequisites_of(concept_name, support, db)
        unmastered = []
        for prereq_name in prereqs:
            prereq = (
                db.query(KGConcept)
                .filter(KGConcept.name == prereq_name, KGConcept.support == support)
                .first()
            )
            if not prereq:
                unmastered.append(prereq_name)
                continue
            row = (
                db.query(KGUserMastery)
                .filter(
                    KGUserMastery.user_id == user_id,
                    KGUserMastery.concept_id == prereq.id,
                )
                .first()
            )
            if (row.mastery if row else 0.0) < threshold:
                unmastered.append(prereq_name)
        return unmastered

    def find_prerequisites(
        self,
        concept_name: str,
        support: str,
        db: Session,
    ) -> List[str]:
        """Return concepts that depend on concept_name (outgoing 'requires' edges where target == concept_name).

        Kept for backward-compatibility — prefer get_prerequisites_of() for new code.
        """
        concept = (
            db.query(KGConcept)
            .filter(KGConcept.name == concept_name, KGConcept.support == support)
            .first()
        )
        if not concept:
            return []

        relations = (
            db.query(KGRelation)
            .filter(
                KGRelation.target_id == concept.id,
                KGRelation.relation == "requires",
            )
            .all()
        )
        source_ids = [r.source_id for r in relations]
        if not source_ids:
            return []

        dependents = db.query(KGConcept).filter(KGConcept.id.in_(source_ids)).all()
        return [d.name for d in dependents]
