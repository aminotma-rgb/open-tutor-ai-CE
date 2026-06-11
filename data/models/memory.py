"""Memory and Knowledge Graph ORM models."""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Float,
    Integer,
    JSON,
    ForeignKey,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from data.database import Base


class Memory(Base):
    __tablename__ = "opentutorai_memory"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    memory_type = Column(
        String(50), nullable=False, index=True
    )  # episodic | behavioral | procedural | session_summary
    content = Column(Text, nullable=False)
    memory_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "memory_metadata": self.memory_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class KGConcept(Base):
    __tablename__ = "opentutorai_kg_concept"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    support = Column(String(255), nullable=False, index=True)
    difficulty = Column(Float, default=0.5, nullable=False)
    last_seen = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    relations_from = relationship(
        "KGRelation",
        foreign_keys="KGRelation.source_id",
        back_populates="source",
        cascade="all, delete-orphan",
    )
    relations_to = relationship(
        "KGRelation",
        foreign_keys="KGRelation.target_id",
        back_populates="target",
        cascade="all, delete-orphan",
    )
    masteries = relationship(
        "KGUserMastery",
        back_populates="concept",
        cascade="all, delete-orphan",
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "support": self.support,
            "difficulty": self.difficulty,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class KGRelation(Base):
    __tablename__ = "opentutorai_kg_relation"
    __table_args__ = (
        UniqueConstraint("source_id", "target_id", "relation", name="uq_kg_relation"),
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id = Column(
        String(36), ForeignKey("opentutorai_kg_concept.id", ondelete="CASCADE"), nullable=False
    )
    target_id = Column(
        String(36), ForeignKey("opentutorai_kg_concept.id", ondelete="CASCADE"), nullable=False
    )
    relation = Column(String(100), nullable=False)  # e.g. "requires", "extends"

    source = relationship("KGConcept", foreign_keys=[source_id], back_populates="relations_from")
    target = relationship("KGConcept", foreign_keys=[target_id], back_populates="relations_to")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
        }


class KGUserMastery(Base):
    __tablename__ = "opentutorai_kg_user_mastery"
    __table_args__ = (
        UniqueConstraint("user_id", "concept_id", name="uq_kg_user_mastery"),
    )

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), nullable=False, index=True)
    concept_id = Column(
        String(36), ForeignKey("opentutorai_kg_concept.id", ondelete="CASCADE"), nullable=False
    )
    mastery = Column(Float, default=0.0, nullable=False)  # 0.0 → 1.0
    attempts = Column(Integer, default=0, nullable=False)
    last_error = Column(Text, nullable=True)
    last_seen = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    concept = relationship("KGConcept", back_populates="masteries")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "concept_id": self.concept_id,
            "mastery": self.mastery,
            "attempts": self.attempts,
            "last_error": self.last_error,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
