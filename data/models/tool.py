"""Tool ORM model — stores Open WebUI-compatible tool source code."""

from datetime import datetime
from sqlalchemy import Column, String, Text, DateTime, JSON
from data.database import Base


class Tool(Base):
    __tablename__ = "otai_tools"

    id = Column(String(255), primary_key=True)
    user_id = Column(String(255), nullable=True, index=True)
    name = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    meta = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "content": self.content,
            "meta": self.meta or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
