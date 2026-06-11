"""ToolsService — CRUD for OTAI tool records."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy.orm import Session

from data.models.tool import Tool


class ToolsService:
    def __init__(self, db: Session):
        self.db = db

    def list_all(self) -> List[Tool]:
        return self.db.query(Tool).order_by(Tool.name).all()

    def get_by_id(self, tool_id: str) -> Optional[Tool]:
        return self.db.query(Tool).filter(Tool.id == tool_id).first()

    def upsert(
        self,
        tool_id: str,
        name: str,
        content: str,
        meta: dict,
        user_id: Optional[str] = None,
    ) -> Tool:
        tool = self.db.query(Tool).filter(Tool.id == tool_id).first()
        if tool:
            tool.name = name
            tool.content = content
            tool.meta = meta
            tool.updated_at = datetime.utcnow()
        else:
            tool = Tool(
                id=tool_id,
                user_id=user_id,
                name=name,
                content=content,
                meta=meta,
            )
            self.db.add(tool)
        self.db.commit()
        self.db.refresh(tool)
        return tool

    def delete(self, tool_id: str) -> bool:
        tool = self.db.query(Tool).filter(Tool.id == tool_id).first()
        if not tool:
            return False
        self.db.delete(tool)
        self.db.commit()
        return True
