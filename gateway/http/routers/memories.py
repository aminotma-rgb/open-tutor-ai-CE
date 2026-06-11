"""Memories router — /api/v1/memories/* matching memories/index.ts UI client."""

import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from data.database import get_db
from data.models import User
from data.models.memory import Memory
from gateway.http.dependencies import get_current_user

router = APIRouter(prefix="/memories", tags=["memories"])


class MemoryCreateRequest(BaseModel):
    content: str
    memory_type: Optional[str] = "episodic"
    memory_metadata: Optional[Dict[str, Any]] = None


class MemoryUpdateRequest(BaseModel):
    content: Optional[str] = None
    memory_metadata: Optional[Dict[str, Any]] = None


class MemoryQueryRequest(BaseModel):
    content: str


@router.get("/")
def list_memories(
    memory_type: Optional[str] = None,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Memory).filter(Memory.user_id == user.id)
    if memory_type:
        query = query.filter(Memory.memory_type == memory_type)
    return [m.to_dict() for m in query.order_by(Memory.created_at.desc()).all()]


@router.post("/add")
def add_memory(
    body: MemoryCreateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    mem = Memory(
        id=str(uuid.uuid4()),
        user_id=user.id,
        memory_type=body.memory_type,
        content=body.content,
        memory_metadata=body.memory_metadata,
    )
    db.add(mem)
    db.commit()
    db.refresh(mem)
    return mem.to_dict()


@router.post("/query")
def query_memories(
    body: MemoryQueryRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    term = body.content.lower()
    all_memories = (
        db.query(Memory)
        .filter(Memory.user_id == user.id)
        .order_by(Memory.created_at.desc())
        .all()
    )
    matched = [m.to_dict() for m in all_memories if term in m.content.lower()]
    return {"documents": [m["content"] for m in matched], "metadatas": matched, "distances": []}


@router.post("/{memory_id}/update")
def update_memory(
    memory_id: str,
    body: MemoryUpdateRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    mem = db.query(Memory).filter(Memory.id == memory_id, Memory.user_id == user.id).first()
    if not mem:
        raise HTTPException(status_code=404, detail="Memory not found")
    if body.content is not None:
        mem.content = body.content
    if body.memory_metadata is not None:
        mem.memory_metadata = body.memory_metadata
    db.commit()
    db.refresh(mem)
    return mem.to_dict()


@router.delete("/delete/user")
def delete_user_memories(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    db.query(Memory).filter(Memory.user_id == user.id).delete()
    db.commit()
    return {"status": True}


@router.delete("/{memory_id}")
def delete_memory(
    memory_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    mem = db.query(Memory).filter(Memory.id == memory_id, Memory.user_id == user.id).first()
    if not mem:
        raise HTTPException(status_code=404, detail="Memory not found")
    db.delete(mem)
    db.commit()
    return {"id": memory_id}
