"""Tools router — /api/v1/tools/* matching tools/index.ts UI client."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from data.database import get_db
from data.models import User
from gateway.http.dependencies import get_current_user
from ai.tools.service import ToolsService

router = APIRouter(prefix="/tools", tags=["tools"])


def get_tools_service(db: Session = Depends(get_db)) -> ToolsService:
    return ToolsService(db)


class ToolCreateBody(BaseModel):
    id: Optional[str] = None
    name: str = ""
    content: str = ""
    meta: Dict[str, Any] = {}


# ── List / export ─────────────────────────────────────────────────────────────


@router.get("/")
def list_tools(
    user: User = Depends(get_current_user),
    svc: ToolsService = Depends(get_tools_service),
):
    return [t.to_dict() for t in svc.list_all()]


@router.get("/list")
def list_tools_public(
    user: User = Depends(get_current_user),
    svc: ToolsService = Depends(get_tools_service),
):
    return [t.to_dict() for t in svc.list_all()]


@router.get("/export")
def export_tools(
    user: User = Depends(get_current_user),
    svc: ToolsService = Depends(get_tools_service),
):
    return [t.to_dict() for t in svc.list_all()]


# ── CRUD ──────────────────────────────────────────────────────────────────────


@router.post("/create", status_code=status.HTTP_201_CREATED)
def create_tool(
    body: ToolCreateBody,
    user: User = Depends(get_current_user),
    svc: ToolsService = Depends(get_tools_service),
):
    tool_id = body.id or str(uuid.uuid4())
    tool = svc.upsert(
        tool_id=tool_id,
        name=body.name,
        content=body.content,
        meta=body.meta,
        user_id=user.id,
    )
    return tool.to_dict()


@router.get("/id/{tool_id}")
def get_tool(
    tool_id: str,
    user: User = Depends(get_current_user),
    svc: ToolsService = Depends(get_tools_service),
):
    tool = svc.get_by_id(tool_id)
    if not tool:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Tool not found"
        )
    return tool.to_dict()


@router.post("/id/{tool_id}/update")
def update_tool(
    tool_id: str,
    body: ToolCreateBody,
    user: User = Depends(get_current_user),
    svc: ToolsService = Depends(get_tools_service),
):
    tool = svc.upsert(
        tool_id=tool_id,
        name=body.name,
        content=body.content,
        meta=body.meta,
        user_id=user.id,
    )
    return tool.to_dict()


@router.delete("/id/{tool_id}/delete")
def delete_tool(
    tool_id: str,
    user: User = Depends(get_current_user),
    svc: ToolsService = Depends(get_tools_service),
):
    svc.delete(tool_id)
    return {"id": tool_id}


# ── Valves (pass-through stubs — no valve config on OTAI tools) ───────────────


@router.get("/id/{tool_id}/valves")
def get_valves(tool_id: str, user: User = Depends(get_current_user)):
    return {}


@router.get("/id/{tool_id}/valves/spec")
def get_valves_spec(tool_id: str, user: User = Depends(get_current_user)):
    return {}


@router.post("/id/{tool_id}/valves/update")
def update_valves(
    tool_id: str, body: Dict[str, Any], user: User = Depends(get_current_user)
):
    return body


@router.get("/id/{tool_id}/valves/user")
def get_user_valves(tool_id: str, user: User = Depends(get_current_user)):
    return {}


@router.get("/id/{tool_id}/valves/user/spec")
def get_user_valves_spec(tool_id: str, user: User = Depends(get_current_user)):
    return {}


@router.post("/id/{tool_id}/valves/user/update")
def update_user_valves(
    tool_id: str, body: Dict[str, Any], user: User = Depends(get_current_user)
):
    return body
