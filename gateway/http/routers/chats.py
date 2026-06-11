"""Chats router — /api/v1/chats/*."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from learning.sessions.service import ChatsService
from common.exceptions import AuthorizationError, NotFoundError
from data.database import get_db
from data.models import User
from gateway.http.dependencies import get_current_user
from ai.summarization.service import SummarizationService, extract_exchanges

router = APIRouter(prefix="/chats", tags=["chats"])


def get_chats_service(db: Session = Depends(get_db)) -> ChatsService:
    return ChatsService(db)


class NewChatRequest(BaseModel):
    chat: Dict[str, Any]


class TagRequest(BaseModel):
    name: str


# ── List endpoints ────────────────────────────────────────────────────────────


@router.get("/")
async def get_chats(
    page: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    skip = ((page or 1) - 1) * 50 if page else 0
    return [c.to_title_id() for c in svc.list_for_user(current_user.id, skip=skip)]


@router.get("/list")
async def get_chat_list(
    page: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    skip = ((page or 1) - 1) * 50 if page else 0
    return [c.to_title_id() for c in svc.list_for_user(current_user.id, skip=skip)]


@router.get("/pinned")
async def get_pinned(
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    return [c.to_title_id() for c in svc.list_pinned(current_user.id)]


@router.get("/archived")
async def get_archived(
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    return [c.to_title_id() for c in svc.list_archived(current_user.id)]


@router.get("/all")
async def get_all(
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    return [c.to_dict() for c in svc.list_all(current_user.id)]


@router.get("/all/archived")
async def get_all_archived(
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    return [c.to_dict() for c in svc.list_archived(current_user.id)]


@router.get("/all/tags")
async def get_all_tags(
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    return [{"name": t} for t in svc.get_all_tags(current_user.id)]


@router.get("/all/db")
async def get_all_db(
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    return [c.to_dict() for c in svc.list_all(current_user.id)]


@router.get("/tags")
async def get_tags(
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    return [{"name": t} for t in svc.get_all_tags(current_user.id)]


@router.post("/tags")
async def get_chats_by_tag(
    body: TagRequest,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    return [c.to_title_id() for c in svc.list_by_tag(current_user.id, body.name)]


@router.get("/search")
async def search_chats(
    q: str = Query(""),
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    return [c.to_title_id() for c in svc.search(current_user.id, q)]


@router.get("/list/user/{user_id}")
async def get_user_chats(
    user_id: str,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    if not current_user.is_admin and current_user.id != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden")
    return [c.to_title_id() for c in svc.list_all(user_id)]


@router.get("/folder/{folder_id}")
async def get_folder_chats(
    folder_id: str,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    return [c.to_dict() for c in svc.list_by_folder(current_user.id, folder_id)]


# ── Share ─────────────────────────────────────────────────────────────────────


@router.get("/share/{share_id}")
async def get_shared_chat(
    share_id: str,
    svc: ChatsService = Depends(get_chats_service),
):
    chat = svc.get_by_share_id(share_id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Shared chat not found"
        )
    return chat.to_dict()


# ── Create / Import ───────────────────────────────────────────────────────────


@router.post("/new")
async def create_chat(
    body: NewChatRequest,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    return svc.create(current_user.id, body.chat).to_dict()


@router.post("/import")
async def import_chats(
    body: List[Dict[str, Any]],
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    return [svc.create(current_user.id, c.get("chat", c)).to_dict() for c in body]


# ── Bulk actions ──────────────────────────────────────────────────────────────


@router.delete("/")
async def delete_all_chats(
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    count = svc.delete_all(current_user.id)
    return {"count": count}


@router.post("/archive/all")
async def archive_all(
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    return {"count": svc.archive_all(current_user.id)}


# ── Chat completion tracking (replaces /api/chat/completed) ───────────────────


def _run_summarization(
    user_id: str,
    chat_id: str,
    support_id: str,
    exchanges: list,
    db: Session,
) -> None:
    """Background task — summarize session if threshold reached."""
    try:
        summarization_svc = SummarizationService()
        summarization_svc.summarize_session(
            user_id=user_id,
            support_id=support_id,
            exchanges=exchanges,
            db=db,
        )
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "Background summarization failed for chat %s: %s", chat_id, exc
        )


@router.post("/completed")
async def chat_completed(
    body: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
    db: Session = Depends(get_db),
):
    """Record chat completion — trigger session summarization if threshold reached."""
    chat_id = body.get("chat_id")
    if not chat_id:
        return {"status": "recorded"}

    try:
        chat = svc.get(chat_id, current_user.id)
    except (NotFoundError, AuthorizationError):
        return {"status": "recorded"}

    exchanges = extract_exchanges(chat.chat)
    support_id = (chat.meta or {}).get("support_id") or chat_id

    summarization_svc = SummarizationService()
    if summarization_svc.should_summarize(exchanges):
        background_tasks.add_task(
            _run_summarization,
            user_id=current_user.id,
            chat_id=chat_id,
            support_id=support_id,
            exchanges=exchanges,
            db=db,
        )
        return {"status": "recorded", "summarization": "scheduled"}

    return {"status": "recorded"}


# ── Chat actions (replaces /api/chat/actions/{action_id}) ────────────────────


@router.post("/actions/{action_id}")
async def chat_action(
    action_id: str,
    body: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    """Execute a chat action (title generation, summarization, etc.).

    This keeps action execution inside the OpenTutorAI chats namespace.
    """
    # Placeholder - actual action handling depends on action_id
    # Common actions: "generate_title", "summarize", etc.
    return {"action_id": action_id, "status": "executed"}


# ── Per-chat CRUD ─────────────────────────────────────────────────────────────


@router.get("/{id}")
async def get_chat(
    id: str,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    try:
        return svc.get(id, current_user.id).to_dict()
    except (NotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.post("/{id}")
async def update_chat(
    id: str,
    body: NewChatRequest,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    try:
        return svc.update(id, current_user.id, body.chat).to_dict()
    except (NotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.delete("/{id}")
async def delete_chat(
    id: str,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    try:
        svc.delete(id, current_user.id)
        return {"id": id}
    except (NotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.post("/{id}/archive")
async def archive_chat(
    id: str,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    try:
        return svc.archive(id, current_user.id).to_dict()
    except (NotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.post("/{id}/pin")
async def pin_chat(
    id: str,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    try:
        return svc.pin(id, current_user.id).to_dict()
    except (NotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.get("/{id}/pinned")
async def get_pinned_status(
    id: str,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    try:
        return svc.get_pinned_status(id, current_user.id)
    except (NotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.post("/{id}/share")
async def share_chat(
    id: str,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    try:
        return svc.share(id, current_user.id).to_dict()
    except (NotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.delete("/{id}/share")
async def unshare_chat(
    id: str,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    try:
        return svc.unshare(id, current_user.id).to_dict()
    except (NotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.post("/{id}/folder")
async def set_folder(
    id: str,
    body: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    try:
        return svc.set_folder(id, current_user.id, body.get("folder_id")).to_dict()
    except (NotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.post("/{id}/clone")
async def clone_chat(
    id: str,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    try:
        original = svc.get(id, current_user.id)
        cloned_data = dict(original.chat or {})
        cloned_data["title"] = f"Clone of {original.title}"
        return svc.create(current_user.id, cloned_data).to_dict()
    except (NotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.post("/{id}/clone/shared")
async def clone_shared_chat(
    id: str,  # share_id of the publicly shared chat
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    chat = svc.get_by_share_id(id)
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Shared chat not found"
        )
    cloned_data = dict(chat.chat or {})
    cloned_data["title"] = f"Clone of {chat.title}"
    return svc.create(current_user.id, cloned_data).to_dict()


# ── Tags per chat ─────────────────────────────────────────────────────────────


@router.get("/{id}/tags")
async def get_chat_tags(
    id: str,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    try:
        return svc.get_tags(id, current_user.id)
    except (NotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.post("/{id}/tags")
async def add_tag(
    id: str,
    body: TagRequest,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    try:
        return svc.add_tag(id, current_user.id, body.name).to_dict()
    except (NotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.delete("/{id}/tags")
async def delete_tag(
    id: str,
    body: TagRequest,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    try:
        return svc.remove_tag(id, current_user.id, body.name).to_dict()
    except (NotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.get("/{id}/tags/all")
async def get_all_chat_tags(
    id: str,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    try:
        return svc.get_tags(id, current_user.id)
    except (NotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)


@router.delete("/{id}/tags/all")
async def delete_all_tags(
    id: str,
    current_user: User = Depends(get_current_user),
    svc: ChatsService = Depends(get_chats_service),
):
    try:
        return svc.remove_all_tags(id, current_user.id).to_dict()
    except (NotFoundError, AuthorizationError) as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=e.message)
