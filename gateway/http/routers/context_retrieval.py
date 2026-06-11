"""Context retrieval router — /api/v1/context/*

Exposes document indexing, semantic retrieval and collection listing
backed by ContextRetrievalService (ChromaDB + SQL memory).
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ai.retrieval.context_retrieval import ContextRetrievalService
from data.database import get_db
from data.models import User
from gateway.http.dependencies import get_current_user

router = APIRouter(prefix="/context", tags=["context-retrieval"])


def _svc() -> ContextRetrievalService:
    return ContextRetrievalService()


# ── Request / response schemas ────────────────────────────────────────────────


class RetrieveRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    collection_name: Optional[str] = "pedagogical_docs"


class MemoryRetrieveRequest(BaseModel):
    query: str
    memory_types: Optional[List[str]] = None
    limit: Optional[int] = 10


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.post("/index")
async def index_document(
    file: UploadFile = File(...),
    collection_name: str = "pedagogical_docs",
    user: User = Depends(get_current_user),
    svc: ContextRetrievalService = Depends(_svc),
):
    """Upload a PDF or text file and index it into ChromaDB."""
    import tempfile, os

    suffix = os.path.splitext(file.filename or "")[1] or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        chunks = svc.index_document(
            file_path=tmp_path,
            metadata={"filename": file.filename},
            user_id=user.id,
            collection_name=collection_name,
        )
    finally:
        os.unlink(tmp_path)

    if chunks == 0:
        raise HTTPException(
            status_code=422, detail="No text could be extracted from the file"
        )

    return {"chunks_indexed": chunks, "filename": file.filename}


@router.post("/retrieve")
def retrieve_documents(
    body: RetrieveRequest,
    user: User = Depends(get_current_user),
    svc: ContextRetrievalService = Depends(_svc),
):
    """Semantic search over indexed documents."""
    return svc.retrieve_pedagogical_documents(
        user_id=user.id,
        query=body.query,
        top_k=body.top_k,
        collection_name=body.collection_name,
    )


@router.post("/retrieve/memory")
def retrieve_memory(
    body: MemoryRetrieveRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    svc: ContextRetrievalService = Depends(_svc),
):
    """Text search over user memories in SQL."""
    return svc.retrieve_internal_memory(
        user_id=user.id,
        query=body.query,
        memory_types=body.memory_types,
        limit=body.limit,
        db=db,
    )


@router.get("/collections")
def list_collections(
    user: User = Depends(get_current_user),
    svc: ContextRetrievalService = Depends(_svc),
):
    """List ChromaDB collections with document counts."""
    return svc.list_collections()
