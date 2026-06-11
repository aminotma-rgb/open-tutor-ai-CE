"""Context Retrieval Service — ChromaDB RAG + SQL memory retrieval."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from data.models.memory import Memory

_CHROMA_CLIENT = None
_VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vector_db")
_EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


def get_chroma_client():
    """Lazy singleton — PersistentClient at data/vector_db/."""
    global _CHROMA_CLIENT
    if _CHROMA_CLIENT is None:
        import chromadb

        os.makedirs(_VECTOR_DB_PATH, exist_ok=True)
        _CHROMA_CLIENT = chromadb.PersistentClient(path=_VECTOR_DB_PATH)
    return _CHROMA_CLIENT


class ContextRetrievalService:
    """ChromaDB-backed RAG indexing/retrieval + SQL memory retrieval."""

    # ── Collection management ─────────────────────────────────────────────────

    def get_or_create_collection(self, name: str):
        """Return (or create) a ChromaDB collection with sentence-transformer embeddings."""
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        client = get_chroma_client()
        ef = SentenceTransformerEmbeddingFunction(model_name=_EMBED_MODEL)
        return client.get_or_create_collection(name=name, embedding_function=ef)

    # ── Document indexing ─────────────────────────────────────────────────────

    def index_document(
        self,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_id: str = "",
        collection_name: str = "pedagogical_docs",
    ) -> int:
        """Extract text → chunk → embed → upsert into ChromaDB. Returns chunk count."""
        text = self._extract_text(file_path)
        if not text.strip():
            return 0

        chunks = self._chunk_text(text)
        if not chunks:
            return 0

        collection = self.get_or_create_collection(collection_name)
        base_meta = metadata or {}
        base_meta["user_id"] = user_id
        base_meta["file_path"] = file_path

        filename = os.path.basename(file_path)
        ids = [f"{filename}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{**base_meta, "chunk_index": i} for i in range(len(chunks))]

        collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
        return len(chunks)

    # ── Pedagogical document retrieval ────────────────────────────────────────

    def retrieve_pedagogical_documents(
        self,
        user_id: str,
        query: str,
        top_k: int = 5,
        collection_name: str = "pedagogical_docs",
    ) -> List[Dict[str, Any]]:
        """Semantic search over indexed documents. Returns list of {id, content, metadata, score}."""
        if not query.strip():
            return []

        try:
            collection = self.get_or_create_collection(collection_name)
        except Exception:
            return []

        try:
            results = collection.query(query_texts=[query], n_results=top_k)
        except Exception:
            return []

        docs = []
        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i, doc_id in enumerate(ids):
            dist = distances[i] if i < len(distances) else 2.0
            score = max(0.0, 1.0 - dist / 2.0)
            docs.append(
                {
                    "id": doc_id,
                    "content": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "score": round(score, 4),
                }
            )
        return docs

    # ── Internal memory retrieval ─────────────────────────────────────────────

    def retrieve_internal_memory(
        self,
        user_id: str,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 10,
        db: Session = None,
    ) -> List[Dict[str, Any]]:
        """Substring search over opentutorai_memory SQL rows."""
        if db is None:
            return []

        q = db.query(Memory).filter(Memory.user_id == user_id)
        if memory_types:
            q = q.filter(Memory.memory_type.in_(memory_types))
        if query.strip():
            q = q.filter(Memory.content.ilike(f"%{query}%"))

        rows = q.order_by(Memory.created_at.desc()).limit(limit).all()
        return [r.to_dict() for r in rows]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _chunk_text(
        self,
        text: str,
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> List[str]:
        """Word-level sliding window chunker."""
        words = text.split()
        if not words:
            return []
        if len(words) <= chunk_size:
            return [" ".join(words)]

        chunks = []
        step = chunk_size - overlap
        for start in range(0, len(words), step):
            chunk = words[start : start + chunk_size]
            if chunk:
                chunks.append(" ".join(chunk))
            if start + chunk_size >= len(words):
                break
        return chunks

    def _extract_text(self, file_path: str) -> str:
        """Extract text from a .pdf or plain text file."""
        if not os.path.exists(file_path):
            return ""

        if file_path.lower().endswith(".pdf"):
            try:
                from pypdf import PdfReader

                reader = PdfReader(file_path)
                return "\n".join(
                    page.extract_text() or "" for page in reader.pages
                )
            except Exception:
                return ""

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""

    # ── Collection listing ────────────────────────────────────────────────────

    def list_collections(self) -> List[Dict[str, Any]]:
        """List all ChromaDB collections with their document counts."""
        client = get_chroma_client()
        collections = client.list_collections()
        result = []
        for col in collections:
            try:
                c = client.get_collection(col.name)
                result.append({"name": col.name, "count": c.count()})
            except Exception:
                result.append({"name": col.name, "count": 0})
        return result
