# Phase 2 — Context Retrieval Engine

**Status :** ✅ Implémenté  
**Date :** 2026-06-11  
**Tests :** 15 tests (test_phase2_context_retrieval.py)

---

## Objectifs

Brancher ChromaDB comme backend RAG et exposer deux surfaces de récupération
utilisées par tous les agents :

1. **Récupération de documents pédagogiques** — recherche sémantique sur les
   fichiers indexés (PDF, texte) via ChromaDB + embeddings sentence-transformers.
2. **Récupération de mémoire interne** — recherche textuelle sur les lignes SQL
   `opentutorai_memory` (créées en Phase 1).
3. **Endpoints REST** — indexer des documents, récupérer, lister les collections.

---

## Fichiers créés

| Fichier | Rôle |
|---------|------|
| `ai/retrieval/context_retrieval.py` | `ContextRetrievalService` — client ChromaDB, chunking, indexation, recherche RAG, récupération mémoire SQL |
| `gateway/http/routers/context_retrieval.py` | Router REST `/api/v1/context/*` (4 endpoints) |
| `tests/test_phase2_context_retrieval.py` | 15 tests unitaires |

## Fichiers modifiés

| Fichier | Modification |
|---------|-------------|
| `requirements.txt` | Ajout de `langgraph>=0.2.0`, `langgraph-checkpoint-sqlite>=0.1.0` |
| `gateway/http/app.py` | Import et enregistrement du router `context_retrieval` |

---

## `ContextRetrievalService` — API publique

| Méthode | Description |
|---------|-------------|
| `get_chroma_client()` | Singleton lazy — `PersistentClient` dans `data/vector_db/` |
| `get_or_create_collection(name)` | Crée/récupère une collection ChromaDB avec `SentenceTransformerEmbeddingFunction` |
| `index_document(file_path, metadata, user_id, collection_name)` | Extrait le texte → découpe en chunks (500 mots, 50 overlap) → embed → upsert. Retourne le nombre de chunks. |
| `retrieve_pedagogical_documents(user_id, query, top_k, collection_name)` | Recherche sémantique, retourne `{id, content, metadata, score}` |
| `retrieve_internal_memory(user_id, query, memory_types, limit, db)` | Recherche substring sur `opentutorai_memory` SQL |
| `list_collections()` | Liste les collections ChromaDB avec leur nombre de documents |
| `_chunk_text(text, chunk_size, overlap)` | Découpage par fenêtre glissante sur les mots |
| `_extract_text(file_path)` | Lit `.pdf` (via pypdf) ou fichiers texte plat |

---

## Endpoints REST

| Méthode | Chemin | Description |
|---------|--------|-------------|
| `POST` | `/api/v1/context/index` | Upload fichier (PDF/txt), indexation dans ChromaDB. Retourne `{chunks_indexed, filename}` |
| `POST` | `/api/v1/context/retrieve` | Recherche sémantique. Body : `{query, top_k, collection_name}` |
| `POST` | `/api/v1/context/retrieve/memory` | Recherche mémoire SQL. Body : `{query, memory_types, limit}` |
| `GET`  | `/api/v1/context/collections` | Liste les collections ChromaDB avec le nombre de documents |

---

## Décisions de conception

| Décision | Justification |
|----------|--------------|
| **Modèle d'embedding :** `all-MiniLM-L6-v2` | Rapide, bonne qualité, déjà installé via `sentence-transformers`. Configurable via variable d'env `EMBED_MODEL`. |
| **Distance → score :** `score = max(0, 1 - dist/2)` | ChromaDB retourne une distance L2 ; conversion pour que tous les scores soient dans `[0, 1]`. |
| **Taille des chunks :** 500 mots / 50 overlap | Équilibre richesse du contexte et précision des embeddings. |
| **Récupération mémoire** : substring SQL (pas d'embeddings) | Rapide et suffisant pour la fenêtre de contexte de 10 mémoires utilisée par les agents. |
| **Client lazy :** `get_chroma_client()` | Initialisation au premier appel, évite le coût au démarrage. |
| **Stockage ChromaDB :** `data/vector_db/` | Configurable via variable d'env `VECTOR_DB_PATH`. |
| **Isolation des tests :** `chromadb.EphemeralClient()` + `monkeypatch` | Chaque test utilise un client en mémoire, sans effet de bord sur les données persistées. |

---

## Dépendances ajoutées

| Package | Version | Usage |
|---------|---------|-------|
| `langgraph` | `>=0.2.0` | Orchestration agentique (Phases 3+) |
| `langgraph-checkpoint-sqlite` | `>=0.1.0` | Persistance état LangGraph (SqliteSaver) |

> `chromadb`, `sentence-transformers` et `pypdf` étaient déjà présents dans `requirements.txt`.

---

## Intégration avec Phase 1

`retrieve_internal_memory()` requête directement la table `opentutorai_memory`
créée en Phase 1. Le filtre `memory_types` permet de cibler
`episodic`, `behavioral`, `procedural` ou `session_summary`.

---

## Variables d'environnement

| Variable | Défaut | Usage |
|----------|--------|-------|
| `VECTOR_DB_PATH` | `./data/vector_db` | Répertoire persistance ChromaDB |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Modèle sentence-transformers pour les embeddings |

---

## Phase suivante

**Phase 3 — Dynamic Context Manager** : `ContextManager.build_agent_context()`
fusionne documents RAG + mémoires + résumé de session en un seul `AgentContext`,
filtré par pertinence, récence (`max_age_days=14`) et budget token
(`max_context_tokens=3000`).
