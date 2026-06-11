# Phase 3 — Session Summarization Service

**Status :** ✅ Implémenté  
**Date :** 2026-06-11  
**Tests :** 18 tests (test_phase3_summarization.py)

---

## Objectifs

Implémenter un système de résumé automatique de session qui :

1. **Détecte le bon moment** — déclenchement dual : ≥ 5 échanges OU ≥ 1200 tokens.
2. **Appelle le LLM configuré** — provider-agnostique (Ollama puis OpenAI-compatible).
3. **Met en cache** les résumés dans `opentutorai_memory` (`memory_type=session_summary`).
4. **Branche `POST /chats/completed`** — l'endpoint stub est maintenant fonctionnel.

---

## Fichiers créés

| Fichier | Rôle |
|---------|------|
| `ai/summarization/__init__.py` | Package marker |
| `ai/summarization/service.py` | `SummarizationService` + `extract_exchanges()` |
| `tests/test_phase3_summarization.py` | 18 tests unitaires |

## Fichiers modifiés

| Fichier | Modification |
|---------|-------------|
| `config/settings.py` | Ajout de `SUMMARIZATION_CONFIG` |
| `gateway/http/routers/chats.py` | `POST /chats/completed` branché, imports `BackgroundTasks` |

---

## `SummarizationService` — API publique

| Méthode | Description |
|---------|-------------|
| `extract_exchanges(chat_json)` | Extraction défensive de `Chat.chat` (JSON libre) → `list[dict]` |
| `count_tokens(exchanges)` | tiktoken `cl100k_base`, fallback char/4 |
| `should_summarize(exchanges, cfg)` | Dual trigger : count ≥ seuil OU tokens ≥ seuil |
| `get_cached_summary(user_id, support_id, db)` | Lecture `opentutorai_memory` (session_summary) |
| `cache_summary(user_id, support_id, summary, db)` | Écriture `opentutorai_memory` |
| `invalidate_cache(user_id, support_id, db)` | Supprime les anciens résumés avant régénération |
| `_call_llm(prompt, db, model)` | Ollama `/api/generate` → OpenAI-compatible `/chat/completions` |
| `summarize_session(user_id, support_id, exchanges, db, model, force)` | Point d'entrée principal |

---

## Déclencheur dual

```python
if len(exchanges) >= exchanges_per_summary:   # défaut : 5
    return True
if count_tokens(exchanges) >= auto_summarize_token_limit:  # défaut : 1200
    return True
```

---

## `POST /chats/completed` — comportement réel

```
1. Charge le Chat depuis la DB (svc.get)
2. Extrait les messages via extract_exchanges(chat.chat)
3. Résout support_id = chat.meta["support_id"] ?? chat_id
4. Appelle should_summarize(exchanges)
5. Si True → ajoute _run_summarization en BackgroundTask
6. Retourne {"status": "recorded", "summarization": "scheduled"} ou {"status": "recorded"}
```

La tâche de fond capture toutes les exceptions et les log en WARNING sans bloquer la réponse HTTP.

---

## Appel LLM (provider-agnostique)

1. Ollama : `POST {url}/api/generate` — `{"model": model, "prompt": ..., "stream": false}`
2. OpenAI-compatible : `POST {url}/chat/completions` — format messages standard

En cas d'échec sur tous les providers → fallback `"Session avec N échanges."` mis en cache.

---

## Cache dans `opentutorai_memory`

| Champ | Valeur |
|-------|--------|
| `memory_type` | `session_summary` |
| `memory_metadata` | `{"support_id": "<support_id>"}` |
| `content` | Texte du résumé |

`invalidate_cache()` supprime les anciens résumés avant d'écrire le nouveau (évite l'accumulation).  
La lecture filtre en Python sur `memory_metadata["support_id"]` pour compatibilité SQLite.

---

## Variables d'environnement

| Variable | Défaut | Usage |
|----------|--------|-------|
| `SUMMARY_EXCHANGES_THRESHOLD` | `5` | Seuil en nombre de messages |
| `SUMMARY_TOKEN_THRESHOLD` | `1200` | Seuil en tokens |

---

## Décisions de conception

| Décision | Justification |
|----------|--------------|
| **`extract_exchanges` séparée** | `Chat.chat` est un blob JSON libre — extraction défensive nécessaire |
| **Dual trigger** | Un chat court mais dense en tokens doit aussi être résumé |
| **BackgroundTask** | Ne bloque pas la réponse HTTP du frontend |
| **Filtre Python sur JSON** | `JSON_EXTRACT` SQLite peu fiable cross-version ; filtre Python plus sûr |
| **Fallback minimal** | Évite les retry storms : un résumé vide serait re-tenté en boucle |
| **`support_id` = `chat.meta["support_id"]` ?? `chat_id`** | Le support peut ne pas être renseigné dans `meta` |

---

## Intégration avec Phase 1 & 2

- Écrit dans `opentutorai_memory` (Phase 1) avec `memory_type=session_summary`
- `ContextRetrievalService.retrieve_internal_memory(memory_types=["session_summary"])` permet de récupérer ces résumés (Phase 2)
- `SummarizationService.get_cached_summary()` est le point d'entrée direct pour Phase 4

---

## Phase suivante

**Phase 4 — Dynamic Context Manager** : `ContextManager.build_agent_context()` fusionne
documents RAG + mémoires + résumé de session (`get_cached_summary()`) en un seul `AgentContext`,
filtré par pertinence, récence et budget token.
