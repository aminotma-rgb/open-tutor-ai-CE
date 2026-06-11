"""Register OTAI pedagogical tools in the DB at application startup."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def register_otai_tools() -> None:
    """Upsert the 4 OTAI tools. Silent on any error — must not block startup."""
    try:
        from data.database import get_db
        from ai.tools.service import ToolsService
        from ai.tools.owui_tools import OTAI_TOOLS

        db = next(get_db())
        try:
            svc = ToolsService(db)
            for tool_def in OTAI_TOOLS:
                svc.upsert(
                    tool_id=tool_def["id"],
                    name=tool_def["name"],
                    content=tool_def["content"],
                    meta=tool_def["meta"],
                )
            log.info("OTAI tools registered: %s", [t["id"] for t in OTAI_TOOLS])
        finally:
            db.close()
    except Exception as exc:
        log.warning("OTAI tools registration failed (non-blocking): %s", exc)
