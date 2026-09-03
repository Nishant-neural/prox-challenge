"""
FastAPI Application
===================
Endpoints:

  POST /api/chat          — main SSE streaming chat
  GET  /api/health        — health + store stats
  GET  /api/image/{path}  — serve extracted images
  GET  /api/page/{num}    — serve page screenshots
  GET  /api/tables        — list all ingested tables
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.config import settings
from backend.agent import OmniProAgent
from backend.retrieval.vector_store import VectorStore
from backend.retrieval.table_store import TableStore

# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Vulcan OmniPro 220 Expert Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-initialised singletons ────────────────────────────────────────────────
# Stores are expensive to init (model load + DB connect) so we do it once.

_vector_store: VectorStore | None = None
_table_store: TableStore | None = None
_agent: OmniProAgent | None = None


def get_stores():
    global _vector_store, _table_store, _agent
    if _vector_store is None:
        _vector_store = VectorStore()
        _table_store = TableStore()
        _agent = OmniProAgent(_vector_store, _table_store)
    return _vector_store, _table_store, _agent


# ═══════════════════════════════════════════════════════════════════════════════
# Request / response models
# ═══════════════════════════════════════════════════════════════════════════════

class Message(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    SSE streaming chat endpoint.
    
    Events emitted:
      data: {"type": "text_delta",  "content": "..."}
      data: {"type": "tool_call",   "name": "...", "input": {...}}
      data: {"type": "tool_result", "name": "...", "result": {...}}
      data: {"type": "artifact",    "artifact_type": "...", "data": {...}}
      data: {"type": "done",        "usage": {...}, "artifacts": [...]}
      data: {"type": "error",       "message": "..."}
    """
    _, _, agent = get_stores()

    history = [{"role": m.role, "content": m.content} for m in request.history]

    async def event_generator():
        async for event in agent.run(request.message, history):
            yield {"data": json.dumps(event)}

    return EventSourceResponse(event_generator())


@app.get("/api/health")
def health():
    try:
        vs, ts, _ = get_stores()
        stats = vs.collection_stats()
        tables = ts.get_all_tables_summary()
        return {
            "status": "ok",
            "vector_store": stats,
            "table_count": len(tables),
        }
    except Exception as exc:
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(exc)})


@app.get("/api/image/{image_id}")
def get_image(image_id: str):
    """Serve a raw extracted image by its image_id."""
    # Search images directory for matching file
    for ext in ["png", "jpg", "jpeg", "bmp", "webp"]:
        path = settings.images_dir / f"{image_id}.{ext}"
        if path.exists():
            return FileResponse(str(path))
    raise HTTPException(status_code=404, detail=f"Image {image_id} not found")


@app.get("/api/page/{page_num}")
def get_page_screenshot(page_num: int):
    """Serve a rendered page screenshot."""
    path = settings.screenshots_dir / f"page_{page_num:04d}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Page {page_num} screenshot not found")
    return FileResponse(str(path))


@app.get("/api/tables")
def list_tables():
    """List all ingested tables with page and section metadata."""
    _, ts, _ = get_stores()
    return ts.get_all_tables_summary()


@app.get("/api/tables/{table_id}")
def get_table(table_id: str):
    """Get a specific table by ID."""
    _, ts, _ = get_stores()
    table = ts.get_table_by_id(table_id)
    if not table:
        raise HTTPException(status_code=404, detail=f"Table {table_id} not found")
    return table


@app.get("/api/metadata")
def get_metadata():
    if not settings.metadata_path.exists():
        raise HTTPException(status_code=404, detail="Metadata not found - run ingest first")
    return JSONResponse(content=json.loads(settings.metadata_path.read_text(encoding="utf-8")))


# ── Dev entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
