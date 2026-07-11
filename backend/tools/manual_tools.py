"""
Tool definitions and executor.
Schemas drive the Anthropic tool_use API.
Executor calls into the retrieval layer.
"""
from __future__ import annotations
import base64
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.retrieval import vector_store as vs
from backend.retrieval import table_store as ts


TOOL_SCHEMAS = [
    {
        "name": "search_manual",
        "description": "Semantic search over Vulcan OmniPro 220 manual text. Use for "
                       "concepts, setup steps, safety info, and troubleshooting.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "lookup_table",
        "description": "Exact/fuzzy query over structured welding tables (duty cycles, "
                       "voltage/current settings, wire speed). Use for specific numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "process":      {"type": "string", "enum": ["MIG", "TIG", "Stick", "Flux Core"]},
                "material":     {"type": "string"},
                "thickness_mm": {"type": "number"},
                "voltage":      {"type": "number"},
                "current":      {"type": "number"},
            },
        },
    },
    {
        "name": "find_diagram",
        "description": "Find diagrams, schematics, wiring diagrams, or photos from the manual.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query":      {"type": "string"},
                "image_type": {"type": "string",
                               "enum": ["diagram","photo","schematic","wiring","weld_sample"]},
                "top_k":      {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_manual_page",
        "description": "Retrieve a specific page screenshot for citation.",
        "input_schema": {
            "type": "object",
            "properties": {"page": {"type": "integer"}},
            "required": ["page"],
        },
    },
    {
        "name": "generate_artifact",
        "description": "Signal the frontend to render an interactive widget. "
                       "Types: duty_cycle_calculator, polarity_diagram, "
                       "settings_configurator, troubleshooting_wizard, wire_feed_explainer.",
        "input_schema": {
            "type": "object",
            "properties": {
                "artifact_type": {"type": "string",
                                  "enum": ["duty_cycle_calculator", "polarity_diagram",
                                           "settings_configurator", "troubleshooting_wizard",
                                           "wire_feed_explainer"]},
                "data":  {"type": "object"},
                "title": {"type": "string"},
            },
            "required": ["artifact_type", "data"],
        },
    },
]


# ── Executor ───────────────────────────────────────────────────────────────────

def execute(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    handlers = {
        "search_manual":   _search_manual,
        "lookup_table":    _lookup_table,
        "find_diagram":    _find_diagram,
        "get_manual_page": _get_manual_page,
        "generate_artifact": _generate_artifact,
    }
    return handlers[tool_name](args)


def _search_manual(args: dict) -> dict:
    docs = vs.search_text(args["query"], k=min(args.get("top_k", 5), 10))
    return {"results": [{"page": d.metadata.get("page"), "section": d.metadata.get("section"),
                         "content": d.page_content} for d in docs]}


def _lookup_table(args: dict) -> dict:
    return {"results": ts.query(**{k: v for k, v in args.items() if v is not None})}


def _find_diagram(args: dict) -> dict:
    docs = vs.search_images(args["query"], k=min(args.get("top_k", 3), 6))
    results = []
    for d in docs:
        img_path = settings.knowledge_dir / d.metadata.get("image_path", "")
        b64 = base64.b64encode(img_path.read_bytes()).decode() if img_path.exists() else None
        results.append({**d.metadata, "image_b64": b64})
    return {"results": results}


def _get_manual_page(args: dict) -> dict:
    path = settings.screenshots_dir / f"page_{args['page']:04d}.png"
    if not path.exists():
        return {"error": f"Page {args['page']} screenshot not found — run ingest first"}
    return {"page": args["page"], "screenshot_b64": base64.b64encode(path.read_bytes()).decode()}


def _generate_artifact(args: dict) -> dict:
    return {**args, "__render_artifact": True}
