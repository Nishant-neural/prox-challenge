"""
Agent Tool Definitions
======================
Five tools the Claude orchestrator can call:

  1. search_manual      — semantic search over text chunks
  2. lookup_table       — exact/fuzzy query over SQLite structured tables
  3. find_diagram       — semantic search over image captions
  4. get_manual_page    — return a page screenshot path + surrounding text
  5. generate_artifact  — signal the frontend to render an interactive widget

Each tool is defined as:
  • A JSON schema for the Anthropic tool_use API
  • An executor function that takes (args, stores) → result dict
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from backend.config import settings

# ── Tool schemas (Anthropic tool_use format) ───────────────────────────────────

TOOL_SCHEMAS = [
    {
        "name": "search_manual",
        "description": (
            "Semantically search the Vulcan OmniPro 220 manual text for relevant passages. "
            "Use this for conceptual questions, setup instructions, safety information, "
            "process explanations, and troubleshooting steps."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query describing what information is needed.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of passages to return (default 5, max 10).",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "lookup_table",
        "description": (
            "Query structured welding tables for exact parameter values: "
            "duty cycles, recommended voltage/current, wire feed speed, gas mix, "
            "material settings. Use when the user asks for specific numbers or settings."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "process": {
                    "type": "string",
                    "enum": ["MIG", "TIG", "Stick", "Flux Core"],
                    "description": "Welding process to filter by.",
                },
                "material": {
                    "type": "string",
                    "description": "Base metal (e.g. steel, stainless, aluminum).",
                },
                "thickness_mm": {
                    "type": "number",
                    "description": "Material thickness in millimetres.",
                },
                "voltage": {
                    "type": "number",
                    "description": "Target voltage (volts).",
                },
                "current": {
                    "type": "number",
                    "description": "Target current (amps).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "find_diagram",
        "description": (
            "Search for relevant diagrams, schematics, wiring diagrams, polarity diagrams, "
            "or welding example photos from the manual. "
            "Use when the user asks how something looks or how to wire something."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Description of the diagram or image to find.",
                },
                "image_type": {
                    "type": "string",
                    "enum": ["diagram", "photo", "schematic", "wiring", "weld_sample", "unknown"],
                    "description": "Optional filter by image type.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of images to return (default 3).",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_manual_page",
        "description": (
            "Retrieve the screenshot of a specific manual page for citation. "
            "Use when you know the page number from a previous search result "
            "and want to show the user the source page."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "page": {
                    "type": "integer",
                    "description": "Page number (1-indexed).",
                },
            },
            "required": ["page"],
        },
    },
    {
        "name": "generate_artifact",
        "description": (
            "Instruct the frontend to render an interactive artifact widget. "
            "Use when a static text answer is insufficient and an interactive tool "
            "would better serve the user. "
            "Available types: duty_cycle_calculator, polarity_diagram, "
            "settings_configurator, troubleshooting_wizard, wire_feed_explainer, "
            "duty_cycle_visualizer."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "artifact_type": {
                    "type": "string",
                    "enum": [
                        "duty_cycle_calculator",
                        "polarity_diagram",
                        "settings_configurator",
                        "troubleshooting_wizard",
                        "wire_feed_explainer",
                        "duty_cycle_visualizer",
                    ],
                    "description": "Which interactive widget to render.",
                },
                "data": {
                    "type": "object",
                    "description": (
                        "Seed data for the artifact. "
                        "For polarity_diagram: {process, polarity, connections:[]}. "
                        "For settings_configurator: {process, material, thickness_mm}. "
                        "For troubleshooting_wizard: {symptom, possible_causes:[]}. "
                        "For duty_cycle_calculator: {current, voltage, duty_cycle_pct}. "
                        "For duty_cycle_visualizer: {duty_cycle_pct, weld_minutes, cool_minutes}."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Human-readable title for the artifact panel.",
                },
            },
            "required": ["artifact_type", "data"],
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Tool executors
# ═══════════════════════════════════════════════════════════════════════════════

class ToolExecutor:
    """
    Holds references to the retrieval stores and executes tool calls.
    Instantiated once and shared across the request lifecycle.
    """

    def __init__(self, vector_store, table_store):
        self._vs = vector_store
        self._ts = table_store

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        dispatch = {
            "search_manual": self._search_manual,
            "lookup_table": self._lookup_table,
            "find_diagram": self._find_diagram,
            "get_manual_page": self._get_manual_page,
            "generate_artifact": self._generate_artifact,
        }
        fn = dispatch.get(tool_name)
        if not fn:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return fn(tool_input)
        except Exception as exc:
            return {"error": str(exc)}

    # ── search_manual ──────────────────────────────────────────────────────────

    def _search_manual(self, args: dict) -> dict:
        query = args["query"]
        top_k = min(args.get("top_k", 5), 10)
        hits = self._vs.search_text(query, top_k=top_k)
        return {
            "query": query,
            "results": [
                {
                    "chunk_id": h.payload.get("chunk_id"),
                    "page": h.page,
                    "section": h.section,
                    "content": h.content,
                    "score": round(h.score, 4),
                }
                for h in hits
            ],
        }

    # ── lookup_table ───────────────────────────────────────────────────────────

    def _lookup_table(self, args: dict) -> dict:
        rows = self._ts.query(
            process=args.get("process"),
            material=args.get("material"),
            thickness_mm=args.get("thickness_mm"),
            voltage=args.get("voltage"),
            current=args.get("current"),
        )
        return {
            "query_params": args,
            "results": rows,
            "count": len(rows),
        }

    # ── find_diagram ───────────────────────────────────────────────────────────

    def _find_diagram(self, args: dict) -> dict:
        query = args["query"]
        top_k = min(args.get("top_k", 3), 6)
        image_type = args.get("image_type")
        hits = self._vs.search_images(query, top_k=top_k, image_type=image_type)
        results = []
        for h in hits:
            payload = h.payload
            img_path = settings.knowledge_dir / payload.get("image_path", "")
            # Return base64 for inline display + metadata
            b64 = None
            if img_path.exists():
                with open(img_path, "rb") as f:
                    b64 = base64.standard_b64encode(f.read()).decode()
            results.append({
                "image_id": payload.get("image_id"),
                "page": payload.get("page"),
                "image_type": payload.get("image_type"),
                "caption": payload.get("caption"),
                "description": payload.get("description"),
                "keywords": payload.get("keywords", []),
                "image_path": payload.get("image_path"),
                "image_b64": b64,
                "score": round(h.score, 4),
            })
        return {"query": query, "results": results}

    # ── get_manual_page ────────────────────────────────────────────────────────

    def _get_manual_page(self, args: dict) -> dict:
        page = args["page"]
        screenshot_path = settings.screenshots_dir / f"page_{page:04d}.png"
        if not screenshot_path.exists():
            return {"error": f"Screenshot for page {page} not found. Run the ingestion pipeline first."}

        with open(screenshot_path, "rb") as f:
            b64 = base64.standard_b64encode(f.read()).decode()

        # Also pull text chunks from this page for context
        hits = self._vs.search_text(f"page {page}", top_k=3, page_filter=page)
        context_text = "\n\n".join(h.content for h in hits)

        return {
            "page": page,
            "screenshot_b64": b64,
            "screenshot_path": str(screenshot_path.relative_to(settings.knowledge_dir.parent)),
            "page_text_preview": context_text[:500] if context_text else "",
        }

    # ── generate_artifact ──────────────────────────────────────────────────────

    def _generate_artifact(self, args: dict) -> dict:
        """
        This tool doesn't actually render anything server-side.
        It returns a structured artifact spec that the frontend interprets.
        """
        return {
            "artifact_type": args["artifact_type"],
            "title": args.get("title", args["artifact_type"].replace("_", " ").title()),
            "data": args.get("data", {}),
            "__render_artifact": True,   # sentinel for frontend
        }
