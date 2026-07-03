"""
Claude Agent Orchestrator
=========================
Implements a multi-turn agentic loop using the Anthropic tool_use API.

The agent:
  1. Receives the user message + conversation history
  2. Decides which tools to call (may call multiple in parallel rounds)
  3. Executes the tools via ToolExecutor
  4. Feeds results back to Claude
  5. Repeats until Claude emits a final text response (stop_reason = "end_turn")

Yields SSE events so the frontend can stream the response live.

Key design decisions
--------------------
• Tool results are injected as "tool_result" content blocks per the Anthropic spec.
• Artifacts are extracted from generate_artifact tool results and forwarded as a
  separate SSE event type so the frontend can render them alongside the text.
• The system prompt gives Claude a detailed persona and decision framework.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import AsyncGenerator, Any

import anthropic

sys.path.insert(0, str(Path(__file__).parent.parent))
from backend.config import settings
from backend.tools.manual_tools import TOOL_SCHEMAS, ToolExecutor

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are the Vulcan OmniPro 220 Expert Assistant — a digital technician with deep knowledge
of this multi-process welder (MIG, Flux Core, TIG, Stick).

## Your Persona
You think and communicate like an experienced welder standing next to the user in their shop.
You are precise, safety-conscious, and always cite the manual when giving specifications.

## Decision Framework — Think before responding
For every user question, decide which combination of tools to call:

• Factual / conceptual question → search_manual
• Specific parameter request (voltage, amps, wire speed) → lookup_table
• "How does X look?" / wiring / polarity → find_diagram
• "Show me page N" / source citation → get_manual_page
• Complex setup / troubleshooting / settings → generate_artifact (after gathering data)
• Ambiguous question → ask ONE clarifying question before calling tools

## Tool calling rules
- Call tools in a logical sequence: gather text context first, then images, then generate artifacts.
- You may call multiple tools in a single round.
- Always call find_diagram when discussing wiring, polarity, or physical setup.
- Always call generate_artifact for settings questions (welding 3mm steel → settings_configurator).
- Always call generate_artifact for troubleshooting (porosity, spatter, arc issues → troubleshooting_wizard).
- Always call generate_artifact for duty cycle questions → duty_cycle_calculator.
- Cite page numbers for every claim you make using manual search results.

## Response format
- Lead with the most important information.
- Use numbered steps for procedures.
- Call out ⚠️ SAFETY warnings prominently.
- At the end of your response, list: "Sources: Page X, Page Y"
- Never make up specifications. If the manual doesn't cover it, say so.

## What you know about this machine
- Model: Vulcan OmniPro 220
- Processes: MIG, Flux Core, DC TIG, Stick
- Input power: 120V/240V dual voltage
- Output: 20-220A
- Wire sizes: 0.023"–0.045" solid, 0.030"–0.045" flux core
- Gas: 75/25 Ar/CO2 for MIG; 100% Argon for TIG
""".strip()


# ═══════════════════════════════════════════════════════════════════════════════
# Agent
# ═══════════════════════════════════════════════════════════════════════════════

class OmniProAgent:
    def __init__(self, vector_store, table_store):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._executor = ToolExecutor(vector_store, table_store)

    # ── Main entry point ───────────────────────────────────────────────────────

    async def run(
        self,
        user_message: str,
        history: list[dict],
    ) -> AsyncGenerator[dict, None]:
        """
        Async generator that yields SSE-compatible event dicts:

          {"type": "text_delta",   "content": "..."}
          {"type": "tool_call",    "name": "...", "input": {...}}
          {"type": "tool_result",  "name": "...", "result": {...}}
          {"type": "artifact",     "artifact_type": "...", "data": {...}, "title": "..."}
          {"type": "done",         "usage": {...}}
        """
        messages = list(history)
        messages.append({"role": "user", "content": user_message})

        artifacts: list[dict] = []
        iteration = 0

        while iteration < settings.max_agent_iterations:
            iteration += 1
            response = self._client.messages.create(
                model=settings.claude_model,
                max_tokens=4096,
                temperature=settings.agent_temperature,
                system=SYSTEM_PROMPT,
                tools=TOOL_SCHEMAS,
                messages=messages,
            )

            # ── Process response content blocks ────────────────────────────────
            tool_use_blocks: list[dict] = []
            text_so_far = ""

            for block in response.content:
                if block.type == "text":
                    text_so_far += block.text
                    yield {"type": "text_delta", "content": block.text}

                elif block.type == "tool_use":
                    yield {
                        "type": "tool_call",
                        "name": block.name,
                        "input": block.input,
                        "tool_use_id": block.id,
                    }
                    tool_use_blocks.append({
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    })

            # ── If no tool calls → Claude is done ─────────────────────────────
            if response.stop_reason == "end_turn" or not tool_use_blocks:
                yield {
                    "type": "done",
                    "usage": {
                        "input_tokens": response.usage.input_tokens,
                        "output_tokens": response.usage.output_tokens,
                    },
                    "artifacts": artifacts,
                }
                return

            # ── Execute all tool calls ─────────────────────────────────────────
            # Append the assistant turn with tool_use blocks
            messages.append({
                "role": "assistant",
                "content": response.content,   # raw list of blocks
            })

            tool_results_content = []
            for tool_call in tool_use_blocks:
                result = self._executor.execute(tool_call["name"], tool_call["input"])

                # Intercept artifact specs — send to frontend separately
                if result.get("__render_artifact"):
                    artifact_event = {
                        "type": "artifact",
                        "artifact_type": result["artifact_type"],
                        "title": result["title"],
                        "data": result["data"],
                    }
                    artifacts.append(artifact_event)
                    yield artifact_event
                    # Still give Claude a confirmation it was generated
                    result_for_claude = {
                        "status": "artifact_generated",
                        "artifact_type": result["artifact_type"],
                        "title": result["title"],
                    }
                else:
                    result_for_claude = result

                yield {
                    "type": "tool_result",
                    "name": tool_call["name"],
                    "tool_use_id": tool_call["id"],
                    "result": _truncate_for_log(result_for_claude),
                }

                # Serialize result for Claude (strip base64 to save tokens)
                tool_results_content.append({
                    "type": "tool_result",
                    "tool_use_id": tool_call["id"],
                    "content": json.dumps(_strip_binary(result_for_claude)),
                })

            # Append tool results and loop
            messages.append({"role": "user", "content": tool_results_content})

        yield {"type": "error", "message": "Max iterations reached without a final answer."}


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _strip_binary(obj: Any) -> Any:
    """Remove base64 image data from tool results before feeding back to Claude (saves tokens)."""
    if isinstance(obj, dict):
        return {
            k: ("[base64_image_omitted]" if k.endswith("_b64") else _strip_binary(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_strip_binary(i) for i in obj]
    return obj


def _truncate_for_log(obj: Any, max_len: int = 300) -> Any:
    """Truncate for the SSE log event so we don't flood the client."""
    s = json.dumps(obj)
    if len(s) > max_len:
        return {"_truncated": s[:max_len] + "…"}
    return obj
