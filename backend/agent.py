"""
Claude agent loop with tool execution and SSE-friendly events.
"""
from __future__ import annotations

import json
from typing import Any, AsyncGenerator

import anthropic

from backend.config import settings
from backend.tools.manual_tools import TOOL_SCHEMAS, execute

_client = anthropic.Anthropic(api_key=settings.anthropic_api_key) if settings.anthropic_api_key else None

SYSTEM = """
You are the Vulcan OmniPro 220 Expert, a digital welding technician.
Think like an experienced welder: precise, safety-conscious, cite the manual.

Decision rules:
- Concept or procedure question -> search_manual
- Specific numbers (V, A, %) -> lookup_table
- Wiring, polarity, diagrams, or photos -> find_diagram
- "Show page N" -> get_manual_page
- Settings or troubleshooting -> generate_artifact after gathering data
- Ambiguous -> ask one clarifying question

Always cite page numbers and surface safety warnings clearly.
""".strip()


async def run(message: str, history: list[dict]) -> AsyncGenerator[dict, None]:
    if _client is None:
        yield {"type": "error", "message": "ANTHROPIC_API_KEY is not configured."}
        return

    messages = [*history, {"role": "user", "content": message}]

    for _ in range(settings.max_agent_iterations):
        response = _client.messages.create(
            model=settings.claude_model,
            max_tokens=4096,
            temperature=settings.agent_temperature,
            system=SYSTEM,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        tool_calls = []
        for block in response.content:
            if block.type == "text":
                yield {"type": "text_delta", "content": block.text}
            elif block.type == "tool_use":
                yield {"type": "tool_call", "name": block.name, "input": block.input}
                tool_calls.append(block)

        if response.stop_reason == "end_turn" or not tool_calls:
            yield {"type": "done", "usage": response.usage.model_dump()}
            return

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for call in tool_calls:
            result = execute(call.name, call.input)

            if result.pop("__render_artifact", False):
                yield {"type": "artifact", **result}
                result = {"status": "artifact_queued", "type": result.get("artifact_type")}

            yield {"type": "tool_result", "name": call.name, "result": _slim(result)}
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": json.dumps(_strip_b64(result)),
                }
            )

        messages.append({"role": "user", "content": tool_results})

    yield {"type": "error", "message": "Max iterations reached"}


def _strip_b64(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: "[image]" if k.endswith("_b64") else _strip_b64(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_b64(item) for item in obj]
    return obj


def _slim(obj: Any, max_chars: int = 300) -> Any:
    serialized = json.dumps(obj)
    if len(serialized) <= max_chars:
        return obj
    return {"_preview": serialized[:max_chars] + "..."}


class OmniProAgent:
    def __init__(self, *_args, **_kwargs):
        pass

    async def run(self, message: str, history: list[dict]) -> AsyncGenerator[dict, None]:
        async for event in run(message, history):
            yield event
