from __future__ import annotations

import asyncio
import os
import urllib.request
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

COMPACT_TOOLS = {
    "get_health",
    "find_languages",
    "get_saved_voices_catalog",
    "inspect_speech",
    "talk_simple",
}
ADVANCED_TOOLS = COMPACT_TOOLS | {
    "talk_advanced",
    "create_cloned_voice_profile",
    "create_designed_voice_profile",
    "delete_saved_voice",
}


def payload(result: Any, action: str) -> dict[str, Any]:
    if getattr(result, "is_error", False):
        detail = "\n".join(str(getattr(item, "text", "")) for item in result.content)
        raise AssertionError(f"{action} failed: {detail}")
    structured = getattr(result, "structured_content", None)
    if not isinstance(structured, dict):
        raise AssertionError(f"{action} returned no structured result")
    return structured


async def tool_names(url: str) -> set[str]:
    async with (
        streamable_http_client(url) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        return {tool.name for tool in (await session.list_tools()).tools}


async def main() -> int:
    compact_url = os.getenv("LOCAL_MCP_URL", "http://127.0.0.1:8808/mcp/")
    advanced_url = os.getenv("LOCAL_ADVANCED_MCP_URL", "http://127.0.0.1:8808/mcp/advanced/")
    async with (
        streamable_http_client(compact_url) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        await session.initialize()
        tools = {tool.name: tool for tool in (await session.list_tools()).tools}
        if set(tools) != COMPACT_TOOLS:
            raise AssertionError(f"unexpected compact tools: {sorted(tools)}")
        if tools["talk_simple"].input_schema.get("required", []) != ["text"]:
            raise AssertionError("talk_simple should require only text")
        health = payload(await session.call_tool("get_health", arguments={}), "health")
        if health["mcp"]["tier"] != "compact":
            raise AssertionError("compact endpoint reported the wrong MCP tier")
        speech = payload(
            await session.call_tool(
                "talk_simple",
                arguments={"text": "The compact VoxCPMTTS MCP endpoint is working correctly."},
            ),
            "simple speech",
        )
        if any(key in speech for key in ("audio", "audio_bytes", "data", "base64")):
            raise AssertionError("MCP generation unexpectedly returned raw audio")
        with urllib.request.urlopen(speech["download_url"], timeout=30) as response:
            downloaded = response.read()
            content_type = response.headers.get_content_type()
        if len(downloaded) < 1_000 or content_type != "audio/mpeg":
            raise AssertionError(f"temporary MP3 was invalid: {len(downloaded)} bytes, {content_type}")

    advanced_tools = await tool_names(advanced_url)
    if advanced_tools != ADVANCED_TOOLS:
        raise AssertionError(f"unexpected advanced tools: {sorted(advanced_tools)}")
    print("PASS MCP catalogs, linked audio generation, and capability download")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
