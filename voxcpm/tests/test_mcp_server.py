from __future__ import annotations

import asyncio
from array import array
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from voxcpm.artifacts import ArtifactStore
from voxcpm.mcp_server import attach_mcp, create_mcp_server
from voxcpm.settings import RuntimeSettingsStore


class FakeTTSRequest(BaseModel):
    text: str
    input_type: str = "text"
    language: str = "auto"
    voice_profile: str | None = None
    control: str | None = None
    ref_audio: str | None = None
    ref_text: str | None = None
    clone_mode: str = "auto"
    output_format: str = "mp3"
    seed: int | None = 42
    randomize_seed: bool = False
    cfg_value: float = 2.0
    inference_timesteps: int = 10
    normalize_text: bool = False
    normalize_loudness: bool = False
    protect_long_audio: bool = True
    denoise: bool = False


class FakeRuntime:
    APP_VERSION = "test"
    SUPPORTED_LANGUAGES = ["English", "Polish"]
    MAX_RANDOM_SEED = 2**32 - 1
    OUTPUT_FORMATS = {"mp3": {"extension": "mp3", "media_type": "audio/mpeg"}}
    REFERENCE_AUDIO_SUFFIXES = {".wav", ".mp3", ".opus"}
    TTSRequest = FakeTTSRequest

    def __init__(self, root: Path) -> None:
        self.RUNTIME_SETTINGS = RuntimeSettingsStore(root / "settings.json")
        self.VOICE_PROFILE_DIR = root / "voices"
        self.GPU_MONITOR = SimpleNamespace(request_snapshot=lambda: {"gpus": []})

    @staticmethod
    def get_status_payload():
        return {"msg": "pong"}

    @staticmethod
    def voice_profile_payloads():
        return []

    @staticmethod
    def normalize_profile_name(name: str) -> str:
        return name.strip().lower()

    @staticmethod
    def synthesize_complete_payload(payload):
        return payload.output_format, 24_000, array("f", [0.0] * 24_000), 42, []

    @staticmethod
    def encode_audio_bytes(waveform, output_format, sample_rate):
        return b"mp3-data"


def simple_arguments(**overrides):
    values = {"text": "Hello from MCP.", "language": "auto", "voice_id": "auto", "ttl_seconds": 300}
    values.update(overrides)
    return values


def test_compact_and_advanced_tool_catalogs_are_separate(tmp_path) -> None:
    runtime = FakeRuntime(tmp_path)
    store = ArtifactStore(tmp_path / "output")
    with patch("voxcpm.mcp_server._RUNTIME", runtime):
        compact = create_mcp_server(
            artifact_store=store,
            base_url="http://testserver",
            advanced=False,
            input_directory=tmp_path / "input",
        )
        advanced = create_mcp_server(
            artifact_store=store,
            base_url="http://testserver",
            advanced=True,
            input_directory=tmp_path / "input",
        )
        compact_tools = {tool.name: tool for tool in asyncio.run(compact.list_tools())}
        advanced_tools = {tool.name: tool for tool in asyncio.run(advanced.list_tools())}

    core = {"get_health", "find_languages", "get_saved_voices_catalog", "inspect_speech", "talk_simple"}
    assert set(compact_tools) == core
    assert set(advanced_tools) == core | {
        "talk_advanced",
        "create_cloned_voice_profile",
        "create_designed_voice_profile",
        "delete_saved_voice",
    }
    simple_schema = compact_tools["talk_simple"].input_schema
    assert set(simple_schema["properties"]) == set(simple_arguments())
    assert simple_schema["required"] == ["text"]
    assert simple_schema["properties"]["ttl_seconds"]["default"] == 3600
    assert "verbatim" in simple_schema["properties"]["text"]["description"]


def test_simple_generation_returns_expiring_link_not_audio(tmp_path) -> None:
    runtime = FakeRuntime(tmp_path)
    store = ArtifactStore(tmp_path / "output")
    with patch("voxcpm.mcp_server._RUNTIME", runtime):
        server = create_mcp_server(
            artifact_store=store,
            base_url="http://testserver",
            advanced=False,
            input_directory=tmp_path / "input",
        )
        result = asyncio.run(server.call_tool("talk_simple", simple_arguments()))

    payload = result.structured_content
    token = payload["download_url"].rsplit("/", 1)[1]
    artifact, path = store.resolve(token)
    assert not result.is_error
    assert path.read_bytes() == b"mp3-data"
    assert artifact.mime_type == "audio/mpeg"
    assert "text" not in payload
    assert "data" not in payload


def test_compact_and_advanced_gates_follow_persisted_settings(tmp_path) -> None:
    runtime = FakeRuntime(tmp_path)
    store = ArtifactStore(tmp_path / "output")
    app = FastAPI()
    with patch.dict(
        "os.environ",
        {"VOXCPM_MCP_ALLOWED_HOSTS": "testserver", "VOXCPM_MCP_BASE_URL": "http://testserver"},
    ):
        attach_mcp(api_app=app, runtime=runtime, artifact_store=store)
        request = {
            "headers": {
                "Accept": "application/json, text/event-stream",
                "MCP-Protocol-Version": "2025-06-18",
            },
            "json": {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0"},
                },
            },
        }
        with TestClient(app) as client:
            assert client.post("/mcp/", **request).status_code == 403
            runtime.RUNTIME_SETTINGS.set_mcp_access(enabled=True, advanced_enabled=False)
            compact = client.post("/mcp/", **request)
            assert client.post("/mcp/advanced/", **request).status_code == 403
            runtime.RUNTIME_SETTINGS.set_mcp_access(enabled=True, advanced_enabled=True)
            advanced = client.post("/mcp/advanced/", **request)

    assert compact.status_code == 200
    assert advanced.status_code == 200
    assert compact.json()["result"]["serverInfo"]["name"] == "voxcpmtts-compact"
    assert advanced.json()["result"]["serverInfo"]["name"] == "voxcpmtts-advanced"
