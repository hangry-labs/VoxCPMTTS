from __future__ import annotations

import asyncio

import httpx
import pytest

from voxcpm.remote_audio import RemoteAudioError, RemoteAudioFetcher


def test_remote_audio_download_is_bounded() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"12345", request=request))
    fetcher = RemoteAudioFetcher(
        max_bytes=4,
        allowed_hosts=["audio.example"],
        transport=transport,
        resolver=lambda host, port: ["8.8.8.8"],
    )
    with pytest.raises(RemoteAudioError, match="limit"):
        asyncio.run(fetcher.fetch("https://audio.example/reference.wav"))


def test_remote_audio_redirect_destination_is_revalidated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            302,
            headers={"location": "http://169.254.169.254/metadata.wav"},
            request=request,
        )

    fetcher = RemoteAudioFetcher(
        max_bytes=1024,
        allowed_hosts=["*"],
        transport=httpx.MockTransport(handler),
        resolver=lambda host, port: [host if host[0].isdigit() else "8.8.8.8"],
    )
    with pytest.raises(RemoteAudioError, match="link-local"):
        asyncio.run(fetcher.fetch("https://audio.example/reference.wav"))
