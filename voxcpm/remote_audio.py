"""Bounded HTTP(S) retrieval for MCP reference-audio inputs."""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import httpx


class RemoteAudioError(ValueError):
    """An actionable failure while retrieving remote audio."""


@dataclass(frozen=True)
class DownloadedAudio:
    payload: bytes
    filename: str
    content_type: str | None


AddressResolver = Callable[[str, int], Iterable[str]]


def _system_resolver(host: str, port: int) -> list[str]:
    try:
        return [str(ipaddress.ip_address(host))]
    except ValueError:
        pass
    addresses = {result[4][0] for result in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)}
    return sorted(addresses)


def _is_blocked_address(value: str) -> bool:
    address = ipaddress.ip_address(value.split("%", 1)[0])
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return bool(address.is_link_local or address.is_multicast or address.is_unspecified or address.is_reserved)


def _host_is_allowed(host: str, port: int, patterns: tuple[str, ...]) -> bool:
    normalized = host.casefold().rstrip(".")
    authority = f"[{normalized}]:{port}" if ":" in normalized else f"{normalized}:{port}"
    for raw_pattern in patterns:
        pattern = raw_pattern.casefold().strip().rstrip(".")
        if pattern == "*" or pattern in {normalized, authority}:
            return True
        if pattern.startswith("*."):
            suffix = pattern[1:]
            if normalized.endswith(suffix) and normalized != suffix[1:]:
                return True
    return False


class RemoteAudioFetcher:
    REDIRECT_STATUSES = {301, 302, 303, 307, 308}

    def __init__(
        self,
        *,
        max_bytes: int,
        allowed_hosts: Iterable[str],
        timeout_seconds: float = 60.0,
        max_redirects: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
        resolver: AddressResolver | None = None,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("Remote audio download limit must be greater than zero")
        if timeout_seconds <= 0:
            raise ValueError("Remote audio timeout must be greater than zero")
        if max_redirects < 0:
            raise ValueError("Remote audio redirect limit cannot be negative")
        self.max_bytes = max_bytes
        self.allowed_hosts = tuple(item.strip() for item in allowed_hosts if item.strip())
        self.timeout_seconds = timeout_seconds
        self.max_redirects = max_redirects
        self.transport = transport
        self.resolver = resolver or _system_resolver

    @property
    def enabled(self) -> bool:
        return bool(self.allowed_hosts)

    async def _validated_url(self, value: str | httpx.URL) -> httpx.URL:
        try:
            url = value if isinstance(value, httpx.URL) else httpx.URL(value.strip())
            port = url.port
        except (httpx.InvalidURL, ValueError) as exc:
            raise RemoteAudioError("audio_location must be a valid HTTP or HTTPS URL") from exc
        if url.scheme not in {"http", "https"}:
            raise RemoteAudioError("audio_location must use http:// or https://")
        if not url.host:
            raise RemoteAudioError("audio_location must include a hostname")
        if url.username or url.password:
            raise RemoteAudioError("audio_location must not contain embedded credentials")
        resolved_port = int(port or (443 if url.scheme == "https" else 80))
        if not _host_is_allowed(url.host, resolved_port, self.allowed_hosts):
            raise RemoteAudioError("The audio URL host is not allowed by VOXCPM_MCP_AUDIO_URL_ALLOWED_HOSTS")
        try:
            addresses = await asyncio.to_thread(lambda: list(self.resolver(str(url.host), resolved_port)))
        except (OSError, ValueError) as exc:
            raise RemoteAudioError("The audio URL hostname could not be resolved") from exc
        if not addresses:
            raise RemoteAudioError("The audio URL hostname resolved to no addresses")
        try:
            blocked = any(_is_blocked_address(address) for address in addresses)
        except ValueError as exc:
            raise RemoteAudioError("The audio URL hostname resolved to an invalid address") from exc
        if blocked:
            raise RemoteAudioError(
                "The audio URL resolves to a link-local, reserved, multicast, or unspecified address"
            )
        return url

    async def fetch(self, audio_location: str) -> DownloadedAudio:
        if not self.enabled:
            raise RemoteAudioError("Audio URL input is disabled because no allowed URL hosts are configured")
        current = await self._validated_url(audio_location)
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_seconds),
                follow_redirects=False,
                transport=self.transport,
                trust_env=False,
            ) as client:
                for redirect_count in range(self.max_redirects + 1):
                    async with client.stream(
                        "GET", current, headers={"Accept": "audio/*, application/octet-stream;q=0.9"}
                    ) as response:
                        if response.status_code in self.REDIRECT_STATUSES:
                            location = response.headers.get("location")
                            if not location:
                                raise RemoteAudioError("The audio URL returned a redirect without a Location header")
                            if redirect_count >= self.max_redirects:
                                raise RemoteAudioError(
                                    f"The audio URL exceeded the {self.max_redirects} redirect limit"
                                )
                            current = await self._validated_url(current.join(location))
                            continue
                        if response.status_code >= 400:
                            raise RemoteAudioError(f"The audio URL returned HTTP {response.status_code}")
                        declared = response.headers.get("content-length")
                        if declared:
                            try:
                                declared_size = int(declared)
                            except ValueError as exc:
                                raise RemoteAudioError(
                                    "The audio URL returned an invalid Content-Length header"
                                ) from exc
                            if declared_size > self.max_bytes:
                                raise RemoteAudioError(self._size_error())
                        chunks: list[bytes] = []
                        downloaded = 0
                        async for chunk in response.aiter_bytes():
                            downloaded += len(chunk)
                            if downloaded > self.max_bytes:
                                raise RemoteAudioError(self._size_error())
                            chunks.append(chunk)
                        payload = b"".join(chunks)
                        if not payload:
                            raise RemoteAudioError("The audio URL returned an empty file")
                        content_type = response.headers.get("content-type")
                        if content_type:
                            content_type = content_type.split(";", 1)[0].strip() or None
                        return DownloadedAudio(
                            payload=payload,
                            filename=(Path(unquote(current.path)).name or "remote-audio")[:255],
                            content_type=content_type,
                        )
        except RemoteAudioError:
            raise
        except httpx.TimeoutException as exc:
            raise RemoteAudioError(
                f"The audio URL download exceeded the {self.timeout_seconds:g} second timeout"
            ) from exc
        except httpx.HTTPError as exc:
            raise RemoteAudioError("The audio URL could not be downloaded") from exc
        raise RemoteAudioError("The audio URL could not be downloaded")

    def _size_error(self) -> str:
        return f"Remote audio exceeds the configured {self.max_bytes / (1024 * 1024):g} MB limit"
