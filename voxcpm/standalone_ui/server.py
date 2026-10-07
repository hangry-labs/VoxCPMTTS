from __future__ import annotations

import html
import mimetypes
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from voxcpm.standalone_ui.gpu import GPU_MONITOR


PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
REPO_ROOT = PACKAGE_DIR.parents[1]
SOURCE_ASSET_DIR = REPO_ROOT / "assets"
PACKAGED_ASSET_DIR = PACKAGE_DIR / "assets"
ASSET_DIR = SOURCE_ASSET_DIR if SOURCE_ASSET_DIR.is_dir() else PACKAGED_ASSET_DIR

mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("font/woff2", ".woff2")


def _read_version_file() -> str:
    for version_path in (REPO_ROOT / "VERSION", PACKAGE_DIR / "VERSION"):
        try:
            version = version_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if version:
            return version
    return "0.0.0"


def _index_response() -> HTMLResponse:
    index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    rendered = index_html.replace("{{UI_VERSION}}", html.escape(_read_version_file()))
    return HTMLResponse(rendered, headers={"Cache-Control": "no-cache"})


def attach_ui(*, api_app: FastAPI) -> FastAPI:
    """Attach the offline browser workspace to the existing TTS API."""
    development_assets = os.getenv("VOXCPM_UI_DEV", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    @api_app.middleware("http")
    async def disable_development_asset_cache(request, call_next):
        response = await call_next(request)
        if development_assets and (
            request.url.path == "/"
            or request.url.path.startswith("/static/")
            or request.url.path.startswith("/assets/")
        ):
            response.headers["Cache-Control"] = "no-store"
        return response

    api_app.mount("/static", StaticFiles(directory=STATIC_DIR), name="ui-static")
    if ASSET_DIR.is_dir():
        api_app.mount("/assets", StaticFiles(directory=ASSET_DIR), name="ui-assets")

    @api_app.get("/", include_in_schema=False)
    async def index() -> HTMLResponse:
        return _index_response()

    @api_app.get("/system/gpu", tags=["System"], summary="Current GPU telemetry")
    def gpu() -> JSONResponse:
        return JSONResponse(GPU_MONITOR.request_snapshot(), headers={"Cache-Control": "no-store"})

    return api_app
