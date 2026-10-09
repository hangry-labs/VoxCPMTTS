from __future__ import annotations

import html
import json
import mimetypes
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from voxcpm.standalone_ui.gpu import GPU_MONITOR


PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
LOCALES_DIR = STATIC_DIR / "locales"
ENGLISH_CATALOG_PATH = LOCALES_DIR / "en.json"
REPO_ROOT = PACKAGE_DIR.parents[1]
SOURCE_ASSET_DIR = REPO_ROOT / "assets"
PACKAGED_ASSET_DIR = PACKAGE_DIR / "assets"
ASSET_DIR = SOURCE_ASSET_DIR if SOURCE_ASSET_DIR.is_dir() else PACKAGED_ASSET_DIR

mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("font/woff2", ".woff2")

UI_LOCALES = (
    {"code": "en", "name": "English", "path": "/en", "direction": "ltr", "browserLanguage": "en-US"},
    {"code": "nb", "name": "Norsk bokmål", "path": "/nb", "direction": "ltr", "browserLanguage": "nb-NO"},
    {"code": "pl", "name": "Polski", "path": "/pl", "direction": "ltr", "browserLanguage": "pl-PL"},
    {"code": "ja", "name": "日本語", "path": "/ja", "direction": "ltr", "browserLanguage": "ja-JP"},
    {"code": "zh", "name": "简体中文", "path": "/zh", "direction": "ltr", "browserLanguage": "zh-CN"},
    {"code": "es", "name": "Español", "path": "/es", "direction": "ltr", "browserLanguage": "es-ES"},
)
UI_DEFAULT_LOCALE = "en"


def _read_version_file() -> str:
    for version_path in (REPO_ROOT / "VERSION", PACKAGE_DIR / "VERSION"):
        try:
            version = version_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if version:
            return version
    return "0.0.0"


@lru_cache(maxsize=1)
def _english_catalog() -> dict[str, str]:
    payload = json.loads(ENGLISH_CATALOG_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The English UI catalog must be a JSON object.")
    return {str(key): str(value) for key, value in payload.items()}


@lru_cache(maxsize=len(UI_LOCALES))
def _locale_catalog(locale: str) -> dict[str, str]:
    catalog = dict(_english_catalog())
    if locale == UI_DEFAULT_LOCALE:
        return catalog
    path = LOCALES_DIR / f"{locale}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"The {locale} UI catalog must be a JSON object.")
    catalog.update({str(key): str(value) for key, value in payload.items()})
    return catalog


def _index_response(locale: str) -> HTMLResponse:
    locale_entry = next(item for item in UI_LOCALES if item["code"] == locale)
    bootstrap = {
        "locale": locale,
        "defaultLocale": UI_DEFAULT_LOCALE,
        "locales": UI_LOCALES,
        "storageKey": "voxcpmtts-ui-locale-v1",
        "messages": _locale_catalog(locale),
    }
    bootstrap_json = json.dumps(bootstrap, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    rendered = (
        index_html.replace("{{UI_VERSION}}", html.escape(_read_version_file()))
        .replace("{{UI_LOCALE}}", html.escape(locale))
        .replace("{{UI_DIRECTION}}", str(locale_entry["direction"]))
        .replace("{{UI_BOOTSTRAP}}", bootstrap_json)
    )
    return HTMLResponse(rendered, headers={"Cache-Control": "no-cache"})


def attach_ui(*, api_app: FastAPI) -> FastAPI:
    """Attach the offline browser workspace to the existing TTS API."""
    development_assets = os.getenv("VOXCPM_UI_DEV", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    locale_paths = {str(item["path"]) for item in UI_LOCALES}

    @api_app.middleware("http")
    async def disable_development_asset_cache(request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-store" if development_assets else "no-cache"
        elif development_assets and (
            request.url.path == "/"
            or request.url.path in locale_paths
            or request.url.path.startswith("/assets/")
        ):
            response.headers["Cache-Control"] = "no-store"
        return response

    api_app.mount("/static", StaticFiles(directory=STATIC_DIR), name="ui-static")
    if ASSET_DIR.is_dir():
        api_app.mount("/assets", StaticFiles(directory=ASSET_DIR), name="ui-assets")

    @api_app.get("/", include_in_schema=False)
    async def index() -> HTMLResponse:
        return _index_response(UI_DEFAULT_LOCALE)

    def locale_handler(locale: str):
        async def localized_index() -> HTMLResponse:
            return _index_response(locale)

        return localized_index

    for locale_entry in UI_LOCALES:
        locale = str(locale_entry["code"])
        api_app.add_api_route(
            str(locale_entry["path"]),
            locale_handler(locale),
            methods=["GET"],
            include_in_schema=False,
            name=f"ui-{locale}",
        )

    @api_app.get("/system/gpu", tags=["System"], summary="Current GPU telemetry")
    def gpu() -> JSONResponse:
        return JSONResponse(GPU_MONITOR.request_snapshot(), headers={"Cache-Control": "no-store"})

    return api_app
