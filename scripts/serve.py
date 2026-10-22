"""Deployable TSE server: the clip-mixer API, live streaming and the web pages on one port.

Run it on the GPU machine and expose it with a Cloudflare tunnel (scripts/run_public.ps1).
The pages in webui/ can then be used straight from the tunnel URL or from a static host
such as Vercel (they call this server cross-origin; see webui/config.js).

Endpoints:
    GET  /api/info                  models, sample rate, whether a passcode is needed
    GET  /api/models                registry entries with their test metrics
    GET  /api/random?n=3&gender=any random LibriSpeech test speakers (clip mixer)
    POST /api/separate              mix the given sources and extract the target
    WS   /ws?code=...               live streaming (protocol in tse.serving.sessions.Session)
    GET  /, /mixer.html, /live.html the pages

With --passcode (or env TSE_PASSCODE), API calls must send it as the X-Passcode header and the
WebSocket as ?code=. At most --max-live streams run at once.

Example:
    python scripts/serve.py --port 8000 --passcode faculty2026
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import secrets
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn  # noqa: E402
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from tse.config import PROJECT_ROOT, load_config, resolve_paths  # noqa: E402
from tse.demo_app import SpeakerPool, load_speaker_pool  # noqa: E402
from tse.serving import ModelRegistry  # noqa: E402
from tse.serving.registry import DEFAULT_REGISTRY  # noqa: E402
from tse.serving.sessions import DemoState, Session, dispatch  # noqa: E402
from tse.utils import resolve_device, setup_logging  # noqa: E402

log = logging.getLogger("serve")

WEBUI = PROJECT_ROOT / "webui"
MAX_BODY_BYTES = 64 * 1024 * 1024


class _WSAdapter:
    """Gives a FastAPI WebSocket the ``send(str | bytes)`` method Session expects."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws

    async def send(self, data: str | bytes) -> None:
        if isinstance(data, bytes):
            await self.ws.send_bytes(data)
        else:
            await self.ws.send_text(data)


def create_app(registry: ModelRegistry, pool: SpeakerPool | None, passcode: str | None = None,
               allow_origins: list[str] | None = None, max_live: int = 2) -> FastAPI:
    """Build the application (separate from main() so tests can use it)."""
    origins = allow_origins or ["*"]
    app = FastAPI(title="Target speaker extraction", docs_url=None, redoc_url=None)
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type", "X-Passcode"])
    state = DemoState(registry, pool)
    live = {"n": 0}

    def authorised(code: str | None) -> bool:
        return passcode is None or (code is not None and secrets.compare_digest(code, passcode))

    def deny() -> JSONResponse:
        return JSONResponse({"error": "passcode required or wrong"}, status_code=401)

    def bad(e: Exception, status: int = 400) -> JSONResponse:
        return JSONResponse({"error": str(e)}, status_code=status)

    @app.get("/api/info")
    def info() -> dict[str, Any]:
        return {**state.info(), "passcode_required": passcode is not None,
                "live_available": True, "max_live": max_live, "live_in_use": live["n"]}

    @app.get("/api/models")
    def models(request: Request) -> Any:
        if not authorised(request.headers.get("x-passcode")):
            return deny()
        return {"default": registry.default_id, "models": registry.available()}

    @app.get("/api/random")
    async def random(request: Request, n: int = 3, gender: str = "any") -> Any:
        if not authorised(request.headers.get("x-passcode")):
            return deny()
        try:
            return await asyncio.to_thread(state.random, n, gender)
        except ValueError as e:
            return bad(e)

    @app.post("/api/separate")
    async def separate(request: Request) -> Any:
        if not authorised(request.headers.get("x-passcode")):
            return deny()
        if int(request.headers.get("content-length", 0)) > MAX_BODY_BYTES:
            return bad(ValueError("request too large"), 413)
        try:
            req = await request.json()
            return await asyncio.to_thread(state.separate, req)
        except (ValueError, KeyError, TypeError) as e:
            return bad(e)
        except RuntimeError as e:  # e.g. CUDA out of memory
            log.exception("separation failed")
            return bad(e, 500)

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket) -> None:
        origin = ws.headers.get("origin")
        if "*" not in origins and origin is not None and origin not in origins:
            await ws.close(code=4403)
            return
        if not authorised(ws.query_params.get("code")):
            await ws.close(code=4401)
            return
        await ws.accept()
        if live["n"] >= max_live:
            await ws.send_text('{"type": "error", "message": "the server is busy (too many live '
                               'streams); try again in a minute"}')
            await ws.close(code=4429)
            return
        live["n"] += 1
        session = Session(_WSAdapter(ws), registry, pool)
        worker = asyncio.create_task(session.worker())
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None:
                    session.on_audio(msg["bytes"])
                elif msg.get("text") is not None:
                    await dispatch(session, msg["text"])
        except WebSocketDisconnect:
            pass
        finally:
            worker.cancel()
            live["n"] -= 1

    app.mount("/", StaticFiles(directory=WEBUI, html=True), name="webui")
    return app


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "proposed.yaml"),
                    help="config whose data section locates the dataset for test mixtures")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--host", default="127.0.0.1",
                    help="keep 127.0.0.1 behind a tunnel; nothing else can reach it then")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--passcode", default=os.environ.get("TSE_PASSCODE"),
                    help="require this passcode (default: env TSE_PASSCODE; none = open)")
    ap.add_argument("--allow-origin", action="append", default=None,
                    help="allowed web origin, e.g. https://tse.vercel.app (repeatable; default *)")
    ap.add_argument("--max-live", type=int, default=2, help="concurrent live streams")
    args = ap.parse_args()
    setup_logging()

    registry = ModelRegistry(args.registry, resolve_device(args.device))
    if not registry.available():
        raise SystemExit(f"no model in {args.registry} has an existing checkpoint")
    pool = load_speaker_pool(resolve_paths(load_config(args.config), args.data_root)["data"],
                             args.split)
    app = create_app(registry, pool, args.passcode, args.allow_origin, args.max_live)
    log.info("Models: %s | passcode %s | origins %s", ", ".join(m["id"] for m in registry.available()),
             "on" if args.passcode else "off", args.allow_origin or ["*"])
    print(f"\n  TSE server on http://{args.host}:{args.port}   (Ctrl+C to stop)\n", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning",
                ws_max_size=32 * 1024 * 1024)


if __name__ == "__main__":
    main()
