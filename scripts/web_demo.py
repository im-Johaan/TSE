"""Local web demo: give 2-4 separate speaker recordings, mix them, extract the target.

Open the printed URL in a browser. For each speaker you can upload a file, record from
the microphone, or draw random LibriSpeech test speakers. Pick the target, give a
reference clip of that speaker (a different recording), and the app mixes the sources,
runs the model and shows SI-SDR against every source.

The browser decodes and resamples audio to the model rate (8 kHz mono), so any format the
browser can play works (wav, flac, mp3, m4a, webm, ...). Only the Python standard library
is used for the server; it binds to localhost by default.

Example:
    python scripts/web_demo.py                 # then open http://127.0.0.1:7860
    python scripts/web_demo.py --registry my_models.yaml --port 8000
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


from tse.config import PROJECT_ROOT, load_config, resolve_paths  # noqa: E402
from tse.demo_app import SpeakerPool  # noqa: E402
from tse.serving import ModelRegistry  # noqa: E402
from tse.serving.registry import DEFAULT_REGISTRY  # noqa: E402
from tse.serving.sessions import DemoState, b64_to_f32, f32_to_b64  # noqa: E402,F401
from tse.utils import resolve_device, setup_logging  # noqa: E402

log = logging.getLogger("web_demo")

PAGE = PROJECT_ROOT / "webui" / "mixer.html"
MAX_BODY_BYTES = 64 * 1024 * 1024
STATIC = {f.name: ("application/javascript" if f.suffix == ".js" else "text/html; charset=utf-8")
          for f in PAGE.parent.iterdir() if f.suffix in (".js", ".html")}


def make_handler(state: DemoState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # route to logging
            log.debug("%s - %s", self.address_string(), fmt % args)

        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj: Any, status: int = HTTPStatus.OK) -> None:
            self._send(status, json.dumps(obj).encode("utf-8"), "application/json")

        def _error(self, status: int, msg: str) -> None:
            self._json({"error": msg}, status)

        def do_GET(self) -> None:  # noqa: N802 (http.server API)
            url = urlparse(self.path)
            name = url.path.lstrip("/")
            if url.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif name in STATIC:  # shared page assets (config.js, common.js, other pages)
                self._send(HTTPStatus.OK, (PAGE.parent / name).read_bytes(), STATIC[name])
            elif url.path == "/api/info":
                self._json(state.info())
            elif url.path == "/api/random":
                q = parse_qs(url.query)
                try:
                    self._json(state.random(int(q.get("n", ["3"])[0]), q.get("gender", ["any"])[0]))
                except ValueError as e:
                    self._error(HTTPStatus.BAD_REQUEST, str(e))
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/api/separate":
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0 or n > MAX_BODY_BYTES:
                self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "request too large or empty")
                return
            try:
                req = json.loads(self.rfile.read(n))
                self._json(state.separate(req))
            except (ValueError, KeyError, TypeError) as e:
                self._error(HTTPStatus.BAD_REQUEST, str(e))
            except RuntimeError as e:  # e.g. CUDA out of memory
                log.exception("separation failed")
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(e))

    return Handler


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY),
                    help="models.yaml listing the models to offer")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "proposed.yaml"),
                    help="config whose data section locates the dataset for random speakers")
    ap.add_argument("--data-root", default=None, help="override data.root")
    ap.add_argument("--split", default="test", help="split to draw random speakers from")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-browser", action="store_true", help="don't open a browser tab")
    args = ap.parse_args()
    setup_logging()

    registry = ModelRegistry(args.registry, resolve_device(args.device))
    if not registry.available():
        raise SystemExit(f"no model in {args.registry} has an existing checkpoint")
    cfg = resolve_paths(load_config(args.config), args.data_root)
    pool = None
    try:
        pool = SpeakerPool(cfg["data"], args.split)
    except (FileNotFoundError, OSError, KeyError, ValueError) as e:
        log.warning("Random test speakers disabled: %s", e)

    state = DemoState(registry, pool)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(state))
    url = f"http://{args.host}:{args.port}"
    log.info("Models: %s", ", ".join(m["label"] for m in registry.available()))
    print(f"\n  TSE demo running at {url}   (Ctrl+C to stop)\n", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
