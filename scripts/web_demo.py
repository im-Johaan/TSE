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
import base64
import json
import logging
import sys
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from tse.config import PROJECT_ROOT, load_config, resolve_paths  # noqa: E402
from tse.demo_app import SpeakerPool, mix_sources, score_extraction  # noqa: E402
from tse.serving import ModelRegistry  # noqa: E402
from tse.serving.registry import DEFAULT_REGISTRY  # noqa: E402
from tse.utils import resolve_device, setup_logging  # noqa: E402

log = logging.getLogger("web_demo")

PAGE = PROJECT_ROOT / "webui" / "index.html"
MAX_SECONDS = 60.0
MAX_BODY_BYTES = 64 * 1024 * 1024


def b64_to_f32(s: str) -> np.ndarray:
    """Decode a base64 string of little-endian float32 samples."""
    x = np.frombuffer(base64.b64decode(s), dtype="<f4").astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("audio contains NaN/Inf")
    return x


def f32_to_b64(x: np.ndarray) -> str:
    """Encode samples as base64 little-endian float32."""
    return base64.b64encode(np.ascontiguousarray(x, dtype="<f4").tobytes()).decode("ascii")


class DemoState:
    """Models, data pool and settings shared by all requests."""

    def __init__(self, registry: ModelRegistry, pool: SpeakerPool | None) -> None:
        self.registry = registry
        self.pool = pool
        self.sample_rate = registry.get().sample_rate  # also preloads the default model
        self.rng = np.random.default_rng()

    def info(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "max_seconds": MAX_SECONDS,
            "models": [{"id": m["id"], "label": m["label"] + (" (default)" if m["default"] else "")}
                       for m in self.registry.available()],
            "random_available": self.pool is not None,
            "device": str(self.registry.device),
        }

    def random(self, n: int, gender_mode: str) -> dict[str, Any]:
        if self.pool is None:
            raise ValueError("random test speakers are unavailable (dataset not found)")
        if not 2 <= n <= 4:
            raise ValueError("n must be 2-4")
        spk = self.pool.random_speakers(n, self.rng, gender_mode)
        return {"speakers": [{
            "speaker": s["speaker"], "gender": s["gender"], "utt": s["utt"],
            "ref_utt": s["ref_utt"], "audio": f32_to_b64(s["audio"]),
            "ref_audio": f32_to_b64(s["ref_audio"]),
        } for s in spk]}

    def separate(self, req: dict[str, Any]) -> dict[str, Any]:
        sources = [b64_to_f32(s) for s in req["sources"]]
        if not 2 <= len(sources) <= 4:
            raise ValueError("give 2-4 speakers")
        target = int(req["target"])
        if not 0 <= target < len(sources):
            raise ValueError("target index out of range")
        enrollment = b64_to_f32(req["enrollment"])
        if len(enrollment) < self.sample_rate // 2:
            raise ValueError("the reference clip must be at least 0.5 s long")
        gains = [float(g) for g in req.get("gains_db", [0.0] * len(sources))]
        model_id = req.get("model") or self.registry.default_id
        if model_id not in self.registry.entries:
            raise ValueError(f"unknown model {model_id!r}")
        backend = self.registry.get(model_id)
        if backend.sample_rate != self.sample_rate:
            raise ValueError("this model uses a different sample rate than the page")
        limit = int(MAX_SECONDS * self.sample_rate)
        if max(len(s) for s in sources) > limit:
            raise ValueError(f"each recording must be at most {MAX_SECONDS:.0f} s")
        mixture, scaled = mix_sources(sources, gains, req.get("length_mode", "longest"))
        t0 = time.perf_counter()
        est = backend.extract(mixture, backend.enroll(enrollment))
        elapsed = time.perf_counter() - t0
        scores = score_extraction(est, scaled, mixture, target)
        peak = float(np.abs(est).max())
        est_play = est / peak * 0.9 if peak > 0 else est  # SI-SDR output scale is arbitrary
        log.info("separate: %d speakers, %.1f s, model %s, target %d -> SI-SDRi %.2f dB (%.0f ms)",
                 len(sources), len(mixture) / self.sample_rate, model_id, target,
                 scores["si_sdri"], elapsed * 1e3)
        return {
            "mixture": f32_to_b64(mixture),
            "estimate": f32_to_b64(est_play),
            "target": f32_to_b64(scaled[target]),
            "seconds": len(mixture) / self.sample_rate,
            "inference_ms": elapsed * 1e3,
            **scores,
        }


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
            if url.path in ("/", "/index.html"):
                self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
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
