"""Real-time target speaker extraction in the browser.

Open the printed URL (Chrome or Edge). Give a reference recording of the target speaker,
then stream audio in, either live from the microphone (use headphones to avoid feedback) or
"simulated live" from a file or a random 2-4 speaker test mixture played at real-time speed.
The extracted voice plays back continuously. The model can be switched while streaming; the
list comes from models.yaml (see tse.serving.registry).

The browser runs its audio graph at the model sample rate (8 kHz), so the server receives
and returns raw float32 PCM frames over a WebSocket without resampling. Processing uses the
sliding-window streamer (tse.serving.streaming): latency = hop + lookahead + compute + audio
buffering, adjustable live.

Example:
    python scripts/realtime_server.py            # then open http://127.0.0.1:7870
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import sys
import time
import webbrowser
from http import HTTPStatus
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from websockets.asyncio.server import ServerConnection, serve  # noqa: E402
from websockets.exceptions import ConnectionClosed  # noqa: E402
from websockets.http11 import Request, Response  # noqa: E402

from tse.config import PROJECT_ROOT, load_config, resolve_paths  # noqa: E402
from tse.demo_app import SpeakerPool, mix_sources  # noqa: E402
from tse.serving import ChunkedStream, ModelRegistry  # noqa: E402
from tse.serving.registry import DEFAULT_REGISTRY  # noqa: E402
from tse.utils import resolve_device, setup_logging  # noqa: E402

log = logging.getLogger("realtime")

PAGE = PROJECT_ROOT / "webui" / "realtime.html"
MAX_BACKLOG_S = 1.0     # drop the oldest input if processing falls this far behind
STATS_EVERY_S = 0.5
HOPS = (0.1, 0.2, 0.25, 0.5)
LOOKAHEADS = (0.0, 0.1, 0.2, 0.3, 0.5)


def b64_to_f32(s: str) -> np.ndarray:
    x = np.frombuffer(base64.b64decode(s), dtype="<f4").astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("audio contains NaN/Inf")
    return x


def f32_to_b64(x: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(x, dtype="<f4").tobytes()).decode("ascii")


class Session:
    """One browser connection: chosen model, speaker profile, live stream and input backlog."""

    def __init__(self, ws: ServerConnection, registry: ModelRegistry,
                 pool: SpeakerPool | None) -> None:
        self.ws, self.registry, self.pool = ws, registry, pool
        self.model_id = registry.default_id
        self.backend = None
        self.reference: np.ndarray | None = None
        self.profile = None
        self.stream: ChunkedStream | None = None
        self.settings = {"window": 3.0, "hop": 0.2, "lookahead": 0.1}
        self.backlog = np.zeros(0, dtype=np.float32)
        self.wake = asyncio.Event()
        self.dropped_s = 0.0
        self.rng = np.random.default_rng()

    async def send(self, obj: dict[str, Any]) -> None:
        await self.ws.send(json.dumps(obj))

    async def error(self, msg: str) -> None:
        await self.send({"type": "error", "message": msg})

    # ---- control messages --------------------------------------------------------------
    async def ensure_backend(self) -> None:
        if self.backend is None or self.backend.model_id != self.model_id:
            self.backend = await asyncio.to_thread(self.registry.get, self.model_id)

    async def on_hello(self, _: dict) -> None:
        await self.ensure_backend()
        await self.send({"type": "init", "models": self.registry.available(),
                         "model": self.model_id, "model_info": self.backend.info(),
                         "sample_rate": self.backend.sample_rate, "settings": self.settings,
                         "hops": HOPS, "lookaheads": LOOKAHEADS,
                         "random_available": self.pool is not None})

    async def on_set_model(self, msg: dict) -> None:
        old_sr = self.backend.sample_rate if self.backend else None
        self.model_id = msg["id"]
        t0 = time.perf_counter()
        await self.ensure_backend()
        if old_sr is not None and self.backend.sample_rate != old_sr:
            raise ValueError("this model uses a different sample rate; reload the page")
        if self.reference is not None:
            self.profile = await asyncio.to_thread(self.backend.enroll, self.reference)
            if self.stream is not None:  # hot swap, input history kept
                await asyncio.to_thread(self.stream.rebind, self.backend, self.profile)
        log.info("Switched to model %s", self.model_id)
        await self.send({"type": "model", "model": self.model_id, "model_info": self.backend.info(),
                         "switch_ms": 1000 * (time.perf_counter() - t0)})

    async def on_reference(self, msg: dict) -> None:
        ref = b64_to_f32(msg["audio"])
        await self.ensure_backend()
        self.profile = await asyncio.to_thread(self.backend.enroll, ref)
        self.reference = ref
        if self.stream is not None:
            await asyncio.to_thread(self.stream.rebind, self.backend, self.profile)
        await self.send({"type": "enrolled", "seconds": len(ref) / self.backend.sample_rate})

    def _new_stream(self) -> ChunkedStream:
        st = self.backend.new_stream(self.profile, **self.settings)
        if self.stream is not None:  # keep the input history across setting changes
            st.history = self.stream.history.copy()
        return st

    async def on_start(self, msg: dict) -> None:
        if self.profile is None:
            raise ValueError("give a reference recording first")
        self._apply_settings(msg)
        self.stream = None
        self.backlog = np.zeros(0, dtype=np.float32)
        self.dropped_s = 0.0
        self.stream = self._new_stream()
        await self.send({"type": "started", "settings": self.stream.settings()})

    def _apply_settings(self, msg: dict) -> None:
        for key, allowed in (("hop", HOPS), ("lookahead", LOOKAHEADS)):
            if key in msg:
                v = float(msg[key])
                if v not in allowed:
                    raise ValueError(f"{key} must be one of {allowed}")
                self.settings[key] = v

    async def on_settings(self, msg: dict) -> None:
        self._apply_settings(msg)
        if self.stream is not None:
            self.stream = self._new_stream()
        await self.send({"type": "settings", "settings": dict(self.settings)})

    async def on_stop(self, _: dict) -> None:
        self.stream = None
        self.backlog = np.zeros(0, dtype=np.float32)
        await self.send({"type": "stopped"})

    async def on_random_mixture(self, msg: dict) -> None:
        if self.pool is None:
            raise ValueError("random test mixtures are unavailable (dataset not found)")
        n = int(msg.get("n", 3))
        if not 2 <= n <= 4:
            raise ValueError("n must be 2-4")
        spk = await asyncio.to_thread(self.pool.random_speakers, n, self.rng,
                                      msg.get("gender", "any"))
        mix, srcs = mix_sources([s["audio"] for s in spk], [0.0] * n, "longest")
        await self.send({
            "type": "mixture", "mixture": f32_to_b64(mix), "target": f32_to_b64(srcs[0]),
            "reference": f32_to_b64(spk[0]["ref_audio"]),
            "speakers": [{"speaker": s["speaker"], "gender": s["gender"]} for s in spk],
        })

    HANDLERS = {"hello": on_hello, "set_model": on_set_model, "reference": on_reference,
                "start": on_start, "settings": on_settings, "stop": on_stop,
                "random_mixture": on_random_mixture}

    # ---- audio --------------------------------------------------------------------------
    def on_audio(self, data: bytes) -> None:
        if self.stream is None:
            return
        x = np.frombuffer(data, dtype="<f4")
        self.backlog = np.concatenate([self.backlog, x])
        limit = int(MAX_BACKLOG_S * self.stream.sr)
        if len(self.backlog) > limit:  # we are behind real time: drop the oldest input
            drop = len(self.backlog) - limit
            self.dropped_s += drop / self.stream.sr
            self.backlog = self.backlog[drop:]
        self.wake.set()

    async def worker(self) -> None:
        """Process the backlog in hop-sized steps and send the output back."""
        last_stats = 0.0
        while True:
            await self.wake.wait()
            self.wake.clear()
            while self.stream is not None and len(self.backlog) >= self.stream.hop:
                st = self.stream
                n = (len(self.backlog) // st.hop) * st.hop
                x, self.backlog = self.backlog[:n], self.backlog[n:]
                try:
                    y = await asyncio.to_thread(st.push, x)
                except (ValueError, RuntimeError) as e:
                    log.exception("streaming failed")
                    self.stream = None
                    await self.error(f"streaming stopped: {e}")
                    break
                if len(y):
                    await self.ws.send(np.ascontiguousarray(y, dtype="<f4").tobytes())
                now = time.perf_counter()
                if now - last_stats > STATS_EVERY_S:
                    last_stats = now
                    await self.send({"type": "stats", "compute_ms": st.compute_ms,
                                     "algorithmic_ms": st.latency_ms,
                                     "backlog_ms": 1000 * len(self.backlog) / st.sr,
                                     "dropped_s": self.dropped_s, "model": self.model_id})


def make_handler(registry: ModelRegistry, pool: SpeakerPool | None):
    async def handler(ws: ServerConnection) -> None:
        session = Session(ws, registry, pool)
        worker = asyncio.create_task(session.worker())
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    session.on_audio(msg)
                    continue
                try:
                    req = json.loads(msg)
                    fn = Session.HANDLERS.get(req.get("type"))
                    if fn is None:
                        raise ValueError(f"unknown message type {req.get('type')!r}")
                    await fn(session, req)
                except (ValueError, KeyError, TypeError, FileNotFoundError) as e:
                    await session.error(str(e))
                except RuntimeError as e:  # e.g. CUDA out of memory
                    log.exception("request failed")
                    await session.error(str(e))
        except ConnectionClosed:
            pass
        finally:
            worker.cancel()
    return handler


def process_request(connection: ServerConnection, request: Request) -> Response | None:
    """Serve the page over plain HTTP; only /ws is upgraded to a WebSocket."""
    if request.path == "/ws":
        return None
    if request.path in ("/", "/index.html"):
        resp = connection.respond(HTTPStatus.OK, PAGE.read_text(encoding="utf-8"))
        del resp.headers["Content-Type"]
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        resp.headers["Cache-Control"] = "no-store"
        return resp
    return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")


async def run(args: argparse.Namespace) -> None:
    registry = ModelRegistry(args.registry, resolve_device(args.device))
    await asyncio.to_thread(registry.get)  # preload the default model
    pool = None
    try:
        cfg = resolve_paths(load_config(args.config), args.data_root)
        pool = SpeakerPool(cfg["data"], args.split)
    except (FileNotFoundError, OSError, KeyError, ValueError) as e:
        log.warning("Random test mixtures disabled: %s", e)
    url = f"http://{args.host}:{args.port}"
    async with serve(make_handler(registry, pool), args.host, args.port,
                     process_request=process_request, max_size=32 * 1024 * 1024):
        print(f"\n  Real-time TSE running at {url}   (Ctrl+C to stop)\n", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        await asyncio.get_running_loop().create_future()  # run forever


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY), help="models.yaml")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "proposed.yaml"),
                    help="config whose data section locates the dataset for test mixtures")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7870)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    setup_logging()
    logging.getLogger("websockets").setLevel(logging.WARNING)
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        log.info("Stopped")


if __name__ == "__main__":
    main()
