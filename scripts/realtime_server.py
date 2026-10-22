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
import logging
import sys
import webbrowser
from http import HTTPStatus
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from websockets.asyncio.server import ServerConnection, serve  # noqa: E402
from websockets.exceptions import ConnectionClosed  # noqa: E402
from websockets.http11 import Request, Response  # noqa: E402

from tse.config import PROJECT_ROOT, load_config, resolve_paths  # noqa: E402
from tse.demo_app import SpeakerPool  # noqa: E402
from tse.serving import ModelRegistry  # noqa: E402
from tse.serving.registry import DEFAULT_REGISTRY  # noqa: E402
from tse.serving.sessions import Session, dispatch  # noqa: E402
from tse.utils import resolve_device, setup_logging  # noqa: E402

log = logging.getLogger("realtime")

PAGE = PROJECT_ROOT / "webui" / "live.html"
def make_handler(registry: ModelRegistry, pool: SpeakerPool | None):
    async def handler(ws: ServerConnection) -> None:
        session = Session(ws, registry, pool)
        worker = asyncio.create_task(session.worker())
        try:
            async for msg in ws:
                if isinstance(msg, bytes):
                    session.on_audio(msg)
                else:
                    await dispatch(session, msg)
        except ConnectionClosed:
            pass
        finally:
            worker.cancel()
    return handler


def process_request(connection: ServerConnection, request: Request) -> Response | None:
    """Serve the page over plain HTTP; only /ws is upgraded to a WebSocket."""
    path = urlparse(request.path).path
    if path == "/ws":
        return None
    name = path.lstrip("/")
    files = {f.name: f for f in PAGE.parent.iterdir() if f.suffix in (".js", ".html")}
    if path in ("/", "/index.html") or name in files:
        f = PAGE if path in ("/", "/index.html") else files[name]
        resp = connection.respond(HTTPStatus.OK, f.read_text(encoding="utf-8"))
        del resp.headers["Content-Type"]
        resp.headers["Content-Type"] = ("application/javascript" if f.suffix == ".js"
                                        else "text/html; charset=utf-8")
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
