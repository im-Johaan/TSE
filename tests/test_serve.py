"""Deployable FastAPI server (scripts/serve.py): pages, API, passcode, CORS, WebSocket."""

from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from test_serving import _registry_file, _tiny_ckpt  # noqa: E402
from tse.demo_app import SpeakerPool  # noqa: E402
from tse.serving import ModelRegistry  # noqa: E402


def _serve_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "serve.py"
    spec = importlib.util.spec_from_file_location("serve", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def make_client(tmp_path: Path, fake_librimix: dict):
    reg = ModelRegistry(_registry_file(tmp_path, {"m1": _tiny_ckpt(tmp_path)}), torch.device("cpu"))
    pool = SpeakerPool(fake_librimix["data_cfg"], "test")
    srv = _serve_module()

    def make(**kw) -> TestClient:
        return TestClient(srv.create_app(reg, pool, **kw))
    return make


def _b64(x: np.ndarray) -> str:
    return base64.b64encode(x.astype("<f4").tobytes()).decode()


def test_pages_and_open_api(make_client) -> None:
    c = make_client()
    for page in ("/", "/mixer.html", "/live.html", "/common.js", "/config.js"):
        assert c.get(page).status_code == 200, page
    info = c.get("/api/info").json()
    assert info["passcode_required"] is False and info["models"][0]["id"] == "m1"
    spk = c.get("/api/random", params={"n": 2}).json()["speakers"]
    res = c.post("/api/separate", json={"sources": [s["audio"] for s in spk], "target": 0,
                                        "enrollment": spk[0]["ref_audio"], "model": "m1"})
    assert res.status_code == 200 and np.isfinite(res.json()["si_sdri"])
    assert c.post("/api/separate", json={"sources": [], "target": 0}).status_code == 400
    assert c.get("/api/models").json()["default"] == "m1"


def test_passcode_and_cors(make_client) -> None:
    c = make_client(passcode="s3cret", allow_origins=["https://tse.example.app"])
    assert c.get("/api/info").json()["passcode_required"] is True       # info stays public
    assert c.get("/api/random").status_code == 401
    assert c.get("/api/random", headers={"X-Passcode": "wrong"}).status_code == 401
    assert c.get("/api/random", params={"n": 2}, headers={"X-Passcode": "s3cret"}).status_code == 200
    pre = c.options("/api/separate", headers={"Origin": "https://tse.example.app",
                                              "Access-Control-Request-Method": "POST",
                                              "Access-Control-Request-Headers": "content-type,x-passcode"})
    assert pre.headers.get("access-control-allow-origin") == "https://tse.example.app"
    bad = c.options("/api/separate", headers={"Origin": "https://evil.example",
                                              "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers
    with pytest.raises(WebSocketDisconnect) as e:
        with c.websocket_connect("/ws?code=wrong") as ws:
            ws.receive_text()
    assert e.value.code == 4401
    with pytest.raises(WebSocketDisconnect) as e:
        with c.websocket_connect("/ws?code=s3cret", headers={"origin": "https://evil.example"}) as ws:
            ws.receive_text()
    assert e.value.code == 4403


def test_websocket_live_stream_and_limit(make_client) -> None:
    c = make_client(max_live=1)
    with c.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "hello"}))
        assert json.loads(ws.receive_text())["type"] == "init"
        ws.send_text(json.dumps({"type": "reference", "audio": _b64(np.random.randn(16000))}))
        assert json.loads(ws.receive_text())["type"] == "enrolled"
        ws.send_text(json.dumps({"type": "start", "hop": 0.2, "lookahead": 0.1}))
        assert json.loads(ws.receive_text())["type"] == "started"
        x = np.random.randn(8000).astype("<f4")
        for i in range(0, len(x), 160):
            ws.send_bytes(x[i:i + 160].tobytes())
        got = 0
        while got < 4800:
            m = ws.receive()
            if m.get("bytes"):
                got += len(m["bytes"]) // 4
        assert got % 1600 == 0
        with c.websocket_connect("/ws") as ws2:                       # second stream: busy
            msg = json.loads(ws2.receive_text())
            assert msg["type"] == "error" and "busy" in msg["message"]
