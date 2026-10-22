"""Serving layer: backends, sliding-window streaming, the registry and the real-time server."""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import small_model_cfg
from tse.config import DEFAULTS
from tse.models import build_model
from tse.serving import ModelRegistry, SpeakerProfile, TSEBackend, register_backend
from tse.utils import save_checkpoint


@register_backend("identity_test")
class IdentityBackend(TSEBackend):
    """Returns its input times ``scale``: streaming must then reproduce the input exactly."""

    def __init__(self, model_id: str = "id", label: str = "identity", device=None,
                 scale: float = 1.0, **_) -> None:
        super().__init__(model_id, label, device or torch.device("cpu"))
        self.scale = scale

    def enroll(self, reference: np.ndarray) -> SpeakerProfile:
        return SpeakerProfile(self.model_id, None, len(reference) / self.sample_rate)

    def extract(self, mixture: np.ndarray, profile: SpeakerProfile) -> np.ndarray:
        self._check(profile)
        return mixture * self.scale


def _tiny_ckpt(tmp_path: Path, name: str = "tiny.pt", seed: int = 0) -> Path:
    torch.manual_seed(seed)
    cfg = {"model": small_model_cfg(), "data": dict(DEFAULTS["data"])}
    path = tmp_path / name
    save_checkpoint(path, build_model(cfg["model"], n_speakers=3), cfg, {"a": 0, "b": 1, "c": 2})
    return path


def _registry_file(tmp_path: Path, ckpts: dict[str, Path]) -> Path:
    lines = [f"default: {next(iter(ckpts))}", "models:"]
    for mid, c in ckpts.items():
        lines += [f"- id: {mid}", f"  label: model {mid}", "  backend: ueconv",
                  f"  checkpoint: '{c.as_posix()}'"]
    p = tmp_path / "models.yaml"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


@pytest.mark.parametrize("hop,lookahead", [(0.1, 0.0), (0.2, 0.1), (0.25, 0.25)])
@pytest.mark.parametrize("scale", [1.0, -3.0])
def test_stream_reproduces_identity_model_exactly(hop: float, lookahead: float, scale: float) -> None:
    be = IdentityBackend(scale=scale)
    st = be.new_stream(be.enroll(np.ones(8000)), window=1.0, hop=hop, lookahead=lookahead,
                       min_context=0.0)
    x = np.random.default_rng(0).standard_normal(8000 * 4).astype(np.float32)
    y = st.process_all(x, chunk=137)               # odd chunk size: hops straddle chunks
    assert y.shape == x.shape
    # level normalisation undoes the arbitrary output scale; alignment is sample-exact
    skip = int((hop + lookahead) * 8000)
    np.testing.assert_allclose(y[skip:], x[skip:], atol=1e-5)
    assert st.latency_ms == pytest.approx(1000 * (hop + lookahead))


def test_stream_outputs_whole_hops_and_counts_match() -> None:
    be = IdentityBackend()
    st = be.new_stream(be.enroll(np.ones(8000)), hop=0.2, lookahead=0.1)
    total = 0
    for n in (100, 1500, 37, 4000):
        out = st.push(np.zeros(n, dtype=np.float32))
        assert len(out) % st.hop == 0
        total += len(out)
    assert total == (100 + 1500 + 37 + 4000) // st.hop * st.hop


def test_stream_rebind_checks_profile_and_keeps_history() -> None:
    a, b = IdentityBackend("a"), IdentityBackend("b")
    st = a.new_stream(a.enroll(np.ones(8000)), min_context=0.0)
    st.push(np.ones(8000, dtype=np.float32))
    hist = st.history.copy()
    with pytest.raises(ValueError, match="enroll again"):
        b.extract(np.ones(10, dtype=np.float32), a.enroll(np.ones(8000)))
    st.rebind(b, b.enroll(np.ones(8000)))
    np.testing.assert_array_equal(st.history, hist)
    assert len(st.push(np.ones(1600, dtype=np.float32))) == 1600


def test_registry_lists_loads_and_promotes(tmp_path: Path) -> None:
    c1, c2 = _tiny_ckpt(tmp_path, "a.pt", 0), _tiny_ckpt(tmp_path, "b.pt", 1)
    reg = ModelRegistry(_registry_file(tmp_path, {"m1": c1}), torch.device("cpu"))
    assert [m["id"] for m in reg.available()] == ["m1"] and reg.default_id == "m1"
    be = reg.get()
    prof = be.enroll(np.random.randn(16000).astype(np.float32))
    out = be.extract(np.random.randn(8013).astype(np.float32), prof)
    assert out.shape == (8013,) and np.isfinite(out).all()
    reg.add({"id": "m2", "label": "two", "backend": "ueconv", "checkpoint": str(c2)},
            make_default=True)
    reg2 = ModelRegistry(reg.path, torch.device("cpu"))      # persisted
    assert reg2.default_id == "m2" and set(reg2.entries) == {"m1", "m2"}
    assert reg2.available()[0]["id"] == "m2"                  # default listed first
    with pytest.raises(ValueError):
        reg.add({"id": "m2", "label": "dup", "backend": "ueconv", "checkpoint": str(c2)})
    with pytest.raises(ValueError, match="unknown backend"):
        reg.add({"id": "m3", "label": "x", "backend": "nope", "checkpoint": str(c2)})
    with pytest.raises(ValueError, match="enroll again"):     # profiles are per model
        reg.get("m2").extract(np.zeros(8000, np.float32), prof)


def test_ueconv_stream_runs(tmp_path: Path) -> None:
    reg = ModelRegistry(_registry_file(tmp_path, {"m": _tiny_ckpt(tmp_path)}), torch.device("cpu"))
    be = reg.get()
    prof = be.enroll(np.random.randn(16000).astype(np.float32))
    x = np.random.default_rng(1).standard_normal(8000 * 2).astype(np.float32)
    st = be.new_stream(prof, window=2.0, hop=0.25, lookahead=0.5, min_context=0.0)
    y = st.process_all(x)
    assert y.shape == x.shape and np.isfinite(y).all() and st.calls > 0


def _load_realtime():
    path = Path(__file__).resolve().parents[1] / "scripts" / "realtime_server.py"
    spec = importlib.util.spec_from_file_location("realtime_server", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_realtime_server_round_trip_and_hot_swap(tmp_path: Path) -> None:
    pytest.importorskip("websockets")
    from websockets.asyncio.client import connect
    from websockets.asyncio.server import serve

    rt = _load_realtime()
    reg = ModelRegistry(_registry_file(tmp_path, {"m1": _tiny_ckpt(tmp_path, "a.pt", 0),
                                                  "m2": _tiny_ckpt(tmp_path, "b.pt", 1)}),
                        torch.device("cpu"))

    async def scenario() -> None:
        async with serve(rt.make_handler(reg, None), "127.0.0.1", 0,
                         process_request=rt.process_request) as server:
            port = server.sockets[0].getsockname()[1]
            async with connect(f"ws://127.0.0.1:{port}/ws", max_size=2 ** 24) as ws:
                async def expect(kind: str) -> dict:
                    while True:
                        m = await asyncio.wait_for(ws.recv(), 30)
                        if isinstance(m, str):
                            d = json.loads(m)
                            if d["type"] == kind:
                                return d
                            assert d["type"] != "error", d

                await ws.send(json.dumps({"type": "hello"}))
                init = await expect("init")
                assert init["model"] == "m1" and init["sample_rate"] == 8000
                await ws.send(json.dumps({"type": "start"}))
                assert "reference" in (await expect("error"))["message"]
                ref = base64.b64encode(np.random.randn(16000).astype("<f4").tobytes()).decode()
                await ws.send(json.dumps({"type": "reference", "audio": ref}))
                await expect("enrolled")
                await ws.send(json.dumps({"type": "start", "hop": 0.2, "lookahead": 0.1}))
                await expect("started")
                got, swapped = 0, False
                x = np.random.randn(8000 * 2).astype("<f4")
                for i in range(0, len(x), 160):
                    await ws.send(x[i:i + 160].tobytes())
                    if i == 8000:
                        await ws.send(json.dumps({"type": "set_model", "id": "m2"}))
                while got < 8000 or not swapped:
                    m = await asyncio.wait_for(ws.recv(), 30)
                    if isinstance(m, bytes):
                        got += len(m) // 4
                    elif json.loads(m)["type"] == "model":
                        swapped = True
                assert got % 1600 == 0
                await ws.send(json.dumps({"type": "settings", "hop": 0.33}))
                assert "hop must be" in (await expect("error"))["message"]

    asyncio.run(scenario())


def test_realtime_page_served() -> None:
    rt = _load_realtime()
    assert rt.PAGE.is_file() and "AudioWorklet" in rt.PAGE.read_text(encoding="utf-8")
