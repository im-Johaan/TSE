"""Web demo: mixing, scoring, the random-speaker pool and an HTTP round trip."""

from __future__ import annotations

import importlib.util
import json
import threading
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import small_model_cfg
from tse.config import DEFAULTS
from tse.data.dynamic_mix import active_rms
from tse.demo_app import SpeakerPool, mix_sources, score_extraction
from tse.models import build_model
from tse.serving import ModelRegistry
from tse.utils import save_checkpoint


def _load_web_demo():
    path = Path(__file__).resolve().parents[1] / "scripts" / "web_demo.py"
    spec = importlib.util.spec_from_file_location("web_demo", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def tiny_ckpt(tmp_path: Path) -> str:
    torch.manual_seed(0)
    cfg = {"model": small_model_cfg(), "data": dict(DEFAULTS["data"])}
    path = tmp_path / "tiny.pt"
    save_checkpoint(path, build_model(cfg["model"], n_speakers=3), cfg, {"a": 0, "b": 1, "c": 2})
    return str(path)


def _registry(tmp_path: Path, ckpt: str) -> ModelRegistry:
    reg = tmp_path / "models.yaml"
    reg.write_text(f"default: tiny\nmodels:\n- id: tiny\n  label: tiny\n  backend: ueconv\n"
                   f"  checkpoint: '{Path(ckpt).as_posix()}'\n", encoding="utf-8")
    return ModelRegistry(reg, torch.device("cpu"))


def test_mix_sources_levels_lengths_and_sum() -> None:
    rng = np.random.default_rng(0)
    srcs = [rng.standard_normal(8000).astype(np.float32) * 0.01,
            rng.standard_normal(12000).astype(np.float32) * 0.5,
            rng.standard_normal(6000).astype(np.float32)]
    mix, scaled = mix_sources(srcs, [0.0, 0.0, 6.0], "longest")
    assert len(mix) == 12000 and all(len(s) == 12000 for s in scaled)
    np.testing.assert_allclose(np.sum(scaled, axis=0), mix, atol=1e-6)
    r = [active_rms(s) for s in scaled]
    assert r[0] == pytest.approx(r[1], rel=1e-3)          # equal level at 0 dB gain
    assert r[2] / r[0] == pytest.approx(10 ** (6 / 20), rel=1e-3)
    assert np.abs(mix).max() <= 0.9 + 1e-6
    mix_s, _ = mix_sources(srcs, [0.0] * 3, "shortest")
    assert len(mix_s) == 6000
    with pytest.raises(ValueError):
        mix_sources([np.zeros(100, np.float32), srcs[0]], [0.0, 0.0])


def test_score_extraction_finds_closest_source() -> None:
    rng = np.random.default_rng(1)
    srcs = [rng.standard_normal(4000).astype(np.float32) for _ in range(3)]
    mix = np.sum(srcs, axis=0)
    est = srcs[2] + 0.05 * srcs[0]
    s = score_extraction(est, srcs, mix, target=2)
    assert s["closest_source"] == 2
    assert s["si_sdri"] == pytest.approx(s["si_sdr_out"] - s["si_sdr_in"])
    assert s["si_sdri"] > 10
    assert score_extraction(est, srcs, mix, target=0)["closest_source"] == 2


def test_random_speakers(fake_librimix: dict) -> None:
    pool = SpeakerPool(fake_librimix["data_cfg"], "test")
    rng = np.random.default_rng(0)
    for mode in ("any", "same", "mixed"):
        spk = pool.random_speakers(3, rng, mode)
        assert len({s["speaker"] for s in spk}) == 3
        for s in spk:
            assert s["speaker"] in fake_librimix["speakers"]["test"]
            assert s["utt"] != s["ref_utt"] and len(s["audio"]) and len(s["ref_audio"])
        if mode == "mixed":
            assert len({s["gender"] for s in spk}) == 2


def test_http_round_trip(tiny_ckpt: str, fake_librimix: dict, tmp_path: Path) -> None:
    wd = _load_web_demo()
    state = wd.DemoState(_registry(tmp_path, tiny_ckpt), SpeakerPool(fake_librimix["data_cfg"], "test"))
    server = wd.ThreadingHTTPServer(("127.0.0.1", 0), wd.make_handler(state))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        info = json.load(urllib.request.urlopen(base + "/api/info"))
        assert info["models"] == [{"id": "tiny", "label": "tiny (default)"}]
        rnd = json.load(urllib.request.urlopen(base + "/api/random?n=3&gender=any"))
        spk = rnd["speakers"]
        body = json.dumps({"sources": [s["audio"] for s in spk], "gains_db": [0, 0, 0],
                           "target": 1, "enrollment": spk[1]["ref_audio"], "model": "tiny"}).encode()
        req = urllib.request.Request(base + "/api/separate", body,
                                     {"Content-Type": "application/json"})
        res = json.load(urllib.request.urlopen(req))
        n = len(wd.b64_to_f32(res["mixture"]))
        assert len(wd.b64_to_f32(res["estimate"])) == n == len(wd.b64_to_f32(res["target"]))
        assert len(res["si_sdr_vs_sources"]) == 3 and np.isfinite(res["si_sdri"])
        bad = urllib.request.Request(base + "/api/separate", json.dumps(
            {"sources": [spk[0]["audio"]], "target": 0, "enrollment": spk[0]["ref_audio"]}).encode())
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(bad)
        assert e.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
