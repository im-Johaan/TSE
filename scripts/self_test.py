"""Check that this machine can run the demos: GPU, models, extraction quality and live speed.

Builds a 3-speaker mixture from the test speakers (dataset or bundled demo pack), extracts the
target with the default model, and times sliding-window streaming at the live defaults.
Exits with a non-zero code if something essential fails.

Example:
    python scripts/self_test.py
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tse.config import PROJECT_ROOT, load_config, resolve_paths  # noqa: E402
from tse.demo_app import load_speaker_pool, mix_sources  # noqa: E402
from tse.metrics import si_sdr_np  # noqa: E402
from tse.serving import ModelRegistry  # noqa: E402


def main() -> int:
    """Run the checks; return the exit code."""
    logging.basicConfig(level=logging.WARNING)
    print(f"  torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name}, {p.total_memory / 2**30:.1f} GB")
    else:
        print("  WARNING: no CUDA GPU visible to PyTorch; the demos will run on the CPU "
              "(the mixer works, live mode will probably be too slow)")

    reg = ModelRegistry()
    models = reg.available()
    if not models:
        print("  FAIL: no model checkpoint found (see models.yaml)")
        return 1
    print("  models: " + ", ".join(m["id"] + (" (default)" if m["default"] else "") for m in models))
    be = reg.get()

    pool = load_speaker_pool(resolve_paths(load_config(PROJECT_ROOT / "configs" / "proposed.yaml"))["data"])
    if pool is None:
        print("  FAIL: no test speakers (no dataset and no demo_assets/)")
        return 1
    rng = np.random.default_rng(0)
    offline, streamed, ms = [], [], []
    for trial in range(5):
        spk = pool.random_speakers(3, rng)
        mix, src = mix_sources([s["audio"] for s in spk], [0.0, 0.0, 0.0], "longest")
        prof = be.enroll(spk[0]["ref_audio"])
        base = si_sdr_np(mix, src[0])
        offline.append(si_sdr_np(be.extract(mix, prof), src[0]) - base)
        st = be.new_stream(prof)                       # live defaults: hop 200 ms, lookahead 100 ms
        t0 = time.perf_counter()
        y = st.process_all(mix)
        if trial:                                      # first run includes CUDA warm-up
            ms.append(1000 * (time.perf_counter() - t0) / max(st.calls, 1))
        streamed.append(si_sdr_np(y, src[0]) - base)
    hop_ms = 1000 * st.hop / st.sr
    per_hop = float(np.mean(ms))
    print(f"  3-speaker test mixtures ({len(offline)}): SI-SDRi offline {np.mean(offline):.1f} dB, "
          f"streamed {np.mean(streamed):.1f} dB")
    print(f"  live speed: {per_hop:.0f} ms of compute per {hop_ms:.0f} ms hop "
          f"({100 * per_hop / hop_ms:.0f}% of real time)")
    if per_hop > 0.8 * hop_ms:
        print("  WARNING: live mode may stutter on this machine; use a larger hop (500 ms) "
              "or the smaller R=3 model")
    else:
        print("  OK: fast enough for live mode")
    if np.mean(offline) < 2.0:
        print("  FAIL: extraction quality is far below expectations")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
