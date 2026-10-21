"""Evaluation metrics: SI-SDR(i), SDR(i), PESQ, STOI and aggregation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from .losses import si_sdr

log = logging.getLogger(__name__)

try:
    import fast_bss_eval
    HAVE_BSS = True
except ImportError:
    HAVE_BSS = False
try:
    from pesq import pesq as _pesq
    HAVE_PESQ = True
except ImportError:
    HAVE_PESQ = False
try:
    from pystoi import stoi as _stoi
    HAVE_STOI = True
except ImportError:
    HAVE_STOI = False


def available_metrics() -> dict[str, bool]:
    """Which optional metrics can be computed in this environment."""
    return {"sdr": HAVE_BSS, "pesq": HAVE_PESQ, "stoi": HAVE_STOI}


def si_sdr_np(est: np.ndarray, ref: np.ndarray) -> float:
    """SI-SDR of two 1-D numpy arrays (same formula as the training loss)."""
    return float(si_sdr(torch.from_numpy(est), torch.from_numpy(ref)))


def utterance_metrics(est: np.ndarray, ref: np.ndarray, mix: np.ndarray, sample_rate: int,
                      extra: bool = True) -> dict[str, float]:
    """All metrics for one utterance. ``extra`` enables SDR/PESQ/STOI when available."""
    m: dict[str, float] = {}
    m["si_sdr_out"] = si_sdr_np(est, ref)
    m["si_sdr_in"] = si_sdr_np(mix, ref)
    m["si_sdri"] = m["si_sdr_out"] - m["si_sdr_in"]
    if not extra:
        return m
    if HAVE_BSS:
        ref64 = ref.astype(np.float64)[None]
        sdr_out = float(fast_bss_eval.sdr(ref64, est.astype(np.float64)[None], load_diag=1e-10)[0])
        sdr_in = float(fast_bss_eval.sdr(ref64, mix.astype(np.float64)[None], load_diag=1e-10)[0])
        m.update(sdr_out=sdr_out, sdr_in=sdr_in, sdri=sdr_out - sdr_in)
    if HAVE_PESQ and sample_rate in (8000, 16000):
        mode = "nb" if sample_rate == 8000 else "wb"
        try:
            m["pesq_out"] = float(_pesq(sample_rate, ref, est, mode))
            m["pesq_in"] = float(_pesq(sample_rate, ref, mix, mode))
        except Exception as err:  # pesq raises its own error types on silent input
            log.warning("PESQ failed: %s", err)
    if HAVE_STOI:
        m["stoi_out"] = float(_stoi(ref, est, sample_rate, extended=False))
        m["stoi_in"] = float(_stoi(ref, mix, sample_rate, extended=False))
    return m


def summarize(values: np.ndarray) -> dict[str, float]:
    """Mean / median / std of a 1-D array."""
    v = np.asarray(values, dtype=np.float64)
    return {"mean": float(v.mean()), "median": float(np.median(v)), "std": float(v.std()),
            "n": int(len(v))}


def aggregate(per_utt: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-utterance metrics into the summary reported by ``evaluate.py``."""
    keys = [k for k in per_utt[0] if isinstance(per_utt[0][k], float)]
    out: dict[str, Any] = {k: summarize(np.array([u[k] for u in per_utt if k in u])) for k in keys}
    sdri = np.array([u["si_sdri"] for u in per_utt])
    out["nsr"] = float((sdri < 0).mean())
    out["frac_si_sdri_below_5db"] = float((sdri < 5).mean())
    return out
