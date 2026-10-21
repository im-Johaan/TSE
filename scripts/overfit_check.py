"""Sanity check: a model must be able to overfit 32 real training samples.

Loads 32 fixed (mixture crop, enrollment, target) triplets from the training split, trains
for ``--steps`` steps on them and asserts that the SI-SDR on those samples improves by more
than ``--min-gain`` dB (measured in eval mode before and after).

Example:
    python scripts/overfit_check.py --config configs/proposed.yaml
    python scripts/overfit_check.py --config configs/baseline_tdspeakerbeam.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tse.config import load_config, resolve_paths  # noqa: E402
from tse.data.dataset import TSEDataset, collate_tse, prepare_splits  # noqa: E402
from tse.engine import train_step  # noqa: E402
from tse.losses import TSELoss, si_sdr  # noqa: E402
from tse.models import build_model  # noqa: E402
from tse.utils import count_parameters, resolve_device, seed_everything, setup_logging  # noqa: E402

log = logging.getLogger("overfit_check")


@torch.no_grad()
def eval_si_sdr(model: torch.nn.Module, batch: dict, device: torch.device) -> tuple[float, float]:
    """Mean SI-SDR and SI-SDRi of the model on a fixed batch (eval mode)."""
    model.eval()
    est = model(batch["mixture"].to(device), batch["enrollment"].to(device),
                compute_interference=False)["est"]
    out = si_sdr(est, batch["target"].to(device))
    inp = si_sdr(batch["mixture"], batch["target"]).to(device)
    return float(out.mean()), float((out - inp).mean())


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="configs/proposed.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--n-samples", type=int, default=32)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--min-gain", type=float, default=5.0)
    ap.add_argument("--out", default=None, help="optional JSON file for the result")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    setup_logging()
    cfg = resolve_paths(load_config(args.config, args.set), args.data_root)
    seed_everything(cfg["seed"])
    device = resolve_device(cfg["train"]["device"])
    d = cfg["data"]

    data = prepare_splits(d)
    s = data.samples["train"]
    rng = np.random.default_rng(cfg["seed"])
    subset = s.iloc[np.sort(rng.choice(len(s), args.n_samples, replace=False))]
    ds = TSEDataset(subset, d["root"], data.pools["train"], data.spk2label, train=True,
                    sample_rate=d["sample_rate"], segment_seconds=d["segment_seconds"],
                    enroll_seconds=d["enroll_seconds_train"])
    fixed = collate_tse([ds[i] for i in range(len(ds))])  # crops + enrollments drawn once
    n_spk = len(data.spk2label)

    model = build_model(cfg["model"], n_spk, d["sample_rate"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = TSELoss(cfg["loss"]["lambda_ce"], cfg["loss"]["lambda_int"])
    sdr0, imp0 = eval_si_sdr(model, fixed, device)
    log.info("%s (%.2fM params): initial SI-SDR %.2f dB (SI-SDRi %.2f)", cfg["model"]["type"],
             count_parameters(model) / 1e6, sdr0, imp0)
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    skipped = 0
    for step in range(1, args.steps + 1):
        idx = torch.randperm(args.n_samples, generator=g)[:args.batch_size]
        batch = {k: (v[idx] if torch.is_tensor(v) else v) for k, v in fixed.items()}
        res = train_step(model, batch, loss_fn, opt, device, cfg["train"]["amp"],
                         cfg["train"]["grad_clip"])
        skipped += res is None
        if step % 50 == 0:
            sdr, imp = eval_si_sdr(model, fixed, device)
            log.info("step %d  train-batch SI-SDR %.2f  | all-32 eval SI-SDR %.2f (SI-SDRi %.2f)",
                     step, res["si_sdr"] if res else float("nan"), sdr, imp)
    sdr1, imp1 = eval_si_sdr(model, fixed, device)
    gain = sdr1 - sdr0
    result = {"model": cfg["model"]["type"], "config": args.config, "steps": args.steps,
              "n_samples": args.n_samples, "initial_si_sdr": sdr0, "final_si_sdr": sdr1,
              "initial_si_sdri": imp0, "final_si_sdri": imp1, "gain_db": gain,
              "skipped_steps": skipped, "time_s": time.time() - t0, "passed": gain > args.min_gain}
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(f"FAILED: SI-SDR improved by only {gain:.2f} dB (< {args.min_gain})")
    print(f"PASSED: SI-SDR improved by {gain:.2f} dB")


if __name__ == "__main__":
    main()
