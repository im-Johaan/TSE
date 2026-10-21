"""3-speaker TSE demo: add a third (unseen) test speaker to Libri2Mix test mixtures.

For each example, the target is one speaker of a Libri2Mix test mixture and the enrollment is
that sample's fixed test enrollment. A third speaker (a test speaker different from both) is
added at the same RMS as the target (0 dB). The model runs on the original 2-speaker mixture
and on the 3-speaker mixture with the same enrollment, so the effect of the extra interferer
is directly visible.

Outputs (in ``--out-dir``):
  ex{k}/  mix_2spk.wav mix_3spk.wav enrollment.wav target_clean.wav est_2spk.wav est_3spk.wav
          interferer_1.wav interferer_2.wav
  examples.csv, stats_per_sample.csv, summary.json, index.html (listening page)

The model was trained on 2-speaker mixtures only; this is an out-of-distribution test.
This is an ad-hoc construction, not the official Libri3Mix test set.

Example:
    python scripts/demo_3spk.py --ckpt runs/proposed/best.pt
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from tse.config import PROJECT_ROOT  # noqa: E402
from tse.data.dataset import TSEDataset, prepare_splits, read_audio  # noqa: E402
from tse.data.index import load_speaker_genders  # noqa: E402
from tse.metrics import si_sdr_np  # noqa: E402
from tse.utils import load_model_from_checkpoint, resolve_device, setup_logging  # noqa: E402

log = logging.getLogger("demo_3spk")


def rms(x: np.ndarray) -> float:
    """RMS over the non-silent (non-zero-padded) part of ``x``."""
    active = x[np.abs(x) > 1e-6]
    return float(np.sqrt(np.mean(active ** 2))) if len(active) else 0.0


def add_third_speaker(item: dict, pool: dict, root: Path, rng: np.random.Generator,
                      gain_db: float = 0.0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return (3-speaker mixture, scaled third source, info) for one 2-speaker test item."""
    mix = item["mixture"].numpy()
    tgt = item["target"].numpy()
    meta = item["meta"]
    other = meta["mixture_id"].split("_")[1 if meta["role"] == "s1" else 0].split("-")[0]
    cands = [s for s in sorted(pool) if s not in (meta["speaker_id"], other)]
    spk3 = cands[int(rng.integers(len(cands)))]
    utt3, path3, _ = pool[spk3][int(rng.integers(len(pool[spk3])))]
    s3 = read_audio(root / path3)
    n = len(mix)
    if len(s3) >= n:
        start = int(rng.integers(len(s3) - n + 1))
        s3 = s3[start:start + n]
    else:  # shorter: place at a random offset, zero-padded
        off = int(rng.integers(n - len(s3) + 1))
        s3 = np.pad(s3, (off, n - len(s3) - off))
    s3 = s3 * (rms(tgt) / max(rms(s3), 1e-8)) * 10 ** (gain_db / 20)
    return (mix + s3).astype(np.float32), s3.astype(np.float32), {
        "other_speaker": other, "third_speaker": spk3, "third_utt": utt3}


@torch.no_grad()
def extract(model: torch.nn.Module, mix: np.ndarray, enr: torch.Tensor,
            device: torch.device) -> np.ndarray:
    """Run the model on one mixture (numpy) with one enrollment (tensor)."""
    x = torch.from_numpy(mix)[None].to(device)
    return model(x, enr[None].to(device), compute_interference=False)["est"][0].cpu().numpy()


def write_norm(path: Path, x: np.ndarray, sr: int, peak: float = 0.9) -> None:
    """Write ``x`` peak-normalised (the model output scale is arbitrary under SI-SDR)."""
    sf.write(str(path), (x / max(np.abs(x).max(), 1e-8) * peak).astype(np.float32), sr)


def build_html(rows: list[dict], summary: dict, out: Path) -> None:
    """Write a self-contained local listening page that references the wav files."""
    def audio(rel: str) -> str:
        return f'<audio controls preload="none" src="{html.escape(rel)}"></audio>'

    cards = []
    for r in rows:
        k = r["example"]
        cards.append(f"""
<section class="card">
  <h2>Example {k}</h2>
  <p class="meta">Target speaker <b>{r['target_speaker']}</b> ({r['target_gender']}) &middot;
  interferers {r['other_speaker']} ({r['other_gender']}) + <b>{r['third_speaker']}</b>
  ({r['third_gender']}, added) &middot; {r['length_s']:.1f} s</p>
  <table>
    <tr><th></th><th>2 speakers (as trained)</th><th>3 speakers (new)</th></tr>
    <tr><td>Mixture (input)</td><td>{audio(f'ex{k}/mix_2spk.wav')}</td><td>{audio(f'ex{k}/mix_3spk.wav')}</td></tr>
    <tr><td>Model output</td><td>{audio(f'ex{k}/est_2spk.wav')}</td><td>{audio(f'ex{k}/est_3spk.wav')}</td></tr>
    <tr><td>SI-SDR in &rarr; out</td><td>{r['si_sdr_in_2spk']:.1f} &rarr; {r['si_sdr_out_2spk']:.1f} dB</td>
        <td>{r['si_sdr_in_3spk']:.1f} &rarr; {r['si_sdr_out_3spk']:.1f} dB</td></tr>
    <tr><td>SI-SDRi</td><td class="num">{r['si_sdri_2spk']:+.1f} dB</td><td class="num">{r['si_sdri_3spk']:+.1f} dB</td></tr>
  </table>
  <div class="refs">
    <div><span>Reference (enrollment, 6 s)</span>{audio(f'ex{k}/enrollment.wav')}</div>
    <div><span>Clean target (ground truth)</span>{audio(f'ex{k}/target_clean.wav')}</div>
    <div><span>Added 3rd speaker alone</span>{audio(f'ex{k}/interferer_2.wav')}</div>
  </div>
</section>""")
    s2, s3 = summary["stats"]["2spk"], summary["stats"]["3spk"]
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3-Speaker TSE Demo</title>
<style>
:root {{ --bg:#fafaf9; --fg:#1c1917; --muted:#57534e; --card:#fff; --line:#e7e5e4; --acc:#0f766e; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#1c1917; --fg:#f5f5f4; --muted:#a8a29e; --card:#292524; --line:#44403c; --acc:#5eead4; }} }}
body {{ background:var(--bg); color:var(--fg); font:15px/1.5 system-ui,sans-serif; margin:0; padding:24px 16px; }}
main {{ max-width:980px; margin:0 auto; }}
h1 {{ margin:0 0 4px; font-size:24px; }} h2 {{ margin:0 0 4px; font-size:18px; }}
.lead, .meta {{ color:var(--muted); }}
.card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:16px; margin:16px 0; }}
table {{ border-collapse:collapse; width:100%; }} th, td {{ text-align:left; padding:6px 8px; border-top:1px solid var(--line); vertical-align:middle; }}
th {{ color:var(--muted); font-weight:600; }} .num {{ font-weight:700; color:var(--acc); }}
audio {{ width:100%; max-width:300px; height:36px; }}
.refs {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:12px; }}
.refs div {{ display:flex; flex-direction:column; gap:4px; flex:1 1 240px; }} .refs span {{ color:var(--muted); font-size:13px; }}
.stats td:not(:first-child), .stats th:not(:first-child) {{ text-align:right; font-variant-numeric:tabular-nums; }}
@media (max-width:640px) {{ td, th {{ display:block; border:0; padding:4px 0; }} tr {{ display:block; border-top:1px solid var(--line); padding:6px 0; }} }}
</style></head><body><main>
<h1>3-speaker target speaker extraction</h1>
<p class="lead">Checkpoint: <code>{html.escape(summary['checkpoint'])}</code> (epoch {summary['ckpt_epoch']},
val SI-SDRi {summary['ckpt_val_si_sdri']:.2f} dB on 2-speaker dev). The model was trained on
<b>2-speaker</b> mixtures only. Each 3-speaker mixture is a Libri2Mix test mixture plus a third unseen test
speaker at the target's level (0 dB). The same reference is used for both versions. All speakers are unseen in
training. Audio files are peak-normalised for listening, since the output scale of an SI-SDR-trained model
is arbitrary.</p>
<section class="card"><h2>Paired comparison on {s2['n']} random test samples</h2>
<table class="stats"><tr><th></th><th>2 speakers</th><th>3 speakers</th></tr>
<tr><td>Input SI-SDR (mean)</td><td>{s2['si_sdr_in_mean']:.2f} dB</td><td>{s3['si_sdr_in_mean']:.2f} dB</td></tr>
<tr><td>Output SI-SDR (mean)</td><td>{s2['si_sdr_out_mean']:.2f} dB</td><td>{s3['si_sdr_out_mean']:.2f} dB</td></tr>
<tr><td>SI-SDRi mean / median</td><td>{s2['si_sdri_mean']:.2f} / {s2['si_sdri_median']:.2f} dB</td><td>{s3['si_sdri_mean']:.2f} / {s3['si_sdri_median']:.2f} dB</td></tr>
<tr><td>NSR (SI-SDRi &lt; 0)</td><td>{s2['nsr'] * 100:.1f}%</td><td>{s3['nsr'] * 100:.1f}%</td></tr>
<tr><td>Output SI-SDR &lt; 0 dB</td><td>{s2['out_below_0'] * 100:.1f}%</td><td>{s3['out_below_0'] * 100:.1f}%</td></tr>
</table></section>
{''.join(cards)}
</main></body></html>"""
    out.write_text(page, encoding="utf-8")


def summarize(df: pd.DataFrame, tag: str) -> dict:
    """Summary statistics for one condition."""
    return {"n": int(len(df)), "si_sdr_in_mean": float(df[f"si_sdr_in_{tag}"].mean()),
            "si_sdr_out_mean": float(df[f"si_sdr_out_{tag}"].mean()),
            "si_sdri_mean": float(df[f"si_sdri_{tag}"].mean()),
            "si_sdri_median": float(df[f"si_sdri_{tag}"].median()),
            "nsr": float((df[f"si_sdri_{tag}"] < 0).mean()),
            "out_below_0": float((df[f"si_sdr_out_{tag}"] < 0).mean())}


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ckpt", default=str(PROJECT_ROOT / "runs" / "proposed" / "best.pt"))
    ap.add_argument("--n-examples", type=int, default=6, help="examples saved as audio")
    ap.add_argument("--n-stats", type=int, default=300, help="random samples for the paired stats")
    ap.add_argument("--gain-db", type=float, default=0.0, help="3rd speaker level vs the target")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "demo_3spk"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    setup_logging()

    device = resolve_device(args.device)
    model, ckpt = load_model_from_checkpoint(args.ckpt, device)
    st = ckpt.get("state", {})
    cfg = ckpt["config"]
    d = cfg["data"]
    sr = d["sample_rate"]
    data = prepare_splits(d)
    genders = load_speaker_genders(d.get("gender_file"), Path(d["root"])) or {}
    samples = data.samples["test"]
    ds = TSEDataset(samples, d["root"], None, data.spk2label, train=False, sample_rate=sr,
                    enroll_seconds=d["enroll_seconds_eval"])
    rng = np.random.default_rng(args.seed)
    n_total = max(args.n_stats, args.n_examples)
    picks = rng.choice(len(ds), size=n_total, replace=False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for j, i in enumerate(picks):
        item = ds[int(i)]
        meta = item["meta"]
        mix3, s3, info = add_third_speaker(item, data.pools["test"], Path(d["root"]), rng,
                                           args.gain_db)
        mix2, tgt = item["mixture"].numpy(), item["target"].numpy()
        peak = max(np.abs(mix3).max(), 1e-8)
        if peak > 0.99:  # avoid clipping in the saved files; SI-SDR is scale-invariant
            mix3, s3 = mix3 * 0.99 / peak, s3 * 0.99 / peak
        est2 = extract(model, mix2, item["enrollment"], device)
        est3 = extract(model, mix3, item["enrollment"], device)
        row = {
            "example": j + 1 if j < args.n_examples else None, "mixture_id": meta["mixture_id"],
            "role": meta["role"], "target_speaker": meta["speaker_id"],
            "other_speaker": info["other_speaker"], "third_speaker": info["third_speaker"],
            "third_utt": info["third_utt"], "enroll_utt": meta["enroll_utt"],
            "target_gender": genders.get(meta["speaker_id"], "?"),
            "other_gender": genders.get(info["other_speaker"], "?"),
            "third_gender": genders.get(info["third_speaker"], "?"),
            "length_s": len(mix2) / sr,
            "si_sdr_in_2spk": si_sdr_np(mix2, tgt), "si_sdr_out_2spk": si_sdr_np(est2, tgt),
            "si_sdr_in_3spk": si_sdr_np(mix3, tgt), "si_sdr_out_3spk": si_sdr_np(est3, tgt),
        }
        for t in ("2spk", "3spk"):
            row[f"si_sdri_{t}"] = row[f"si_sdr_out_{t}"] - row[f"si_sdr_in_{t}"]
        rows.append(row)
        if j < args.n_examples:
            ex = out_dir / f"ex{j + 1}"
            ex.mkdir(exist_ok=True)
            write_norm(ex / "mix_2spk.wav", mix2, sr)
            write_norm(ex / "mix_3spk.wav", mix3, sr)
            write_norm(ex / "enrollment.wav", item["enrollment"].numpy(), sr)
            write_norm(ex / "target_clean.wav", tgt, sr)
            write_norm(ex / "est_2spk.wav", est2, sr)
            write_norm(ex / "est_3spk.wav", est3, sr)
            write_norm(ex / "interferer_1.wav", item["interference"].numpy(), sr)
            write_norm(ex / "interferer_2.wav", s3, sr)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stats_per_sample.csv", index=False)
    examples = df[df["example"].notna()].copy()
    examples["example"] = examples["example"].astype(int)
    examples.to_csv(out_dir / "examples.csv", index=False)
    same3 = df["third_gender"] == df["target_gender"]
    summary = {
        "checkpoint": str(Path(args.ckpt).resolve()), "ckpt_epoch": st.get("epoch"),
        "ckpt_val_si_sdri": st.get("val", {}).get("si_sdri", float("nan")),
        "third_speaker_gain_db": args.gain_db, "seed": args.seed,
        "stats": {"2spk": summarize(df, "2spk"), "3spk": summarize(df, "3spk")},
        "3spk_by_third_speaker_gender": {
            "same_as_target": summarize(df[same3], "3spk"),
            "different_from_target": summarize(df[~same3], "3spk")},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    build_html(examples.to_dict("records"), summary, out_dir / "index.html")

    print(f"\nCheckpoint epoch {summary['ckpt_epoch']} (val SI-SDRi "
          f"{summary['ckpt_val_si_sdri']:.2f} dB on 2-speaker dev)")
    cols = ["example", "target_speaker", "target_gender", "other_gender", "third_gender",
            "length_s", "si_sdr_in_2spk", "si_sdri_2spk", "si_sdr_in_3spk", "si_sdr_out_3spk",
            "si_sdri_3spk"]
    print(examples[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print(f"\nPaired stats on {len(df)} random test samples:")
    for t in ("2spk", "3spk"):
        s = summary["stats"][t]
        print(f"  {t}: input SI-SDR {s['si_sdr_in_mean']:6.2f}  output {s['si_sdr_out_mean']:6.2f}  "
              f"SI-SDRi mean {s['si_sdri_mean']:6.2f} median {s['si_sdri_median']:6.2f}  "
              f"NSR {s['nsr'] * 100:5.1f}%  out<0dB {s['out_below_0'] * 100:5.1f}%")
    for k, s in summary["3spk_by_third_speaker_gender"].items():
        print(f"  3spk, 3rd speaker gender {k:<22} n={s['n']:<4} SI-SDRi {s['si_sdri_mean']:6.2f}  "
              f"NSR {s['nsr'] * 100:5.1f}%")
    print(f"\nListening page: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
