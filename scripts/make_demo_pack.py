"""Build demo_assets/: a few MB of LibriSpeech test-speaker clips for the demos.

The demos' "Random speakers" / "Test mixture" buttons normally draw from the Libri2Mix test
set. This script copies a small, fixed selection (speakers never seen in training, 8 kHz
clean sources) into demo_assets/ so those buttons also work on a machine without the
dataset. The servers fall back to it automatically (tse.demo_app.load_speaker_pool).

Example:
    python scripts/make_demo_pack.py --speakers 16 --utts 3
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from tse.config import PROJECT_ROOT, load_config, resolve_paths  # noqa: E402
from tse.data.dataset import read_audio  # noqa: E402
from tse.demo_app import SpeakerPool  # noqa: E402
from tse.utils import setup_logging  # noqa: E402

log = logging.getLogger("make_demo_pack")

README = """# Demo pack

{n_spk} speakers x {n_utt} utterances from the LibriSpeech test-clean speakers, taken from
the clean Libri2Mix test sources at 8 kHz. These speakers were never seen in training. The
demos use these clips for "Random speakers" / "Test mixture" when the full dataset is absent.

Source: LibriSpeech (Panayotov et al., 2015), CC BY 4.0, https://www.openslr.org/12, via
LibriMix (Cosentino et al., 2020), https://github.com/JorisCos/LibriMix.
Regenerate with `python scripts/make_demo_pack.py`.
"""


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--speakers", type=int, default=16, help="number of speakers (half per gender)")
    ap.add_argument("--utts", type=int, default=3, help="utterances per speaker")
    ap.add_argument("--min-seconds", type=float, default=4.0)
    ap.add_argument("--max-seconds", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(PROJECT_ROOT / "demo_assets"))
    args = ap.parse_args()
    setup_logging()

    cfg = resolve_paths(load_config(PROJECT_ROOT / "configs" / "proposed.yaml"))
    pool = SpeakerPool(cfg["data"], "test")
    sr = cfg["data"]["sample_rate"]
    rng = np.random.default_rng(args.seed)
    lo, hi = int(args.min_seconds * sr), int(args.max_seconds * sr)
    by_gender: dict[str, list[str]] = {}
    for s in pool.speakers:
        long_utts = [u for u in pool.pool[s] if u[2] >= lo]
        if len(long_utts) >= args.utts:
            by_gender.setdefault(pool.genders.get(s, "?"), []).append(s)
    chosen = []
    for g in sorted(by_gender):
        spks = by_gender[g]
        k = min(len(spks), args.speakers // max(len(by_gender), 1))
        chosen += [spks[int(i)] for i in sorted(rng.choice(len(spks), size=k, replace=False))]

    out = Path(args.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for spk in chosen:
        utts = [u for u in pool.pool[spk] if u[2] >= lo]
        for i in sorted(rng.choice(len(utts), size=args.utts, replace=False)):
            utt, path, _ = utts[int(i)]
            x = read_audio(pool.root / path)[:hi]
            rel = f"audio/{utt}.wav"
            sf.write(out / rel, x, sr, subtype="PCM_16")
            rows.append({"speaker": spk, "gender": pool.genders.get(spk, "?"), "utt": utt,
                         "path": rel, "length": len(x)})
    with open(out / "speakers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["speaker", "gender", "utt", "path", "length"])
        w.writeheader()
        w.writerows(rows)
    (out / "README.md").write_text(README.format(n_spk=len(chosen), n_utt=args.utts), encoding="utf-8")
    size = sum(f.stat().st_size for f in (out / "audio").iterdir()) / 1e6
    genders = {g: sum(pool.genders.get(s) == g for s in chosen) for g in by_gender}
    print(f"Wrote {len(rows)} clips of {len(chosen)} speakers {genders} to {out} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
