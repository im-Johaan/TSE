"""Compare test metrics of several checkpoints on the 2-, 3- and 4-speaker test sets.

Reads ``test_metrics.json``, ``Libri3Mix_test_metrics.json`` and ``Libri4Mix_test_metrics.json``
from each given run directory (missing files are shown as '-').

Example:
    python scripts/compare_results.py saved_models/proposed_2spk_ep74 runs/finetune_multispk
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SETS = [("2 spk (Libri2Mix)", "test_metrics.json"),
        ("3 spk (Libri3Mix)", "Libri3Mix_test_metrics.json"),
        ("4 spk (Libri4Mix*)", "Libri4Mix_test_metrics.json")]
COLS = [("SI-SDRi", lambda o: o["si_sdri"]["mean"], "{:6.2f}"),
        ("median", lambda o: o["si_sdri"]["median"], "{:6.2f}"),
        ("SI-SDR out", lambda o: o["si_sdr_out"]["mean"], "{:6.2f}"),
        ("SDRi", lambda o: o["sdri"]["mean"] if "sdri" in o else float("nan"), "{:6.2f}"),
        ("STOI", lambda o: o["stoi_out"]["mean"] if "stoi_out" in o else float("nan"), "{:5.3f}"),
        ("NSR %", lambda o: 100 * o["nsr"], "{:5.1f}"),
        ("<5dB %", lambda o: 100 * o["frac_si_sdri_below_5db"], "{:5.1f}")]


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories containing *_metrics.json")
    args = ap.parse_args()
    width = max(len(Path(r).name) for r in args.runs) + 2
    header = f"{'test set':<20}{'model':<{width}}" + "".join(f"{c[0]:>11}" for c in COLS)
    print(header)
    print("-" * len(header))
    for set_name, fname in SETS:
        for r in args.runs:
            f = Path(r) / fname
            cells = ["-".rjust(11)] * len(COLS)
            if f.is_file():
                o = json.loads(f.read_text())["overall"]
                cells = [fmt.format(get(o)).rjust(11) for _, get, fmt in COLS]
            print(f"{set_name:<20}{Path(r).name:<{width}}" + "".join(cells))
        print()
    print("* Libri4Mix = our 4-speaker extension of Libri3Mix (not an official benchmark).")


if __name__ == "__main__":
    main()
