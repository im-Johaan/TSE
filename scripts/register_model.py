"""Add a trained model to models.yaml, compare it with the current default, and promote it.

The demos (scripts/web_demo.py, scripts/realtime_server.py) offer every model listed in
models.yaml and start with the `default` one, so swapping in a better model needs no code
changes. The comparison uses the test metrics found next to the checkpoint (or in
--metrics-dir): SI-SDRi and wrong-speaker rate (NSR) on the 2-, 3- and 4-speaker test sets.

Examples:
    # register and compare, without changing the default
    python scripts/register_model.py --id r6_judge --label "R=6 + ECAPA judge" \\
        --ckpt experiments/ecapa_judge/judge/best.pt
    # promote an already registered model
    python scripts/register_model.py --set-default r6_judge
    # just print the comparison of all registered models
    python scripts/register_model.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tse.config import PROJECT_ROOT  # noqa: E402
from tse.serving.registry import DEFAULT_REGISTRY, ModelRegistry, read_metrics  # noqa: E402
from tse.utils import setup_logging  # noqa: E402

SETS = ("2spk", "3spk", "4spk")


def rel(path: str) -> str:
    """Store paths relative to the project root when possible (portable models.yaml)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def table(reg: ModelRegistry, highlight: str | None = None) -> None:
    """Print SI-SDRi / NSR per test set for every registered model."""
    print(f"\n{'model':28s}" + "".join(f"{s + ' SI-SDRi':>14s}{'NSR':>7s}" for s in SETS))
    base = read_metrics(reg.entries[reg.default_id])
    for mid, e in reg.entries.items():
        m = read_metrics(e)
        tag = " *default" if mid == reg.default_id else (" <- new" if mid == highlight else "")
        row = f"{(mid + tag)[:28]:28s}"
        for s in SETS:
            if s in m:
                d = m[s]["si_sdri"] - base[s]["si_sdri"] if s in base and mid != reg.default_id else None
                delta = f"({d:+.2f})" if d is not None else ""
                row += f"{m[s]['si_sdri']:7.2f}{delta:>7s}{100 * m[s]['nsr']:6.1f}%"
            else:
                row += f"{'-':>14s}{'-':>7s}"
        print(row)
    print("(SI-SDRi in dB, change vs the default in brackets; NSR = share of outputs with "
          "SI-SDRi < 0, i.e. failures / wrong speaker)")


def main() -> None:
    """Entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    ap.add_argument("--id", help="new model id (letters, digits, _)")
    ap.add_argument("--label", help="name shown in the demos")
    ap.add_argument("--ckpt", help="checkpoint path")
    ap.add_argument("--backend", default="ueconv")
    ap.add_argument("--metrics-dir", default=None,
                    help="folder with *_test_metrics.json (default: the checkpoint's folder)")
    ap.add_argument("--notes", default="")
    ap.add_argument("--make-default", action="store_true", help="also make it the default")
    ap.add_argument("--replace", action="store_true", help="overwrite an existing id")
    ap.add_argument("--set-default", metavar="ID", help="promote an already registered model")
    ap.add_argument("--list", action="store_true", help="only print the comparison table")
    args = ap.parse_args()
    setup_logging()

    reg = ModelRegistry(args.registry)
    new_id = None
    if args.set_default:
        old = reg.default_id
        reg.set_default(args.set_default)
        print(f"Default model: {old} -> {args.set_default}")
    elif not args.list:
        if not (args.id and args.label and args.ckpt):
            ap.error("--id, --label and --ckpt are required to register a model")
        entry = {"id": args.id, "label": args.label, "backend": args.backend,
                 "checkpoint": rel(args.ckpt)}
        if args.metrics_dir:
            entry["metrics_dir"] = rel(args.metrics_dir)
        if args.notes:
            entry["notes"] = args.notes
        if not read_metrics(entry):
            print("warning: no test metrics found for this model; evaluate it first "
                  "(scripts/evaluate.py) to compare it fairly")
        reg.add(entry, make_default=args.make_default, replace=args.replace)
        new_id = args.id
        print(f"Registered {args.id} in {reg.path}" + (" as the default" if args.make_default else ""))
    table(reg, new_id)


if __name__ == "__main__":
    main()
