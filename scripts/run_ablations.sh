#!/usr/bin/env bash
# Train + evaluate every ablation of the proposed model (one run per switch).
#
# Usage (from tse_project/, Git Bash on Windows):
#   bash scripts/run_ablations.sh                       # all ablations, full recipe
#   MAX_EPOCHS=30 bash scripts/run_ablations.sh         # shorter budget for every run
#   ONLY="abl_no_mgfa abl_mgfa_scalar" bash scripts/run_ablations.sh
#   DATA_ROOT=/path/to/Libri2Mix bash scripts/run_ablations.sh
#   CONFIG=configs/debug.yaml EVAL_ARGS="--max-samples 10" bash scripts/run_ablations.sh   # smoke test
#
# Already-finished runs (with test_metrics.json) are skipped; interrupted runs resume.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x .venv/Scripts/python.exe ]]; then PYTHON=.venv/Scripts/python.exe
  elif [[ -x .venv/bin/python ]]; then PYTHON=.venv/bin/python
  else PYTHON=python; fi
fi
CONFIG=${CONFIG:-configs/proposed.yaml}
read -r -a EVAL_EXTRA <<< "${EVAL_ARGS:-}"
EXTRA=()
[[ -n "${MAX_EPOCHS:-}" ]] && EXTRA+=("train.max_epochs=${MAX_EPOCHS}")
ROOT_ARG=()
[[ -n "${DATA_ROOT:-}" ]] && ROOT_ARG=(--data-root "${DATA_ROOT}")

declare -A ABL=(
  [abl_sep_tcn]="model.separator=tcn"
  [abl_fusion_mult]="model.fusion=mult"
  [abl_fusion_concat]="model.fusion=concat"
  [abl_inject_input]="model.inject=input_only"
  [abl_no_ges_int]="model.use_ges_interference=false loss.lambda_int=0.0"
  [abl_no_mgfa]="model.use_mgfa=false"
  [abl_mgfa_scalar]="model.mgfa_beta=scalar"
  [abl_no_ce]="loss.lambda_ce=0.0"
  [abl_ecapa]="model.spk_encoder=ecapa_frozen"
)
ORDER=(abl_sep_tcn abl_fusion_mult abl_fusion_concat abl_inject_input abl_no_ges_int
       abl_no_mgfa abl_mgfa_scalar abl_no_ce abl_ecapa)
[[ -n "${ONLY:-}" ]] && read -r -a ORDER <<< "${ONLY}"

for name in "${ORDER[@]}"; do
  if [[ "$name" == abl_ecapa ]] && ! "$PYTHON" -c "import speechbrain" 2>/dev/null; then
    echo ">>> skipping $name: speechbrain is not installed"; continue
  fi
  if [[ -f "runs/$name/test_metrics.json" ]]; then
    echo ">>> $name already evaluated, skipping"; continue
  fi
  echo ">>> training $name (${ABL[$name]})"
  resume=()
  [[ -f "runs/$name/last.pt" ]] && resume=(--resume)
  # shellcheck disable=SC2086
  "$PYTHON" scripts/train.py --config "$CONFIG" --name "$name" "${ROOT_ARG[@]}" "${resume[@]}" \
      --set ${ABL[$name]} "${EXTRA[@]}"
  "$PYTHON" scripts/evaluate.py --ckpt "runs/$name/best.pt" "${ROOT_ARG[@]}" "${EVAL_EXTRA[@]}"
done

echo ">>> summary (test SI-SDRi mean / NSR)"
"$PYTHON" - <<'EOF'
import json, pathlib
for p in sorted(pathlib.Path("runs").glob("*/test_metrics.json")):
    o = json.loads(p.read_text())["overall"]
    print(f"{p.parent.name:<26} SI-SDRi {o['si_sdri']['mean']:6.2f} dB   NSR {o['nsr']*100:5.2f}%")
EOF
