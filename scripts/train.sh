#!/usr/bin/env bash
# Run within an existing single-GPU allocation. This script submits no jobs.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
export PYTHONPATH="$repo:$repo/asset${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
python_bin=${PYTHON:-python}
stage=${1:-all}
if (( $# )); then shift; fi
data=${DATA_ROOT:-data/LibriSpeech}
boundaries=${BOUNDARY_ROOT:-data/boundaries/char}
cache=${MODEL_CACHE:-.cache/huggingface}
seed=${SEED:-3407}
out=${OUTPUT_ROOT:-results/asset_ls960}
[[ "$out" == *ls960* ]] || { echo 'OUTPUT_ROOT must contain ls960 (all three training splits).' >&2; exit 1; }
splits='["train-clean-100","train-clean-360","train-other-500"]'
common=(configs/asset.yaml --seed "$seed" --data_folder "$data"
  --boundary_target_dir "$boundaries" --cache_folder "$cache"
  --train_splits "$splits")
echo "LS960 training splits: $splits"
echo "Outputs: $out; model cache: $cache; seed: $seed"
echo 'Default rate channels: aux=True reward=False; task-NLL rewards enabled'
echo 'Recommended allocation: 1 x A100 80 GB; all components share that GPU; stages run sequentially (maximum 1 GPU).'

# Resolve a complete, best-validation checkpoint; never infer it from a date.
best_checkpoint() {
  "$python_bin" - "$1" "$2" <<'PY'
import math
import sys
from pathlib import Path
import yaml
root, metric = Path(sys.argv[1]), sys.argv[2]
required = ['segmenter.ckpt'] if metric == 'boundary_f1' else ['llm.ckpt', 'proj.ckpt', 'normalize.ckpt']
choices = []
for meta in sorted(root.glob('CKPT*/CKPT.yaml')):
    data = yaml.safe_load(meta.read_text())
    value = float(data.get(metric, float('nan')))
    if math.isfinite(value) and all((meta.parent / f).is_file() for f in required):
        choices.append((value if metric == 'WER' else -value, str(meta.parent.resolve())))
if not choices:
    raise SystemExit(f'No complete checkpoint with {metric} metadata in {root}')
print(min(choices)[1])
PY
}

run_stage() {
  local phase=$1
  shift
  local cold decoder
  case "$phase" in
    decoder)
      # Necessary recognizer initialization with transcript-informed boundaries.
      "$python_bin" asset/train.py "${common[@]}" \
        --output_folder "$out/decoder/$seed" --segmenter_mode joint \
        --freeze_boundary_policy True --boundary_source alignment \
        --warmstart_target alignment --segment_pooling mean \
        --warmup_fraction_of_epoch null --warmup_epochs 0 \
        --skip_final_test True "$@"
      ;;
    coldstart)
      "$python_bin" asset/train.py "${common[@]}" \
        --output_folder "$out/coldstart/$seed" --segmenter_mode coldstart \
        --warmstart_target alignment --segment_pooling mean \
        --warmup_fraction_of_epoch null --warmup_epochs 0 "$@"
      ;;
    joint)
      cold=${SEGMENTER_CHECKPOINT:-$(best_checkpoint "$out/coldstart/$seed/save" boundary_f1)/segmenter.ckpt}
      decoder=${DECODER_CHECKPOINT_DIR:-$(best_checkpoint "$out/decoder/$seed/save" WER)}
      "$python_bin" asset/train.py "${common[@]}" \
        --output_folder "$out/joint/$seed" --segmenter_init_checkpoint "$cold" \
        --decoder_init_ckpt_dir "$decoder" "$@"
      ;;
    eval)
      [[ -d "$out/joint/$seed/save" ]] || { echo 'No joint save directory; train first.' >&2; exit 1; }
      "$python_bin" asset/train.py "${common[@]}" \
        --output_folder "$out/joint/$seed" --eval_only True "$@"
      ;;
    *) echo 'Usage: scripts/train.sh {all|decoder|coldstart|joint|eval} [HyperPyYAML overrides]' >&2; return 2 ;;
  esac
}
if [[ "$stage" == all ]]; then
  run_stage decoder "$@"
  run_stage coldstart "$@"
  run_stage joint "$@"
else
  run_stage "$stage" "$@"
fi
