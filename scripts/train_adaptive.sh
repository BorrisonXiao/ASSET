#!/usr/bin/env bash
# WIP continuation within an existing single-GPU allocation; no job submission.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo"
source_checkpoint=${1:?Usage: scripts/train_adaptive.sh SOURCE_CHECKPOINT_DIR [HyperPyYAML overrides]}
shift
export PYTHONPATH="$repo:$repo/asset${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
python_bin=${PYTHON:-python}
data=${DATA_ROOT:-data/LibriSpeech}
cache=${MODEL_CACHE:-.cache/huggingface}
root=${OUTPUT_ROOT:-results/asset_ls960_adaptive}
seed=${SEED:-3407}
arm=${ARM:-adaptive}
case "$arm" in adaptive|original|fixed_lower) ;; *) echo 'ARM must be adaptive, original, or fixed_lower' >&2; exit 1 ;; esac
[[ "$root" == *ls960* ]] || { echo 'OUTPUT_ROOT must contain ls960.' >&2; exit 1; }
output="$root/$arm/$seed"
config="$output/controller.yaml"
controller=${CONTROLLER_CONFIG:-configs/adaptive.yaml}
read -r max_steps interval < <("$python_bin" - "$controller" "$source_checkpoint" "$config" <<'PY'
import sys
from pathlib import Path
import yaml
template, source, destination = map(Path, sys.argv[1:])
source = source.resolve(strict=True)
for name in ('segmenter.ckpt', 'llm.ckpt', 'proj.ckpt', 'normalize.ckpt', 'model_optimizer.ckpt', 'CKPT.yaml'):
    if not (source / name).is_file():
        raise SystemExit(f'Missing source checkpoint component: {source / name}')
config = yaml.safe_load(template.read_text())
config['source_checkpoint'] = str(source)
if destination.exists() and yaml.safe_load(destination.read_text()) != config:
    raise SystemExit('Refusing to change the controller or source checkpoint of an existing run; use a new OUTPUT_ROOT.')
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(yaml.safe_dump(config, sort_keys=False))
print(config['max_steps'], config['validation_interval'])
PY
)
splits='["train-clean-100","train-clean-360","train-other-500"]'
echo "LS960 training splits: $splits"
echo "Output: $output; source: $source_checkpoint; arm: $arm; seed: $seed"
echo "New optimizer updates: $max_steps; validate every $interval updates"
echo 'Default rate channels: aux=True reward=False; task-NLL rewards remain enabled'
echo 'Recommended allocation: 1 x A100 80 GB; all model components share it; maximum 1 GPU, no overlapping jobs.'
exec "$python_bin" asset/train_adaptive.py configs/asset.yaml \
  --continuation-config "$config" --continuation-arm "$arm" \
  --seed "$seed" --output_folder "$output" --data_folder "$data" \
  --cache_folder "$cache" --train_splits "$splits" \
  --dev_splits '["dev-clean"]' --test_splits '["dev-other"]' \
  --test_csv "[\"$output/dev-other.csv\"]" \
  --segmenter_init_checkpoint "$source_checkpoint/segmenter.ckpt" \
  --decoder_init_ckpt_dir "$source_checkpoint" \
  --warmup_epochs 0 --warmup_optimizer_steps 0 --warmup_fraction_of_epoch null \
  --validate_at_warmup_end False --validation_interval_optimizer_steps "$interval" \
  --optimizer_step_limit "$max_steps" --number_of_epochs 10000 \
  --lr_decoder 0.00002 --lr_decoder_warmup 0.00002 --lr_segmenter 0.000005 \
  --ckpt_interval_minutes 30 --noprogressbar "$@"
