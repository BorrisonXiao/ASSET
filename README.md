# ASSET

**Adaptive Speech Segmentation via Reinforcement Learning**

[Paper (arXiv:2610.14309)](https://arxiv.org/abs/2610.14309) · [Citation](#citation)

ASSET learns hard, variable-length speech segments from transcription feedback.
A frozen WavLM-Large encoder produces frame features; a local-history
autoregressive Transformer predicts segment boundaries; a residual BiGRU pools
each segment into an audio token. A projection feeds these tokens to a frozen
Llama-3.2-1B-Instruct base with trainable LoRA adapters. Joint training combines
recognizer cross-entropy, on-policy GRPO with transcription-NLL rewards, and an
auxiliary rate penalty.

This repository contains the final English LibriSpeech training path, boundary
preparation, and checkpoint evaluation. The `main` branch uses a fixed rate band.
The validation-driven rate curriculum is a **work in progress** on
[`wip/adaptive-training`](https://github.com/BorrisonXiao/ASSET/tree/wip/adaptive-training).
It has shown superior compression in preliminary experiments: about 18.5% fewer
test-other audio tokens with a small WER increase. The branch README gives the
comparison and its limitations.

## Installation

Use Python 3.11 and a CUDA-capable PyTorch environment. The checked runtime uses
PyTorch/torchaudio 2.7.1, Transformers 5.14.1, HyperPyYAML 1.2.3, and Accelerate
1.14.0. For the CUDA 11.8 build used during development:

```bash
git clone https://github.com/BorrisonXiao/ASSET.git
cd ASSET
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.7.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -e .
```

Accept the access terms for
[`meta-llama/Llama-3.2-1B-Instruct`](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
and authenticate with `hf auth login` before the first model load. WavLM is
[`microsoft/wavlm-large`](https://huggingface.co/microsoft/wavlm-large).
Model weights download into `MODEL_CACHE`, or `.cache/huggingface` by default.
For a fully populated offline cache, set `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`.

The included `speechbrain/` directory is a reduced runtime with local changes;
do not install another SpeechBrain distribution into the same environment.

## Data and boundary targets

Obtain [LibriSpeech](https://www.openslr.org/12) separately. `DATA_ROOT` must point
to a directory containing the three training splits (`train-clean-100`,
`train-clean-360`, `train-other-500`), `dev-clean`, `test-clean`, and `test-other`.
Keep the original speaker/chapter directory layout and `.trans.txt` files.

```bash
export DATA_ROOT=/path/to/LibriSpeech
export BOUNDARY_ROOT=/path/to/boundaries/char
export MODEL_CACHE=/path/to/model-cache
export OUTPUT_ROOT=/path/to/results/asset_ls960
export SEED=3407

# Character-CTC forced alignments for recognizer and policy initialization.
# --out_root is the parent of the char directory in BOUNDARY_ROOT.
python asset/prepare_boundaries.py \
  --audio_root "$DATA_ROOT" --out_root "$(dirname "$BOUNDARY_ROOT")" \
  --levels char --splits train-clean-100 train-clean-360 train-other-500 \
  dev-clean --device cuda
```

Boundary generation uses torchaudio's `WAV2VEC2_ASR_BASE_960H` model, which
downloads separately through PyTorch's cache (`TORCH_HOME` controls that cache).
Each `<utterance-id>.pt` is a one-dimensional uint8 tensor containing one 0/1
label per encoder frame; 1 starts a segment. Frame 0 always starts a segment.
The generator verifies the alignment frame grid. These transcript-informed
targets are used only during initialization; final inference requires audio.
CSV manifests are generated locally in each run directory.

## Training and evaluation

Run the following inside an existing GPU allocation. The launcher submits no
cluster jobs and uses one GPU at a time. The development configuration targets
**1 × A100 80 GB**: WavLM, the policy, BiGRU, projection, and Llama/LoRA share that
GPU. Stages execute sequentially, so maximum concurrent use is one GPU.
Smaller GPUs may require a lower `--max_batch_length_train` and higher
`--grad_accumulation_factor`; those settings have not been validated here.

```bash
bash scripts/train.sh all
```

The stages can also run separately:

```bash
bash scripts/train.sh decoder    # Recognizer CE on character-aligned mean pools
bash scripts/train.sh coldstart  # Boundary-policy BCE against character targets
bash scripts/train.sh joint      # Decoder/pooler warmup, then CE + GRPO + rate loss
bash scripts/train.sh eval       # Recover best dev-clean WER checkpoint and test
```

`decoder` trains the projection and LoRA parameters needed to initialize the
recognizer. `coldstart` trains the Transformer boundary policy. `joint` selects
their checkpoints by dev-clean WER and boundary F1, respectively. It adds a
BiGRU whose residual output projection starts at zero, preserving mean pooling
at initialization. The first 20% of joint-training updates adapt the recognizer
and pooler with the boundary policy frozen; subsequent updates train all four
components. WavLM and the Llama base stay frozen throughout.

Outputs live under `$OUTPUT_ROOT/{decoder,coldstart,joint}/$SEED/`. Checkpoints
are in `save/`, logs in `train_log.txt`, and final per-utterance test WER records
in `wer_results/`. Plan for several GB of checkpoints and a separate model
cache; corpus audio is stored separately. Repeating a stage with the same
output path resumes its checkpoint, so use a new output root for a new run.
Keep the configuration and rate channel unchanged when resuming.

To use existing initialization checkpoints, set `SEGMENTER_CHECKPOINT` to the
exact `segmenter.ckpt` file and `DECODER_CHECKPOINT_DIR` to the directory
containing `llm.ckpt`, `proj.ckpt`, and `normalize.ckpt`, then run `joint`.
Additional HyperPyYAML overrides follow the stage name:

```bash
SEED=3408 bash scripts/train.sh all --max_batch_length_train 200
```

## Canonical defaults

[`configs/asset.yaml`](configs/asset.yaml) is the single source of model and
joint-training defaults. The launcher overrides only stage behavior, paths,
and seed; it explicitly prints all three LS960 training splits. Each stage
defaults to one full pass over the 960h training set. This is a release recipe;
no new full training run was performed to certify these cleaned defaults.

| Setting | Default |
|---|---|
| Encoder / features | Frozen WavLM-Large / 1,024 dimensions |
| Decoder | Llama-3.2-1B-Instruct, frozen base, LoRA rank 16 |
| Boundary policy | Transformer AR, width 256, 4 layers, 4 heads, FFN 1,024 |
| Policy history / dropout / KV cache | 64 frames / 0 / preallocated |
| Pooling | Mean + residual BiGRU, hidden size 128, 1 layer |
| GRPO | 4 fresh samples per utterance; standard-deviation normalization |
| Task reward | Negative teacher-forced transcription NLL |
| Rate penalty | Auxiliary loss only; band 0.15–0.25 kept-frame ratio |
| Nominal rate band | 7.5–12.5 audio tokens/s at a 50 Hz encoder grid |
| Decoder / policy learning rates | 2e-4 / 5e-5, AdamW, weight decay 0 |
| Warmup | First 20% of joint-training optimizer updates |
| Training batch budget / accumulation | 300 seconds / 1 |
| Precision / gradient clipping | BF16 / norm 1 |
| Evaluation batch size | 8 |
| Entropy bonus | 0 |

The auxiliary rate objective for the AR policy uses conditional boundary
probabilities along sampled histories as a differentiable surrogate. The
realized token rate counts hard segments. Greedy decoding packs valid audio
prefixes and uses per-utterance duration limits independent of the number of
pooled tokens, avoiding a rate-dependent decoding budget.

`rate_channel: aux` is the current default. `reward` and `both` are explicit
experimental overrides. Historical ASSET and adaptive-continuation results
used reward-side rate penalties; this release does not claim that an auxiliary
run reproduces those numbers. A 100h configuration is a separate replication,
not an LS960 scale-up.

## Repository layout

```text
asset/                  Final trainer, policy, poolers, decoder and data helpers
configs/asset.yaml      Canonical final-system configuration
scripts/train.sh        Portable initialization / training / evaluation launcher
speechbrain/            Required SpeechBrain runtime subset
LICENSE, NOTICE         License and upstream attribution
```

Datasets, model weights, experiment outputs, reports, debug tools, baseline
launchers, manuscript drafts, and the research workspace's test suite are not
part of this release. The runtime subset retains original author notices;
see [NOTICE](NOTICE) and the [Apache-2.0 license](LICENSE).

## Citation

If you use ASSET in your research, please cite our
[paper](https://arxiv.org/abs/2610.14309):

```bibtex
@misc{xiao2026asset,
  title         = {{ASSET}: Adaptive Speech Segmentation via Reinforcement Learning},
  author        = {Xiao, Cihan and Tseng, Yuan and Wiesner, Matthew and Raj, Desh and
                   Garcia Perera, Leibny Paola and Khudanpur, Sanjeev and Harwath, David},
  year          = {2026},
  eprint        = {2610.14309},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.14309}
}
```
