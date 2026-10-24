"""LS960 continuation with paired dev validation and a held rate penalty.

Uses joint CE + task-reward GRPO + auxiliary rate loss by default. This entry
point controls the rate band's upper edge and checkpoint selection. Test splits
are deliberately not evaluated while choosing the pilot curriculum.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import yaml

import speechbrain as sb
from speechbrain.utils.logger import get_logger
from adaptive_frequency import FrequencyConfig, FrequencyController
from train import SegmenterASR, main as base_main

logger = get_logger(__name__)


class RandomState:
    """Preserve training randomness across validation and checkpoint recovery."""

    @staticmethod
    def capture():
        return dict(python=random.getstate(), numpy=np.random.get_state(),
                    torch=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

    @staticmethod
    def restore(state):
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"]:
            torch.cuda.set_rng_state_all(state["cuda"])

    def save(self, path):
        torch.save(self.capture(), path)

    def load(self, path, end_of_epoch=False):
        del end_of_epoch
        # Only our own trusted local checkpoint includes Python/NumPy RNG tuples.
        self.restore(torch.load(path, map_location="cpu", weights_only=False))


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def configure_rate_channel(hparams):
    """Honor explicit overrides; never overwrite the requested loss routing.

    Older continuation manifests predate this field and used reward-only. Refuse
    to silently switch their objective if an old directory is resumed.
    """
    choice = str(hparams.get("rate_channel", "aux")).lower()
    choice = "aux" if choice == "auto" else choice
    if choice not in {"aux", "reward", "both"}:
        raise ValueError(f"Unknown rate_channel: {choice!r}")
    previous = Path(hparams["output_folder"]) / "continuation_config.json"
    if previous.is_file():
        saved = json.loads(previous.read_text()).get("rate_channel", "reward")
        if saved != choice:
            raise ValueError(
                f"Cannot change rate_channel on resume ({saved} -> {choice}); "
                "use a new output directory for a new experiment"
            )
    hparams["rate_channel"] = choice
    return choice


class AdaptiveSegmenterASR(SegmenterASR):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frequency = FrequencyController(FrequencyConfig(**self.hparams.frequency_config))
        self.checkpointer.add_recoverable(
            "frequency", self.frequency,
            custom_save_hook=FrequencyController.save,
            custom_load_hook=FrequencyController.load,
        )
        self.checkpointer.add_recoverable(
            "random_state", RandomState(),
            custom_save_hook=RandomState.save, custom_load_hook=RandomState.load,
        )

    def _rate_band(self):
        return self.frequency.rate_band

    def make_dataloader(self, dataset, stage, ckpt_prefix="dataloader-", **kwargs):
        # Iterator construction consumes a worker seed. Keep that separate from
        # model RNG so restarting a mid-epoch loader does not change rollouts.
        kwargs.setdefault("generator", torch.Generator().manual_seed(int(self.hparams.seed)))
        return super().make_dataloader(dataset, stage, ckpt_prefix=ckpt_prefix, **kwargs)

    def configure_validation_datasets(self, valid_data, test_datasets):
        if set(test_datasets) != {"dev-other"}:
            raise ValueError("Pilot continuation may only load dev-clean/dev-other")
        self._frequency_datasets = {"dev-clean": valid_data, "dev-other": test_datasets["dev-other"]}
        limit = int(self.hparams.continuation_smoke_dev_limit)
        if limit:
            # Small, deterministic integration diagnostic only; never production.
            self._frequency_datasets = {
                name: data.filtered_sorted(select_n=limit)
                for name, data in self._frequency_datasets.items()
            }

    def configure_step_validation(self, valid_set, loader_kwargs):
        del valid_set, loader_kwargs
        self._frequency_loaders = {
            name: self.make_dataloader(data, stage=sb.Stage.VALID, ckpt_prefix=None,
                                      **self.hparams.valid_dataloader_opts)
            for name, data in self._frequency_datasets.items()
        }
        self._step_validation_set = self._frequency_loaders["dev-clean"]
        self._step_validation_enable = not self.noprogressbar
        self._last_step_validation = None

    def on_fit_start(self):
        if self.distributed_launch or self.data_parallel_backend:
            raise ValueError("This continuation entry point supports one GPU per run")
        if self.optimizer_step_limit != self.frequency.config.max_steps:
            raise ValueError("Run-option optimizer_step_limit must match continuation max_steps")
        super().on_fit_start()
        if not self.frequency.initialized:
            if self.optimizer_step != 0:
                raise ValueError("New continuation must start at optimizer step zero")
            source = Path(self.frequency.config.source_checkpoint)
            state = torch.load(source / "model_optimizer.ckpt", map_location="cpu", weights_only=True)
            self.optimizer.load_state_dict(state)
            logger.info("Loaded matched source optimizer: %s", source)
        else:
            self._last_step_validation = self.frequency.state["last_step"]
            logger.info("Resumed continuation step=%d target_hz=%.4f EMA=%.4f",
                        self.optimizer_step, self.frequency.target_hz,
                        self.frequency.state["ema_other_wer"])
        # Loading optimizer moments also restores old LRs: override those here.
        self.optimizer.param_groups[0]["lr"] = float(self.hparams.lr_segmenter)
        self.optimizer.param_groups[1]["lr"] = float(self.hparams.lr_decoder)
        logger.info("Continuation rate penalty channels=%s", sorted(self._rate_channels()))
        if not all(p.requires_grad for p in self.modules.segmenter.parameters()):
            raise ValueError("Boundary policy and pooler must both remain trainable")
        if not all(p.requires_grad for p in self.modules.proj.parameters()):
            raise ValueError("Projection must remain trainable")
        if not any(p.requires_grad for p in self.modules.llm.parameters()):
            raise ValueError("LoRA must remain trainable")
        if any(p.requires_grad for p in self.modules.ssl.parameters()):
            raise ValueError("SSL encoder must stay frozen")
        logger.info("Continuation optimizer LRs=%s; new-step budget=%d",
                    [g["lr"] for g in self.optimizer.param_groups], self.optimizer_step_limit)

    def on_stage_start(self, stage, epoch):
        super().on_stage_start(stage, epoch)
        if stage == sb.Stage.TRAIN and not self.frequency.initialized:
            # Here the epoch counter is 1, but no training update has happened.
            self._run_step_validation("initial_checkpoint")

    def on_stage_end(self, stage, stage_loss, epoch):
        if stage != sb.Stage.VALID:
            return super().on_stage_end(stage, stage_loss, epoch)
        if not self._rho_samples:
            raise RuntimeError("Validation produced no boundary-rate measurements")
        self._frequency_metrics[self._frequency_split] = dict(
            loss=float(stage_loss),
            WER=float(self.wer_metric.summarize("error_rate")),
            CER=float(self.cer_metric.summarize("error_rate")),
            rate_hz=self.frequency.config.frame_hz * sum(self._rho_samples) / len(self._rho_samples),
            utterances=len(self._rho_samples),
        )

    def _run_step_validation(self, reason):
        """Validate both splits, restore training state, THEN checkpoint it."""
        names = ("step", "current_epoch", "_stage_started_at", "_rho_samples", "_reward_std_samples")
        saved = {name: getattr(self, name) for name in names}
        rng = RandomState.capture()
        modes = [(module, module.training) for module in self.modules.modules()]
        train_stats = self._partial_train_stats()
        started = time.monotonic()
        self._accumulate_train_memory_peaks()
        self._frequency_metrics = {}
        try:
            for split, loader in self._frequency_loaders.items():
                self._frequency_split = split
                self.step = 0
                logger.info("Continuation validation: %s new_step=%d reason=%s",
                            split, self.optimizer_step, reason)
                self._fit_valid(loader, epoch=saved["current_epoch"],
                                enable=self._step_validation_enable)
        finally:
            for name, value in saved.items():
                setattr(self, name, value)
            for module, training in modes:
                module.training = training
            RandomState.restore(rng)
            self._step_validation_seconds_in_epoch += time.monotonic() - started
            if self.device_type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

        clean, other = (self._frequency_metrics[s] for s in ("dev-clean", "dev-other"))
        for metrics in (clean, other):
            if any(not math.isfinite(v) for v in metrics.values()):
                raise RuntimeError("Nonfinite validation metric")
        decision = self.frequency.observe(int(self.optimizer_step), clean["WER"], other["WER"], other["rate_hz"])
        self._last_step_validation = int(self.optimizer_step)
        event = dict(decision, reason=reason, arm=self.frequency.config.arm,
                     rate_channel=getattr(self.hparams, "rate_channel", "aux"),
                     splits=self._frequency_metrics, source_optimizer_step=self.hparams.source_optimizer_step,
                     training_batch_step=self.step,
                     lr=[group["lr"] for group in self.optimizer.param_groups])
        self.hparams.train_logger.log_stats(
            stats_meta={"new_optimizer_step": self.optimizer_step, "arm": self.frequency.config.arm,
                        "target_hz": self.frequency.target_hz, "action": decision["action"],
                        "ema_other_WER": decision["ema_other_wer"]},
            train_stats=train_stats,
            valid_stats={f"{split}_{key}": value for split, stats in self._frequency_metrics.items()
                         for key, value in stats.items()},
        )
        # Keep every validation snapshot (~282 MB each), including step 0 fallback.
        self.checkpointer.save_checkpoint(
            meta={"optimizer_step": int(self.optimizer_step), "validation_step": int(self.optimizer_step),
                  "WER": clean["WER"], "frequency": event},
            end_of_epoch=False, name=f"continuation-step-{self.optimizer_step:06d}",
        )
        with (Path(self.hparams.output_folder) / "frequency_events.jsonl").open("a") as stream:
            stream.write(json.dumps(event, allow_nan=False) + "\n")
        self.write_selection()
        logger.info("FREQUENCY_DECISION %s", json.dumps(decision))
        pause = int(self.hparams.continuation_smoke_pause_step)
        if pause and self.optimizer_step == pause:
            logger.info("SMOKE_CHECKPOINT_SAVED: intentional exit to test resume")
            raise SystemExit(0)

    def write_selection(self):
        s = self.frequency.state
        step = s["best_step"]
        write_json(Path(self.hparams.output_folder) / "selected_checkpoint.json", dict(
            checkpoint=str(Path(self.hparams.save_folder) / f"CKPT+continuation-step-{step:06d}"),
            selected_new_step=step, measured_dev_other_hz=s["best_other_hz"],
            initial_checkpoint_fallback=step == 0, controller=s,
            completed_new_steps=int(self.optimizer_step),
            completed=int(self.optimizer_step) == self.frequency.config.max_steps,
            test_sets_evaluated=False,
        ))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--continuation-config", required=True)
    parser.add_argument("--continuation-arm", choices=("adaptive", "original", "fixed_lower"), required=True)
    parser.add_argument("--continuation-smoke-dev-limit", type=int, default=0)
    parser.add_argument("--continuation-smoke-pause-step", type=int, default=0)
    args, remaining = parser.parse_known_args()
    config_data = yaml.safe_load(Path(args.continuation_config).read_text())
    config_data["arm"] = args.continuation_arm
    config = FrequencyConfig(**config_data)
    source = Path(config.source_checkpoint)
    for filename in ("segmenter.ckpt", "llm.ckpt", "proj.ckpt", "normalize.ckpt", "model_optimizer.ckpt", "CKPT.yaml"):
        if not (source / filename).is_file():
            raise FileNotFoundError(source / filename)

    def configure(hparams):
        required = dict(segmenter_mode="joint", segmenter_backbone="transformer_ar",
                        segment_pooling="bigru_residual", rl_update_mode="combined_on_policy",
                        segmenter_ar_history_window=64, grpo_k=4, grpo_normalize_std=True,
                        rate_mode="band", lambda_cap=1.0, pg_weight=1.0,
                        entropy_coeff_init=0.0, entropy_coeff_final=0.0,
                        freeze_boundary_policy=False, freeze_decoder_in_joint=False,
                        warmup_epochs=0, warmup_optimizer_steps=0,
                        fixed_rate_k=None, eval_only=False, test_batch_size=8,
                        lr_segmenter=5e-6, lr_decoder=2e-5)
        for key, value in required.items():
            if hparams.get(key) != value:
                raise ValueError(f"Continuation requires {key}={value}, got {hparams.get(key)}")
        if set(hparams["train_splits"]) != {"train-clean-100", "train-clean-360", "train-other-500"}:
            raise ValueError("LS960 requires all three training splits")
        if "ls960" not in hparams["output_folder"]:
            raise ValueError("LS960 output path must contain ls960")
        if (Path(hparams["segmenter_init_checkpoint"]).resolve() != (source / "segmenter.ckpt").resolve()
                or Path(hparams["decoder_init_ckpt_dir"]).resolve() != source.resolve()):
            raise ValueError("Segmenter, decoder and optimizer must come from the same checkpoint")
        if hparams["validation_interval_optimizer_steps"] != config.validation_interval:
            raise ValueError("Validation interval must match controller configuration")
        rate_channel = configure_rate_channel(hparams)
        hparams.update(frequency_config=asdict(config), skip_final_test=True,
                       continuation_smoke_dev_limit=args.continuation_smoke_dev_limit,
                       continuation_smoke_pause_step=args.continuation_smoke_pause_step,
                       source_optimizer_step=yaml.safe_load((source / "CKPT.yaml").read_text())["optimizer_step"])
        Path(hparams["output_folder"]).mkdir(parents=True, exist_ok=True)
        write_json(Path(hparams["output_folder"]) / "continuation_config.json",
                   dict(asdict(config), rate_channel=rate_channel))

    sys.argv = [sys.argv[0], *remaining]
    brain = base_main(AdaptiveSegmenterASR, configure)
    if brain.optimizer_step != config.max_steps:
        raise RuntimeError("Training ended before the requested maximum update count")
    brain.write_selection()
    logger.info("CONTINUATION_COMPLETE: %d new optimizer steps; test sets untouched", brain.optimizer_step)
    return brain


if __name__ == "__main__":
    main()
