"""Validation-only rate-target controller; never an early-stopping rule.

WER is in absolute percentage points. Rate is 50 * mean utterance kept ratio,
the same rate statistic used by the existing segmenter trainer.
"""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class FrequencyConfig:
    arm: str = "adaptive"
    max_steps: int = 8000
    validation_interval: int = 500
    ema_alpha: float = 0.2
    wer_tolerance_pp: float = 0.1
    consecutive_checks: int = 2
    rate_step_hz: float = 0.5
    min_target_hz: float = 7.5
    original_target_hz: float = 12.5
    frame_hz: float = 50.0
    source_checkpoint: str = ""

    def __post_init__(self):
        if self.arm not in {"adaptive", "original", "fixed_lower"}:
            raise ValueError("Unknown continuation arm")
        for key in ("max_steps", "validation_interval", "consecutive_checks"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("ema_alpha", "wer_tolerance_pp", "rate_step_hz",
                    "min_target_hz", "original_target_hz", "frame_hz"):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"Nonfinite {key}")
        if not 0 < self.ema_alpha <= 1 or self.wer_tolerance_pp < 0:
            raise ValueError("Invalid EMA alpha or WER tolerance")
        if not 0 < self.min_target_hz <= self.original_target_hz <= self.frame_hz:
            raise ValueError("Invalid rate band")
        if self.rate_step_hz <= 0:
            raise ValueError("rate_step_hz must be positive")


class FrequencyController:
    """Monotone target with independent quality and target-advance streaks."""

    def __init__(self, config):
        self.config = config
        self.state = None

    @property
    def initialized(self):
        return self.state is not None

    @property
    def target_hz(self):
        return (self.state["target_hz"] if self.initialized
                else self.config.original_target_hz)

    @property
    def rate_band(self):
        return (self.config.min_target_hz / self.config.frame_hz,
                self.target_hz / self.config.frame_hz)

    def observe(self, step, clean_wer, other_wer, other_hz):
        """Update once per paired validation; return an auditable decision."""
        cfg = self.config
        values = (clean_wer, other_wer, other_hz)
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Validation metrics must be finite and nonnegative")
        if other_hz > cfg.frame_hz:
            raise ValueError("Measured rate exceeds the encoder frame rate")
        if not isinstance(step, int) or step < 0 or step > cfg.max_steps:
            raise ValueError("Invalid continuation step")
        if not self.initialized:
            if step != 0:
                raise ValueError("Measure the initial checkpoint at step zero")
            target = min(cfg.original_target_hz, max(cfg.min_target_hz, other_hz))
            if cfg.arm == "original":
                target = cfg.original_target_hz
            elif cfg.arm == "fixed_lower":
                target = max(cfg.min_target_hz, target - cfg.rate_step_hz)
            self.state = dict(
                initial_clean_wer=clean_wer, initial_other_wer=other_wer,
                initial_other_hz=other_hz, ema_other_wer=other_wer,
                target_hz=target, last_step=0, quality_streak=0,
                advance_streak=0, best_step=0, best_other_hz=other_hz,
            )
            return dict(self.state, step=step, clean_wer=clean_wer,
                        other_wer=other_wer, other_hz=other_hz,
                        target_before_hz=cfg.original_target_hz,
                        action="initialize", quality_ok=True,
                        rate_reached=other_hz <= target, selected=True,
                        training_done=False)

        s = self.state
        if step <= s["last_step"]:
            raise ValueError("Validation steps must increase strictly")
        before = s["target_hz"]
        s["ema_other_wer"] = (cfg.ema_alpha * other_wer
                               + (1 - cfg.ema_alpha) * s["ema_other_wer"])
        other_limit = s["initial_other_wer"] + cfg.wer_tolerance_pp
        clean_limit = s["initial_clean_wer"] + cfg.wer_tolerance_pp
        quality_ok = (s["ema_other_wer"] <= other_limit
                      and other_wer <= other_limit and clean_wer <= clean_limit)
        rate_reached = other_hz <= before
        s["quality_streak"] = s["quality_streak"] + 1 if quality_ok else 0
        s["advance_streak"] = (s["advance_streak"] + 1
                               if quality_ok and rate_reached else 0)
        selected = (s["quality_streak"] >= cfg.consecutive_checks
                    and other_hz < s["best_other_hz"])
        if selected:
            s["best_step"], s["best_other_hz"] = step, other_hz

        done = step == cfg.max_steps
        action = "hold"
        if (cfg.arm == "adaptive" and not done
                and s["advance_streak"] >= cfg.consecutive_checks
                and before > cfg.min_target_hz):
            s["target_hz"] = max(cfg.min_target_hz, before - cfg.rate_step_hz)
            s["advance_streak"] = 0
            action = "lower_target"
        s["last_step"] = step
        return dict(s, step=step, clean_wer=clean_wer, other_wer=other_wer,
                    other_hz=other_hz, target_before_hz=before,
                    action=action, quality_ok=quality_ok,
                    rate_reached=rate_reached, selected=selected,
                    training_done=done)

    def save(self, path):
        Path(path).write_text(json.dumps(
            {"version": 1, "config": asdict(self.config), "state": self.state},
            indent=2, allow_nan=False) + "\n")

    def load(self, path, end_of_epoch=False):
        del end_of_epoch
        data = json.loads(Path(path).read_text())
        if data["version"] != 1 or data["config"] != asdict(self.config):
            raise ValueError("Continuation configuration changed on resume")
        self.state = data["state"]
