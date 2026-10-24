#!/usr/bin/env python3
"""ASSET: supervised boundary initialization, then decoder CE + on-policy GRPO.

WavLM and the Llama base remain frozen. The boundary policy, residual pooler,
projection and LoRA parameters are trainable. The default rate penalty is an
auxiliary loss; K=4 sampled segmentations retain transcription-NLL rewards.
"""

import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from batch_invariant_decode import (
    PerUtteranceEosLogitsProcessor,
    compact_position_ids,
    duration_decode_limits,
    left_pack_valid_prefix,
)
from hyperpyyaml import load_hyperpyyaml
from segment_pooling import (
    fixed_rate_boundary_targets,
    lengths_to_padding_mask,
    padding_mask_to_lengths,
)
from segmenter import (
    boundary_bce_loss,
    boundary_prf,
    entropy_bonus,
    group_advantage,
    grpo_pg_loss,
    rate_loss,
    realized_kept_ratio,
    sampled_rate_penalty,
)
from speechllm import (  # reuse the base recipe's plumbing
    ASR,
    dataio_prepare,
    get_multimodal_attention_mask,
    loader_runtime_options,
)
from transformers import LogitsProcessorList

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def resolve_fractional_warmup_steps(num_batches, grad_accumulation, fraction):
    """Convert a first-epoch fraction into an optimizer-step boundary."""
    if not 0.0 <= float(fraction) < 1.0:
        raise ValueError("warmup_fraction_of_epoch must be in [0, 1)")
    if int(num_batches) < 1 or int(grad_accumulation) < 1:
        raise ValueError("num_batches and grad_accumulation must be positive")
    updates = math.ceil(int(num_batches) / int(grad_accumulation))
    if float(fraction) == 0.0:
        return 0
    return max(1, int(round(updates * float(fraction))))


def resolve_effective_grad_accumulation(
    hparams_value, run_option_value, run_option_was_overridden
):
    """Resolve the accumulation factor before constructing ``Brain``.

    SpeechBrain run options override HyperPyYAML values inside ``Brain``, but the
    source hparams dictionary does not receive that override. Runtime-derived
    schedules must therefore resolve the effective value explicitly.
    """
    value = run_option_value if run_option_was_overridden else hparams_value
    value = int(value)
    if value < 1:
        raise ValueError("grad_accumulation_factor must be positive")
    return value


def resolve_optimizer_step_limit(value):
    """Normalize SpeechBrain's optional CLI step limit to an integer."""
    if value is None:
        return None
    value = int(value)
    if value < 1:
        raise ValueError("optimizer_step_limit must be positive")
    return value


def resolve_fixed_rate_k(value):
    """Normalize the optional fixed-boundary control to a positive integer."""
    if value is None:
        return None
    value = int(value)
    if value < 1:
        raise ValueError("fixed_rate_k must be positive")
    return value


def resolve_joint_boundary_source(value):
    """Normalize an optional externally supplied joint-mode boundary source."""

    if value is None or str(value).strip().lower() in {"", "none"}:
        return None
    source = str(value).strip().lower()
    if source != "alignment":
        raise ValueError(
            "Joint external boundaries currently support boundary_source="
            f"'alignment', not {value!r}"
        )
    return source


def alignment_boundary_targets(targets, pad_mask):
    """Validate and pad-mask precomputed one-per-frame boundaries."""

    if targets.ndim != 2 or targets.shape != pad_mask.shape:
        raise ValueError(
            f"boundary_target shape {tuple(targets.shape)} does not match "
            f"encoder padding mask {tuple(pad_mask.shape)}"
        )
    valid_targets = targets[~pad_mask]
    if valid_targets.numel() and not torch.all(
        (valid_targets == 0) | (valid_targets == 1)
    ):
        raise ValueError(
            "boundary_target must contain only 0/1 on valid frames"
        )
    boundary = targets.long().masked_fill(pad_mask, -1)
    if boundary.size(1) and not torch.all(boundary[:, 0] == 1):
        raise ValueError("boundary_target must open a segment at frame zero")
    return boundary


def freeze_boundary_policy_parameters(segmenter):
    """Freeze boundary decisions while leaving any trainable pooler unfrozen.

    The segmenter owns both the boundary policy and the pooler. Matching a
    frozen-policy control therefore cannot freeze the whole module: BiGRU
    pooling must still receive decoder-CE gradients. Parameter-free mean
    pooling legitimately returns zero trainable pooler parameters.
    """
    frozen = 0
    trainable_pooler = 0
    for name, parameter in segmenter.named_parameters():
        if name.startswith("pooler."):
            trainable_pooler += parameter.numel()
            continue
        parameter.requires_grad = False
        frozen += parameter.numel()
    return frozen, trainable_pooler


def step_validation_reason(
    optimizer_step,
    interval,
    *,
    warmup_step=0,
    max_steps=None,
    last_validated_step=None,
):
    """Return why a step validation is due, or ``None``.

    In addition to a regular interval, the decoder-to-RL transition and the
    final optimizer step are validation points. This keeps the schedule useful
    when either boundary does not land on an interval multiple.
    """
    step = int(optimizer_step)
    interval = int(interval)
    warmup_step = int(warmup_step or 0)
    max_steps = None if max_steps is None else int(max_steps)
    if interval < 1:
        raise ValueError("validation interval must be positive")
    if step < 1 or step == last_validated_step:
        return None
    if warmup_step > 0 and step == warmup_step:
        return "warmup_end"
    if step % interval == 0:
        return "interval"
    if max_steps is not None and step >= max_steps:
        return "final"
    return None


def _levenshtein(a, b):
    """Edit distance between two sequences (lists/strings)."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


class SegmenterASR(ASR):
    """Brain that trains a learned segmenter (cold-start BCE, then joint GRPO)."""

    # ------------------------------------------------------------------ helpers
    def _encoder_features(self, batch):
        """Frozen decoder/segmenter SSL features on one shared frame grid.

        The default path reuses ``modules.ssl`` for both consumers. When a
        ``modules.segmenter_ssl`` is configured, it supplies only the boundary
        policy input while the decoder continues to pool ``modules.ssl`` features.
        Both encoders must emit exactly the same number of frames because a boundary
        index predicted on one stream is applied directly to the other stream.

        Returns ``(decoder_feats, segmenter_feats, rel_lens)``.
        """
        if bool(getattr(self.hparams, "use_feats", False)):
            if getattr(batch, "feats", None) is None:
                raise ValueError("`use_feats=True` but batch has no `feats`.")
            if getattr(self.modules, "segmenter_ssl", None) is not None:
                raise ValueError(
                    "Dual-SSL mode is incompatible with `use_feats=True`: the "
                    "batch provides only one cached feature stream."
                )
            feats, feat_lens = batch.feats
            return feats, feats, feat_lens
        wavs, wav_lens = batch.sig
        wavs = self.hparams.normalize(wavs, wav_lens)
        decoder_feats = self.modules.ssl(wavs, wav_lens)
        segmenter_ssl = getattr(self.modules, "segmenter_ssl", None)
        segmenter_feats = (
            decoder_feats
            if segmenter_ssl is None
            else segmenter_ssl(wavs, wav_lens)
        )
        if decoder_feats.shape[:2] != segmenter_feats.shape[:2]:
            raise RuntimeError(
                "Decoder and segmenter SSL frame grids do not match: "
                f"decoder={tuple(decoder_feats.shape)}, "
                f"segmenter={tuple(segmenter_feats.shape)}. Boundaries cannot be "
                "transferred safely between these streams."
            )
        if segmenter_ssl is not None and not getattr(
            self, "_dual_ssl_grid_logged", False
        ):
            logger.info(
                "Dual-SSL frame grid verified: T=%d; segmenter_dim=%d; "
                "decoder_dim=%d",
                decoder_feats.size(1),
                segmenter_feats.size(2),
                decoder_feats.size(2),
            )
            self._dual_ssl_grid_logged = True
        return decoder_feats, segmenter_feats, wav_lens

    def _run_decoder(self, feats, pad_mask, boundary, batch, want_hyps=False):
        """Pool by ``boundary`` -> proj -> LLM. Returns dict(llm_logits, seg_lens, hyps).

        Runs under the caller's grad context (grad on for the decoder-CE path, wrapped in
        ``no_grad`` by the caller for the reward samples).
        """
        seg_feats, seg_pad = self.modules.segmenter.pool(
            feats, pad_mask, boundary
        )
        return self._run_decoder_from_segments(
            seg_feats, seg_pad, batch, want_hyps=want_hyps
        )

    def _run_decoder_from_segments(
        self, seg_feats, seg_pad, batch, want_hyps=False
    ):
        """Run projection and LLM from an already pooled acoustic prefix."""
        tokens_bos, tokens_bos_lens = batch.tokens_bos
        seg_lens = padding_mask_to_lengths(seg_pad)
        projected = self.modules.proj(seg_feats)
        txt_embds = self.txt_embedding(tokens_bos)
        multimodal = torch.cat(
            [txt_embds[:, 0].unsqueeze(1), projected, txt_embds[:, 1:]], dim=1
        )
        attn = get_multimodal_attention_mask(
            projected, seg_lens, txt_embds, tokens_bos_lens, self.device
        )
        llm_logits = self.modules.llm(
            inputs_embeds=multimodal,
            attention_mask=attn,
            position_ids=compact_position_ids(attn),
        ).logits
        out = {
            "llm_logits": llm_logits,
            "seg_pad": seg_pad,
            "seg_lens": seg_lens,
        }
        if want_hyps:
            prompt_len = batch.prompt_len
            n = projected.shape[1] + int(prompt_len[0].item())
            out["hyps"] = self.modules.searcher(
                multimodal[:, :n],
                seg_lens,
                attn[:, :n],
                decode_durations=self._decode_durations(batch),
            )
        return out

    def _decode_replicated(
        self,
        feats,
        pad_mask,
        boundary,
        tokens_bos,
        tokens_bos_lens,
        prompt_len,
        decode_durations,
        want_hyps,
    ):
        """Decode for a K-replicated rollout batch from explicit tensors (no ``batch``).

        Mirrors :meth:`_run_decoder` but takes already-replicated ``(K*B, ...)`` tensors,
        so all K samples of every utterance are pooled + decoded in **one** batched
        forward/generation instead of a Python loop of K sequential decodes.
        """
        seg_feats, seg_pad = self.modules.segmenter.pool(
            feats, pad_mask, boundary
        )
        seg_lens = padding_mask_to_lengths(seg_pad)
        projected = self.modules.proj(seg_feats)
        txt_embds = self.txt_embedding(tokens_bos)
        multimodal = torch.cat(
            [txt_embds[:, 0].unsqueeze(1), projected, txt_embds[:, 1:]], dim=1
        )
        attn = get_multimodal_attention_mask(
            projected, seg_lens, txt_embds, tokens_bos_lens, self.device
        )
        if want_hyps:
            # KV-cached HF generate for the free-running reward decode. The SpeechBrain
            # greedy searcher re-runs the FULL prefix+generated forward every step (no KV
            # cache => O(L^2), compute-bound) which dominated the CER step; generate reuses
            # past_key_values (O(L)). All sequences share prefix length n and end with the
            # prompt, so batched generation is well-formed (interior segment-pad handled by
            # the attention mask). LoRA is active inside adapted_model.
            n = projected.shape[1] + int(prompt_len[0].item())
            prefix, prefix_mask = left_pack_valid_prefix(
                multimodal[:, :n], attn[:, :n]
            )
            decode_limits = duration_decode_limits(
                decode_durations,
                float(self.hparams.max_decode_tokens_per_second),
                int(self.hparams.max_decode_token_margin),
                int(self.hparams.max_decode_tokens),
                int(self.hparams.min_decode_tokens),
            )
            max_new = int(decode_limits.max().item())
            logits_processor = LogitsProcessorList(
                [
                    PerUtteranceEosLogitsProcessor(
                        decode_limits, int(self.hparams.eos_index)
                    )
                ]
            )
            gen = self.modules.llm.adapted_model.generate(
                inputs_embeds=prefix,
                attention_mask=prefix_mask,
                position_ids=compact_position_ids(prefix_mask),
                max_new_tokens=max_new,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                logits_processor=logits_processor,
                eos_token_id=int(self.hparams.eos_index),
                pad_token_id=int(self.hparams.pad_token),
            )
            return (gen,)  # mimic searcher's hyps[0] == token ids
        return self.modules.llm(
            inputs_embeds=multimodal,
            attention_mask=attn,
            position_ids=compact_position_ids(attn),
        ).logits

    def _utterance_nll_rep(self, llm_logits, tokens_eos):
        """Per-utterance teacher-forced NLL ``(N,)`` for a replicated rollout batch."""
        n_audio = llm_logits.shape[1] - tokens_eos.shape[1]
        target = torch.cat(
            [
                torch.full(
                    (tokens_eos.shape[0], n_audio),
                    self.hparams.ignore_index,
                    device=self.device,
                ),
                tokens_eos,
            ],
            dim=1,
        ).long()
        ce = F.cross_entropy(
            llm_logits.reshape(-1, llm_logits.shape[-1]),
            target.reshape(-1),
            ignore_index=self.hparams.ignore_index,
            reduction="none",
        ).reshape(llm_logits.shape[0], -1)
        valid = (target != self.hparams.ignore_index).float()
        return ce.sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def _rollout_rewards(self, feats, pad_mask, sampled, batch, reward_kind):
        """Detached rewards ``(B, K)`` for K sampled segmentations via ONE batched decode.

        Replicates the utterance batch K times along dim 0, concatenates the K sampled
        boundaries, and does a single batched decode (free-running for cer/wer, or a
        teacher-forced forward for nll). This turns K sequential decodes into one wide
        batched generation — the main cost amortization for the CER reward.
        """
        K = len(sampled)
        Bsz = feats.size(0)
        boundary_cat = torch.cat(sampled, dim=0)  # (K*B, T)
        feats_rep = feats.repeat(K, 1, 1)
        pad_rep = pad_mask.repeat(K, 1)
        tokens_bos, tokens_bos_lens = batch.tokens_bos
        tb_rep = tokens_bos.repeat(K, 1)
        tbl_rep = tokens_bos_lens.repeat(K)
        if reward_kind == "nll":
            llm_logits = self._decode_replicated(
                feats_rep,
                pad_rep,
                boundary_cat,
                tb_rep,
                tbl_rep,
                None,
                None,
                want_hyps=False,
            )
            tokens_eos, _ = batch.tokens_eos
            err = self._utterance_nll_rep(llm_logits, tokens_eos.repeat(K, 1))
        else:  # cer / wer -- free-running batched generation
            hyps = self._decode_replicated(
                feats_rep,
                pad_rep,
                boundary_cat,
                tb_rep,
                tbl_rep,
                batch.prompt_len.repeat(K),
                self._decode_durations(batch).repeat(K),
                want_hyps=True,
            )
            err = self._utterance_error(
                hyps, batch, reward_kind, refs=list(batch.wrd) * K
            )
            if not getattr(self, "_rl_dbg_printed", False):
                self._rl_dbg_printed = True
                preds = self.tokenizer.batch_decode(
                    hyps[0], skip_special_tokens=True
                )
                for i in range(min(3, len(preds))):
                    logger.info(
                        "reward-decode sanity | %s=%.3f | hyp=%r | ref=%r",
                        reward_kind,
                        float(err[i]),
                        preds[i],
                        (list(batch.wrd) * K)[i],
                    )
        # err order is [s0:b0..bB-1, s1:b0..bB-1, ...] -> (K, B) -> (B, K).
        return (-err).reshape(K, Bsz).t().contiguous()








    def _target_tokens(self, llm_logits, batch):
        """Text targets aligned to ``llm_logits`` (audio positions -> ignore_index)."""
        tokens_eos, _ = batch.tokens_eos
        n_audio = llm_logits.shape[1] - tokens_eos.shape[1]
        return torch.cat(
            [
                torch.full(
                    (tokens_eos.shape[0], n_audio),
                    self.hparams.ignore_index,
                    device=self.device,
                ),
                tokens_eos,
            ],
            dim=1,
        ).long()

    def _decoder_ce(self, llm_logits, batch):
        """Mean cross-entropy over text tokens (decoder-training loss / monitor)."""
        target = self._target_tokens(llm_logits, batch)
        return F.cross_entropy(
            llm_logits.reshape(-1, llm_logits.shape[-1]),
            target.reshape(-1),
            ignore_index=self.hparams.ignore_index,
        )

    def _utterance_nll(self, llm_logits, batch):
        """Per-utterance teacher-forced NLL ``(B,)`` (the RL reward is ``-NLL``)."""
        target = self._target_tokens(llm_logits, batch)
        ce = F.cross_entropy(
            llm_logits.reshape(-1, llm_logits.shape[-1]),
            target.reshape(-1),
            ignore_index=self.hparams.ignore_index,
            reduction="none",
        ).reshape(llm_logits.shape[0], -1)
        valid = (target != self.hparams.ignore_index).float()
        return ce.sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

    def _rollout_quality_reward(self, feats, pad_mask, boundary, batch, kind):
        """Quality reward ``(B,)`` for ONE sampled segmentation.

        Called ``K`` times per joint training batch, always under ``no_grad``:
        GRPO scores each rollout with the current decoder and the reward is
        detached. Extracted as its own method so a task whose answer is not a
        transcript can supply its own reward without forking the rollout loop
        (see ``MultitaskSegmenterASR``); the ASR path is unchanged.
        """
        if kind == "nll":
            deck = self._run_decoder(feats, pad_mask, boundary, batch)
            return -self._utterance_nll(deck["llm_logits"], batch)
        # cer / wer -- free-running decode
        deck = self._run_decoder(
            feats, pad_mask, boundary, batch, want_hyps=True
        )
        return -self._utterance_error(deck["hyps"], batch, kind)

    def _utterance_error(self, hyps, batch, kind="cer", refs=None):
        """Per-utterance free-running decode error (CER/WER) ``(N,)`` for the RL reward.

        ``hyps`` are the searcher's decoded token ids; the reference is ``refs`` (defaults
        to ``batch.wrd``; pass an explicit list for a K-replicated rollout batch).
        Character-level (cer) is denser -> lower-variance reward than word-level (wer).
        """
        preds = self.tokenizer.batch_decode(hyps[0], skip_special_tokens=True)
        if refs is None:
            refs = batch.wrd
        errs = []
        for pred, ref in zip(preds, refs):
            if kind == "wer":
                p, r = pred.split(), ref.split()
            else:  # cer
                p, r = list(pred.strip()), list(ref.strip())
            errs.append(_levenshtein(p, r) / max(len(r), 1))
        return torch.tensor(errs, device=self.device, dtype=torch.float32)

    def _entropy_coeff(self):
        """Linearly annealed entropy weight over the joint (post-warmup) epochs."""
        init = float(self.hparams.entropy_coeff_init)
        final = float(self.hparams.entropy_coeff_final)
        warm = int(self.hparams.warmup_epochs)
        total = int(self.hparams.number_of_epochs)
        span = max(total - warm, 1)
        frac = min(max((self.current_epoch - warm) / span, 0.0), 1.0)
        return init + frac * (final - init)

    def _is_warmup(self, stage):
        if self.hparams.segmenter_mode != "joint" or stage != sb.Stage.TRAIN:
            return False
        # The frozen-policy attribution arm is decoder/pooler CE adaptation for
        # the entire run. Reuse the established warmup forward/objective path,
        # which does not construct sampled rollouts or a policy-gradient loss.
        if bool(getattr(self.hparams, "freeze_boundary_policy", False)):
            return True
        step_limit = getattr(self.hparams, "warmup_optimizer_steps", None)
        if step_limit is not None:
            return int(getattr(self, "optimizer_step", 0)) < int(step_limit)
        return self.current_epoch <= int(self.hparams.warmup_epochs)

    def _set_decoder_phase_lr(self, is_warmup):
        """Apply and log the decoder LR when a within-epoch phase changes."""
        if self.hparams.segmenter_mode != "joint" or not hasattr(
            self, "optimizer"
        ):
            return
        lr = float(
            getattr(self.hparams, "lr_decoder_warmup", self.hparams.lr_decoder)
            if is_warmup
            else self.hparams.lr_decoder
        )
        phase = "warmup" if is_warmup else "joint-RL"
        if getattr(self, "_decoder_phase", None) != phase:
            self.optimizer.param_groups[1]["lr"] = lr
            self._decoder_phase = phase
            logger.info(
                "Decoder phase=%s optimizer_step=%d learning_rate=%g",
                phase,
                int(getattr(self, "optimizer_step", 0)),
                lr,
            )

    def _rate_mode(self):
        """Resolve the current rate objective, including the legacy switch."""
        return getattr(self.hparams, "rate_mode", None) or (
            "pin"
            if bool(getattr(self.hparams, "rate_two_sided", False))
            else "captax"
        )

    def _rate_band(self):
        """``(rho_lo, rho_hi)`` this batch's rate terms are held to.

        Single source of truth for both rate channels. Overridable so a
        multi-task run can hold different tasks to different bands without
        forking the rate machinery (see ``MultitaskSegmenterASR``).
        """
        return (
            float(getattr(self.hparams, "rho_lo", 0.0)),
            float(getattr(self.hparams, "rho_hi", 1.0)),
        )

    def _sampled_rate_terms(self, sampled, pad_mask):
        """Rate penalties and ratios ``(B, K)`` for full-prefix AR rollouts."""
        K = len(sampled)
        B = pad_mask.size(0)
        boundary_cat = torch.cat(sampled, dim=0)
        pad_rep = pad_mask.repeat(K, 1)
        rho_lo, rho_hi = self._rate_band()
        penalty, rho = sampled_rate_penalty(
            boundary_cat,
            pad_rep,
            float(self.hparams.rho_star),
            float(self.hparams.lambda_cap),
            float(self.hparams.lambda_press),
            rho_floor=float(getattr(self.hparams, "rho_floor", 0.0)),
            lambda_floor=float(getattr(self.hparams, "lambda_floor", 0.0)),
            mode=self._rate_mode(),
            rho_lo=rho_lo,
            rho_hi=rho_hi,
        )
        return (
            penalty.reshape(K, B).t().contiguous(),
            rho.reshape(K, B).t().contiguous(),
        )

    def _sample_full_ar_k(self, segmenter_feats, pad_mask, K):
        """Sample K histories in one cached ``K*B`` policy rollout."""
        rollout = self.modules.segmenter.sample_boundaries(
            segmenter_feats.repeat(K, 1, 1),
            pad_mask.repeat(K, 1),
            deterministic=False,
        )
        return list(rollout.boundaries.chunk(K, dim=0))

    def _rate_channels(self):
        """Which channel(s) the rate penalty reaches the policy through.

        ``reward``
            Subtracted from each rollout's GRPO reward, on the *realized* rho.
            Under std-normalized GRPO this channel is lambda-invariant whenever
            the quality term is flat across the group, so its weight is largely
            inert exactly when the policy is doing well.
        ``aux``
            A differentiable term added to the loss, on the *expected* rho. This
            is the only channel in which the rate weight scales the gradient.
        ``both``
            Both of the above. The two must measure the same quantity or they
            aim at different points inside one band -- see
            ``segmenter.expected_kept_ratio``.
        ``auto``
            Alias for aux-only, independent of the backbone. Since 2026-09-20,
            aux is the project default; reward/both require an explicit opt-in.
            Historical runs retain their frozen sources and recorded settings.
        """
        choice = str(getattr(self.hparams, "rate_channel", "aux")).lower()
        if choice == "auto":
            return frozenset({"aux"})
        if choice == "both":
            return frozenset({"reward", "aux"})
        if choice in ("reward", "aux"):
            return frozenset({choice})
        raise ValueError(
            f"rate_channel must be one of auto/reward/aux/both, got {choice!r}"
        )

    def _full_ar_rate_loss(self, score_logits, *, padding_mask):
        """Differentiable rate term for a full-AR policy (zero if unused).

        A full-AR policy has no closed-form marginal ``E[rho]``: the boundary at
        frame t depends on the whole realized prefix. What *is* available is the
        per-frame conditional probability along each sampled history, which
        ``score_boundaries`` already returns for the policy-gradient term. Summing
        those gives ``E[#segments | realized prefix]`` averaged over the K
        rollouts.

        That is a surrogate for the marginal, not the marginal itself, and any
        report should call it the conditional expectation. It is consistent in
        the sense that matters here: it pushes on exactly the per-frame
        probabilities the policy gradient already acts on. It is also why
        ``expected_kept_ratio`` had to start counting the implicit first segment
        -- otherwise this term and the reward's realized rho would aim at points
        ``1/T`` apart inside the same band.

        Use the acoustic padding mask, NOT the action-valid mask: frame 0 is
        forced, but still contributes to the acoustic duration denominator.
        ``expected_kept_ratio`` already counts its implicit segment once and
        ignores its boundary probability.
        """
        if "aux" not in self._rate_channels():
            return score_logits.sum() * 0.0
        rho_lo, rho_hi = self._rate_band()
        loss, _ = rate_loss(
            score_logits,
            padding_mask,
            float(self.hparams.rho_star),
            float(self.hparams.lambda_cap),
            float(self.hparams.lambda_press),
            rho_floor=float(getattr(self.hparams, "rho_floor", 0.0)),
            lambda_floor=float(getattr(self.hparams, "lambda_floor", 0.0)),
            mode=self._rate_mode(),
            rho_lo=rho_lo,
            rho_hi=rho_hi,
        )
        return loss

    def _full_ar_pg_entropy(self, segmenter_feats, pad_mask, sampled, adv):
        """Score all K completed histories in one parallel causal forward pass.

        Returns ``(pg, entropy, aux_rate)``; ``aux_rate`` is exactly zero unless
        the ``aux`` rate channel is enabled.
        """
        K = len(sampled)
        boundary_cat = torch.cat(sampled, dim=0)
        pad_rep = pad_mask.repeat(K, 1)
        score = self.modules.segmenter.score_boundaries(
            segmenter_feats.repeat(K, 1, 1),
            pad_rep,
            boundary_cat,
        )
        advantage_cat = adv.t().reshape(-1)
        pg = grpo_pg_loss(
            score.logits,
            score.boundaries,
            advantage_cat,
            valid_mask=score.valid_mask,
        )
        ent = entropy_bonus(score.logits, ~score.valid_mask)
        aux_rate = self._full_ar_rate_loss(score.logits, padding_mask=pad_rep)
        return pg, ent, aux_rate

    # ------------------------------------------------------------------ forward
    def compute_forward(self, batch, stage):
        benchmark = stage == sb.Stage.TEST and bool(
            getattr(self.hparams, "inference_benchmark_file", None)
        )
        benchmark_events = None
        benchmark_wall_start = None
        if benchmark:
            if self.device_type == "cuda":
                torch.cuda.synchronize(self.device)
                benchmark_events = [
                    torch.cuda.Event(enable_timing=True) for _ in range(4)
                ]
                benchmark_events[0].record()
            benchmark_wall_start = time.perf_counter()

        batch = batch.to(self.device, non_blocking=True)
        feats, segmenter_feats, feat_lens = self._encoder_features(batch)
        if benchmark_events is not None:
            benchmark_events[1].record()
        benchmark_encoder_end = time.perf_counter()
        T = feats.size(1)
        pad_mask = lengths_to_padding_mask(feat_lens, T)
        full_ar = bool(getattr(self.modules.segmenter, "is_full_ar", False))

        if self.hparams.segmenter_mode == "coldstart":
            if full_ar:
                targets, _ = batch.boundary_target
                targets = targets.long().masked_fill(pad_mask, -1)
                logits = self.modules.segmenter.teacher_forced_logits(
                    segmenter_feats, pad_mask, targets
                )
            else:
                logits = self.modules.segmenter.boundary_logits(
                    segmenter_feats, pad_mask
                )
            return {
                "mode": "coldstart",
                "logits": logits,
                "pad_mask": pad_mask,
                "segmenter_feats": segmenter_feats,
                "full_ar": full_ar,
            }

        # joint: decoder path on a fixed-rate control or the deterministic
        # (argmax) learned segmentation.
        boundary_source = resolve_joint_boundary_source(
            getattr(self.hparams, "boundary_source", None)
        )
        fixed_rate_k = resolve_fixed_rate_k(
            getattr(self.hparams, "fixed_rate_k", None)
        )
        if boundary_source == "alignment":
            targets, _ = batch.boundary_target
            argmax_b = alignment_boundary_targets(targets, pad_mask)
            logits = None
        elif fixed_rate_k is not None:
            argmax_b = fixed_rate_boundary_targets(pad_mask, fixed_rate_k)
            logits = None
        elif full_ar:
            with torch.no_grad():
                greedy = self.modules.segmenter.sample_boundaries(
                    segmenter_feats, pad_mask, deterministic=True
                )
            argmax_b = greedy.boundaries
            logits = greedy.logits
        else:
            logits = self.modules.segmenter.boundary_logits(
                segmenter_feats, pad_mask
            )
            argmax_b = self.modules.segmenter.sample_boundary(
                logits.detach(), pad_mask, mode="argmax"
            )
        if benchmark_events is not None:
            benchmark_events[2].record()
        benchmark_segmenter_end = time.perf_counter()
        dec = self._run_decoder(
            feats,
            pad_mask,
            argmax_b,
            batch,
            want_hyps=(stage != sb.Stage.TRAIN),
        )
        out = {
            "mode": "joint",
            "logits": logits,
            "pad_mask": pad_mask,
            "argmax_boundary": argmax_b,
            "llm_logits": dec["llm_logits"],
            "seg_pad": dec["seg_pad"],
            "seg_lens": dec["seg_lens"],
            "hyps": dec.get("hyps"),
            "is_warmup": self._is_warmup(stage),
            "segmenter_feats": segmenter_feats,
            "full_ar": full_ar,
        }
        if benchmark:
            benchmark_decoder_end = time.perf_counter()
            timing = {
                "wall_seconds": benchmark_decoder_end - benchmark_wall_start,
                "encoder_wall_seconds": (
                    benchmark_encoder_end - benchmark_wall_start
                ),
                "segmenter_wall_seconds": (
                    benchmark_segmenter_end - benchmark_encoder_end
                ),
                "decoder_wall_seconds": (
                    benchmark_decoder_end - benchmark_segmenter_end
                ),
            }
            if benchmark_events is not None:
                benchmark_events[3].record()
                benchmark_events[3].synchronize()
                timing.update(
                    {
                        "gpu_seconds": benchmark_events[0].elapsed_time(
                            benchmark_events[3]
                        )
                        / 1000.0,
                        "encoder_gpu_seconds": benchmark_events[0].elapsed_time(
                            benchmark_events[1]
                        )
                        / 1000.0,
                        "segmenter_gpu_seconds": benchmark_events[
                            1
                        ].elapsed_time(benchmark_events[2])
                        / 1000.0,
                        "decoder_gpu_seconds": benchmark_events[2].elapsed_time(
                            benchmark_events[3]
                        )
                        / 1000.0,
                    }
                )
                # The wall clock must include completion of all queued GPU work.
                timing["wall_seconds"] = (
                    time.perf_counter() - benchmark_wall_start
                )
            out["inference_timing"] = timing

        # joint (post-warmup) train: K sampled segmentations -> detached rewards.
        #   segmenter_reward=nll: teacher-forced -NLL (cheap, but under-rewards audio).
        #   segmenter_reward=cer/wer: free-running decode error (requires the audio).
        if stage == sb.Stage.TRAIN and not out["is_warmup"]:
            reward_kind = getattr(self.hparams, "segmenter_reward", "nll")
            K = int(self.hparams.grpo_k)
            quality_rewards, sampled = [], []
            if full_ar:
                with torch.no_grad():
                    sampled = self._sample_full_ar_k(
                        segmenter_feats, pad_mask, K
                    )
            for k in range(K):
                if full_ar:
                    bk = sampled[k]
                else:
                    bk = self.modules.segmenter.sample_boundary(
                        logits.detach(), pad_mask, mode="sample"
                    )
                with torch.no_grad():
                    r = self._rollout_quality_reward(
                        feats, pad_mask, bk, batch, reward_kind
                    )
                quality_rewards.append(r)
                if not full_ar:
                    sampled.append(bk)
            quality_rewards = torch.stack(quality_rewards, dim=1)
            if full_ar:
                rate_penalties, sampled_rhos = self._sampled_rate_terms(
                    sampled, pad_mask
                )
                rewards = (
                    quality_rewards - rate_penalties
                    if "reward" in self._rate_channels()
                    else quality_rewards
                )
                out["rate_penalties"] = rate_penalties
                out["sampled_rhos"] = sampled_rhos
            else:
                rewards = quality_rewards
            out["quality_rewards"] = quality_rewards
            out["rewards"] = rewards
            out["sampled_boundaries"] = sampled
        return out

    # --------------------------------------------------------------- objectives
    def compute_objectives(self, predictions, batch, stage):
        if predictions["mode"] == "coldstart":
            return self._obj_coldstart(predictions, batch, stage)
        return self._obj_joint(predictions, batch, stage)

    def _obj_coldstart(self, predictions, batch, stage):
        logits, pad_mask = predictions["logits"], predictions["pad_mask"]
        targets, _ = batch.boundary_target
        targets = targets.long().masked_fill(pad_mask, -1)
        if predictions["full_ar"] and targets.size(1):
            targets = targets.clone()
            targets[:, 0] = -1
        loss = boundary_bce_loss(logits, targets)
        if stage != sb.Stage.TRAIN:
            if predictions["full_ar"]:
                pred_b = self.modules.segmenter.sample_boundaries(
                    predictions["segmenter_feats"],
                    pad_mask,
                    deterministic=True,
                ).boundaries
            else:
                pred_b = self.modules.segmenter.sample_boundary(
                    logits, pad_mask, mode="argmax"
                )
            p, r, f1 = boundary_prf(pred_b, targets)
            self.prf_stats["precision"].append(p)
            self.prf_stats["recall"].append(r)
            self.prf_stats["f1"].append(f1)
            self._track_rho_from_boundary(pred_b, pad_mask)
        return loss

    def _obj_joint(self, predictions, batch, stage):
        llm_logits = predictions["llm_logits"]
        dec_ce = self._decoder_ce(llm_logits, batch)

        if stage != sb.Stage.TRAIN:
            self._append_wer(predictions["hyps"], llm_logits, batch)
            self._track_rho_from_boundary(
                predictions["argmax_boundary"], predictions["pad_mask"]
            )
            if stage == sb.Stage.TEST:
                self._record_inference_benchmark(predictions, batch)
            return dec_ce

        # warmup: decoder only (segmenter gets no gradient).
        if predictions["is_warmup"]:
            self._track_rho_from_boundary(
                predictions["argmax_boundary"], predictions["pad_mask"]
            )
            self._log_train(dec_ce=dec_ce)
            return dec_ce

        # joint: decoder CE + GRPO pg + rate + annealed entropy.
        logits, pad_mask = predictions["logits"], predictions["pad_mask"]
        rewards = predictions["rewards"]  # (B, K)
        adv = group_advantage(
            rewards, normalize_std=bool(self.hparams.grpo_normalize_std)
        )
        if predictions["full_ar"]:
            pg, ent, rl_rate = self._full_ar_pg_entropy(
                predictions["segmenter_feats"],
                pad_mask,
                predictions["sampled_boundaries"],
                adv,
            )
            # Aux-only is the default: GRPO sees task rewards, while rl_rate
            # supplies the separate differentiable rate gradient. An explicit
            # reward-only run instead has exactly zero rl_rate.
            rho_bar = predictions["sampled_rhos"].mean(dim=1)
        else:
            pg = logits.sum() * 0.0
            for k, bk in enumerate(predictions["sampled_boundaries"]):
                policy_logits = self.modules.segmenter.conditioned_logits(
                    logits, bk
                )
                pg = pg + grpo_pg_loss(policy_logits, bk, adv[:, k])
            pg = pg / len(predictions["sampled_boundaries"])
            # rate_mode: pin | captax | band. Back-compat: if unset, derive from
            # the old rate_two_sided flag (True -> pin, False -> captax).
            aux_rho_lo, aux_rho_hi = self._rate_band()
            rl_rate, rho_bar = rate_loss(
                logits,
                pad_mask,
                float(self.hparams.rho_star),
                float(self.hparams.lambda_cap),
                float(self.hparams.lambda_press),
                rho_floor=float(getattr(self.hparams, "rho_floor", 0.0)),
                lambda_floor=float(getattr(self.hparams, "lambda_floor", 0.0)),
                mode=self._rate_mode(),
                rho_lo=aux_rho_lo,
                rho_hi=aux_rho_hi,
                probabilities=self.modules.segmenter.boundary_marginals(
                    logits, pad_mask
                ),
            )
            if "aux" not in self._rate_channels():
                rl_rate = rl_rate * 0.0
            ent = self.modules.segmenter.policy_entropy(logits, pad_mask)
        beta_h = self._entropy_coeff()  # 0 when the entropy bonus is dropped
        # Multi-task: add the decoder CE (co-train proj+LoRA) unless the decoder is
        # frozen for this ablation (single-task: RL vs a fixed warmup decoder).
        freeze_dec = bool(
            getattr(self.hparams, "freeze_decoder_in_joint", False)
        )
        total = float(self.hparams.pg_weight) * pg + rl_rate - beta_h * ent
        if not freeze_dec:
            total = total + dec_ce
        self._track_rho_from_boundary(
            predictions["argmax_boundary"], predictions["pad_mask"]
        )
        reward_std = rewards.std(dim=1)
        self._track_reward_std(reward_std)
        self._log_train(
            dec_ce=dec_ce,
            pg=pg,
            rate=rl_rate,
            entropy=ent,
            beta_h=beta_h,
            reward=rewards.mean(),
            quality_reward=predictions["quality_rewards"].mean(),
            # GRPO's whole signal is the spread of rewards WITHIN an utterance's
            # K rollouts: advantage = (r - mean) / (std + 1e-6). When that std
            # reaches zero the advantage is exactly zero and the policy can never
            # move again -- the absorbing state that killed the first RL runs
            # (ADR-005), which stayed invisible for seven epochs. These two are
            # the leading indicator, so they are logged every step.
            reward_std=reward_std.mean(),
            zero_variance_share=(reward_std <= 1e-6).float().mean(),
            sampled_rate_penalty=(
                predictions["rate_penalties"].mean()
                if predictions["full_ar"]
                else 0.0
            ),
            exp_rho=rho_bar.mean(),
        )
        return total

    # --------------------------------------------------------- batched fit_batch (RL)
    def fit_batch(self, batch):
        """Select the reference or batched on-policy GRPO update.

        ``on_policy`` is the original, validated implementation and uses SpeechBrain's
        accumulation-aware ``fit_batch``. ``batched_on_policy`` changes only reward
        evaluation: K segmentations are decoded in one wide batch, followed by one
        plain-GRPO backward pass that respects the same gradient accumulation schedule.
        ``combined_on_policy`` is the full-prefix Transformer-AR path: greedy and K
        sampled histories share one sequential policy loop and K rewards share one
        decoder forward. No rollout is reused across optimizer steps.
        """
        is_warmup = self._is_warmup(sb.Stage.TRAIN)
        self._set_decoder_phase_lr(is_warmup)
        is_rl = self.hparams.segmenter_mode == "joint" and not is_warmup
        update_mode = getattr(self.hparams, "rl_update_mode", "auto")
        full_ar = bool(getattr(self.modules.segmenter, "is_full_ar", False))
        if update_mode == "auto":
            update_mode = "combined_on_policy" if full_ar else "on_policy"
        if not is_rl or update_mode == "on_policy":
            return super().fit_batch(batch)
        if update_mode == "combined_on_policy":
            if not full_ar:
                raise ValueError(
                    "combined_on_policy requires segmenter_backbone=transformer_ar"
                )
            return self._fit_batch_full_ar_combined_on_policy(batch)
        if update_mode != "batched_on_policy":
            raise ValueError(
                "rl_update_mode must be 'auto', 'on_policy', "
                "'batched_on_policy', or 'combined_on_policy', "
                f"got {update_mode!r}"
            )
        if full_ar:
            raise ValueError(
                "Use combined_on_policy, not batched_on_policy, for transformer_ar"
            )
        return self._fit_batch_batched_on_policy(batch)


    def _fit_batch_full_ar_combined_on_policy(self, batch):
        """One exact on-policy update with grouped policy and reward batches.

        The expensive sequential dimension is still the frame axis.  We widen
        its batch to contain one greedy lane and K sampled lanes per utterance,
        then evaluate all K detached rewards in one decoder call.  The policy
        loss remains a differentiable parallel re-score of those same realized
        histories, and SpeechBrain's optimizer/accumulation schedule is unchanged.
        """
        should_step = (self.step % self.grad_accumulation_factor) == 0
        self.on_fit_batch_start(batch, should_step=should_step)
        schedule_logs = getattr(self, "_combined_schedule_logs", 0)
        if schedule_logs < self.grad_accumulation_factor:
            logger.info(
                "Combined Transformer-AR on-policy update: minibatch_step=%d; "
                "optimizer_step=%s; K=%d",
                self.step,
                should_step,
                int(self.hparams.grpo_k),
            )
            self._combined_schedule_logs = schedule_logs + 1

        batch = batch.to(self.device, non_blocking=True)
        K = int(self.hparams.grpo_k)
        reward_kind = getattr(self.hparams, "segmenter_reward", "nll")
        freeze_dec = bool(
            getattr(self.hparams, "freeze_decoder_in_joint", False)
        )

        # Generate one on-policy group and evaluate each sampled trajectory once.
        with torch.no_grad(), self.training_ctx:
            feats, segmenter_feats, feat_lens = self._encoder_features(batch)
            T = feats.size(1)
            pad_mask = lengths_to_padding_mask(feat_lens, T)
            group = self.modules.segmenter.sample_boundary_group(
                segmenter_feats, pad_mask, num_samples=K
            )
            argmax_b = group.greedy.boundaries
            sampled = list(group.sampled_boundaries)
            quality_rewards = self._rollout_rewards(
                feats, pad_mask, sampled, batch, reward_kind
            )
            rate_penalties, sampled_rhos = self._sampled_rate_terms(
                sampled, pad_mask
            )
            rewards = (
                quality_rewards - rate_penalties
                if "reward" in self._rate_channels()
                else quality_rewards
            )
            advantage = group_advantage(
                rewards,
                normalize_std=bool(self.hparams.grpo_normalize_std),
            )

        # Re-score the realized histories once with gradients, exactly as in the
        # reference full-AR objective. The decoder CE still uses the greedy path.
        with self.no_sync(not should_step):
            with self.training_ctx:
                pg, entropy, rl_rate = self._full_ar_pg_entropy(
                    segmenter_feats, pad_mask, sampled, advantage
                )
                beta_h = self._entropy_coeff()
                total = (
                    float(self.hparams.pg_weight) * pg
                    + rl_rate
                    - beta_h * entropy
                )
                if not freeze_dec:
                    dec = self._run_decoder(feats, pad_mask, argmax_b, batch)
                    dec_ce = self._decoder_ce(dec["llm_logits"], batch)
                    total = total + dec_ce
                else:
                    dec_ce = torch.zeros((), device=self.device)
            scaled = self.scaler.scale(total / self.grad_accumulation_factor)
            self.check_loss_isfinite(scaled)
            scaled.backward()

        if should_step:
            self.optimizers_step()

        self._track_rho_from_boundary(argmax_b, pad_mask)
        reward_std = rewards.std(dim=1)
        self._track_reward_std(reward_std)
        self._log_train(
            dec_ce=dec_ce,
            pg=pg,
            rate=rl_rate,
            entropy=entropy,
            beta_h=beta_h,
            reward=rewards.mean(),
            quality_reward=quality_rewards.mean(),
            reward_std=reward_std.mean(),
            zero_variance_share=(reward_std <= 1e-6).float().mean(),
            sampled_rate_penalty=rate_penalties.mean(),
            exp_rho=sampled_rhos.mean(),
        )
        self.on_fit_batch_end(batch, {}, total, should_step=should_step)
        return total.detach().cpu()

    def _fit_batch_batched_on_policy(self, batch):
        should_step = (self.step % self.grad_accumulation_factor) == 0
        self.on_fit_batch_start(batch, should_step=should_step)
        schedule_logs = getattr(self, "_rl_schedule_logs", 0)
        if schedule_logs < self.grad_accumulation_factor:
            logger.info(
                "Batched on-policy accumulation: minibatch_step=%d; optimizer_step=%s",
                self.step,
                should_step,
            )
            self._rl_schedule_logs = schedule_logs + 1
        batch = batch.to(self.device, non_blocking=True)
        K = int(self.hparams.grpo_k)
        reward_kind = getattr(self.hparams, "segmenter_reward", "cer")
        freeze_dec = bool(
            getattr(self.hparams, "freeze_decoder_in_joint", False)
        )
        pg_weight = float(self.hparams.pg_weight)
        rate_mode = getattr(self.hparams, "rate_mode", None) or (
            "pin"
            if bool(getattr(self.hparams, "rate_two_sided", False))
            else "captax"
        )

        # --- 1. rollout: generate ONCE (no grad) -- feats, old policy, K samples, decode.
        with torch.no_grad(), self.training_ctx:
            feats, segmenter_feats, feat_lens = self._encoder_features(batch)
            T = feats.size(1)
            pad_mask = lengths_to_padding_mask(feat_lens, T)
            logits_old = self.modules.segmenter.boundary_logits(
                segmenter_feats, pad_mask
            )
            sampled = [
                self.modules.segmenter.sample_boundary(
                    logits_old, pad_mask, mode="sample"
                )
                for _ in range(K)
            ]
            rewards = self._rollout_rewards(
                feats, pad_mask, sampled, batch, reward_kind
            )  # (B, K)
        adv = group_advantage(
            rewards, normalize_std=bool(self.hparams.grpo_normalize_std)
        )

        # --- 2. exactly one on-policy loss/backward; accumulate like SpeechBrain.
        with self.no_sync(not should_step):
            with self.training_ctx:
                logits = self.modules.segmenter.boundary_logits(
                    segmenter_feats, pad_mask
                )
                pg = logits.sum() * 0.0
                for k, bk in enumerate(sampled):
                    policy_logits = self.modules.segmenter.conditioned_logits(
                        logits, bk
                    )
                    pg = pg + grpo_pg_loss(policy_logits, bk, adv[:, k])
                pg = pg / K
                rl_rate, rho_bar = rate_loss(
                    logits,
                    pad_mask,
                    float(self.hparams.rho_star),
                    float(self.hparams.lambda_cap),
                    float(self.hparams.lambda_press),
                    rho_floor=float(getattr(self.hparams, "rho_floor", 0.0)),
                    lambda_floor=float(
                        getattr(self.hparams, "lambda_floor", 0.0)
                    ),
                    mode=rate_mode,
                    rho_lo=float(getattr(self.hparams, "rho_lo", 0.0)),
                    rho_hi=float(getattr(self.hparams, "rho_hi", 1.0)),
                    probabilities=self.modules.segmenter.boundary_marginals(
                        logits, pad_mask
                    ),
                )
                total = pg_weight * pg + rl_rate
                if not freeze_dec:
                    argmax_b = self.modules.segmenter.sample_boundary(
                        logits.detach(), pad_mask, mode="argmax"
                    )
                    dec = self._run_decoder(feats, pad_mask, argmax_b, batch)
                    dec_ce = self._decoder_ce(dec["llm_logits"], batch)
                    total = total + dec_ce
                else:
                    dec_ce = torch.zeros((), device=self.device)
            scaled = self.scaler.scale(total / self.grad_accumulation_factor)
            self.check_loss_isfinite(scaled)
            scaled.backward()

        if should_step:
            self.optimizers_step()

        # --- logging (once/batch): rho from the argmax path + loss components.
        with torch.no_grad(), self.training_ctx:
            argmax_b = self.modules.segmenter.sample_boundary(
                self.modules.segmenter.boundary_logits(
                    segmenter_feats, pad_mask
                ),
                pad_mask,
                mode="argmax",
            )
        self._track_rho_from_boundary(argmax_b, pad_mask)
        reward_std = rewards.std(dim=1)
        self._track_reward_std(reward_std)
        self._log_train(
            dec_ce=dec_ce,
            pg=pg,
            rate=rl_rate,
            entropy=0.0,
            beta_h=0.0,
            reward=rewards.mean(),
            reward_std=reward_std.mean(),
            zero_variance_share=(reward_std <= 1e-6).float().mean(),
            exp_rho=rho_bar.mean(),
        )
        self.on_fit_batch_end(batch, {}, total, should_step=should_step)
        return total.detach().cpu()

    # ------------------------------------------------------------ metric helpers
    def _track_reward_std(self, reward_std):
        """Record the per-utterance within-group reward std (collapse warning).

        Guarded because optimized update paths assemble rewards without
        going through a normal stage start; a missing accumulator must not turn
        a diagnostic into a crash.
        """
        if not hasattr(self, "_reward_std_samples"):
            self._reward_std_samples = []
        self._reward_std_samples.extend(reward_std.detach().tolist())

    def _track_rho_from_boundary(self, boundary, pad_mask):
        """Record realized rho, using the SAME definition the rate terms optimize."""
        self._rho_samples.extend(
            realized_kept_ratio(boundary, pad_mask).tolist()
        )

    def _append_wer(self, hyps, llm_logits, batch):
        ids = batch.id
        target = self._target_tokens(llm_logits, batch).masked_fill(
            self._target_tokens(llm_logits, batch) == self.hparams.ignore_index,
            self.tokenizer.pad_token_id,
        )
        preds = self.tokenizer.batch_decode(hyps[0], skip_special_tokens=True)
        preds_words = [p.split(" ") for p in preds]
        targets = self.tokenizer.batch_decode(target, skip_special_tokens=True)
        targets_words = [t.split(" ") for t in targets]
        self.cer_metric.append(ids, preds_words, targets_words)
        self.wer_metric.append(ids, preds_words, targets_words)

    @staticmethod
    def _percentile(values, q):
        """Nearest-rank percentile without adding a statistics dependency."""
        if not values:
            return None
        ordered = sorted(float(value) for value in values)
        index = round((len(ordered) - 1) * float(q))
        return ordered[index]

    def _record_inference_benchmark(self, predictions, batch):
        """Collect latency, rate, duration, and utterance-error diagnostics."""
        state = getattr(self, "_inference_benchmark", None)
        timing = predictions.get("inference_timing")
        if state is None or timing is None:
            return

        wavs, wav_lens = batch.sig
        durations = (
            wav_lens.detach().float().cpu() * float(wavs.size(1)) / 16000.0
        ).tolist()
        # Count the pooler's actual valid outputs.  This respects its frame-0
        # guard and therefore avoids double-counting a boundary at frame zero.
        segment_counts = (
            (~predictions["seg_pad"].detach()).sum(dim=1).float().cpu().tolist()
        )
        hyps = self.tokenizer.batch_decode(
            predictions["hyps"][0], skip_special_tokens=True
        )
        refs = list(batch.wrd)

        state["batch_count"] += 1
        warmup = state["batch_count"] <= int(
            getattr(self.hparams, "inference_warmup_batches", 5)
        )
        if not warmup:
            state["measured_batch_count"] += 1
            state["measured_utterances"] += len(durations)
            state["measured_audio_seconds"] += sum(durations)
            for name, value in timing.items():
                state.setdefault(name, []).append(float(value))

        for utterance_id, duration, n_segments, hyp, ref in zip(
            list(batch.id), durations, segment_counts, hyps, refs
        ):
            ref_words = str(ref).split()
            hyp_words = str(hyp).split()
            errors = _levenshtein(hyp_words, ref_words)
            token_hz = float(n_segments) / max(float(duration), 1.0e-9)
            if duration < 5.0:
                bucket = "<5s"
            elif duration < 10.0:
                bucket = "5-10s"
            elif duration < 20.0:
                bucket = "10-20s"
            else:
                bucket = ">=20s"
            state["utterances"].append(
                {
                    "id": str(utterance_id),
                    "duration_seconds": float(duration),
                    "segments": int(n_segments),
                    "token_hz": token_hz,
                    "word_errors": int(errors),
                    "reference_words": len(ref_words),
                    "wer": float(errors) / max(len(ref_words), 1),
                    "duration_bucket": bucket,
                    "hypothesis": str(hyp),
                    "reference": str(ref),
                }
            )

    def _write_inference_benchmark(
        self, stage_seconds, memory_stats, stage_stats
    ):
        """Append one compact, reproducible benchmark summary per test split."""
        state = getattr(self, "_inference_benchmark", None)
        output = getattr(self.hparams, "inference_benchmark_file", None)
        if state is None or not output or not if_main_process():
            return

        utterances = state["utterances"]
        token_hz = [row["token_hz"] for row in utterances]
        total_audio = sum(row["duration_seconds"] for row in utterances)
        total_segments = sum(row["segments"] for row in utterances)
        duration_buckets = {}
        for bucket in ("<5s", "5-10s", "10-20s", ">=20s"):
            rows = [
                row for row in utterances if row["duration_bucket"] == bucket
            ]
            bucket_audio = sum(row["duration_seconds"] for row in rows)
            duration_buckets[bucket] = {
                "utterances": len(rows),
                "audio_seconds": bucket_audio,
                "token_hz": (
                    sum(row["segments"] for row in rows) / bucket_audio
                    if bucket_audio
                    else None
                ),
                "mean_utterance_wer": (
                    sum(row["wer"] for row in rows) / len(rows)
                    if rows
                    else None
                ),
            }

        wall = state.get("wall_seconds", [])
        measured_audio = state["measured_audio_seconds"]
        measured_wall = sum(wall)
        result = {
            "split": str(getattr(self.hparams, "evaluation_split", "unknown")),
            "decoding_protocol": "batch_invariant_left_packed_duration_cap_v1",
            "max_decode_tokens_per_second": float(
                self.hparams.max_decode_tokens_per_second
            ),
            "max_decode_token_margin": int(
                self.hparams.max_decode_token_margin
            ),
            "max_decode_tokens": int(self.hparams.max_decode_tokens),
            "batch_size": int(self.hparams.test_dataloader_opts["batch_size"]),
            "warmup_batches": int(
                getattr(self.hparams, "inference_warmup_batches", 5)
            ),
            "utterances": len(utterances),
            "audio_seconds": total_audio,
            "stage_seconds_including_data_and_metrics": float(stage_seconds),
            "stage_rtf": float(stage_seconds) / max(total_audio, 1.0e-9),
            "measured_batches": state["measured_batch_count"],
            "measured_utterances": state["measured_utterances"],
            "measured_audio_seconds": measured_audio,
            "measured_forward_seconds": measured_wall,
            "forward_rtf": measured_wall / max(measured_audio, 1.0e-9),
            "utterances_per_second": state["measured_utterances"]
            / max(measured_wall, 1.0e-9),
            "batch_latency_seconds_median": self._percentile(wall, 0.5),
            "batch_latency_seconds_p95": self._percentile(wall, 0.95),
            "component_gpu_seconds": {
                name.removesuffix("_gpu_seconds"): sum(state.get(name, []))
                for name in (
                    "encoder_gpu_seconds",
                    "segmenter_gpu_seconds",
                    "decoder_gpu_seconds",
                )
            },
            "token_frequency_hz": {
                "global": total_segments / max(total_audio, 1.0e-9),
                "utterance_mean": sum(token_hz) / max(len(token_hz), 1),
                "utterance_p05": self._percentile(token_hz, 0.05),
                "utterance_median": self._percentile(token_hz, 0.5),
                "utterance_p95": self._percentile(token_hz, 0.95),
            },
            "duration_buckets": duration_buckets,
            "WER": float(stage_stats["WER"]),
            "CER": float(stage_stats["CER"]),
            "hardware": (
                torch.cuda.get_device_name(self.device)
                if self.device_type == "cuda"
                else "CPU"
            ),
            "precision": str(
                getattr(self.hparams, "eval_precision", "unknown")
            ),
            **memory_stats,
        }
        output_dir = os.path.dirname(output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(result) + "\n")
        split = str(result["split"]).replace("/", "_")
        utterance_output = Path(output).with_name(
            f"{Path(output).stem}_{split}_utterances.jsonl"
        )
        with open(utterance_output, "w", encoding="utf-8") as stream:
            for row in utterances:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        logger.info(
            "Inference benchmark: split=%s batch=%d RTF=%.4f token_hz=%.2f",
            result["split"],
            result["batch_size"],
            result["forward_rtf"],
            result["token_frequency_hz"]["global"],
        )

    def _log_train(self, **kw):
        for k, v in kw.items():
            self._train_accum[k] = self._train_accum.get(k, 0.0) + float(v)
        self._train_accum["n"] = self._train_accum.get("n", 0.0) + 1.0

    # --------------------------------------------------- optimizer-step validation
    def configure_step_validation(self, valid_set, loader_kwargs):
        """Build the held-out loader used for optimizer-step validation."""
        interval = int(
            getattr(self.hparams, "validation_interval_optimizer_steps", 0)
        )
        if interval < 1:
            raise ValueError(
                "validation_interval_optimizer_steps must be positive"
            )
        if self.optimizer_step_limit is None:
            raise ValueError(
                "Step validation requires --optimizer_step_limit so a final "
                "validation point is guaranteed."
            )
        self._step_validation_set = self.make_dataloader(
            valid_set,
            stage=sb.Stage.VALID,
            ckpt_prefix=None,
            **loader_kwargs,
        )
        self._step_validation_enable = not self.noprogressbar
        self._last_step_validation = None
        logger.info(
            "Step-controlled training: max_optimizer_steps=%d "
            "validation_interval=%d validate_at_warmup_end=%s",
            int(self.optimizer_step_limit),
            interval,
            bool(getattr(self.hparams, "validate_at_warmup_end", True)),
        )

    def _partial_train_stats(self):
        """Snapshot accumulated train metrics for an in-progress data pass."""
        stats = {
            "loss": float(getattr(self, "avg_train_loss", 0.0)),
            "optimizer_step": int(getattr(self, "optimizer_step", 0)),
        }
        rho_samples = getattr(self, "_rho_samples", [])
        if rho_samples:
            rho = torch.tensor(rho_samples)
            stats["rho_mean"] = float(rho.mean())
            stats["rho_std"] = float(rho.std()) if rho.numel() > 1 else 0.0
        spread_samples = getattr(self, "_reward_std_samples", [])
        if spread_samples:
            spread = torch.tensor(spread_samples)
            stats["reward_std_mean"] = float(spread.mean())
            stats["zero_variance_share"] = float(
                (spread <= 1e-6).float().mean()
            )
        acc = getattr(self, "_train_accum", {})
        n = acc.get("n", 0.0)
        if n > 0:
            for key, value in acc.items():
                if key != "n":
                    stats[key] = value / n
        return stats

    def _accumulate_train_memory_peaks(self):
        """Preserve train peaks across validation, which resets CUDA stats."""
        if self.device_type != "cuda":
            return
        self._train_peak_allocated_max = max(
            getattr(self, "_train_peak_allocated_max", 0),
            torch.cuda.max_memory_allocated(self.device),
        )
        self._train_peak_reserved_max = max(
            getattr(self, "_train_peak_reserved_max", 0),
            torch.cuda.max_memory_reserved(self.device),
        )

    def _run_step_validation(self, reason):
        """Pause training, validate, checkpoint, then restore train state."""
        train_step = self.step
        train_epoch = self.current_epoch
        train_started_at = self._stage_started_at
        train_rho_samples = self._rho_samples
        train_reward_std = self._reward_std_samples
        old_train_stats = getattr(self, "train_stats", None)
        had_train_stats = hasattr(self, "train_stats")
        validation_started_at = time.monotonic()
        self._accumulate_train_memory_peaks()
        self.train_stats = self._partial_train_stats()
        self._in_step_validation = True
        self._step_validation_reason = reason
        logger.info(
            "Running dev validation at optimizer_step=%d reason=%s",
            int(self.optimizer_step),
            reason,
        )
        try:
            self._fit_valid(
                self._step_validation_set,
                epoch=train_epoch,
                enable=self._step_validation_enable and if_main_process(),
            )
            self._last_step_validation = int(self.optimizer_step)
        finally:
            self._in_step_validation = False
            self.step = train_step
            self.current_epoch = train_epoch
            self._rho_samples = train_rho_samples
            self._reward_std_samples = train_reward_std
            self._stage_started_at = train_started_at
            self._step_validation_seconds_in_epoch += (
                time.monotonic() - validation_started_at
            )
            if had_train_stats:
                self.train_stats = old_train_stats
            else:
                del self.train_stats
            self.modules.train()
            if self.device_type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)

    def on_fit_batch_end(self, batch, outputs, loss, should_step):
        """Run held-out validation at configured optimizer-step boundaries."""
        super().on_fit_batch_end(batch, outputs, loss, should_step)
        if not should_step or not hasattr(self, "_step_validation_set"):
            return
        interval = int(self.hparams.validation_interval_optimizer_steps)
        warmup_step = (
            int(getattr(self.hparams, "warmup_optimizer_steps", 0) or 0)
            if bool(getattr(self.hparams, "validate_at_warmup_end", True))
            else 0
        )
        reason = step_validation_reason(
            self.optimizer_step,
            interval,
            warmup_step=warmup_step,
            max_steps=self.optimizer_step_limit,
            last_validated_step=getattr(self, "_last_step_validation", None),
        )
        if reason is not None:
            self._run_step_validation(reason)

    # --------------------------------------------------------------- stage hooks
    def on_stage_start(self, stage, epoch):
        self._stage_started_at = time.monotonic()
        self.current_epoch = epoch if epoch is not None else 1
        if stage == sb.Stage.TRAIN:
            self._step_validation_seconds_in_epoch = 0.0
            self._train_peak_allocated_max = 0
            self._train_peak_reserved_max = 0
        if self.device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        if (
            stage == sb.Stage.TRAIN
            and self.hparams.segmenter_mode == "joint"
            and hasattr(self, "optimizer")
        ):
            self._set_decoder_phase_lr(self._is_warmup(stage))
        self._rho_samples = []
        self._reward_std_samples = []
        if stage == sb.Stage.TRAIN:
            self._train_accum = {}
        if stage != sb.Stage.TRAIN:
            if self.hparams.segmenter_mode == "coldstart":
                self.prf_stats = {"precision": [], "recall": [], "f1": []}
            else:
                self.cer_metric = self.hparams.cer_computer()
                self.wer_metric = self.hparams.error_rate_computer()
        if stage == sb.Stage.TEST and getattr(
            self.hparams, "inference_benchmark_file", None
        ):
            self._inference_benchmark = {
                "batch_count": 0,
                "measured_batch_count": 0,
                "measured_utterances": 0,
                "measured_audio_seconds": 0.0,
                "utterances": [],
            }

    def on_stage_end(self, stage, stage_loss, epoch):
        stage_seconds = time.monotonic() - getattr(
            self, "_stage_started_at", time.monotonic()
        )
        if stage == sb.Stage.TRAIN:
            stage_seconds -= getattr(
                self, "_step_validation_seconds_in_epoch", 0.0
            )
        memory_stats = {}
        if self.device_type == "cuda":
            gib = 1024**3
            peak_allocated = torch.cuda.max_memory_allocated(self.device) / gib
            peak_reserved = torch.cuda.max_memory_reserved(self.device) / gib
            if stage == sb.Stage.TRAIN:
                self._accumulate_train_memory_peaks()
                peak_allocated = self._train_peak_allocated_max / gib
                peak_reserved = self._train_peak_reserved_max / gib
            capacity = (
                torch.cuda.get_device_properties(self.device).total_memory / gib
            )
            memory_stats = {
                "gpu_peak_allocated_gb": peak_allocated,
                "gpu_peak_reserved_gb": peak_reserved,
                "gpu_capacity_gb": capacity,
                "gpu_reserved_headroom_gb": capacity - peak_reserved,
            }
            logger.info(
                "GPU memory: stage=%s epoch=%s peak_allocated=%.2f GiB "
                "peak_reserved=%.2f GiB capacity=%.2f GiB headroom=%.2f GiB",
                stage.name,
                epoch,
                peak_allocated,
                peak_reserved,
                capacity,
                capacity - peak_reserved,
            )
        logger.info(
            "Stage timing: stage=%s epoch=%s seconds=%.3f",
            stage.name,
            epoch,
            stage_seconds,
        )
        timing_file = getattr(self.hparams, "stage_timing_file", None)
        if timing_file and if_main_process():
            os.makedirs(os.path.dirname(timing_file), exist_ok=True)
            with open(timing_file, "a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "stage": stage.name,
                            "epoch": epoch,
                            "seconds": stage_seconds,
                            **memory_stats,
                        }
                    )
                    + "\n"
                )
        stats = {"loss": stage_loss, **memory_stats}
        if self._rho_samples:
            rho = torch.tensor(self._rho_samples)
            stats["rho_mean"] = float(rho.mean())
            stats["rho_std"] = float(rho.std()) if rho.numel() > 1 else 0.0
        if getattr(self, "_reward_std_samples", None):
            spread = torch.tensor(self._reward_std_samples)
            stats["reward_std_mean"] = float(spread.mean())
            # Share of utterances whose K rollouts scored identically: their
            # advantage is exactly zero, so they contribute no gradient. A value
            # trending to 1.0 is the absorbing state forming.
            stats["zero_variance_share"] = float(
                (spread <= 1e-6).float().mean()
            )
        transition_bias = getattr(
            self.modules.segmenter, "transition_bias", None
        )
        if transition_bias is not None:
            bias = transition_bias.detach().float().cpu()
            stats["ar_bias_prev0"] = float(bias[0])
            stats["ar_bias_prev1"] = float(bias[1])
            stats["ar_bias_gap"] = float(bias[1] - bias[0])

        if stage == sb.Stage.TRAIN:
            self.train_stats = stats
            acc = getattr(self, "_train_accum", {})
            n = acc.get("n", 0.0)
            if n > 0:
                for k, v in acc.items():
                    if k != "n":
                        self.train_stats[k] = v / n
            return

        if self.hparams.segmenter_mode == "coldstart":
            for k in ("precision", "recall", "f1"):
                vals = self.prf_stats[k]
                stats["boundary_" + k] = (
                    float(sum(vals) / len(vals)) if vals else 0.0
                )
            key_stat, key_kw = "boundary_f1", {"max_keys": ["boundary_f1"]}
        else:
            stats["CER"] = self.cer_metric.summarize("error_rate")
            stats["WER"] = self.wer_metric.summarize("error_rate")
            key_stat, key_kw = "WER", {"min_keys": ["WER"]}

        if stage == sb.Stage.VALID:
            old_lr = float(
                self.optimizer.param_groups[-1]["lr"]
                if hasattr(self, "optimizer")
                else self.hparams.initial_lr
            )
            stats_meta = {"epoch": epoch, "lr": old_lr}
            checkpoint_meta = {key_stat: stats[key_stat], "epoch": epoch}
            checkpoint_kwargs = key_kw
            if getattr(self, "_in_step_validation", False):
                validation_step = int(self.optimizer_step)
                validation_reason = getattr(
                    self, "_step_validation_reason", "interval"
                )
                stats_meta.update(
                    {
                        "optimizer_step": validation_step,
                        "validation_reason": validation_reason,
                    }
                )
                checkpoint_meta.update(
                    {
                        "optimizer_step": validation_step,
                        "validation_step": validation_step,
                        "validation_reason": validation_reason,
                    }
                )
                checkpoint_kwargs = {
                    **key_kw,
                    "end_of_epoch": False,
                    "ckpt_predicate": lambda checkpoint: (
                        "validation_step" in checkpoint.meta
                    ),
                }
            self.hparams.train_logger.log_stats(
                stats_meta=stats_meta,
                train_stats=getattr(self, "train_stats", {}),
                valid_stats=stats,
            )
            self.checkpointer.save_and_keep_only(
                meta=checkpoint_meta,
                num_to_keep=int(
                    getattr(self.hparams, "checkpoints_to_keep", 1)
                ),
                **checkpoint_kwargs,
            )
        elif stage == sb.Stage.TEST:
            self._write_inference_benchmark(stage_seconds, memory_stats, stats)
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
            if self.hparams.segmenter_mode == "joint" and if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

    def on_evaluate_start(self, max_key=None, min_key=None):
        """Recover the requested retained-checkpoint rank when configured."""

        step = getattr(self.hparams, "eval_checkpoint_step", None)
        if step is not None:
            # Select by optimizer step, not by rank. Rank orders checkpoints by
            # the selection metric, so "the final checkpoint" has no fixed rank
            # and would have to be guessed per run -- and the whole reason to
            # evaluate the final checkpoint is that the metric-selected one is
            # the wrong model to report (ADR-024).
            if self.checkpointer is None:
                raise RuntimeError(
                    "eval_checkpoint_step requires a checkpointer"
                )
            step = int(step)
            matches = [
                c
                for c in self.checkpointer.find_checkpoints()
                if int(c.meta.get("optimizer_step", -1)) == step
            ]
            if not matches:
                available = sorted(
                    int(c.meta.get("optimizer_step", -1))
                    for c in self.checkpointer.find_checkpoints()
                )
                raise ValueError(
                    f"No retained checkpoint at optimizer_step={step}; "
                    f"available steps: {available}"
                )
            self.checkpointer.load_checkpoint(matches[0])
            logger.info(
                "Evaluation loaded checkpoint at optimizer_step=%d path=%s",
                step,
                matches[0].path,
            )
            return None

        rank = getattr(self.hparams, "eval_checkpoint_rank", None)
        if rank is None:
            return super().on_evaluate_start(max_key=max_key, min_key=min_key)
        if self.checkpointer is None:
            raise RuntimeError("eval_checkpoint_rank requires a checkpointer")
        rank = int(rank)
        checkpoints = self.checkpointer.find_checkpoints(
            max_key=max_key,
            min_key=min_key,
        )
        if not 0 <= rank < len(checkpoints):
            raise ValueError(
                f"eval_checkpoint_rank={rank} but only "
                f"{len(checkpoints)} checkpoints are available"
            )
        checkpoint = checkpoints[rank]
        self.checkpointer.load_checkpoint(checkpoint)
        logger.info(
            "Evaluation loaded retained checkpoint rank=%d path=%s meta=%s",
            rank,
            checkpoint.path,
            checkpoint.meta,
        )

    # --------------------------------------------------------------- optimizers
    def on_fit_start(self):
        """Recover exactly, then make the launched learning rates authoritative.

        PyTorch optimizer recovery restores checkpoint param-group learning rates.
        Reapply the current experiment's values so task-specific tuning is not silently
        replaced by the shared warm-up checkpoint's rates.
        """
        super().on_fit_start()
        rate_channels = self._rate_channels()
        logger.info(
            "Rate penalty channels: requested=%s resolved=%s; aux=%s reward=%s",
            getattr(self.hparams, "rate_channel", "aux"),
            sorted(rate_channels),
            "aux" in rate_channels,
            "reward" in rate_channels,
        )
        if "reward" in rate_channels:
            logger.warning("Reward-side rate penalty explicitly enabled; aux-only is the project default")
        freeze_policy = bool(
            getattr(self.hparams, "freeze_boundary_policy", False)
        )
        boundary_source = resolve_joint_boundary_source(
            getattr(self.hparams, "boundary_source", None)
        )
        fixed_rate_k = resolve_fixed_rate_k(
            getattr(self.hparams, "fixed_rate_k", None)
        )
        if boundary_source is not None and fixed_rate_k is not None:
            raise ValueError(
                "boundary_source=alignment and fixed_rate_k are mutually exclusive"
            )
        if boundary_source is not None and not freeze_policy:
            raise ValueError(
                "External alignment boundaries require "
                "freeze_boundary_policy=True"
            )
        if fixed_rate_k is not None and not freeze_policy:
            raise ValueError(
                "fixed_rate_k requires freeze_boundary_policy=True"
            )
        self.optimizer.param_groups[0]["lr"] = float(self.hparams.lr_segmenter)
        if self.hparams.segmenter_mode == "joint":
            self.optimizer.param_groups[1]["lr"] = float(
                self.hparams.lr_decoder
            )
            warmup_steps = int(
                getattr(self.hparams, "warmup_optimizer_steps", 0) or 0
            )
            if self.optimizer_step == 0 and warmup_steps > 0:
                if not self._is_warmup(sb.Stage.TRAIN):
                    raise RuntimeError(
                        "Fractional warmup was resolved but the Brain did not "
                        "enter warmup at optimizer step 0."
                    )
                logger.info(
                    "Fractional warmup assertion passed: optimizer_step=0 "
                    "warmup_optimizer_steps=%d",
                    warmup_steps,
                )
        logger.info(
            "RL update mode=%s; grad_accumulation=%d; lr_segmenter=%g; lr_decoder=%g",
            getattr(self.hparams, "rl_update_mode", "on_policy"),
            self.grad_accumulation_factor,
            float(self.hparams.lr_segmenter),
            float(self.hparams.lr_decoder)
            if self.hparams.segmenter_mode == "joint"
            else 0.0,
        )
        logger.info(
            "Boundary policy updates=%s; boundary source=%s; pooler=%s",
            "frozen" if freeze_policy else "joint-RL after warmup",
            (
                "external alignment"
                if boundary_source == "alignment"
                else (
                    f"fixed k={fixed_rate_k}"
                    if fixed_rate_k is not None
                    else "learned argmax"
                )
            ),
            getattr(self.modules.segmenter, "pooling_name", "unknown"),
        )

    def init_optimizers(self):
        """Single AdamW with per-group LRs (segmenter vs decoder). Mirrors the base
        recipe's ``self.optimizer`` + ``optimizers_dict`` pattern for a safe fit_batch.

        Cold-start trains only the segmenter; joint also trains the decoder (proj +
        LoRA). The SSL encoder is always frozen.
        """
        seg_params = [
            parameter
            for parameter in self.modules.segmenter.parameters()
            if parameter.requires_grad
        ]
        freeze_policy = bool(
            getattr(self.hparams, "freeze_boundary_policy", False)
        )
        if not seg_params and not (
            self.hparams.segmenter_mode == "joint" and freeze_policy
        ):
            raise ValueError(
                "The segmenter optimizer has no trainable parameters"
            )
        groups = [
            {"params": seg_params, "lr": float(self.hparams.lr_segmenter)}
        ]
        if self.hparams.segmenter_mode == "joint":
            dec_params = list(self.modules.proj.parameters()) + [
                p for p in self.modules.llm.parameters() if p.requires_grad
            ]
            groups.append(
                {"params": dec_params, "lr": float(self.hparams.lr_decoder)}
            )
        self.optimizer = torch.optim.AdamW(
            groups, weight_decay=float(self.hparams.weight_decay)
        )
        self.optimizers_dict = {"model_optimizer": self.optimizer}
        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("model_optimizer", self.optimizer)
            if self.hparams.segmenter_mode == "joint":
                # Save the trained decoder only in joint mode (avoids a 2.5 GB
                # untrained-LLM copy per cold-start checkpoint).
                self.checkpointer.add_recoverable("proj", self.modules.proj)
                self.checkpointer.add_recoverable("llm", self.modules.llm)


def _load_coldstart_segmenter(brain, ckpt_dir):
    """Load a cold-start segmenter state dict from the latest CKPT in ``ckpt_dir``."""
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "CKPT*")))
    if not ckpts:
        raise FileNotFoundError(f"No CKPT* under coldstart_ckpt_dir={ckpt_dir}")
    seg_file = os.path.join(ckpts[-1], "segmenter.ckpt")
    _load_segmenter_checkpoint(brain, seg_file)
    logger.info("Loaded cold-start segmenter from %s", seg_file)


def _load_segmenter_checkpoint(brain, seg_file):
    """Load an exact segmenter checkpoint with safe structured-policy upgrade."""
    if not os.path.isfile(seg_file):
        raise FileNotFoundError(f"Missing segmenter checkpoint: {seg_file}")
    state = torch.load(seg_file, map_location="cpu", weights_only=True)
    incompatible = brain.modules.segmenter.load_state_dict(state, strict=False)
    allowed_missing = set()
    if brain.modules.segmenter.sampling_strategy == "autoregressive":
        allowed_missing.add("transition_bias")
    if brain.modules.segmenter.pooling_name not in (
        "mean",
        "meanpool",
        "mean_pool",
    ):
        # Old boundary-policy checkpoints predate trainable pooling. Every new
        # order-aware pooler is a zero-residual extension, so these initialized
        # parameters reproduce mean pooling exactly and are safe to add.
        allowed_missing.update(
            key
            for key in incompatible.missing_keys
            if key.startswith("pooler.")
        )
    missing = set(incompatible.missing_keys) - allowed_missing
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Incompatible segmenter initialization: "
            f"missing={sorted(missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    logger.info(
        "Loaded exact segmenter initialization %s; sampling_strategy=%s; "
        "new_parameters=%s",
        seg_file,
        brain.modules.segmenter.sampling_strategy,
        sorted(allowed_missing & set(incompatible.missing_keys)),
    )


def _load_decoder_checkpoint(brain, ckpt_dir):
    """Initialize projection, LoRA, and normalization from an oracle decoder."""
    required = ("llm.ckpt", "proj.ckpt", "normalize.ckpt")
    missing = [
        name
        for name in required
        if not os.path.isfile(os.path.join(ckpt_dir, name))
    ]
    if missing:
        raise FileNotFoundError(
            f"Decoder checkpoint {ckpt_dir} is missing: {', '.join(missing)}"
        )
    brain.modules.llm.loader(os.path.join(ckpt_dir, "llm.ckpt"), True)
    brain.modules.proj.load_state_dict(
        torch.load(
            os.path.join(ckpt_dir, "proj.ckpt"),
            map_location="cpu",
            weights_only=True,
        )
    )
    brain.modules.normalize._load(
        os.path.join(ckpt_dir, "normalize.ckpt"), True
    )
    logger.info("Loaded oracle decoder initialization from %s", ckpt_dir)


def _attach_segmenter_ssl(hparams):
    """Construct the optional frozen SSL encoder used only by the segmenter.

    Keeping this construction outside HyperPyYAML makes the second encoder truly
    optional: ordinary single-encoder experiments do not instantiate or store a
    duplicate SSL model.
    """
    source = hparams.get("segmenter_ssl_hub")
    if not source:
        return
    if bool(hparams.get("use_feats", False)):
        raise ValueError("segmenter_ssl_hub requires use_feats=False")
    from speechbrain.integrations.huggingface.wav2vec2 import Wav2Vec2

    segmenter_ssl = Wav2Vec2(
        source=source,
        output_norm=bool(hparams.get("segmenter_ssl_output_norm", True)),
        freeze=bool(hparams.get("segmenter_ssl_frozen", True)),
        save_path=hparams["segmenter_ssl_folder"],
        device_map=hparams.get("segmenter_ssl_device", "cuda"),
    )
    hparams["modules"]["segmenter_ssl"] = segmenter_ssl
    logger.info(
        "Attached separate segmenter SSL encoder %s (expected feature dim=%s)",
        source,
        hparams.get("segmenter_input_dim"),
    )


def main(brain_class=SegmenterASR, configure_hparams=None):
    """Run the standard recipe, optionally with a continuation controller."""
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file, encoding="utf-8") as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    if configure_hparams is not None:
        configure_hparams(hparams)

    _attach_segmenter_ssl(hparams)

    sb.utils.distributed.ddp_init_group(run_opts)
    from librispeech_prepare import prepare_librispeech  # noqa

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    run_on_main(
        prepare_librispeech,
        kwargs={
            "data_folder": hparams["data_folder"],
            "tr_splits": hparams["train_splits"],
            "dev_splits": hparams["dev_splits"],
            "te_splits": hparams["test_splits"],
            "save_folder": hparams["output_folder"],
            "merge_lst": hparams["train_splits"],
            "merge_name": "train.csv",
            "skip_prep": hparams["skip_prep"],
        },
    )

    tokenizer = hparams["llm"].tokenizer
    (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_bsampler,
        valid_bsampler,
    ) = dataio_prepare(hparams, tokenizer)

    # Resolve all optimizer-step schedules before Brain construction. Brain
    # copies hparams, so writing these values afterward leaves the live Brain
    # with stale/null schedule values (the 2026-08-24 LS960 warmup bug).
    effective_grad_accumulation = resolve_effective_grad_accumulation(
        hparams["grad_accumulation_factor"],
        run_opts.grad_accumulation_factor,
        "grad_accumulation_factor" in run_opts.overridden_args,
    )
    hparams["grad_accumulation_factor"] = effective_grad_accumulation
    if train_bsampler is not None:
        warmup_fraction = hparams.get("warmup_fraction_of_epoch")
        if hparams["segmenter_mode"] == "joint" and warmup_fraction is not None:
            resolved_steps = resolve_fractional_warmup_steps(
                len(train_bsampler),
                effective_grad_accumulation,
                float(warmup_fraction),
            )
            hparams["warmup_optimizer_steps"] = resolved_steps
            logger.info(
                "Fractional decoder warmup: fraction=%.3f batches=%d "
                "effective_grad_accumulation=%d optimizer_steps=%d",
                float(warmup_fraction),
                len(train_bsampler),
                effective_grad_accumulation,
                resolved_steps,
            )

    asr_brain = brain_class(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )
    asr_brain.optimizer_step_limit = resolve_optimizer_step_limit(
        asr_brain.optimizer_step_limit
    )
    asr_brain.tokenizer = tokenizer
    asr_brain.txt_embedding = (
        asr_brain.raw_modules.llm.model.get_input_embeddings()
    )
    asr_brain.current_epoch = 1

    # Joint mode can initialize the segmenter and decoder independently. Exact
    # checkpoint paths take precedence over the legacy save-directory lookup.
    if hparams["segmenter_mode"] == "joint":
        if hparams.get("segmenter_init_checkpoint"):
            _load_segmenter_checkpoint(
                asr_brain, hparams["segmenter_init_checkpoint"]
            )
        elif hparams.get("coldstart_ckpt_dir"):
            _load_coldstart_segmenter(asr_brain, hparams["coldstart_ckpt_dir"])
        if hparams.get("decoder_init_ckpt_dir"):
            _load_decoder_checkpoint(
                asr_brain, hparams["decoder_init_ckpt_dir"]
            )
        if bool(hparams.get("freeze_boundary_policy", False)):
            frozen, trainable_pooler = freeze_boundary_policy_parameters(
                asr_brain.modules.segmenter
            )
            logger.info(
                "Frozen %d boundary-policy parameters; retained %d trainable "
                "pooler parameters",
                frozen,
                trainable_pooler,
            )
    for p in asr_brain.modules.ssl.parameters():
        p.requires_grad = False
    segmenter_ssl = getattr(asr_brain.modules, "segmenter_ssl", None)
    if segmenter_ssl is not None:
        for p in segmenter_ssl.parameters():
            p.requires_grad = False

    train_dl = hparams["train_dataloader_opts"]
    valid_dl = hparams["valid_dataloader_opts"]
    if train_bsampler is not None:
        cf = train_dl.get("collate_fn")
        train_dl = {
            "batch_sampler": train_bsampler,
            **loader_runtime_options(hparams),
        }
        if cf is not None:
            train_dl["collate_fn"] = cf
    if valid_bsampler is not None:
        cf = valid_dl.get("collate_fn")
        valid_dl = {"batch_sampler": valid_bsampler}
        if cf is not None:
            valid_dl["collate_fn"] = cf

    if hasattr(asr_brain, "configure_validation_datasets"):
        asr_brain.configure_validation_datasets(valid_data, test_datasets)

    if bool(hparams.get("eval_only", False)):
        # ``fit`` normally creates the optimizer and registers the joint decoder
        # recoverables.  Evaluation-only mode needs the latter so the selected
        # checkpoint restores segmenter, projection, LoRA, and normalization.
        asr_brain.init_optimizers()
        logger.info(
            "Evaluation-only mode: skipping fit and recovering for test"
        )
    else:
        fit_valid_data = valid_data
        if int(hparams.get("validation_interval_optimizer_steps", 0)) > 0:
            asr_brain.configure_step_validation(valid_data, valid_dl)
            fit_valid_data = None
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            fit_valid_data,
            train_loader_kwargs=train_dl,
            valid_loader_kwargs=valid_dl,
        )

    # WER eval only meaningful in joint mode.
    if hparams["segmenter_mode"] == "joint" and not hparams.get("skip_final_test", False):
        os.makedirs(hparams["output_wer_folder"], exist_ok=True)
        for k in test_datasets.keys():
            asr_brain.hparams.evaluation_split = k
            asr_brain.hparams.test_wer_file = os.path.join(
                hparams["output_wer_folder"], f"wer_{k}.txt"
            )
            asr_brain.evaluate(
                test_datasets[k],
                min_key="WER",
                test_loader_kwargs=hparams["test_dataloader_opts"],
            )
    return asr_brain


if __name__ == "__main__":
    main()
