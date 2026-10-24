"""Learned segmentation module for the SpeechLLM downstream stack.

A small policy reads speech representations ``(B, T, C)`` and emits **one boundary
logit per frame** (``p_t = sigmoid(logit_t)`` = probability frame ``t`` starts a new
segment). Frames between boundaries are pooled (mean by default, with optional
order-aware residuals in ``segment_pooling.py``) into a single embedding, producing
a shorter sequence handed to the LLM. The policy is
trained with **GRPO** (group-relative REINFORCE) against a per-utterance reward
(``-NLL`` from the jointly-trained decoder), plus a differentiable **rate objective**
(a band penalty by default). The default entropy weight is zero. It is
initialized by a supervised **cold-start** (BCE of the logits vs char-level CTC
boundaries) — see ``train.py``.

Conventions (shared with ``segment_pooling.py``):
    boundary tensor ``(B, T)`` values: ``1`` = boundary (frame starts a new segment),
    ``0`` = continuation, ``-1`` = padding. Frame 0 always starts segment 0 (the
    frame-0 guard lives in ``compute_segment_ids``).

Policy-gradient conventions:
  * one acoustic logit / frame, with either independent Bernoulli sampling or a
    first-order autoregressive policy over consecutive boundary labels;
  * ``gamma = 1`` (a one-step bandit): every frame gets the same utterance advantage,
    so there is **no discounted return**;
  * policy-gradient reduced with ``.mean()`` over valid frames (not ``.sum()``), so the
    effective LR does not track the dynamic-batch token count;
  * **no KL-to-cold-start** — the cold-start is an init only, RL is free to leave it.

A differentiable arm (CIF) is intentionally NOT here: Gumbel-softmax straight-through
cannot flow gradients through the hard mean-pool (the pooled output is piecewise-constant
in the boundaries), so a differentiable policy needs soft pooling (CIF) — deferred.

Authors
-------
 * Speech-LLM segmenter, 2026
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_pooling import build_aggregator, lengths_to_padding_mask

# ---------------------------------------------------------------------------
# Task conditioning (FiLM) — no-op for single-task v1, interface for the future
# ---------------------------------------------------------------------------


class TaskFiLM(nn.Module):
    """Per-task FiLM (scale/shift) on the input features. Identity when single-task.

    Arguments
    ---------
    num_tasks : int
        Number of tasks. ``<= 1`` disables conditioning (pure identity).
    dim : int
        Feature dimension the modulation applies to.
    """

    def __init__(self, num_tasks, dim):
        super().__init__()
        self.enabled = bool(num_tasks and num_tasks > 1)
        if self.enabled:
            self.gamma = nn.Embedding(num_tasks, dim)
            self.beta = nn.Embedding(num_tasks, dim)
            nn.init.ones_(self.gamma.weight)
            nn.init.zeros_(self.beta.weight)

    def forward(self, x, task_id=None):
        """Modulate ``x`` (B, T, C) by task. No-op if disabled or ``task_id`` is None.

        ``task_id`` may be an int, a scalar tensor, or a per-example tensor. A
        scalar is the normal case for this project, where every batch is task
        homogeneous, and it survives the K-fold batch replication the GRPO
        rollout paths perform. A per-example tensor shorter than the batch is
        tiled, which is exactly what replication produces.
        """
        if not self.enabled or task_id is None:
            return x
        if not torch.is_tensor(task_id):
            task_id = torch.tensor(task_id, device=x.device)
        task_id = task_id.to(device=x.device, dtype=torch.long)
        if task_id.dim() == 0:
            task_id = task_id.reshape(1)
        if task_id.numel() == 1:
            task_id = task_id.expand(x.size(0))
        elif task_id.size(0) != x.size(0):
            if x.size(0) % task_id.size(0) != 0:
                raise ValueError(
                    f"Cannot align {task_id.size(0)} task ids to a batch of "
                    f"{x.size(0)}: the batch is not a whole multiple of the "
                    "task-id count."
                )
            task_id = task_id.repeat(x.size(0) // task_id.size(0))
        g = self.gamma(task_id).unsqueeze(1)  # (B, 1, C)
        b = self.beta(task_id).unsqueeze(1)
        return g * x + b


# ---------------------------------------------------------------------------
# Boundary-policy backbones (features (B, T, C) -> per-frame logit (B, T))
# ---------------------------------------------------------------------------


class CNNBoundaryPredictor(nn.Module):
    """1-D CNN boundary head. RF ~= kernel_size + 2 frames (local cues)."""

    def __init__(self, input_dim, hidden_dim=256, kernel_size=7, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            input_dim, hidden_dim, kernel_size, padding=kernel_size // 2
        )
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.head = nn.Conv1d(hidden_dim, 1, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, padding_mask=None):
        """``x`` (B, T, C) -> logits (B, T). ``padding_mask`` unused (conv is local)."""
        h = x.transpose(1, 2)  # (B, C, T)
        h = self.dropout(self.relu(self.conv1(h)))
        h = self.dropout(self.relu(self.conv2(h)))
        return self.head(h).squeeze(1)  # (B, T)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard fixed sinusoidal PE, added to (B, T, d)."""

    def __init__(self, dim, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, dim)

    def forward(self, x):
        """Add PE to ``x`` (B, T, d); grows the buffer if T exceeds max_len."""
        T = x.size(1)
        if T > self.pe.size(1):
            # Fallback for unusually long utterances: extend on the fly.
            dim = self.pe.size(2)
            position = torch.arange(T, device=x.device).unsqueeze(1).float()
            div = torch.exp(
                torch.arange(0, dim, 2, device=x.device).float()
                * (-math.log(10000.0) / dim)
            )
            pe = torch.zeros(T, dim, device=x.device)
            pe[:, 0::2] = torch.sin(position * div)
            pe[:, 1::2] = torch.cos(position * div)
            return x + pe.unsqueeze(0)
        return x + self.pe[:, :T].to(x.dtype)


class TransformerBoundaryPredictor(nn.Module):
    """Pre-norm Transformer-encoder boundary head (global context + rate awareness)."""

    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_layers=4,
        nhead=4,
        ffn_dim=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.pe = SinusoidalPositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x, padding_mask=None):
        """``x`` (B, T, C) -> logits (B, T). ``padding_mask`` (B, T) True at padding."""
        h = self.pe(self.in_proj(x))
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        return self.head(h).squeeze(-1)  # (B, T)


class CausalBoundaryBlock(nn.Module):
    """Small pre-norm causal self-attention block with an explicit KV cache."""

    def __init__(self, hidden_dim, nhead, ffn_dim, dropout=0.0):
        super().__init__()
        if hidden_dim % nhead:
            raise ValueError("hidden_dim must be divisible by nhead")
        self.nhead = nhead
        self.head_dim = hidden_dim // nhead
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = float(dropout)

    def _split_qkv(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.nhead, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        return (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )

    def forward(self, x, history_window=None):
        """Parallel teacher-forced pass over a complete sequence."""
        residual = x
        q, k, v = self._split_qkv(self.norm1(x))
        if history_window is None:
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Boolean SDPA masks use True for entries that participate in attention.
            # This path is an explicit local-history model ablation; the default None
            # retains the original full causal attention kernel and behavior.
            positions = torch.arange(x.size(1), device=x.device)
            distance = positions[:, None] - positions[None, :]
            local_causal = (distance >= 0) & (distance < history_window)
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=local_causal,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        attn = attn.transpose(1, 2).reshape_as(x)
        x = residual + self.out(attn)
        return x + self.ffn(self.norm2(x))

    def step(self, x, cache=None, history_window=None):
        """One cached decoding step; ``x`` has shape ``(B, 1, C)``."""
        residual = x
        q, k, v = self._split_qkv(self.norm1(x))
        if cache is not None:
            old_k, old_v = cache
            k = torch.cat([old_k, k], dim=2)
            v = torch.cat([old_v, v], dim=2)
        if history_window is not None and k.size(2) > history_window:
            k = k[:, :, -history_window:]
            v = v[:, :, -history_window:]
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=0.0, is_causal=False
        )
        attn = attn.transpose(1, 2).reshape_as(x)
        x = residual + self.out(attn)
        return x + self.ffn(self.norm2(x)), (k, v)

    def step_preallocated(self, x, cache, position, history_window=None):
        """One step using indexed KV writes instead of repeated ``torch.cat``."""
        residual = x
        q, k, v = self._split_qkv(self.norm1(x))
        key_cache, value_cache = cache
        key_cache[:, :, position : position + 1].copy_(k)
        value_cache[:, :, position : position + 1].copy_(v)
        start = 0
        if history_window is not None:
            start = max(0, position + 1 - history_window)
        attn = F.scaled_dot_product_attention(
            q,
            key_cache[:, :, start : position + 1],
            value_cache[:, :, start : position + 1],
            dropout_p=0.0,
            is_causal=False,
        )
        attn = attn.transpose(1, 2).reshape_as(x)
        x = residual + self.out(attn)
        return x + self.ffn(self.norm2(x))


@dataclass
class BoundaryPolicyOutput:
    """One closed-loop boundary rollout and its exact conditional scores."""

    boundaries: torch.Tensor
    logits: torch.Tensor
    log_probs: torch.Tensor
    valid_mask: torch.Tensor


@dataclass
class BoundaryPolicyGroupOutput:
    """One greedy rollout plus independent stochastic rollouts.

    The trajectories share a single frame-by-frame policy loop.  Batch lanes
    ``[:B]`` use greedy actions and the following ``num_samples`` groups of
    ``B`` lanes sample from the same conditional policy.
    """

    greedy: BoundaryPolicyOutput
    sampled_boundaries: tuple


class CausalTransformerBoundaryPolicy(nn.Module):
    """Decoder-only policy ``p(b_t | audio, b_<t)`` with cached rollout.

    WavLM features may be bidirectional. "Causal" here refers specifically to the
    boundary-label history: the input at frame ``t`` contains ``b_{t-1}``, and causal
    self-attention can use all earlier label decisions but no future decisions.
    """

    bos_index = 2

    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        num_layers=4,
        nhead=4,
        ffn_dim=1024,
        dropout=0.0,
        max_positions=4096,
        history_window=None,
        cache_mode="concat",
    ):
        super().__init__()
        self.audio_proj = nn.Linear(input_dim, hidden_dim)
        self.boundary_embedding = nn.Embedding(3, hidden_dim)
        self.pe = SinusoidalPositionalEncoding(
            hidden_dim, max_len=max_positions
        )
        self.layers = nn.ModuleList(
            [
                CausalBoundaryBlock(hidden_dim, nhead, ffn_dim, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)
        self.max_positions = int(max_positions)
        if history_window is not None and int(history_window) < 1:
            raise ValueError("history_window must be positive or None")
        if cache_mode not in ("concat", "preallocated"):
            raise ValueError("cache_mode must be 'concat' or 'preallocated'")
        self.history_window = (
            None if history_window is None else int(history_window)
        )
        self.cache_mode = cache_mode

    @staticmethod
    def _normalized_boundaries(boundaries, padding_mask):
        normalized = boundaries.long().masked_fill(padding_mask, -1).clone()
        if normalized.size(1):
            normalized[:, 0] = torch.where(
                padding_mask[:, 0],
                torch.full_like(normalized[:, 0], -1),
                torch.zeros_like(normalized[:, 0]),
            )
        return normalized

    @staticmethod
    def action_mask(padding_mask):
        valid = ~padding_mask
        if valid.size(1):
            valid = valid.clone()
            valid[:, 0] = False
        return valid

    def teacher_forced_logits(self, features, padding_mask, boundaries):
        """Score all actions in parallel using the shifted realized/gold history."""
        T = features.size(1)
        if T > self.max_positions:
            raise ValueError(
                f"Boundary sequence length {T} exceeds max_positions="
                f"{self.max_positions}"
            )
        normalized = self._normalized_boundaries(boundaries, padding_mask)
        previous = torch.zeros_like(normalized)
        if T:
            previous[:, 0] = self.bos_index
        if T > 1:
            previous[:, 1:] = normalized[:, :-1].clamp(min=0, max=1)
        h = self.audio_proj(features) + self.boundary_embedding(previous)
        h = self.pe(h)
        for layer in self.layers:
            h = layer(h, history_window=self.history_window)
        return self.head(self.norm(h)).squeeze(-1)

    def _step(self, feature_t, previous, position, caches):
        if position >= self.max_positions:
            raise ValueError(
                f"Boundary position {position} exceeds max_positions="
                f"{self.max_positions}"
            )
        h = self.audio_proj(feature_t).unsqueeze(1)
        h = h + self.boundary_embedding(previous).unsqueeze(1)
        h = h + self.pe.pe[:, position : position + 1].to(
            device=h.device, dtype=h.dtype
        )
        new_caches = []
        for layer, cache in zip(self.layers, caches):
            if self.cache_mode == "preallocated":
                h = layer.step_preallocated(
                    h,
                    cache,
                    position,
                    history_window=self.history_window,
                )
                new_caches.append(cache)
            else:
                h, cache = layer.step(
                    h, cache, history_window=self.history_window
                )
                new_caches.append(cache)
        return self.head(self.norm(h)).squeeze(-1).squeeze(-1), new_caches

    def _new_caches(self, batch_size, frames, device, dtype):
        """Allocate empty caches for one closed-loop rollout."""
        if self.cache_mode != "preallocated":
            return [None] * len(self.layers)
        caches = []
        for layer in self.layers:
            shape = (batch_size, layer.nhead, frames, layer.head_dim)
            caches.append(
                (
                    torch.empty(shape, device=device, dtype=dtype),
                    torch.empty(shape, device=device, dtype=dtype),
                )
            )
        return caches

    def rollout(self, features, padding_mask, mode="sample", generator=None):
        """Greedy or stochastic cached rollout over the complete boundary sequence."""
        if mode not in ("argmax", "sample"):
            raise ValueError(f"Unknown rollout mode {mode!r}")
        B, T, _ = features.shape
        previous = torch.full(
            (B,), self.bos_index, dtype=torch.long, device=features.device
        )
        caches = self._new_caches(B, T, features.device, features.dtype)
        boundaries, logits, log_probs = [], [], []
        for t in range(T):
            logit, caches = self._step(features[:, t], previous, t, caches)
            active = ~padding_mask[:, t]
            if t == 0:
                action = torch.zeros(
                    B, dtype=torch.long, device=features.device
                )
            elif mode == "argmax":
                action = (logit > 0).long()
            else:
                action = torch.bernoulli(
                    torch.sigmoid(logit.float()), generator=generator
                ).long()
            action = action.masked_fill(~active, -1)
            safe_action = action.clamp(min=0).float()
            logp = -F.binary_cross_entropy_with_logits(
                logit.float(), safe_action, reduction="none"
            )
            action_valid = active & (t > 0)
            boundaries.append(action)
            logits.append(logit)
            log_probs.append(logp.masked_fill(~action_valid, 0.0))
            previous = torch.where(active, action.clamp(min=0), previous)
        if not T:
            empty = features.new_empty((B, 0))
            return BoundaryPolicyOutput(
                empty.long(), empty, empty, (~padding_mask)
            )
        return BoundaryPolicyOutput(
            torch.stack(boundaries, dim=1),
            torch.stack(logits, dim=1),
            torch.stack(log_probs, dim=1),
            self.action_mask(padding_mask),
        )

    def rollout_group(
        self, features, padding_mask, num_samples, generator=None
    ):
        """Run greedy and ``num_samples`` stochastic histories in one loop.

        This preserves the greedy policy and each sample's conditional
        distribution while removing a complete second sequential pass over the
        50 Hz boundary grid.  It is intended for on-policy GRPO, where the
        decoder consumes the greedy history and rewards use the sampled ones.
        """
        num_samples = int(num_samples)
        if num_samples < 1:
            raise ValueError("num_samples must be at least one")
        B, T, _ = features.shape
        copies = num_samples + 1
        grouped_features = features.repeat(copies, 1, 1)
        grouped_padding = padding_mask.repeat(copies, 1)
        grouped_batch = grouped_features.size(0)
        previous = torch.full(
            (grouped_batch,),
            self.bos_index,
            dtype=torch.long,
            device=features.device,
        )
        caches = self._new_caches(
            grouped_batch, T, features.device, features.dtype
        )
        boundaries, greedy_logits, greedy_log_probs = [], [], []
        for t in range(T):
            logit, caches = self._step(
                grouped_features[:, t], previous, t, caches
            )
            active = ~grouped_padding[:, t]
            if t == 0:
                action = torch.zeros(
                    grouped_batch, dtype=torch.long, device=features.device
                )
            else:
                # Draw only the stochastic lanes. Besides avoiding unused random
                # numbers, this preserves the reference rollout's RNG stream for
                # matched baseline/optimized comparisons.
                sampled_action = torch.bernoulli(
                    torch.sigmoid(logit[B:].float()), generator=generator
                ).long()
                action = torch.cat(
                    [(logit[:B] > 0).long(), sampled_action], dim=0
                )
            action = action.masked_fill(~active, -1)
            boundaries.append(action)

            greedy_action = action[:B].clamp(min=0).float()
            greedy_logp = -F.binary_cross_entropy_with_logits(
                logit[:B].float(), greedy_action, reduction="none"
            )
            greedy_valid = active[:B] & (t > 0)
            greedy_logits.append(logit[:B])
            greedy_log_probs.append(greedy_logp.masked_fill(~greedy_valid, 0.0))
            previous = torch.where(active, action.clamp(min=0), previous)

        if not T:
            empty = features.new_empty((B, 0))
            greedy = BoundaryPolicyOutput(
                empty.long(), empty, empty, self.action_mask(padding_mask)
            )
            samples = tuple(empty.long() for _ in range(num_samples))
            return BoundaryPolicyGroupOutput(greedy, samples)

        grouped_boundaries = torch.stack(boundaries, dim=1)
        greedy = BoundaryPolicyOutput(
            grouped_boundaries[:B],
            torch.stack(greedy_logits, dim=1),
            torch.stack(greedy_log_probs, dim=1),
            self.action_mask(padding_mask),
        )
        samples = tuple(grouped_boundaries[B:].chunk(num_samples, dim=0))
        return BoundaryPolicyGroupOutput(greedy, samples)

    def score(self, features, padding_mask, boundaries):
        """Differentiably re-score a completed sequence under its exact prefix."""
        boundaries = self._normalized_boundaries(boundaries, padding_mask)
        logits = self.teacher_forced_logits(features, padding_mask, boundaries)
        valid = self.action_mask(padding_mask)
        log_probs = -F.binary_cross_entropy_with_logits(
            logits.float(), boundaries.clamp(min=0).float(), reduction="none"
        )
        return BoundaryPolicyOutput(
            boundaries, logits, log_probs.masked_fill(~valid, 0.0), valid
        )


# ---------------------------------------------------------------------------
# Segmenter: boundary policy + configurable segment pooling
# ---------------------------------------------------------------------------


class Segmenter(nn.Module):
    """Boundary policy (one logit/frame) + configurable segment pooling.

    Arguments
    ---------
    input_dim : int
        Boundary-policy encoder feature dimension ``C``.
    backbone : str
        ``"cnn"`` or ``"transformer"``.
    hidden_dim, kernel_size, num_layers, nhead, ffn_dim, dropout
        Backbone hyperparameters (kernel_size for cnn; num_layers/nhead/ffn_dim for
        transformer).
    num_tasks : int
        Task-conditioning size (``<=1`` -> FiLM is identity).
    pooling, pooling_input_dim, pooling_hidden_dim, pooling_num_layers
        Segment-pooling configuration. ``pooling='mean'`` preserves the original
        parameter-free behavior. Order-aware residual modes keep the same output
        dimension and initialize to exact mean pooling.
    """

    def __init__(
        self,
        input_dim,
        backbone="cnn",
        hidden_dim=256,
        kernel_size=7,
        num_layers=4,
        nhead=4,
        ffn_dim=1024,
        dropout=0.1,
        num_tasks=1,
        sampling_strategy="bernoulli",
        ar_hidden_dim=None,
        ar_num_layers=None,
        ar_nhead=None,
        ar_ffn_dim=None,
        ar_dropout=0.0,
        ar_max_positions=4096,
        ar_history_window=None,
        ar_cache_mode="concat",
        pooling="mean",
        pooling_input_dim=None,
        pooling_hidden_dim=128,
        pooling_num_layers=1,
        pooling_dropout=0.0,
    ):
        super().__init__()
        self.film = TaskFiLM(num_tasks, input_dim)
        # Multi-task runs set this once per (task-homogeneous) batch so every
        # policy entry point conditions on the same task without threading an
        # argument through each call site. ``None`` -- the default -- keeps
        # single-task ASR behavior byte-for-byte identical.
        self.active_task_id = None
        backbone = (backbone or "cnn").lower()
        if backbone == "cnn":
            self.net = CNNBoundaryPredictor(
                input_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                dropout=dropout,
            )
        elif backbone in ("transformer", "tf"):
            self.net = TransformerBoundaryPredictor(
                input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                nhead=nhead,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
        elif backbone == "transformer_ar":
            self.net = CausalTransformerBoundaryPolicy(
                input_dim,
                hidden_dim=ar_hidden_dim or hidden_dim,
                num_layers=ar_num_layers or num_layers,
                nhead=ar_nhead or nhead,
                ffn_dim=ar_ffn_dim or ffn_dim,
                dropout=ar_dropout,
                max_positions=ar_max_positions,
                history_window=ar_history_window,
                cache_mode=ar_cache_mode,
            )
        else:
            raise ValueError(
                f"Unknown segmenter backbone '{backbone}'. Use 'cnn', 'transformer', "
                "or 'transformer_ar'."
            )
        self.backbone_name = backbone
        self.is_full_ar = backbone == "transformer_ar"
        self.pooling_name = (pooling or "mean").lower()
        pooling_input_dim = pooling_input_dim or input_dim
        self.pooler = build_aggregator(
            self.pooling_name,
            pooling_input_dim,
            hidden_dim=pooling_hidden_dim,
            num_layers=pooling_num_layers,
            dropout=pooling_dropout,
        )
        self.out_dim = self.pooler.out_dim
        sampling_strategy = (sampling_strategy or "bernoulli").lower()
        if sampling_strategy not in ("bernoulli", "autoregressive"):
            raise ValueError(
                "sampling_strategy must be 'bernoulli' or 'autoregressive', "
                f"got {sampling_strategy!r}"
            )
        self.sampling_strategy = (
            "transformer_ar" if self.is_full_ar else sampling_strategy
        )
        if sampling_strategy == "autoregressive" and not self.is_full_ar:
            # Added to the acoustic logit according to the preceding boundary
            # label. Zero initialization exactly recovers the independent policy,
            # so an old char checkpoint remains a behavior-preserving warm start.
            self.transition_bias = nn.Parameter(torch.zeros(2))

    def _resolve_task(self, task_id):
        """Explicit ``task_id`` wins; otherwise fall back to the active task."""
        return self.active_task_id if task_id is None else task_id

    def boundary_logits(self, features, padding_mask, task_id=None):
        """Per-frame boundary logits ``(B, T)`` (padding positions are meaningless)."""
        if self.is_full_ar:
            raise RuntimeError(
                "transformer_ar logits require a boundary prefix; use "
                "teacher_forced_logits(), sample_boundaries(), or score_boundaries()."
            )
        h = self.film(features, self._resolve_task(task_id))
        return self.net(h, padding_mask)

    def teacher_forced_logits(
        self, features, padding_mask, boundaries, task_id=None
    ):
        """Parallel conditional logits for the full-prefix Transformer policy."""
        if not self.is_full_ar:
            raise RuntimeError(
                "teacher_forced_logits is only for transformer_ar"
            )
        return self.net.teacher_forced_logits(
            self.film(features, self._resolve_task(task_id)),
            padding_mask,
            boundaries,
        )

    def sample_boundaries(
        self,
        features,
        padding_mask,
        deterministic=False,
        generator=None,
        task_id=None,
    ):
        """Closed-loop cached rollout for the full-prefix Transformer policy."""
        if not self.is_full_ar:
            raise RuntimeError("sample_boundaries is only for transformer_ar")
        mode = "argmax" if deterministic else "sample"
        return self.net.rollout(
            self.film(features, self._resolve_task(task_id)),
            padding_mask,
            mode=mode,
            generator=generator,
        )

    def sample_boundary_group(
        self,
        features,
        padding_mask,
        num_samples,
        generator=None,
        task_id=None,
    ):
        """Share one closed-loop pass across greedy and sampled histories."""
        if not self.is_full_ar:
            raise RuntimeError(
                "sample_boundary_group is only for transformer_ar"
            )
        return self.net.rollout_group(
            self.film(features, self._resolve_task(task_id)),
            padding_mask,
            num_samples=num_samples,
            generator=generator,
        )

    def score_boundaries(
        self, features, padding_mask, boundaries, task_id=None
    ):
        """Differentiably score realized actions under their full label prefix."""
        if not self.is_full_ar:
            raise RuntimeError("score_boundaries is only for transformer_ar")
        return self.net.score(
            self.film(features, self._resolve_task(task_id)),
            padding_mask,
            boundaries,
        )

    def conditioned_logits(self, logits, boundary):
        """Logits of a realized boundary sequence under the active policy.

        For the autoregressive policy, ``boundary[:, t-1]`` selects the learned
        transition bias for frame ``t``. The virtual label before frame zero is
        a boundary, matching the pooler's forced initial segment.
        """
        if self.sampling_strategy == "bernoulli":
            return logits
        previous = torch.ones_like(boundary)
        previous[:, 1:] = boundary[:, :-1].clamp(min=0)
        return logits + self.transition_bias[previous.long()]

    @staticmethod
    def _prefix_binary_functions(functions):
        """Inclusive prefix composition for binary-state functions.

        ``functions[b,t,s]`` is the output at step ``t`` given previous state
        ``s``. Hillis-Steele composition evaluates a whole Markov sample in
        ``O(log T)`` tensor operations rather than one GPU kernel per frame.
        """
        prefix = functions.long()
        step = 1
        while step < functions.size(1):
            outer = prefix[:, step:]
            inner = prefix[:, :-step]
            composed = torch.gather(outer, -1, inner)
            updated = prefix.clone()
            updated[:, step:] = composed
            prefix = updated
            step *= 2
        return prefix

    @staticmethod
    def _prefix_affine(a, b):
        """Inclusive prefix composition of scalar maps ``x -> a + b*x``."""
        prefix_a, prefix_b = a, b
        step = 1
        while step < a.size(1):
            outer_a, outer_b = prefix_a[:, step:], prefix_b[:, step:]
            inner_a, inner_b = prefix_a[:, :-step], prefix_b[:, :-step]
            composed_a = outer_a + outer_b * inner_a
            composed_b = outer_b * inner_b
            updated_a, updated_b = prefix_a.clone(), prefix_b.clone()
            updated_a[:, step:], updated_b[:, step:] = composed_a, composed_b
            prefix_a, prefix_b = updated_a, updated_b
            step *= 2
        return prefix_a, prefix_b

    def boundary_marginals(self, logits, padding_mask):
        """Differentiable marginal boundary probabilities under the policy."""
        if self.sampling_strategy == "bernoulli":
            return torch.sigmoid(logits.float()).masked_fill(padding_mask, 0.0)
        p0 = torch.sigmoid(logits.float() + self.transition_bias[0])
        p1 = torch.sigmoid(logits.float() + self.transition_bias[1])
        # m_t = p0_t + (p1_t-p0_t) m_{t-1}; compose all affine maps in
        # parallel, starting from the virtual boundary m_{-1}=1.
        a, b = self._prefix_affine(p0, p1 - p0)
        return (a + b).masked_fill(padding_mask, 0.0)

    def policy_entropy(self, logits, padding_mask):
        """Mean conditional entropy, exact for both supported policies."""
        if self.sampling_strategy == "bernoulli":
            return entropy_bonus(logits, padding_mask)
        p0 = torch.sigmoid(logits.float() + self.transition_bias[0])
        p1 = torch.sigmoid(logits.float() + self.transition_bias[1])
        marginals = self.boundary_marginals(logits, padding_mask)
        previous = torch.ones_like(marginals)
        previous[:, 1:] = marginals[:, :-1]

        def binary_entropy(prob):
            return -(
                prob * torch.log(prob.clamp(min=1e-8))
                + (1 - prob) * torch.log((1 - prob).clamp(min=1e-8))
            )

        entropy = (1 - previous) * binary_entropy(
            p0
        ) + previous * binary_entropy(p1)
        valid = ~padding_mask
        return entropy[valid].mean() if valid.any() else logits.sum() * 0.0

    def sample_boundary(
        self, logits, padding_mask, mode="sample", generator=None
    ):
        """Turn logits into a ``{1, 0, -1}`` boundary tensor.

        ``mode='argmax'`` -> ``1[logit > 0]`` (eval / decoder path).
        ``mode='sample'`` -> ``Bernoulli(sigmoid(logit))`` (RL exploration).
        Padding positions are set to ``-1``.
        """
        if mode not in ("argmax", "sample"):
            raise ValueError(
                f"Unknown sample mode '{mode}'. Use 'argmax' or 'sample'."
            )
        if self.sampling_strategy == "bernoulli":
            if mode == "argmax":
                b = (logits > 0).long()
            else:
                p = torch.sigmoid(logits.float())
                b = torch.bernoulli(p, generator=generator).long()
            return b.masked_fill(padding_mask, -1)

        conditional = torch.stack(
            [
                logits.float() + self.transition_bias[0],
                logits.float() + self.transition_bias[1],
            ],
            dim=-1,
        )
        if mode == "argmax":
            functions = conditional > 0
        else:
            uniforms = torch.rand(
                logits.shape,
                dtype=conditional.dtype,
                device=logits.device,
                generator=generator,
            )
            functions = uniforms.unsqueeze(-1) < torch.sigmoid(conditional)
        prefix = self._prefix_binary_functions(functions)
        # The virtual state before frame zero is one (a boundary).
        b = prefix[..., 1]
        return b.masked_fill(padding_mask, -1)

    def pool(self, features, padding_mask, boundary):
        """Pool frames within segments. Returns (seg_feats, seg_padding_mask)."""
        return self.pooler(features, padding_mask, boundary)

    def forward(self, features, padding_mask, task_id=None, deterministic=True):
        """Convenience full pass (used at eval): argmax boundaries -> pooled features.

        Returns ``(seg_features, seg_padding_mask, boundary, logits)``.
        """
        if self.is_full_ar:
            result = self.sample_boundaries(
                features,
                padding_mask,
                deterministic=deterministic,
                task_id=task_id,
            )
            seg_features, seg_pad = self.pool(
                features, padding_mask, result.boundaries
            )
            return (
                seg_features,
                seg_pad,
                result.boundaries,
                result.logits,
            )
        logits = self.boundary_logits(features, padding_mask, task_id)
        mode = "argmax" if deterministic else "sample"
        boundary = self.sample_boundary(logits, padding_mask, mode=mode)
        seg_features, seg_pad = self.pool(features, padding_mask, boundary)
        return seg_features, seg_pad, boundary, logits


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def boundary_bce_loss(logits, targets):
    """Cold-start supervised loss: BCE of per-frame logits vs ``{1,0,-1}`` targets.

    Padding (``targets == -1``) is ignored.
    """
    mask = targets != -1
    if mask.sum() == 0:
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(
        logits[mask].float(), targets[mask].float()
    )


def expected_kept_ratio(logits, padding_mask, probabilities=None):
    """Per-utterance expected kept-ratio ``rho_bar = E[#segments] / T``.

    Differentiable proxy for the (non-differentiable) sampled kept-ratio, and it
    must measure the SAME quantity as :func:`realized_kept_ratio`, because the
    two feed the two rate channels (the auxiliary ``rate_loss`` and the reward's
    ``sampled_rate_penalty``) against one shared band.

    Pooling always opens a segment at frame 0 and one more at every *later*
    frame carrying a boundary -- the frame-0 guard in
    ``segment_pooling.compute_segment_ids`` -- so
    ``E[#segments] = 1 + sum_{t>=1} p_t``.

    Fixed 2026-09-01: this previously returned ``sum_{t>=0} p_t / T``, which both
    omitted the implicit first segment and counted a frame-0 boundary that
    pooling ignores. The two rate channels were therefore aiming at bands offset
    by ``(1 - p_0)/T`` -- about 0.4 Hz on a 2.6 s clip, 0.08 Hz on a 12 s one.
    The autoregressive policy forces ``p_0 = 0``, so the correction there is a
    pure ``+1/T``.

    Returns ``(expected_segments (B,), rho_bar (B,), T (B,))``.
    """
    valid = (~padding_mask).float()
    if probabilities is None:
        probabilities = torch.sigmoid(logits.float())
    p = probabilities.float() * valid
    if p.shape[1] > 0:
        # Frame 0 cannot open a second segment; it already has the implicit one.
        p = torch.cat([torch.zeros_like(p[:, :1]), p[:, 1:]], dim=1)
    T = valid.sum(dim=1).clamp(min=1.0)
    expected_segments = p.sum(dim=1) + 1.0
    return expected_segments, expected_segments / T, T


def realized_kept_ratio(boundary, padding_mask):
    """Actual pooled-token ratio, counting the implicit segment beginning at frame 0."""
    valid = ~padding_mask
    n_frames = valid.sum(dim=1).float().clamp(min=1.0)
    # The implicit first segment is counted exactly once here; each LATER 1 starts
    # one additional segment. A boundary at frame 0 is ignored, matching the
    # frame-0 guard in ``segment_pooling.compute_segment_ids`` -- without this it
    # would be counted twice. The autoregressive policy fixes boundary[:, 0] to
    # zero, so this only changes the (non-production) bernoulli path.
    counted = (boundary == 1) & valid
    if counted.shape[1] > 0:
        counted = torch.cat(
            [torch.zeros_like(counted[:, :1]), counted[:, 1:]], dim=1
        )
    n_segments = counted.sum(dim=1).float() + 1.0
    return n_segments / n_frames


def sampled_rate_penalty(
    boundary,
    padding_mask,
    rho_star,
    lambda_cap,
    lambda_press,
    rho_floor=0.0,
    lambda_floor=0.0,
    mode="captax",
    rho_lo=0.0,
    rho_hi=1.0,
):
    """Per-utterance rate penalty for a realized autoregressive rollout."""
    rho = realized_kept_ratio(boundary, padding_mask)
    if mode == "pin":
        penalty = lambda_cap * (rho - rho_star) ** 2
    elif mode == "band":
        below = (rho_lo - rho).clamp(min=0.0)
        above = (rho - rho_hi).clamp(min=0.0)
        penalty = lambda_cap * (below**2 + above**2)
    elif mode == "captax":
        over = (rho - rho_star).clamp(min=0.0)
        penalty = lambda_cap * over**2 + lambda_press * rho
        if lambda_floor > 0.0:
            penalty = penalty + lambda_floor * (
                (rho_floor - rho).clamp(min=0.0) ** 2
            )
    else:
        raise ValueError(f"Unknown rate mode {mode!r}")
    return penalty, rho


def rate_loss(
    logits,
    padding_mask,
    rho_star,
    lambda_cap,
    lambda_press,
    rho_floor=0.0,
    lambda_floor=0.0,
    mode="captax",
    rho_lo=0.0,
    rho_hi=1.0,
    probabilities=None,
):
    """Rate regularizer on the *expected* kept-ratio ``rho_bar = E[count]/T``.

    Always in **ratio space** (O(1), length-independent; a count-space form scales O(T^2)
    and swamps the other losses). ``mode``:

    * ``"pin"`` (**Option A**): ``lambda_cap * (rho_bar - rho_star)^2`` — firm two-sided pin;
      restoring force from both sides, can't collapse. Gives up "use fewer when doing well".
    * ``"captax"`` (**Option B**): one-sided budget cap ``lambda_cap * relu(rho_bar-rho_star)^2``
      + token-tax ``lambda_press * rho_bar`` + soft lower floor
      ``lambda_floor * relu(rho_floor - rho_bar)^2``.
    * ``"band"`` (**Option C**): free inside ``[rho_lo, rho_hi]`` (no penalty at all), only
      out-of-band is penalized: ``lambda_cap * (relu(rho_lo-rho_bar)^2 + relu(rho_bar-rho_hi)^2)``.
      The quality reward picks the rate within the band; can't collapse below rho_lo.

    Returns ``(loss, rho_bar (B,))``.
    """
    _, rho_bar, _ = expected_kept_ratio(
        logits, padding_mask, probabilities=probabilities
    )
    if mode == "pin":
        return lambda_cap * ((rho_bar - rho_star) ** 2).mean(), rho_bar.detach()
    if mode == "band":
        below = (rho_lo - rho_bar).clamp(min=0.0)
        above = (rho_bar - rho_hi).clamp(min=0.0)
        return lambda_cap * (below**2 + above**2).mean(), rho_bar.detach()
    over = (rho_bar - rho_star).clamp(min=0.0)
    loss = lambda_cap * (over**2).mean() + lambda_press * rho_bar.mean()
    if lambda_floor > 0.0:
        under = (rho_floor - rho_bar).clamp(min=0.0)
        loss = loss + lambda_floor * (under**2).mean()
    return loss, rho_bar.detach()


def entropy_bonus(logits, padding_mask):
    """Mean per-frame Bernoulli entropy over valid frames (to be *added* as a bonus)."""
    valid = ~padding_mask
    p = torch.sigmoid(logits.float())
    ent = -(
        p * torch.log(p.clamp(min=1e-8))
        + (1 - p) * torch.log((1 - p).clamp(min=1e-8))
    )
    ent = ent[valid]
    if ent.numel() == 0:
        return logits.sum() * 0.0
    return ent.mean()


def group_advantage(rewards, normalize_std=True, eps=1e-6):
    """GRPO/RLOO group-relative advantage from per-utterance sampled rewards.

    Arguments
    ---------
    rewards : torch.Tensor
        ``(B, K)`` rewards for K samples of each of B utterances.
    normalize_std : bool
        If True (GRPO), divide by the per-utterance std; else (RLOO) mean-only.

    Returns
    -------
    advantage : torch.Tensor
        ``(B, K)`` group-relative advantages (difficulty cancels within a row).
    """
    mean = rewards.mean(dim=1, keepdim=True)
    adv = rewards - mean
    if normalize_std:
        adv = adv / (rewards.std(dim=1, keepdim=True) + eps)
    return adv


def grpo_pg_loss(logits, boundary, advantage, valid_mask=None):
    """Policy-gradient loss for ONE sampled segmentation (gamma=1, ``.mean()`` reduce).

    ``loss = mean_{valid (b,t)} [ advantage_b * (-log pi(boundary_bt)) ]``

    Minimizing this ascends ``E[advantage * log pi]`` (REINFORCE). Every valid frame
    of an utterance shares that utterance's (detached) advantage — the bandit /
    ``gamma = 1`` credit assignment. ``.mean()`` (not ``.sum()``) keeps the gradient
    scale independent of the dynamic-batch token count.

    Arguments
    ---------
    logits : torch.Tensor
        ``(B, T)`` boundary logits (require grad).
    boundary : torch.Tensor
        ``(B, T)`` sampled boundary ``{1, 0, -1}`` (pad == -1).
    advantage : torch.Tensor
        ``(B,)`` detached per-utterance advantage.
    """
    valid = boundary != -1
    if valid_mask is not None:
        valid = valid & valid_mask
    b = boundary.clamp(min=0).float()
    neg_logp = F.binary_cross_entropy_with_logits(
        logits.float(), b, reduction="none"
    )  # (B, T) == -log pi(b_t)
    per_frame = advantage.detach().unsqueeze(1) * neg_logp  # (B, T)
    per_frame = per_frame[valid]
    if per_frame.numel() == 0:
        return logits.sum() * 0.0
    return per_frame.mean()


def grpo_clipped_pg_loss(logits, boundary, advantage, old_logp, clip_eps=0.2):
    """PPO/GRPO clipped-surrogate policy-gradient for ONE sampled segmentation.

    The **off-policy** generalization of :func:`grpo_pg_loss`: the boundary was sampled
    from an *old* policy ``pi_old`` (during the rollout), and this loss can be applied for
    several gradient steps on the *current* ``pi_theta`` using the per-frame importance
    ratio ``r_t = pi_theta(b_t) / pi_old(b_t)`` with the PPO clip::

        loss = -mean_{valid (b,t)} min( r_t * A_b ,  clip(r_t, 1-eps, 1+eps) * A_b )

    This is exactly ms-swift / TRL GRPO's token-level objective (here a *frame* is a
    "token"), which is what makes ``num_iterations > 1`` (reuse one generation for several
    updates) correct. When it is applied on-policy (``pi_theta == pi_old``, i.e. the first
    update after a fresh rollout) ``r_t == 1`` and the gradient equals :func:`grpo_pg_loss`.

    Arguments
    ---------
    logits : torch.Tensor
        ``(B, T)`` current-policy boundary logits (require grad).
    boundary : torch.Tensor
        ``(B, T)`` sampled boundary ``{1, 0, -1}`` (pad == -1).
    advantage : torch.Tensor
        ``(B,)`` detached per-utterance group-relative advantage.
    old_logp : torch.Tensor
        ``(B, T)`` detached per-frame log-prob of ``boundary`` under ``pi_old``.
    clip_eps : float
        PPO clip range ``eps``.
    """
    valid = boundary != -1
    b = boundary.clamp(min=0).float()
    logp = -F.binary_cross_entropy_with_logits(
        logits.float(), b, reduction="none"
    )  # (B, T) == log pi_theta(b_t)
    ratio = torch.exp(logp - old_logp.detach())  # (B, T)
    adv = advantage.detach().unsqueeze(1)  # (B, 1)
    surrogate = torch.min(
        ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    )
    per_frame = (-surrogate)[valid]
    if per_frame.numel() == 0:
        return logits.sum() * 0.0
    return per_frame.mean()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def boundary_prf(pred, target):
    """Per-batch boundary precision / recall / F1 for class 1. Masks pad on both."""
    p = pred.reshape(-1)
    t = target.reshape(-1)
    mask = (t != -1) & (p != -1)
    p = p[mask]
    t = t[mask]
    tp = ((p == 1) & (t == 1)).sum().float()
    fp = ((p == 1) & (t == 0)).sum().float()
    fn = ((p == 0) & (t == 1)).sum().float()
    precision = tp / (tp + fp) if (tp + fp) > 0 else torch.zeros(())
    recall = tp / (tp + fn) if (tp + fn) > 0 else torch.zeros(())
    denom = precision + recall
    f1 = 2 * precision * recall / denom if denom > 0 else torch.zeros(())
    return float(precision), float(recall), float(f1)


# ---------------------------------------------------------------------------
# Self-tests (CPU)
# ---------------------------------------------------------------------------
