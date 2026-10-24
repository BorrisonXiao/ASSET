"""Batch-invariant helpers for multimodal LLM decoding.

The SpeechLLM recipes keep the text prompt at the end of the batch-wide audio
block.  Without explicit position IDs, a short utterance therefore receives
different prompt positions depending on which longer utterances share its
batch.  The stock SpeechBrain searcher also derives one generation limit from
the padded prefix length.  Both effects can change a hypothesis when only the
batch composition changes.

This module makes valid prefixes contiguous for free-running search, assigns
logical positions by counting valid tokens, and uses a duration-derived limit
for each utterance.  The limit is intentionally independent of the number of
pooled audio tokens, so comparisons across segmentation rates use the same
decoding policy.
"""

from __future__ import annotations

import torch
from torch.distributions import Categorical

from speechbrain.decoders.seq2seq import S2SHuggingFaceLLMGreedySearcher
from speechbrain.utils.data_utils import undo_padding


def compact_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return zero-based positions that count only valid tokens.

    Padding positions receive position zero and are masked from attention.  The
    valid positions are consequently identical whether padding is interior,
    left, or absent.
    """

    mask = attention_mask.bool()
    positions = mask.long().cumsum(dim=-1) - 1
    return positions.masked_fill(~mask, 0)


def left_pack_valid_prefix(
    enc_states: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Remove interior padding and left-pad every valid prefix.

    Left padding makes the final prompt token occupy the last prefix position
    for every row, which is the layout expected by batched autoregressive
    generation from ``inputs_embeds``.
    """

    mask = attention_mask.bool()
    if enc_states.shape[:2] != mask.shape:
        raise ValueError(
            "enc_states and attention_mask must share batch/time dimensions: "
            f"got {tuple(enc_states.shape)} and {tuple(mask.shape)}"
        )
    lengths = mask.sum(dim=1)
    if torch.any(lengths <= 0):
        raise ValueError("Every decoding prefix must contain at least one token")
    max_length = int(lengths.max().item())
    packed = enc_states.new_zeros(
        enc_states.size(0), max_length, enc_states.size(2)
    )
    packed_mask = torch.zeros(
        enc_states.size(0),
        max_length,
        dtype=torch.bool,
        device=enc_states.device,
    )
    for row, length in enumerate(lengths.tolist()):
        valid = enc_states[row, mask[row]]
        packed[row, max_length - length :] = valid
        packed_mask[row, max_length - length :] = True
    return packed, packed_mask


def duration_decode_limits(
    durations: torch.Tensor,
    tokens_per_second: float,
    token_margin: int,
    max_tokens: int,
    min_tokens: int = 8,
) -> torch.Tensor:
    """Compute one absolute generation limit per utterance."""

    if tokens_per_second <= 0:
        raise ValueError("tokens_per_second must be positive")
    if token_margin < 0 or min_tokens < 1 or max_tokens < min_tokens:
        raise ValueError("Invalid duration-based decoding limits")
    limits = torch.ceil(durations.float() * float(tokens_per_second)).long()
    limits = limits + int(token_margin)
    return limits.clamp(min=int(min_tokens), max=int(max_tokens))


class PerUtteranceEosLogitsProcessor:
    """Force EOS independently when each row reaches its duration limit.

    This small stateful processor is intended for one Hugging Face ``generate``
    call.  It supports the replicated rollout path, where a single scalar
    ``max_new_tokens`` would otherwise let short utterances inherit the longest
    row's budget.
    """

    def __init__(self, decode_limits: torch.Tensor, eos_index: int):
        self.decode_limits = decode_limits.long()
        self.eos_index = int(eos_index)
        self.step = 0

    def __call__(self, input_ids, scores):
        force_eos = (self.step + 1) >= self.decode_limits.to(scores.device)
        if force_eos.any():
            scores[force_eos] = -torch.inf
            scores[force_eos, self.eos_index] = 0.0
        self.step += 1
        return scores


class BatchInvariantLLMGreedySearcher(S2SHuggingFaceLLMGreedySearcher):
    """Greedy LLM search with batch-invariant positions and length limits."""

    def __init__(
        self,
        llm_model,
        temperature=0.0,
        max_decode_tokens_per_second=12.0,
        max_decode_token_margin=16,
        max_decode_tokens=512,
        min_decode_tokens=8,
        **kwargs,
    ):
        super().__init__(llm_model=llm_model, temperature=temperature, **kwargs)
        self.max_decode_tokens_per_second = float(
            max_decode_tokens_per_second
        )
        self.max_decode_token_margin = int(max_decode_token_margin)
        self.max_decode_tokens = int(max_decode_tokens)
        self.min_decode_tokens = int(min_decode_tokens)

    def forward_step(
        self, inp_tokens, memory, enc_states, enc_lens, attention_mask
    ):
        """Run one full-prefix step with explicit compact position IDs."""

        if inp_tokens is not None:
            memory = self._update_mem_embeddings(
                inp_tokens.unsqueeze(-1), memory
            )
        multimodal_embds = (
            enc_states
            if memory is None
            else torch.cat([enc_states, memory], dim=1)
        )
        logits = self.llm_model(
            inputs_embeds=multimodal_embds,
            attention_mask=attention_mask,
            position_ids=compact_position_ids(attention_mask),
        ).logits
        return logits[:, -1, :], memory, None

    @torch.no_grad()
    def forward(
        self,
        enc_states,
        wav_len,
        attention_mask=None,
        decode_durations=None,
    ):
        """Decode with per-row limits based on original waveform duration.

        ``wav_len`` remains in the signature for SpeechBrain compatibility but
        does not set the generation budget.  Callers must provide
        ``decode_durations`` in seconds.
        """

        if attention_mask is None:
            raise ValueError("Batch-invariant LLM decoding requires an attention mask")
        if decode_durations is None:
            raise ValueError(
                "Batch-invariant LLM decoding requires per-utterance waveform "
                "durations in seconds"
            )
        enc_states, attention_mask = left_pack_valid_prefix(
            enc_states, attention_mask
        )
        device = enc_states.device
        batch_size = enc_states.size(0)
        durations = torch.as_tensor(
            decode_durations, device=device, dtype=torch.float32
        ).reshape(-1)
        if durations.numel() != batch_size:
            raise ValueError(
                f"Expected {batch_size} decode durations, got {durations.numel()}"
            )
        decode_limits = duration_decode_limits(
            durations,
            self.max_decode_tokens_per_second,
            self.max_decode_token_margin,
            self.max_decode_tokens,
            self.min_decode_tokens,
        )
        max_decode_steps = int(decode_limits.max().item())
        enc_lens = attention_mask.sum(dim=1).int()
        memory = self.reset_mem(batch_size, device=device)
        inp_tokens = (
            enc_states.new_full(
                (batch_size,), int(self.bos_index), dtype=torch.long
            )
            if self.bos_index is not None
            else None
        )

        log_probs_lst = []
        has_ended = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for step in range(max_decode_steps):
            if inp_tokens is not None:
                attention_mask = torch.cat(
                    [
                        attention_mask,
                        torch.ones(
                            batch_size, 1, dtype=torch.bool, device=device
                        ),
                    ],
                    dim=1,
                )
                attention_mask[has_ended, -1] = False

            logits, memory, _ = self.forward_step(
                inp_tokens, memory, enc_states, enc_lens, attention_mask
            )
            if self.temperature == 0:
                inp_tokens = logits.argmax(dim=-1)
            else:
                inp_tokens = Categorical(
                    logits=logits / self.temperature
                ).sample()
            log_probs = torch.nn.functional.log_softmax(
                logits.float(), dim=-1
            )
            log_probs_lst.append(log_probs)

            reached_limit = (step + 1) >= decode_limits
            has_ended = has_ended | (inp_tokens == self.eos_index) | reached_limit
            log_probs[has_ended] = -torch.inf
            inp_tokens[has_ended] = self.eos_index
            if has_ended.all() or self._check_end_condition(memory):
                break

        if not log_probs_lst:
            raise RuntimeError("Decoding produced no steps")
        log_probs = torch.stack(log_probs_lst, dim=1)
        scores, predictions = log_probs.max(dim=-1)
        ended_mask = scores == -torch.inf
        scores[ended_mask] = 0
        predictions[ended_mask] = self.eos_index
        top_hyps, top_lengths, top_scores, top_log_probs = (
            self._get_top_prediction(predictions, scores, log_probs)
        )
        hyps = undo_padding(top_hyps[:, 0], top_lengths)
        return hyps, top_lengths, top_scores, top_log_probs
