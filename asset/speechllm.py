"""Shared SpeechLLM decoder, data loading and batching support for ASSET.

Based on the SpeechLLM recipe by Adel Moumen (2025).
"""

import json
import os
import time
from pathlib import Path

import torch
from batch_invariant_decode import compact_position_ids

import speechbrain as sb
from segment_pooling import (
    apply_blank_mode,
    fixed_rate_boundary_targets,
    lengths_to_padding_mask,
    mean_pool_segments,
    padding_mask_to_lengths,
    segment_separator_mask,
)
from speechbrain.integrations.hdf5.cached_item import CachedHDF5DynamicItem
from speechbrain.utils.distributed import if_main_process, run_on_main
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


def loader_runtime_options(hparams):
    """Return safe worker/prefetch options for PyTorch DataLoader."""
    workers = int(hparams.get("num_workers", 0))
    options = {
        "num_workers": workers,
        "pin_memory": bool(hparams.get("pin_memory", False)),
    }
    if workers > 0:
        options["persistent_workers"] = bool(
            hparams.get("persistent_workers", False)
        )
        options["prefetch_factor"] = int(hparams.get("prefetch_factor", 2))
    return options


def _levenshtein(a, b):
    """Edit distance between two token sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left != right),
                )
            )
        previous = current
    return previous[-1]


def get_multimodal_attention_mask(wav, wav_lens, txt, txt_lens, device):
    """Create attention mask for multimodal sequence.

    Arguments
    ---------
    wav : torch.Tensor
        Audio features tensor of shape (batch_size, L_audio, ...)
    wav_lens : torch.Tensor
        Relative lengths of audio features, shape (batch_size,)
    txt : torch.Tensor
        Text embeddings tensor of shape (batch_size, txt_len, ...)
        This is txt_embds which includes: [start_of_audio, end_of_audio, prompt, bos, text]
    txt_lens : torch.Tensor
        Relative lengths of text tokens, shape (batch_size,)
    device : torch.device
        Device to create the mask on

    Returns
    -------
    attention_mask : torch.Tensor
        Boolean attention mask of shape (batch_size, L_audio + txt_len).

        Important
        ---------
        The actual multimodal embedding order in this recipe is:

            [start_of_audio] + [audio_feats] + [end_of_audio + prompt + bos + text]

        i.e., the first text token (<|start_of_audio|>) is placed *before* audio.
        Therefore, we must build the mask with the same layout:
            position 0              -> <|start_of_audio|>
            positions [1 : 1+L_audio] -> audio feats
            positions [1+L_audio : ]  -> remaining text tokens (txt[:, 1:])
    """
    batch_size = wav.size(0)
    wav_len = wav.size(1)
    txt_len = txt.size(1)
    # Total length matches multimodal_embds: 1 (start token) + L_audio + (txt_len - 1)
    total_len = wav_len + txt_len
    attention_mask = torch.zeros(
        batch_size, total_len, dtype=torch.bool, device=device
    )
    for i in range(batch_size):
        # Match SpeechBrain convention (see S2SGreedySearcher): round relative lengths.
        actual_wav_len = int(torch.round(wav_lens[i] * wav_len).item())
        actual_txt_len = int(torch.round(txt_lens[i] * txt_len).item())

        # (1) start_of_audio token (always valid)
        attention_mask[i, 0] = True

        # (2) audio features
        attention_mask[i, 1 : 1 + actual_wav_len] = True

        # (3) remaining text tokens (exclude the start token already handled above)
        remaining_txt = max(actual_txt_len - 1, 0)
        attention_mask[i, 1 + wav_len : 1 + wav_len + remaining_txt] = True
    return attention_mask


# Define training procedure
class ASR(sb.core.Brain):
    def _decode_durations(self, batch):
        """Original waveform durations used for batch-independent decode caps."""

        wavs, wav_lens = batch.sig
        sample_rate = float(getattr(self.hparams, "decode_sample_rate", 16000))
        return wav_lens.float() * float(wavs.size(1)) / sample_rate

    def get_fixed_boundaries(self, boundary_source, pad_mask, batch):
        """Boundary decisions ``(B, T)`` in ``{1, 0, -1}`` from a non-learned source.

        Arguments
        ---------
        boundary_source : str
            "fixed_rate" (a boundary every `fixed_rate_k` frames) or
            "alignment" (per-frame labels from `batch.boundary_target`).
        pad_mask : torch.Tensor
            (B, T) boolean, True at padding.
        batch : PaddedBatch
            Current batch (used for `boundary_target` in alignment mode).

        Returns
        -------
        boundary : torch.Tensor
            (B, T) long with values {1, 0, -1} (pad == -1).
        """
        if boundary_source == "fixed_rate":
            return fixed_rate_boundary_targets(
                pad_mask, int(self.hparams.fixed_rate_k)
            )
        if boundary_source == "alignment":
            boundary, _ = batch.boundary_target
            T = pad_mask.size(1)
            if boundary.size(1) != T:
                raise ValueError(
                    f"boundary_target has {boundary.size(1)} frames but the "
                    f"encoder produced {T}. Alignment labels must be at the "
                    "encoder frame rate (one {0,1} label per encoder frame)."
                )
            return boundary.long().masked_fill(pad_mask, -1)
        raise ValueError(f"Unknown boundary_source '{boundary_source}'.")

    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities.

        The forward pass processes either cached SSL features or raw audio waveforms,
        projects them to the LLM embedding space, and concatenates with text embeddings
        to form a multimodal sequence.

        Sequence structure:
        [start_of_audio] + [audio_features] + [end_of_audio + prompt + bos + text]

        Arguments
        ---------
        batch : PaddedBatch
            Batch containing audio/features, tokens, and metadata
        stage : sb.Stage
            Current stage (TRAIN, VALID, or TEST)

        Returns
        -------
        logits : torch.Tensor
            Model output logits of shape (batch_size, seq_len, vocab_size)
        hyps : list or None
            Decoded hypotheses (only during validation/test, None during training)
        """
        benchmark = stage == sb.Stage.TEST and bool(
            getattr(self.hparams, "inference_benchmark_file", None)
        )
        benchmark_events = None
        benchmark_wall_start = None
        if benchmark:
            if torch.device(self.device).type == "cuda":
                torch.cuda.synchronize(self.device)
                benchmark_events = [
                    torch.cuda.Event(enable_timing=True) for _ in range(4)
                ]
                benchmark_events[0].record()
            benchmark_wall_start = time.perf_counter()

        batch = batch.to(self.device, non_blocking=True)
        tokens_bos, tokens_bos_lens = batch.tokens_bos
        prompt_len = batch.prompt_len

        use_feats = bool(getattr(self.hparams, "use_feats", False))
        if use_feats:
            if getattr(batch, "feats", None) is None:
                raise ValueError(
                    "`use_feats=True` but the batch does not provide `feats`. "
                    "Check `feats_cache_dir` and the data pipeline."
                )
            audio_feats, audio_feats_lens = batch.feats
        else:
            wavs, wav_lens = batch.sig
            wavs = self.hparams.normalize(wavs, wav_lens)
            audio_feats = self.modules.ssl(wavs, wav_lens)
            audio_feats_lens = wav_lens
        if benchmark_events is not None:
            benchmark_events[1].record()
        benchmark_encoder_end = time.perf_counter()
        boundary_source = getattr(self.hparams, "boundary_source", "none")
        if boundary_source and boundary_source != "none":
            # Fixed-segmentation baseline: mean-pool frames between boundaries
            # instead of the fixed feat_downsampler. Sequence lengths change
            # per utterance, so recompute relative lengths from the pooled mask.
            pad_mask = lengths_to_padding_mask(
                audio_feats_lens, audio_feats.size(1)
            )
            boundary = self.get_fixed_boundaries(
                boundary_source, pad_mask, batch
            )
            audio_down_feats, seg_pad_mask = mean_pool_segments(
                audio_feats, pad_mask, boundary
            )
            # Optional blank/separator-token ablation: drop or replace the CTC
            # separator/"blank" tokens before the decoder (no-op when 'keep').
            blank_mode = getattr(self.hparams, "blank_mode", "keep") or "keep"
            if blank_mode != "keep":
                sep_frame, _ = batch.separator_target
                if sep_frame.size(1) != boundary.size(1):
                    raise ValueError(
                        f"separator_target has {sep_frame.size(1)} frames but "
                        f"boundary has {boundary.size(1)}; they must align."
                    )
                sep_seg = segment_separator_mask(
                    sep_frame.long(), boundary, pad_mask
                )
                audio_down_feats, seg_pad_mask = apply_blank_mode(
                    audio_down_feats,
                    seg_pad_mask,
                    sep_seg,
                    blank_mode,
                    const_vec=getattr(self, "blank_const", None),
                )
            audio_down_lens = padding_mask_to_lengths(seg_pad_mask)
        else:
            # R^L*D -> R^(L/R)*(D*R)
            audio_down_feats = self.modules.feat_downsampler(audio_feats)
            audio_down_lens = audio_feats_lens
        if benchmark_events is not None:
            benchmark_events[2].record()
        benchmark_pooling_end = time.perf_counter()
        # R^D' -> R^llm_emb_size
        projected_audio_feats = self.modules.proj(audio_down_feats)
        txt_embds = self.txt_embedding(tokens_bos)
        multimodal_embds = torch.cat(
            [
                txt_embds[:, 0].unsqueeze(1),  # B, D -> B, 1, D
                projected_audio_feats,
                txt_embds[:, 1:],
            ],
            dim=1,
        )
        # attention_mask should be all the true audio features + all the true text features
        attention_mask = get_multimodal_attention_mask(
            projected_audio_feats,
            audio_down_lens,
            txt_embds,
            tokens_bos_lens,
            self.device,
        )
        logits = self.modules.llm(
            inputs_embeds=multimodal_embds,
            attention_mask=attention_mask,
            position_ids=compact_position_ids(attention_mask),
        ).logits

        hyps = None
        if stage != sb.Stage.TRAIN:
            audio_and_prompt_len = projected_audio_feats.shape[1] + int(
                prompt_len[0].item()
            )
            inputs_embeds = multimodal_embds[:, :audio_and_prompt_len]
            hyps = self.modules.searcher(
                inputs_embeds,
                audio_down_lens,
                attention_mask[:, :audio_and_prompt_len],
                decode_durations=self._decode_durations(batch),
            )
        if benchmark:
            benchmark_decoder_end = time.perf_counter()
            timing = {
                "wall_seconds": benchmark_decoder_end - benchmark_wall_start,
                "encoder_wall_seconds": (
                    benchmark_encoder_end - benchmark_wall_start
                ),
                "pooling_wall_seconds": (
                    benchmark_pooling_end - benchmark_encoder_end
                ),
                "decoder_wall_seconds": (
                    benchmark_decoder_end - benchmark_pooling_end
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
                        "pooling_gpu_seconds": benchmark_events[1].elapsed_time(
                            benchmark_events[2]
                        )
                        / 1000.0,
                        "decoder_gpu_seconds": benchmark_events[2].elapsed_time(
                            benchmark_events[3]
                        )
                        / 1000.0,
                    }
                )
                timing["wall_seconds"] = time.perf_counter() - benchmark_wall_start
            audio_token_counts = torch.round(
                audio_down_lens.detach() * projected_audio_feats.size(1)
            ).long()
            self._current_inference_benchmark = {
                "timing": timing,
                "audio_token_counts": audio_token_counts.cpu().tolist(),
            }
        return logits, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the cross-entropy loss given predictions and targets.

        The loss is computed only on text tokens, with audio feature positions
        masked out using ignore_index. During validation/test, also computes
        CER and WER metrics.

        Arguments
        ---------
        predictions : tuple
            (logits, hyps) from compute_forward
        batch : PaddedBatch
            Batch containing target tokens and metadata
        stage : sb.Stage
            Current stage (TRAIN, VALID, or TEST)

        Returns
        -------
        loss : torch.Tensor
            Cross-entropy loss value
        """
        logits, hyps = predictions
        tokens_eos, _ = batch.tokens_eos
        ids = batch.id

        num_audio_feats = logits.shape[1] - tokens_eos.shape[1]
        # We prepend `ignore_index` to the tokens_eos to ignore them in the loss.
        # This corresponds to the audio features.
        target_tokens = torch.cat(
            [
                torch.full(
                    (tokens_eos.shape[0], num_audio_feats),
                    self.hparams.ignore_index,
                    device=self.device,
                ),
                tokens_eos,
            ],
            dim=1,
        ).long()
        # compute the cross entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            target_tokens.view(-1),
            ignore_index=self.hparams.ignore_index,
        )
        if stage != sb.Stage.TRAIN:
            # replace ignore_index with pad token
            target_tokens = target_tokens.masked_fill(
                target_tokens == self.hparams.ignore_index,
                self.tokenizer.pad_token_id,
            )
            preds = self.tokenizer.batch_decode(
                hyps[0], skip_special_tokens=True
            )
            preds_words = [pred.split(" ") for pred in preds]
            targets = self.tokenizer.batch_decode(
                target_tokens, skip_special_tokens=True
            )
            targets_words = [target.split(" ") for target in targets]
            self.cer_metric.append(ids, preds_words, targets_words)
            self.wer_metric.append(ids, preds_words, targets_words)
            if stage == sb.Stage.TEST:
                self._record_inference_benchmark(batch, hyps)
        return loss

    @staticmethod
    def _percentile(values, q):
        if not values:
            return None
        ordered = sorted(float(value) for value in values)
        return ordered[round((len(ordered) - 1) * float(q))]

    def _record_inference_benchmark(self, batch, hyps):
        """Collect the same per-batch and per-utterance fields as learned runs."""
        state = getattr(self, "_inference_benchmark", None)
        current = getattr(self, "_current_inference_benchmark", None)
        if state is None or current is None:
            return
        wavs, wav_lens = batch.sig
        durations = (
            wav_lens.detach().float().cpu() * float(wavs.size(1)) / 16000.0
        ).tolist()
        predictions = self.tokenizer.batch_decode(
            hyps[0], skip_special_tokens=True
        )
        references = list(batch.wrd)

        state["batch_count"] += 1
        warmup = state["batch_count"] <= int(
            getattr(self.hparams, "inference_warmup_batches", 5)
        )
        if not warmup:
            state["measured_batch_count"] += 1
            state["measured_utterances"] += len(durations)
            state["measured_audio_seconds"] += sum(durations)
            for name, value in current["timing"].items():
                state.setdefault(name, []).append(float(value))

        for utterance_id, duration, count, hyp, ref in zip(
            list(batch.id),
            durations,
            current["audio_token_counts"],
            predictions,
            references,
        ):
            ref_words = str(ref).split()
            hyp_words = str(hyp).split()
            errors = _levenshtein(hyp_words, ref_words)
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
                    "segments": int(count),
                    "token_hz": float(count) / max(float(duration), 1.0e-9),
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
        state = getattr(self, "_inference_benchmark", None)
        output = getattr(self.hparams, "inference_benchmark_file", None)
        if state is None or not output or not if_main_process():
            return
        utterances = state["utterances"]
        token_hz = [row["token_hz"] for row in utterances]
        total_audio = sum(row["duration_seconds"] for row in utterances)
        total_tokens = sum(row["segments"] for row in utterances)
        buckets = {}
        for bucket in ("<5s", "5-10s", "10-20s", ">=20s"):
            rows = [row for row in utterances if row["duration_bucket"] == bucket]
            bucket_audio = sum(row["duration_seconds"] for row in rows)
            buckets[bucket] = {
                "utterances": len(rows),
                "audio_seconds": bucket_audio,
                "token_hz": (
                    sum(row["segments"] for row in rows) / bucket_audio
                    if bucket_audio
                    else None
                ),
                "mean_utterance_wer": (
                    sum(row["wer"] for row in rows) / len(rows) if rows else None
                ),
            }
        wall = state.get("wall_seconds", [])
        measured_wall = sum(wall)
        measured_audio = state["measured_audio_seconds"]
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
            "boundary_source": str(self.hparams.boundary_source),
            "fixed_rate_k": (
                int(self.hparams.fixed_rate_k)
                if self.hparams.boundary_source == "fixed_rate"
                else None
            ),
            "oracle_boundaries_precomputed": self.hparams.boundary_source
            == "alignment",
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
                    "pooling_gpu_seconds",
                    "decoder_gpu_seconds",
                )
            },
            "token_frequency_hz": {
                "global": total_tokens / max(total_audio, 1.0e-9),
                "utterance_mean": sum(token_hz) / max(len(token_hz), 1),
                "utterance_p05": self._percentile(token_hz, 0.05),
                "utterance_median": self._percentile(token_hz, 0.5),
                "utterance_p95": self._percentile(token_hz, 0.95),
            },
            "duration_buckets": buckets,
            "WER": float(stage_stats["WER"]),
            "CER": float(stage_stats["CER"]),
            "hardware": (
                torch.cuda.get_device_name(self.device)
                if torch.device(self.device).type == "cuda"
                else "CPU"
            ),
            "precision": str(getattr(self.hparams, "eval_precision", "unknown")),
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
            "Inference benchmark: split=%s source=%s batch=%d RTF=%.4f Hz=%.2f",
            result["split"],
            result["boundary_source"],
            result["batch_size"],
            result["forward_rtf"],
            result["token_frequency_hz"]["global"],
        )

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch.

        Initializes metrics for validation and test stages.

        Arguments
        ---------
        stage : sb.Stage
            Current stage (TRAIN, VALID, or TEST)
        epoch : int
            Current epoch number
        """
        self._stage_started_at = time.monotonic()
        if torch.device(self.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        if stage != sb.Stage.TRAIN:
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
        """Gets called at the end of an epoch.

        Logs statistics, updates learning rate, and saves checkpoints.

        Arguments
        ---------
        stage : sb.Stage
            Current stage (TRAIN, VALID, or TEST)
        stage_loss : float
            Average loss for this stage
        epoch : int
            Current epoch number
        """
        stage_seconds = time.monotonic() - getattr(
            self, "_stage_started_at", time.monotonic()
        )
        memory_stats = {}
        if torch.device(self.device).type == "cuda":
            gib = 1024**3
            peak_allocated = torch.cuda.max_memory_allocated(self.device) / gib
            peak_reserved = torch.cuda.max_memory_reserved(self.device) / gib
            capacity = torch.cuda.get_device_properties(
                self.device
            ).total_memory / gib
            memory_stats = {
                "gpu_peak_allocated_gb": peak_allocated,
                "gpu_peak_reserved_gb": peak_reserved,
                "gpu_capacity_gb": capacity,
                "gpu_reserved_headroom_gb": capacity - peak_reserved,
            }
        timing_file = getattr(self.hparams, "stage_timing_file", None)
        if timing_file and if_main_process():
            timing_dir = os.path.dirname(timing_file)
            if timing_dir:
                os.makedirs(timing_dir, exist_ok=True)
            with open(timing_file, "a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {
                            "split": str(
                                getattr(self.hparams, "evaluation_split", "")
                            ),
                            "stage": stage.name,
                            "epoch": epoch,
                            "seconds": stage_seconds,
                            **memory_stats,
                        }
                    )
                    + "\n"
                )

        # Compute/store important stats
        stage_stats = {"loss": stage_loss, **memory_stats}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.scheduler(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # Optional: SSL fine-tuning LR scheduling (only when SSL is unfrozen).
            if hasattr(self, "ssl_optimizer") and hasattr(
                self.hparams, "lr_annealing_ssl"
            ):
                old_lr_ssl, new_lr_ssl = self.hparams.lr_annealing_ssl(
                    stage_stats["WER"]
                )
                sb.nnet.schedulers.update_learning_rate(
                    self.ssl_optimizer, new_lr_ssl
                )

            steps = self.optimizer_step
            optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": old_lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"], "epoch": epoch},
                min_keys=["WER"],
            )

        elif stage == sb.Stage.TEST:
            self._write_inference_benchmark(
                stage_seconds, memory_stats, stage_stats
            )
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(
                    self.hparams.test_wer_file, "w", encoding="utf-8"
                ) as w:
                    self.wer_metric.write_stats(w)

    def init_optimizers(self):
        """Initialize optimizers for the model.

        Creates separate optimizers for the main model and optionally for the SSL encoder
        if it's not frozen. Registers optimizers with the checkpointer for resuming training.
        """
        self.optimizer = self.hparams.opt(self.hparams.model.parameters())
        self.optimizers_dict = {"model_optimizer": self.optimizer}

        ssl_frozen = getattr(self.hparams, "ssl_frozen", True)
        if not ssl_frozen:
            self.ssl_optimizer = self.hparams.opt_ssl(
                self.modules.ssl.parameters()
            )
            self.optimizers_dict["ssl_optimizer"] = self.ssl_optimizer

        if self.checkpointer is not None:
            self.checkpointer.add_recoverable("model_optimizer", self.optimizer)
            if not ssl_frozen:
                self.checkpointer.add_recoverable(
                    "ssl_optimizer", self.ssl_optimizer
                )


def dataio_prepare(hparams, tokenizer):
    """Prepares the datasets and dynamic pipelines for the brain class.

    This function sets up the data pipelines for both training and evaluation.
    It handles two modes:
    1. Standard audio mode: loads raw audio files and processes them on-the-fly
    2. Cached features mode: loads pre-extracted SSL features from HDF5 cache

    Arguments
    ---------
    hparams : dict
        Hyperparameters dictionary containing data paths, token indices, etc.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for encoding text tokens

    Returns
    -------
    train_data : DynamicItemDataset
        Training dataset
    valid_data : DynamicItemDataset
        Validation dataset
    test_datasets : dict
        Dictionary of test datasets (keyed by split name)
    tokenizer : transformers.PreTrainedTokenizer
        The tokenizer (returned for convenience)
    train_batch_sampler : DynamicBatchSampler or None
        Batch sampler for training if dynamic batching is enabled
    valid_batch_sampler : DynamicBatchSampler or None
        Batch sampler for validation if dynamic batching is enabled
    """
    data_folder = hparams["data_folder"]
    # Cached-feats mode should be enabled ONLY via the explicit `use_feats` flag.
    # Do not use `hparams["ssl"]` as a boolean (it's a model object).
    use_feats = bool(hparams.get("use_feats", False))

    if use_feats:
        feats_cache_dir = hparams.get("feats_cache_dir", None)
        if not feats_cache_dir:
            raise ValueError(
                "`use_feats=True` requires `feats_cache_dir` to be set "
                "(directory produced by `extract_ssl_feats.py`)."
            )
    else:
        # On-the-fly SSL feature extraction requires an SSL encoder module.
        modules = hparams.get("modules", {})
        if not (isinstance(modules, dict) and "ssl" in modules):
            raise ValueError(
                "`use_feats=False` requires an SSL encoder under `modules.ssl` "
                "to extract features on-the-fly. Either set `use_feats=True` "
                "and provide `feats_cache_dir`, or add `ssl` to `modules`."
            )

    logger.info("use_feats=%s", use_feats)

    # The alignment-boundary pipeline serves both the fixed-pooling baseline
    # (`boundary_source: alignment`) and the segmenter warm-start
    # (`warmstart_target: alignment` in train_speechllm_with_segmenter.py).
    boundary_source = hparams.get("boundary_source", "none") or "none"
    needs_boundary_target = (
        boundary_source == "alignment"
        or hparams.get("warmstart_target", None) == "alignment"
    )
    if needs_boundary_target and not hparams.get("boundary_target_dir"):
        raise ValueError(
            "Alignment boundaries require `boundary_target_dir`: a directory "
            "of per-utterance `<id>.pt` files, each a 1-D {0,1} tensor with "
            "one label per encoder frame (1 = frame starts a new segment)."
        )

    # Blank/separator-token ablations (blank_mode != 'keep') need a per-frame
    # is_separator channel (`separator_target_dir`, from ctc_boundary_align's
    # word_ctc_sep) so the recipe can identify the separator/"blank" tokens.
    blank_mode = hparams.get("blank_mode", "keep") or "keep"
    needs_separator = needs_boundary_target and blank_mode != "keep"
    if needs_separator and not hparams.get("separator_target_dir"):
        raise ValueError(
            f"blank_mode='{blank_mode}' requires `separator_target_dir`: a "
            "directory of per-utterance `<id>.pt` files, each a 1-D {0,1} "
            "tensor with one is_separator label per encoder frame."
        )

    # Token indices and prompt setup
    bos_index = hparams["bos_index"]
    eos_index = hparams["eos_index"]
    pad_index = hparams["pad_token"]

    # Convert special tokens to IDs with error handling
    start_of_audio_token = "<|start_of_audio|>"
    end_of_audio_token = "<|end_of_audio|>"

    start_of_audio_index = tokenizer.convert_tokens_to_ids(start_of_audio_token)
    end_of_audio_index = tokenizer.convert_tokens_to_ids(end_of_audio_token)

    logger.info(
        f"Token indices - BOS: {bos_index}, EOS: {eos_index}, PAD: {pad_index}, "
        f"start_of_audio: {start_of_audio_index}, end_of_audio: {end_of_audio_index}"
    )
    logger.info(f"Prompt: '{hparams['prompt']}'")

    prompt_ids = (
        tokenizer(
            hparams["prompt"], return_tensors="pt", add_special_tokens=False
        )
        .input_ids.view(-1)
        .tolist()
    )

    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens", "prompt_len"
    )
    def text_pipeline(wrd):
        """Process text through tokenization pipeline.

        Creates the following sequence structure:
        tokens_bos: [<|start_of_audio|>, <|end_of_audio|>, prompt_tokens, <bos>?, text_tokens]
        (<bos> is omitted when bos_index is None, e.g. for tokenizers with no BOS token)
        tokens_eos: [text_tokens, <eos>]

        Arguments
        ---------
        wrd : str
            Word/transcription text

        Yields
        ------
        wrd : str
            Original word (unchanged)
        tokens_list : list
            List of token IDs for the text (without special tokens)
        tokens_bos : torch.LongTensor
            Token sequence with start_of_audio, end_of_audio, prompt, bos, and text
        tokens_eos : torch.LongTensor
            Token sequence with text and eos
        tokens : torch.LongTensor
            Token IDs for text only (same as tokens_list but as tensor)
        prompt_len : int
            Length of prompt tokens (start_of_audio + end_of_audio + prompt)
        """
        yield wrd
        tokens_list = tokenizer(wrd, add_special_tokens=False).input_ids
        yield tokens_list
        tokens_bos = torch.LongTensor(
            [start_of_audio_index]
            + [end_of_audio_index]
            + prompt_ids
            + ([bos_index] if bos_index is not None else [])
            + tokens_list
        )
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [eos_index])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens
        prompt_len = len(
            [start_of_audio_index] + [end_of_audio_index] + prompt_ids
        )
        yield prompt_len

    extra_items = []
    if needs_boundary_target:
        boundary_dir = hparams["boundary_target_dir"]

        @sb.utils.data_pipeline.takes("id")
        @sb.utils.data_pipeline.provides("boundary_target")
        def boundary_pipeline(utt_id):
            """Load per-frame {0,1} boundary labels for one utterance."""
            path = os.path.join(boundary_dir, f"{utt_id}.pt")
            return torch.load(path).view(-1).long()

        extra_items.append(boundary_pipeline)

    if needs_separator:
        separator_dir = hparams["separator_target_dir"]

        @sb.utils.data_pipeline.takes("id")
        @sb.utils.data_pipeline.provides("separator_target")
        def separator_pipeline(utt_id):
            """Load per-frame {0,1} is_separator labels for one utterance."""
            path = os.path.join(separator_dir, f"{utt_id}.pt")
            return torch.load(path).view(-1).long()

        extra_items.append(separator_pipeline)

    # Define dynamic items based on mode
    # Note: build_dynamic_items is defined outside the if/else to avoid scope issues
    def build_dynamic_items():
        """Build dynamic items list based on whether we're using cached features or raw audio.

        Returns
        -------
        list
            List of dynamic item pipelines
        """
        if use_feats:
            feats_pipeline = CachedHDF5DynamicItem(
                hparams["feats_cache_dir"],
                file_mode="r",
                takes=["id"],
                provides=["feats"],
                compression="gzip",
            )
            return [text_pipeline, feats_pipeline] + extra_items
        else:

            @sb.utils.data_pipeline.takes("wav")
            @sb.utils.data_pipeline.provides("sig")
            def audio_pipeline(wav):
                """Load audio from file path.

                Arguments
                ---------
                wav : str
                    Path to audio file

                Returns
                -------
                sig : torch.Tensor
                    Audio waveform
                """
                sig = sb.dataio.dataio.read_audio(wav)
                return sig

            return [text_pipeline, audio_pipeline] + extra_items

    # Set output keys based on mode
    if use_feats:
        output_keys = [
            "id",
            "wrd",
            "tokens_bos",
            "tokens_eos",
            "tokens",
            "prompt_len",
            "feats",
        ]
    else:
        output_keys = [
            "id",
            "sig",
            "wrd",
            "tokens_bos",
            "tokens_eos",
            "tokens",
            "prompt_len",
        ]
    if needs_boundary_target:
        output_keys.append("boundary_target")
    if needs_separator:
        output_keys.append("separator_target")

    def _create_dataset(csv_path, sorting="ascending"):
        """Create a dataset from CSV file with optional sorting.

        Arguments
        ---------
        csv_path : str
            Path to CSV file containing dataset metadata
        sorting : str
            Sorting strategy: "ascending", "descending", or "random"

        Returns
        -------
        dataset : DynamicItemDataset
            Configured dataset with dynamic pipelines applied
        """
        assert sorting in ["ascending", "descending", "random"], (
            f"sorting must be one of ['ascending', 'descending', 'random'], got '{sorting}'"
        )

        dataset = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_path,
            replacements={"data_root": data_folder},
            dynamic_items=build_dynamic_items(),
            output_keys=output_keys,
        )
        if sorting == "ascending":
            dataset = dataset.filtered_sorted(sort_key="duration")
            hparams["train_dataloader_opts"]["shuffle"] = False
        elif sorting == "descending":
            dataset = dataset.filtered_sorted(sort_key="duration", reverse=True)
            hparams["train_dataloader_opts"]["shuffle"] = False
        elif sorting == "random":
            pass
        return dataset

    # Create training dataset with sorting logic
    train_data = _create_dataset(
        hparams["train_csv"], sorting=hparams["sorting"]
    )
    valid_data = _create_dataset(hparams["valid_csv"], sorting="ascending")

    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = _create_dataset(csv_file, sorting="ascending")

    # Dynamic batch sampling
    train_batch_sampler = None
    valid_batch_sampler = None
    if hparams["dynamic_batching"]:
        from speechbrain.dataio.sampler import DynamicBatchSampler

        train_batch_sampler = DynamicBatchSampler(
            train_data,
            length_func=lambda x: x["duration"],
            **hparams["dynamic_batch_sampler_train"],
        )
        valid_batch_sampler = DynamicBatchSampler(
            valid_data,
            length_func=lambda x: x["duration"],
            **hparams["dynamic_batch_sampler_valid"],
        )

    return (
        train_data,
        valid_data,
        test_datasets,
        tokenizer,
        train_batch_sampler,
        valid_batch_sampler,
    )
