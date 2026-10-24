"""CTC-forced-alignment boundary labels for LibriSpeech (char / word_ctc).

Generates per-utterance boundary-label files consumed by the alignment
baselines in ``train_speechllm.py`` (``boundary_source: alignment``). One file
per utterance: ``{out_root}/{level}/{utt_id}.pt`` containing a 1-D
``torch.uint8`` tensor with values in ``{0, 1}`` and EXACTLY one label per
encoder output frame (``1`` = frame starts a new segment, ``0`` =
continuation; see ``segment_pooling.py`` for the shared convention). Index 0
is always set to ``1``: frame 0 always starts segment 0 downstream (harmless
no-op there), and it makes ``tensor.sum() == num_segments``.

Method
------
Alignment comes from ``torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H``, a
char-level CTC model trained on LibriSpeech 960h (blank ``'-'`` at index 0,
word separator ``'|'`` at index 1). Its emission frames coincide 1:1 with the
frames of the downstream encoder (microsoft/wavlm-large — validated
empirically, including odd lengths), because both use the same wav2vec2-style
conv feature extractor. Alignment output frame indices therefore ARE the
label indices — no time conversion. Each utterance asserts
``emission_length == num_encoder_frames(num_samples)`` and hard-fails on
mismatch.

Levels
------
``char``
    A boundary at the start frame of EVERY token span, including ``'|'``
    spans: silence/space gets its own segment. CTC blank frames simply extend
    the previous segment, so long inter-word silences mostly attach to the
    preceding ``'|'`` segment, and leading silence before the first char
    belongs to segment 0.
``word_ctc``
    A boundary at the start frame of every ``'|'`` span AND at the start
    frame of the first char span following each ``'|'`` (plus the
    utterance-initial first char span). Words start segments; separators get
    their own segment, mirroring the MFA word level's silence policy.

Token spans from ``merge_tokens`` are disjoint (each covers >= 1 emission
frame), so two spans can never share a start frame — boundary writes never
collapse.

Run directly for sharded multi-GPU generation, e.g.::

    python asset/prepare_boundaries.py --levels char word_ctc \
        --splits train-clean-100 --shard 0 --num_shards 8 --device cuda:0

The downstream training recipe reads the generated ``char/<utterance-id>.pt`` files.

Authors
-------
 * Speech-LLM segmenter, 2026
"""

import argparse
import glob
import os
import time
from collections import defaultdict

import torch
import torchaudio
from torchaudio.functional import forced_align, merge_tokens

SAMPLE_RATE = 16000
FRAME_SEC = 0.02
BLANK_ID = 0
SEP = "|"
CTC_LEVELS = ("char", "word_ctc")

CONV = [(10, 5), (3, 2), (3, 2), (3, 2), (3, 2), (2, 2), (2, 2)]


def num_encoder_frames(num_samples):
    """Number of encoder output frames for a raw 16 kHz sample count.

    Matches both microsoft/wavlm-large (the downstream encoder) and
    torchaudio WAV2VEC2_ASR_BASE_960H (the alignment model) exactly.

    Arguments
    ---------
    num_samples : int
        Number of raw audio samples.

    Returns
    -------
    int
        Number of encoder output frames.
    """
    L = num_samples
    for k, s in CONV:
        L = (L - k) // s + 1
    return L


def collect_utterances(audio_root, split):
    """List all utterances of a LibriSpeech split with their transcripts.

    Arguments
    ---------
    audio_root : str
        LibriSpeech root (contains the split directories).
    split : str
        Split name, e.g. ``"dev-clean"``.

    Returns
    -------
    list of (str, str, str)
        Sorted ``(utt_id, flac_path, transcript)`` tuples.
    """
    split_dir = os.path.join(audio_root, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    utts = []
    pattern = os.path.join(split_dir, "*", "*", "*.trans.txt")
    for trans_path in sorted(glob.glob(pattern)):
        chap_dir = os.path.dirname(trans_path)
        with open(trans_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                utt_id = parts[0]
                text = parts[1] if len(parts) > 1 else ""
                flac = os.path.join(chap_dir, f"{utt_id}.flac")
                utts.append((utt_id, flac, text))
    utts.sort(key=lambda u: u[0])
    return utts


def boundaries_from_spans(spans, num_frames, sep_id, levels):
    """Convert merged CTC token spans into boundary label tensors.

    Arguments
    ---------
    spans : list of torchaudio.functional.TokenSpan
        Output of ``merge_tokens`` (disjoint, monotonic, blank-free).
    num_frames : int
        Number of encoder frames (label length).
    sep_id : int
        Token id of the word separator ``'|'``.
    levels : list of str
        Subset of ``("char", "word_ctc")``.

    Returns
    -------
    dict of str -> torch.Tensor
        Level name to 1-D ``uint8`` boundary tensor of length ``num_frames``.
    """
    out = {}
    if "char" in levels:
        b = torch.zeros(num_frames, dtype=torch.uint8)
        for sp in spans:
            b[sp.start] = 1
        b[0] = 1
        out["char"] = b
    if "word_ctc" in levels:
        b = torch.zeros(num_frames, dtype=torch.uint8)
        prev_was_sep = True
        for sp in spans:
            if sp.token == sep_id:
                b[sp.start] = 1
                prev_was_sep = True
            else:
                if prev_was_sep:
                    b[sp.start] = 1
                prev_was_sep = False
        b[0] = 1
        out["word_ctc"] = b
    return out


def word_ctc_separator_frames(spans, num_frames, sep_id):
    """Per-frame ``is_separator`` label (uint8) for the ``word_ctc`` segmentation.

    Companion to ``boundaries_from_spans('word_ctc')``: for the exact same
    segments, marks which frames belong to a *separator* segment vs a *word*
    segment. Downstream (``blank_mode`` in ``train_speechllm.py``) this is what
    lets the recipe identify and drop/replace the "blank"/gap audio tokens, since
    the ``word_ctc`` ``{0,1}`` boundary tensor alone does not record segment type.

    A segment's type is the type of the boundary that opens it (mirroring
    ``boundaries_from_spans``):

    * ``'|'``-span starts open **separator** segments (inter-word gaps),
    * the first char-span after each ``'|'`` opens a **word** segment,
    * frame 0 always opens segment 0; if no span starts exactly at frame 0 there
      is leading silence before the first word, so segment 0 is a **separator**
      (leading-silence) segment. (Trailing silence has no ``'|'`` after it and
      folds into the last word segment, matching the pooling — so it is *not*
      marked as a separator.)

    Arguments
    ---------
    spans : list of torchaudio.functional.TokenSpan
        Output of ``merge_tokens`` (disjoint, monotonic, blank-free).
    num_frames : int
        Number of encoder frames (label length); matches the ``word_ctc`` tensor.
    sep_id : int
        Token id of the word separator ``'|'``.

    Returns
    -------
    is_sep : torch.Tensor
        1-D ``uint8`` of length ``num_frames``; ``1`` = frame is in a separator
        segment, ``0`` = frame is in a word segment.
    """
    openers = []  # (start_frame, is_sep) in increasing frame order
    prev_was_sep = True
    for sp in spans:
        if sp.token == sep_id:
            openers.append((sp.start, True))  # separator segment
            prev_was_sep = True
        else:
            if prev_was_sep:
                openers.append((sp.start, False))  # word onset
            prev_was_sep = False
    # Frame 0 is always a boundary; if nothing opens at 0 it is leading silence.
    if not openers or openers[0][0] != 0:
        openers.insert(0, (0, True))

    is_sep = torch.zeros(num_frames, dtype=torch.uint8)
    for i, (start, sep) in enumerate(openers):
        end = openers[i + 1][0] if i + 1 < len(openers) else num_frames
        if sep:
            is_sep[start:end] = 1
    return is_sep


def labels_from_spans(spans, labels, sep_id):
    """Per-segment ``(start_frame, text)`` labels mirroring the boundary policy.

    The frames produced here are exactly the boundary positions written by
    :func:`boundaries_from_spans` (aside from the always-on frame 0, which the
    caller handles for any leading silence). Used only for visualization.

    Arguments
    ---------
    spans : list of torchaudio.functional.TokenSpan
        Output of ``merge_tokens`` (disjoint, monotonic, blank-free).
    labels : tuple of str
        The model's label alphabet (``labels[token_id]`` -> character).
    sep_id : int
        Token id of the word separator ``'|'``.

    Returns
    -------
    dict of str -> list of (int, str)
        ``"char"``: one entry per token span (``'|'`` shown as ``'␣'``).
        ``"word_ctc"``: one entry per word (chars joined) and one per separator.
    """
    char_seg = []
    for sp in spans:
        ch = labels[sp.token]
        char_seg.append((sp.start, "␣" if sp.token == sep_id else ch))

    word_seg = []
    run = []
    for sp in spans:
        if sp.token == sep_id:
            if run:
                word_seg.append(
                    (run[0].start, "".join(labels[s.token] for s in run))
                )
                run = []
            word_seg.append((sp.start, "␣"))
        else:
            run.append(sp)
    if run:
        word_seg.append((run[0].start, "".join(labels[s.token] for s in run)))

    return {"char": char_seg, "word_ctc": word_seg}


def segment_labels(utts, device="cuda"):
    """Re-run CTC forced alignment and return per-segment char/word_ctc labels.

    Loads the alignment model once and processes each utterance. Intended for
    visualizing a handful of examples (not full-corpus use).

    Arguments
    ---------
    utts : list of (str, str, str)
        ``(utt_id, flac_path, transcript)`` tuples.
    device : str
        Device for the CTC model (and forced_align when supported).

    Returns
    -------
    dict of str -> dict
        ``utt_id`` -> ``{"char": [(start, text), ...],
        "word_ctc": [(start, text), ...]}`` in span order (increasing frame).
    """
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device).eval()
    labels = bundle.get_labels()
    char_to_idx = {c: i for i, c in enumerate(labels)}
    sep_id = char_to_idx[SEP]
    align_device = _probe_align_device(device)

    out = {}
    for utt_id, flac, text in utts:
        tokens = text.replace(" ", SEP)
        with torch.inference_mode():
            wav, sr = torchaudio.load(flac)
            if sr != SAMPLE_RATE:
                raise ValueError(f"{flac}: sample rate {sr} != {SAMPLE_RATE}")
            emissions, _ = model(wav.to(device))
            if emissions.size(1) != num_encoder_frames(wav.size(1)):
                raise RuntimeError(f"{utt_id}: emission/frame mismatch")
            log_probs = torch.log_softmax(emissions.float(), dim=-1)
            targets = torch.tensor(
                [[char_to_idx[c] for c in tokens]], dtype=torch.int32
            )
            aligned, scores = forced_align(
                log_probs.to(align_device),
                targets.to(align_device),
                blank=BLANK_ID,
            )
            spans = merge_tokens(
                aligned[0].cpu(), scores[0].cpu(), blank=BLANK_ID
            )
        out[utt_id] = labels_from_spans(spans, labels, sep_id)
    return out


def _probe_align_device(device):
    """Return the device forced_align actually runs on (CPU fallback)."""
    if not str(device).startswith("cuda"):
        return "cpu"
    try:
        lp = torch.log_softmax(torch.randn(1, 20, 29, device=device), dim=-1)
        tg = torch.tensor([[2, 3]], dtype=torch.int32, device=device)
        forced_align(lp, tg, blank=BLANK_ID)
        return device
    except Exception as e:
        print(f"forced_align unavailable on {device} ({e}); using CPU.")
        return "cpu"


def _new_level_stats():
    return {
        "files_written": 0,
        "total_segments": 0,
        "total_frames": 0,
        "total_duration": 0.0,
        "sum_kept_ratio": 0.0,
    }


def generate_ctc_boundaries(
    levels,
    splits,
    audio_root="data/LibriSpeech",
    out_root="data/boundaries",
    device="cuda",
    shard=0,
    num_shards=1,
    overwrite=False,
    log_every=250,
):
    """Generate CTC-forced-alignment boundary labels for LibriSpeech splits.

    When both levels are requested, the forced alignment is computed ONCE per
    utterance and both label files are written in the same pass. Utterances
    with ``idx % num_shards != shard`` are left to other shards (multi-GPU
    runs). Existing files are skipped unless ``overwrite`` is set.

    Arguments
    ---------
    levels : list of str
        Subset of ``["char", "word_ctc"]``.
    splits : list of str
        LibriSpeech split names, e.g. ``["dev-clean"]``.
    audio_root : str
        LibriSpeech root directory (read-only).
    out_root : str
        Output root; files go to ``{out_root}/{level}/{utt_id}.pt``.
    device : str
        Device for the CTC model (and forced_align when supported).
    shard : int
        This process's shard index.
    num_shards : int
        Total number of shards.
    overwrite : bool
        Re-generate files that already exist.
    log_every : int
        Progress print interval (processed utterances).

    Returns
    -------
    dict
        Per-split stats: files written per level, skips, alignment failures,
        and per-level segment statistics.
    """
    levels = list(levels)
    for lv in levels:
        if lv not in CTC_LEVELS:
            raise ValueError(f"Unknown CTC level '{lv}'. Use {CTC_LEVELS}.")
    if not levels:
        raise ValueError("No levels requested.")

    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device).eval()
    labels = bundle.get_labels()
    char_to_idx = {c: i for i, c in enumerate(labels)}
    sep_id = char_to_idx[SEP]
    align_device = _probe_align_device(device)

    for level in levels:
        os.makedirs(os.path.join(out_root, level), exist_ok=True)
    # word_ctc also emits a parallel per-frame is_separator channel.
    emit_sep = "word_ctc" in levels
    if emit_sep:
        os.makedirs(os.path.join(out_root, "word_ctc_sep"), exist_ok=True)

    stats = {}
    for split in splits:
        utts = collect_utterances(audio_root, split)
        shard_utts = [u for i, u in enumerate(utts) if i % num_shards == shard]
        s = {
            "num_utts_in_shard": len(shard_utts),
            "skipped_empty": 0,
            "skipped_existing": 0,
            "align_failures": 0,
            "failed_utts": [],
            "unexpected_chars": defaultdict(int),
            "levels": {lv: _new_level_stats() for lv in levels},
        }
        stats[split] = s
        t0 = time.time()
        done = 0

        for utt_id, flac, text in shard_utts:
            done += 1
            if done % log_every == 0:
                rate = done / (time.time() - t0)
                print(
                    f"[{split} shard {shard}/{num_shards}] "
                    f"{done}/{len(shard_utts)} ({rate:.1f} utt/s)"
                )
            out_paths = {
                lv: os.path.join(out_root, lv, f"{utt_id}.pt")
                for lv in levels
            }
            if emit_sep:
                out_paths["word_ctc_sep"] = os.path.join(
                    out_root, "word_ctc_sep", f"{utt_id}.pt"
                )
            if not overwrite and all(
                os.path.exists(p) for p in out_paths.values()
            ):
                s["skipped_existing"] += 1
                continue
            if not text:
                s["skipped_empty"] += 1
                continue

            tokens = text.replace(" ", SEP)
            bad = [c for c in set(tokens) if c not in char_to_idx]
            if bad:
                for c in bad:
                    s["unexpected_chars"][c] += 1
                s["align_failures"] += 1
                s["failed_utts"].append((utt_id, f"unexpected chars {bad}"))
                continue

            with torch.inference_mode():
                wav, sr = torchaudio.load(flac)
                if sr != SAMPLE_RATE:
                    raise ValueError(f"{flac}: sample rate {sr} != 16000")
                num_samples = wav.size(1)
                expected_frames = num_encoder_frames(num_samples)
                emissions, _ = model(wav.to(device))
                if emissions.size(1) != expected_frames:
                    raise RuntimeError(
                        f"{utt_id}: emission has {emissions.size(1)} frames "
                        f"but num_encoder_frames({num_samples}) = "
                        f"{expected_frames}. Frame contract violated."
                    )
                log_probs = torch.log_softmax(emissions.float(), dim=-1)
                targets = torch.tensor(
                    [[char_to_idx[c] for c in tokens]], dtype=torch.int32
                )
                try:
                    aligned, scores = forced_align(
                        log_probs.to(align_device),
                        targets.to(align_device),
                        blank=BLANK_ID,
                    )
                except Exception as e:
                    s["align_failures"] += 1
                    s["failed_utts"].append((utt_id, str(e)))
                    continue
                spans = merge_tokens(
                    aligned[0].cpu(), scores[0].cpu(), blank=BLANK_ID
                )

            boundaries = boundaries_from_spans(
                spans, expected_frames, sep_id, levels
            )
            duration = num_samples / SAMPLE_RATE
            for lv, b in boundaries.items():
                torch.save(b, out_paths[lv])
                n_seg = int(b.sum().item())
                ls = s["levels"][lv]
                ls["files_written"] += 1
                ls["total_segments"] += n_seg
                ls["total_frames"] += expected_frames
                ls["total_duration"] += duration
                ls["sum_kept_ratio"] += n_seg / expected_frames
            if emit_sep:
                is_sep = word_ctc_separator_frames(
                    spans, expected_frames, sep_id
                )
                torch.save(is_sep, out_paths["word_ctc_sep"])

        s["unexpected_chars"] = dict(s["unexpected_chars"])
        elapsed = time.time() - t0
        s["elapsed_sec"] = elapsed
        s["utts_per_sec"] = done / elapsed if elapsed > 0 else 0.0
        _print_split_stats(split, s, shard, num_shards)
    return stats


def _print_split_stats(split, s, shard, num_shards):
    """Print a one-split summary of generation stats."""
    print(
        f"== {split} (shard {shard}/{num_shards}): "
        f"{s['num_utts_in_shard']} utts, "
        f"{s['skipped_empty']} empty-skipped, "
        f"{s['skipped_existing']} existing-skipped, "
        f"{s['align_failures']} align-failures, "
        f"{s['elapsed_sec']:.0f}s ({s['utts_per_sec']:.2f} utt/s)"
    )
    if s["unexpected_chars"]:
        print(f"   unexpected transcript chars: {s['unexpected_chars']}")
    for utt_id, err in s["failed_utts"][:10]:
        print(f"   FAILED {utt_id}: {err}")
    for lv, ls in s["levels"].items():
        n = ls["files_written"]
        if n == 0:
            print(f"   [{lv}] no files written")
            continue
        print(
            f"   [{lv}] {n} files | "
            f"{ls['total_segments'] / ls['total_duration']:.2f} seg/s | "
            f"kept-frame ratio {ls['sum_kept_ratio'] / n:.4f} | "
            f"{ls['total_segments'] / n:.1f} seg/utt"
        )


def validate_boundary_files(
    levels,
    splits,
    audio_root="data/LibriSpeech",
    out_root="data/boundaries",
):
    """Reload every generated file and verify the training-time contract.

    Checks per file: values are a subset of ``{0, 1}``, dtype ``uint8``,
    length equals ``num_encoder_frames`` of the utterance's sample count, and
    ``sum >= 1``. Prints per split and level: #files, #missing, mean
    segments/second, mean kept-frame ratio, mean segments per utterance.

    Arguments
    ---------
    levels : list of str
        Subset of ``["char", "word_ctc"]``.
    splits : list of str
        LibriSpeech split names.
    audio_root : str
        LibriSpeech root directory.
    out_root : str
        Root containing ``{level}/{utt_id}.pt`` files.

    Returns
    -------
    dict
        Per (split, level) validation stats.
    """
    stats = {}
    for split in splits:
        utts = collect_utterances(audio_root, split)
        for level in levels:
            n_files = 0
            n_missing = 0
            total_segments = 0
            total_frames = 0
            total_duration = 0.0
            sum_kept = 0.0
            for utt_id, flac, text in utts:
                path = os.path.join(out_root, level, f"{utt_id}.pt")
                if not os.path.exists(path):
                    n_missing += 1
                    continue
                b = torch.load(path)
                num_samples = torchaudio.info(flac).num_frames
                expected = num_encoder_frames(num_samples)
                assert b.dtype == torch.uint8, f"{path}: dtype {b.dtype}"
                assert b.dim() == 1, f"{path}: dim {b.dim()}"
                vals = set(torch.unique(b).tolist())
                assert vals <= {0, 1}, f"{path}: values {vals}"
                assert b.numel() == expected, (
                    f"{path}: {b.numel()} labels != {expected} frames"
                )
                n_seg = int(b.sum().item())
                assert n_seg >= 1, f"{path}: sum {n_seg} < 1"
                n_files += 1
                total_segments += n_seg
                total_frames += expected
                total_duration += num_samples / SAMPLE_RATE
                sum_kept += n_seg / expected
            stats[(split, level)] = {
                "files": n_files,
                "missing": n_missing,
                "seg_per_sec": (
                    total_segments / total_duration if total_duration else 0.0
                ),
                "mean_kept_ratio": sum_kept / n_files if n_files else 0.0,
                "mean_seg_per_utt": (
                    total_segments / n_files if n_files else 0.0
                ),
            }
            st = stats[(split, level)]
            print(
                f"[VALIDATE {split}/{level}] files={st['files']} "
                f"missing={st['missing']} "
                f"seg/s={st['seg_per_sec']:.2f} "
                f"kept-frame-ratio={st['mean_kept_ratio']:.4f} "
                f"seg/utt={st['mean_seg_per_utt']:.1f}"
            )
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CTC forced-alignment boundary labels for LibriSpeech."
    )
    parser.add_argument(
        "--levels", nargs="+", default=list(CTC_LEVELS), choices=CTC_LEVELS
    )
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument(
        "--audio_root", required=True
    )
    parser.add_argument(
        "--out_root", required=True
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--stats_only",
        action="store_true",
        help="Skip generation; validate existing files and print stats.",
    )
    args = parser.parse_args()

    if args.stats_only:
        validate_boundary_files(
            args.levels, args.splits, args.audio_root, args.out_root
        )
    else:
        generate_ctc_boundaries(
            levels=args.levels,
            splits=args.splits,
            audio_root=args.audio_root,
            out_root=args.out_root,
            device=args.device,
            shard=args.shard,
            num_shards=args.num_shards,
            overwrite=args.overwrite,
        )
