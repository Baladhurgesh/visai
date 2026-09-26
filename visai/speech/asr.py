"""Parakeet (parakeet-mlx) deployment evaluation: RTFx on meeting audio, WER on LibriSpeech."""

from __future__ import annotations

import gc
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from visai.speech.data import load_manifest, prepare_ami, prepare_librispeech
from visai.speech.metrics import wer

PARAKEET_ID = "mlx-community/parakeet-tdt-0.6b-v3"
DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}
STOCK = {"dtype": "bfloat16", "chunk_duration": 120.0, "overlap_duration": 15.0}

SEARCH_SPACE_DOC = """Parakeet-TDT 0.6B (FastConformer encoder + TDT decoder) deployment config keys:
- dtype: bfloat16 (stock) | float16 | float32  (activation/weight dtype)
- bits: weight quantization of Linear layers: 8 | 6 | 4 | null ; group_size: 32 | 64 | 128
- keep_high: module-path substrings kept unquantized, e.g. ["decoder", "joint", "encoder.pre_encode", "encoder.layers.0."]
- chunk_duration: seconds per chunk for long audio (30..300; stock 120) ; overlap_duration: 5..30 (stock 15)
Workload: 10-minute AMI meeting audio transcription (throughput = real-time factor x). Quality: WER on LibriSpeech test-clean.
Attention cost grows with chunk length; overlap is recomputed work; quantization cuts weight bytes.
"""


def validate(cfg: dict[str, Any]) -> list[str]:
    errs = []
    if cfg.get("dtype", "bfloat16") not in DTYPES:
        errs.append(f"dtype must be one of {list(DTYPES)}")
    if cfg.get("bits") not in (None, 4, 6, 8):
        errs.append("bits must be 4, 6, 8 or null")
    if cfg.get("group_size", 64) not in (32, 64, 128):
        errs.append("group_size must be 32, 64 or 128")
    cd = cfg.get("chunk_duration", 120.0)
    if cd is not None and not 30 <= float(cd) <= 300:
        errs.append("chunk_duration must be in [30, 300]")
    od = float(cfg.get("overlap_duration", 15.0))
    if not 5 <= od <= 30 or (cd and od >= float(cd) / 2):
        errs.append("overlap_duration must be in [5, 30] and < chunk_duration/2")
    unknown = set(cfg) - {"dtype", "bits", "group_size", "keep_high", "chunk_duration", "overlap_duration"}
    if unknown:
        errs.append(f"unknown keys {sorted(unknown)}")
    return errs


def load(cfg: dict[str, Any]):
    from parakeet_mlx import from_pretrained

    dtype = DTYPES[cfg.get("dtype", "bfloat16")]
    model = from_pretrained(PARAKEET_ID, dtype=dtype)
    if cfg.get("bits"):
        group = int(cfg.get("group_size", 64))
        keep = tuple(cfg.get("keep_high") or ())

        def pred(path: str, m: nn.Module) -> bool:
            return (
                isinstance(m, nn.Linear)
                and not any(k in path for k in keep)
                and m.weight.shape[-1] % group == 0
            )

        nn.quantize(model, group_size=group, bits=int(cfg["bits"]), class_predicate=pred)
    mx.eval(model.parameters())
    return model


def _free() -> None:
    gc.collect()
    mx.clear_cache()


def _transcribe(model, path: str, cfg: dict[str, Any]) -> str:
    dtype = DTYPES[cfg.get("dtype", "bfloat16")]
    res = model.transcribe(
        path, dtype=dtype, chunk_duration=cfg.get("chunk_duration", 120.0),
        overlap_duration=float(cfg.get("overlap_duration", 15.0)),
    )
    return res.text


def speed(model, cfg: dict[str, Any], runs: int = 2) -> float:
    meeting = load_manifest(prepare_ami())[0]
    _transcribe(model, meeting["path"], cfg)  # warmup
    best = None
    for _ in range(runs):
        t0 = time.perf_counter()
        _transcribe(model, meeting["path"], cfg)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    return meeting["duration"] / best


def evaluate(cfg: dict[str, Any], n_wer: int = 40) -> dict[str, Any]:
    cfg = {**STOCK, **cfg}
    model = load(cfg)
    try:
        mx.reset_peak_memory()
        rtfx = speed(model, cfg)
        rows = load_manifest(prepare_librispeech())[:n_wer]
        hyps = [_transcribe(model, r["path"], cfg) for r in rows]
        w = wer([r["text"] for r in rows], hyps)
        return {"throughput": rtfx, "unit": "x real-time (10-min meeting)", "wer": w, "quality": -w,
                "peak_memory_gb": mx.get_peak_memory() / 2**30, "n_errors": 0}
    finally:
        del model
        _free()


def speed_of(cfg: dict[str, Any]) -> float:
    cfg = {**STOCK, **cfg}
    model = load(cfg)
    try:
        return speed(model, cfg)
    finally:
        del model
        _free()


def describe(m: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
            if k in ("throughput", "wer", "peak_memory_gb", "unit")}
