"""Nemotron-3-Diarization (mlx-audio) deployment evaluation: RTFx + DER on AMI meeting slices."""

from __future__ import annotations

import gc
import time
from dataclasses import replace
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from visai.speech.data import load_audio, load_manifest, prepare_ami
from visai.speech.metrics import der

DIAR_ID = "mlx-community/Nemotron-3-Diarization"
DTYPES = {"bfloat16": mx.bfloat16, "float16": mx.float16, "float32": mx.float32}
PRESETS = ("offline", "low", "very_low", "ultra_low")
STOCK = {"preset": "offline"}

SEARCH_SPACE_DOC = """Nemotron-3-Diarization (Streaming Sortformer, up to 8 speakers, 10 ms frames) deployment config keys:
- dtype: float32 | bfloat16 | float16 (null = checkpoint dtype, stock)
- bits: weight quantization of Linear layers: 8 | 4 | null ; group_size: 32 | 64 | 128
- preset: offline (30.4 s buffer, stock) | low (1.04 s) | very_low (0.64 s) | ultra_low (0.32 s)
- streaming: custom override {"chunk_len": int, "chunk_right_context": int, "fifo_len": int,
  "spkcache_update_period": int} (encoder frames; chunk_len*8 must divide by the output stride)
- threshold: speaker activity threshold 0.2..0.7 (stock 0.5)
- merge_gap: merge a speaker's segments separated by < this many seconds (0..2, stock 0)
- min_duration: drop segments shorter than this (0..1 s, stock 0)
Workload: 2 x 10-minute AMI meetings (throughput = real-time factor x). Quality: DER (lower is better).
Smaller chunks = lower latency but more encoder passes over the speaker cache + FIFO context.
"""


def validate(cfg: dict[str, Any]) -> list[str]:
    errs = []
    if cfg.get("dtype") not in (None, *DTYPES):
        errs.append(f"dtype must be one of {list(DTYPES)} or null")
    if cfg.get("bits") not in (None, 4, 8):
        errs.append("bits must be 4, 8 or null")
    if cfg.get("group_size", 64) not in (32, 64, 128):
        errs.append("group_size must be 32, 64 or 128")
    if cfg.get("preset", "offline") not in PRESETS:
        errs.append(f"preset must be one of {PRESETS}")
    th = float(cfg.get("threshold", 0.5))
    if not 0.2 <= th <= 0.7:
        errs.append("threshold must be in [0.2, 0.7]")
    if not 0.0 <= float(cfg.get("merge_gap", 0.0)) <= 2.0:
        errs.append("merge_gap must be in [0, 2] seconds")
    if not 0.0 <= float(cfg.get("min_duration", 0.0)) <= 1.0:
        errs.append("min_duration must be in [0, 1] seconds")
    unknown = set(cfg) - {"dtype", "bits", "group_size", "preset", "streaming", "threshold", "merge_gap", "min_duration"}
    if unknown:
        errs.append(f"unknown keys {sorted(unknown)}")
    return errs


def load(cfg: dict[str, Any]):
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_audio.vad import load as vad_load

    model = vad_load(DIAR_ID, strict=True)
    if cfg.get("dtype"):
        dt = DTYPES[cfg["dtype"]]
        params = [(k, v.astype(dt) if mx.issubdtype(v.dtype, mx.floating) else v) for k, v in tree_flatten(model.parameters())]
        model.update(tree_unflatten(params))
    if cfg.get("bits"):
        group = int(cfg.get("group_size", 64))
        nn.quantize(model, group_size=group, bits=int(cfg["bits"]),
                    class_predicate=lambda p, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % group == 0)
    model.set_streaming_config(cfg.get("preset", "offline"))
    if cfg.get("streaming"):
        s = cfg["streaming"]
        mc = replace(model.config.modules_config, **{k: int(v) for k, v in s.items()})
        model.config = replace(model.config, modules_config=mc)
    mx.eval(model.parameters())
    return model


def _free() -> None:
    gc.collect()
    mx.clear_cache()


def _diarize(model, audio, cfg) -> list[tuple[float, float, str]]:
    out = model.generate(audio, sample_rate=16000, threshold=float(cfg.get("threshold", 0.5)),
                         min_duration=float(cfg.get("min_duration", 0.0)), merge_gap=float(cfg.get("merge_gap", 0.0)))
    return [(float(s.start), float(s.end), str(s.speaker)) for s in out.segments]


def evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = {**STOCK, **cfg}
    model = load(cfg)
    try:
        meetings = load_manifest(prepare_ami())
        audios = [load_audio(m["path"]) for m in meetings]
        _diarize(model, audios[0][: 16000 * 60], cfg)  # warmup
        mx.reset_peak_memory()
        wall, total_dur, ders = 0.0, 0.0, []
        for m, a in zip(meetings, audios):
            t0 = time.perf_counter()
            hyp = _diarize(model, a, cfg)
            wall += time.perf_counter() - t0
            total_dur += m["duration"]
            ders.append(der([tuple(s) for s in m["segments"]], hyp, m["duration"]))
        d = sum(x["der"] for x in ders) / len(ders)
        return {"throughput": total_dur / wall, "unit": "x real-time (AMI meetings)", "der": d, "quality": -d,
                "missed": sum(x["missed"] for x in ders) / len(ders),
                "false_alarm": sum(x["false_alarm"] for x in ders) / len(ders),
                "confusion": sum(x["confusion"] for x in ders) / len(ders),
                "peak_memory_gb": mx.get_peak_memory() / 2**30, "n_errors": 0}
    finally:
        del model
        _free()


def speed_of(cfg: dict[str, Any]) -> float:
    cfg = {**STOCK, **cfg}
    model = load(cfg)
    try:
        meetings = load_manifest(prepare_ami())
        audios = [load_audio(m["path"]) for m in meetings]
        _diarize(model, audios[0][: 16000 * 60], cfg)
        t0 = time.perf_counter()
        for a in audios:
            _diarize(model, a, cfg)
        return sum(m["duration"] for m in meetings) / (time.perf_counter() - t0)
    finally:
        del model
        _free()


def describe(m: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()
            if k in ("throughput", "der", "missed", "false_alarm", "confusion", "peak_memory_gb", "unit")}
