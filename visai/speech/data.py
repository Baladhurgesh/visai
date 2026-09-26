"""Cache small public eval slices locally (decoded with soundfile, no torch audio stack).

- LibriSpeech test-clean: first N utterances -> WER (ASR quality gate)
- AMI (diarizers-community/ami, ihm, test): first M meetings cropped to `crop_s` -> DER (diarization gate)
  and meeting-length audio for the throughput workload.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from visai.config import OUT

DATA = OUT / "data"
SR = 16000


def _stream(repo: str, config: str, split: str):
    import datasets
    from datasets import Audio

    ds = datasets.load_dataset(repo, config, split=split, streaming=True)
    return ds.cast_column("audio", Audio(decode=False))


def _decode(audio: dict) -> np.ndarray:
    a, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        import soxr

        a = soxr.resample(a, sr, SR)
    return a.astype(np.float32)


def prepare_librispeech(n: int = 100) -> Path:
    out = DATA / "librispeech"
    manifest = out / "manifest.jsonl"
    if manifest.exists() and sum(1 for _ in manifest.open()) >= n:
        return manifest
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, row in enumerate(_stream("openslr/librispeech_asr", "clean", "test")):
        if i >= n:
            break
        a = _decode(row["audio"])
        path = out / f"{i:04d}.flac"
        sf.write(path, a, SR)
        rows.append({"path": str(path), "text": row["text"], "duration": len(a) / SR})
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return manifest


def prepare_ami(n_meetings: int = 2, crop_s: float = 600.0) -> Path:
    out = DATA / "ami"
    manifest = out / "manifest.jsonl"
    if manifest.exists() and sum(1 for _ in manifest.open()) >= n_meetings:
        return manifest
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, row in enumerate(_stream("diarizers-community/ami", "ihm", "test")):
        if i >= n_meetings:
            break
        a = _decode(row["audio"])[: int(crop_s * SR)]
        path = out / f"meeting{i}.flac"
        sf.write(path, a, SR)
        dur = len(a) / SR
        segs = [
            (float(s), float(min(e, dur)), str(spk))
            for s, e, spk in zip(row["timestamps_start"], row["timestamps_end"], row["speakers"])
            if s < dur
        ]
        rows.append({"path": str(path), "duration": dur, "segments": segs})
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return manifest


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def load_audio(path: str) -> np.ndarray:
    a, _ = sf.read(path, dtype="float32")
    return a
