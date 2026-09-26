"""WER (jiwer, normalized) and frame-level DER with optimal speaker mapping (no collar)."""

from __future__ import annotations

import re

import numpy as np


def normalize_text(t: str) -> str:
    t = t.lower().replace("-", " ")
    t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def wer(refs: list[str], hyps: list[str]) -> float:
    import jiwer

    r = [normalize_text(x) or "<empty>" for x in refs]
    h = [normalize_text(x) for x in hyps]
    return float(jiwer.wer(r, h))


def _frames(segs: list[tuple[float, float, str]], n: int, step: float) -> tuple[np.ndarray, list[str]]:
    spks = sorted({s[2] for s in segs})
    m = np.zeros((n, max(len(spks), 1)), dtype=bool)
    idx = {s: i for i, s in enumerate(spks)}
    for start, end, spk in segs:
        a, b = int(round(start / step)), int(round(end / step))
        m[max(a, 0) : min(b, n), idx[spk]] = True
    return m, spks


def der(ref: list[tuple[float, float, str]], hyp: list[tuple[float, float, str]], duration: float, step: float = 0.01) -> dict:
    """DER = (missed + false alarm + confusion) / total reference speech, overlap-aware."""
    from scipy.optimize import linear_sum_assignment

    n = int(np.ceil(duration / step))
    R, _ = _frames(ref, n, step)
    H, _ = _frames(hyp, n, step) if hyp else (np.zeros((n, 1), dtype=bool), [])
    # optimal one-to-one speaker mapping maximizing overlap
    overlap = R.T.astype(np.int64) @ H.astype(np.int64)
    ri, hi = linear_sum_assignment(-overlap)
    correct = np.zeros(n, dtype=np.int64)
    for r, h in zip(ri, hi):
        correct += (R[:, r] & H[:, h]).astype(np.int64)
    n_ref, n_hyp = R.sum(axis=1), H.sum(axis=1)
    total = int(n_ref.sum())
    missed = int(np.maximum(n_ref - n_hyp, 0).sum())
    fa = int(np.maximum(n_hyp - n_ref, 0).sum())
    conf = int((np.minimum(n_ref, n_hyp) - correct).sum())
    d = (missed + fa + conf) / max(total, 1)
    return {"der": d, "missed": missed / max(total, 1), "false_alarm": fa / max(total, 1), "confusion": conf / max(total, 1)}
