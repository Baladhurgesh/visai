"""MLX / Metal backend: candidates are Python modules defining kernel(...) with mx.fast.metal_kernel."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

from visai.config import ROOT
from visai.profiler.hardware import probe

METAL_GUIDE = """Candidate contract (MLX / Metal):
- A Python module defining `kernel(...)` with EXACTLY the op signature. Imports allowed: mlx.core, mlx.nn,
  math, functools, typing. No file/network/OS access, no global mutable state, no caching of outputs.
- Build custom kernels with mx.fast.metal_kernel(name, input_names, output_names, source, header=...).
  `source` is the Metal function BODY; the signature is generated. Referencing `<input>_shape`,
  `thread_position_in_grid`, `threadgroup_position_in_grid`, `thread_position_in_threadgroup`,
  `threads_per_threadgroup`, `thread_index_in_simdgroup`, `simdgroup_index_in_threadgroup` makes them available.
- Call: outs = K(inputs=[...], template=[("T", x.dtype)], grid=(total_threads,1,1), threadgroup=(tg,1,1),
  output_shapes=[...], output_dtypes=[...]). grid counts THREADS (dispatchThreads), not threadgroups.
- Scalars (eps) must be passed as small mx.array inputs. Create kernels ONCE at module import, not per call.
- Use simd_sum / threadgroup memory for reductions; accumulate in float; cast back to T on store.
- Per-call Python overhead matters at decode shapes (rows=1): avoid allocating arrays per call. Constant
  arrays (e.g. a default eps) may be built once at import; never cache inputs or outputs.
- mx.compile is allowed for elementwise fusion of plain mx ops.
"""


class MLXMetalBackend:
    name = "mlx"
    candidate_language = "python+metal"

    def __init__(self, timeout_s: float = 180.0) -> None:
        self.timeout_s = timeout_s
        self.hardware = probe().get("chip", "apple-silicon")

    def _run(self, args: list[str]) -> dict[str, Any]:
        env = dict(os.environ)
        env["VISAI_HIDDEN_SEED"] = str(secrets.randbelow(2**31 - 1) + 1)
        cmd = [sys.executable, "-m", "visai.verify.run_candidate", *args]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s, env=env, cwd=ROOT)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": f"candidate exceeded {self.timeout_s:.0f}s (hang or runaway grid)"}
        out = proc.stdout.strip()
        start = out.find("{")
        if start < 0:
            tail = (proc.stderr or "")[-1500:]
            return {"status": "crash", "error": f"exit={proc.returncode}: {tail}"}
        try:
            payload = json.loads(out[start:])
        except json.JSONDecodeError:
            return {"status": "crash", "error": f"unparseable output: {out[-800:]}"}
        payload["exit_code"] = proc.returncode
        if proc.returncode and "correctness" not in payload:
            payload["status"] = "crash"
            payload["error"] = (proc.stderr or "")[-1500:]
        return payload

    def evaluate(self, op: str, candidate: Path, dtype: str = "float16", mode: str = "all") -> dict[str, Any]:
        return self._run(["--op", op, "--candidate", str(candidate), "--dtype", dtype, "--mode", mode])

    def baseline(self, op: str, dtype: str = "float16") -> dict[str, Any]:
        return _baseline_cached(op, dtype, self.timeout_s)


@lru_cache(maxsize=64)
def _baseline_cached(op: str, dtype: str, timeout_s: float) -> dict[str, Any]:
    cmd = [sys.executable, "-m", "visai.verify.run_candidate", "--op", op, "--dtype", dtype, "--mode", "bench"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=ROOT)
    out = proc.stdout
    return json.loads(out[out.find("{") :]).get("bench", {})
