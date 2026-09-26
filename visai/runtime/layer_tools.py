"""Strands tools + backend for optimizing one bottleneck layer class of a real model."""

from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from pathlib import Path
from typing import Any

from strands import tool

from visai.config import ROOT
from visai.memory.store import idea_allowed, log_event
from visai.runtime.state import LoopState, source_hash

LAYER_GUIDE = """Candidate contract (layer kernel, MLX / Metal):
- A Python module defining `forward(module, *args, **kwargs)` that returns EXACTLY what the layer's own
  __call__ returns (same structure, shapes, dtypes) for the same inputs. `module` is the live layer instance:
  read its weights/attributes (module.weight, module.eps, ...) and call its CHILD modules if useful
  (e.g. module.linear1(x)). Never call module(...) or __call__ (it recurses once patched).
- Speed levers: fused ops (mx.fast.rms_norm, mx.fast.layer_norm, mx.fast.rope,
  mx.fast.scaled_dot_product_attention), mx.compile of pure array helpers (define the helper at module level
  and pass weights as arguments), custom kernels via mx.fast.metal_kernel, removing transposes / copies /
  redundant casts, fusing elementwise epilogues, fewer kernel launches.
- Imports allowed: mlx.*, math, functools, typing, and helpers from mlx_lm.models / mlx_audio / parakeet_mlx.
- No file / OS / network access, no global mutable state, no caching of outputs.
- Gate 1 compares against the layer's own output on REAL captured activations plus hidden perturbations.
"""


class LayerBackend:
    name = "mlx-layer"

    def __init__(self, adapter_key: str, cfg: dict, cls: str, captures: list[str], timeout_s: float = 420.0,
                 model_id: str | None = None):
        self.adapter_key, self.cfg, self.cls, self.captures, self.timeout_s = adapter_key, cfg, cls, captures, timeout_s
        self.model_id = model_id

    def _run(self, extra: list[str]) -> dict[str, Any]:
        env = dict(os.environ, VISAI_HIDDEN_SEED=str(secrets.randbelow(2**31 - 1) + 1))
        cmd = [sys.executable, "-m", "visai.verify.run_module", "--adapter", self.adapter_key,
               "--config", json.dumps(self.cfg), "--cls", self.cls, "--captures", ",".join(self.captures), *extra]
        if self.model_id:
            cmd += ["--model-id", self.model_id]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s, env=env, cwd=ROOT)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "error": f"exceeded {self.timeout_s:.0f}s"}
        out = proc.stdout
        i = out.find("{")
        if i < 0:
            return {"status": "crash", "error": (proc.stderr or "")[-1500:]}
        try:
            return json.loads(out[i:])
        except json.JSONDecodeError:
            return {"status": "crash", "error": out[-800:]}

    def evaluate(self, op: str, candidate: Path, dtype: str = "", mode: str = "all") -> dict[str, Any]:
        return self._run(["--candidate", str(candidate), "--mode", mode])

    def baseline(self) -> dict[str, Any]:
        return self._run(["--mode", "bench"]).get("bench", {})


def make_layer_tools(state: LoopState, spec: dict[str, Any], best: dict[str, Any]) -> list:
    """`best` is mutated: {"robust_speedup": float, "label": str, "path": str}."""

    @tool
    def layer_spec() -> dict:
        """Return the target layer: class source, parameters, attributes, children, captured input signatures, profile share."""
        return spec

    @tool
    def check_idea(idea: str) -> dict:
        """Check an idea against persistent do-not-repeat memory.

        Args:
            idea: label, idea class and the concrete change in one sentence.
        """
        return idea_allowed(idea, scope=state.target)

    @tool
    def run_correctness(label: str, idea_class: str, idea: str, source: str) -> dict:
        """Write a layer candidate and run Gate 1 against the model's own layer on real captured activations + hidden perturbations.

        Args:
            label: snake_case file stem unique per idea.
            idea_class: fusion | fewer_launches | vectorize | reduction_strategy | precision | layout | other.
            idea: what changes and why it should be faster.
            source: full Python module defining forward(module, *args, **kwargs).
        """
        slot = state.slot
        slot.labels.add(label)
        path = state.path_for(label)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
        res = state.backend.evaluate(state.op, path, mode="gate")
        g = res.get("correctness") or {"status": res.get("status", "crash"), "error": res.get("error")}
        if g.get("status") == "pass":
            slot.passed[label] = source_hash(source)
        slot.last_correctness = {"label": label, "cls": idea_class, "idea": idea, **g}
        log_event(state.run_id, "gate1", target=state.target, label=label, status=g.get("status"),
                  n_pass=g.get("n_pass"), n=g.get("n"), attempt=slot.correctness_attempts)
        keep = ("status", "n", "n_pass", "n_fail", "n_hidden_fail", "hidden_failure_kinds", "visible_failures",
                "first_error", "problems", "error")
        out = {k: g.get(k) for k in keep if g.get(k) not in (None, [], "")}
        out["attempts_left"] = state.max_attempts_per_slot - slot.correctness_attempts
        return out

    @tool
    def run_benchmark(label: str) -> dict:
        """Benchmark a candidate that passed Gate 1 against the original layer, twice (confirmation).

        Args:
            label: the label that passed run_correctness.
        """
        slot = state.slot
        path = state.path_for(label)
        runs = []
        for _ in range(2):
            b = state.backend.evaluate(state.op, path, mode="bench").get("bench")
            if not b:
                return {"status": "error", "error": "benchmark failed"}
            runs.append(b)
        sp = [r["runtime_us"] / r["candidate_us"] for r in runs]
        robust = min(sp)
        improved = robust > 1.0 and robust > best.get("robust_speedup", 1.0)
        attempt = slot.last_correctness or {}
        result = {
            "label": label, "cls": attempt.get("cls"), "idea": attempt.get("idea"), "correctness": "pass", "n_errors": 0,
            "throughput": state.stock["throughput"] * robust, "unit": state.stock.get("unit"),
            "speedup": sp[0], "confirm_speedup": sp[1], "robust_speedup": robust,
            "confirmed": robust > 1.0, "improved_over_best": improved,
            "latency_us": runs[0]["candidate_us"], "baseline_us": runs[0]["runtime_us"],
            "timed_output_ok": all(r.get("timed_output_ok") for r in runs),
            "per_shape": runs[0]["shapes"], "source": path.read_text(),
        }
        if not result["timed_output_ok"]:
            result.update({"correctness": "fail", "n_errors": 1, "confirmed": False, "improved_over_best": False})
        if result["improved_over_best"]:
            from visai.memory.kernels import register_kernel

            kid = register_kernel(
                result["source"], status="layer_winner", kind="layer", adapter=spec.get("adapter"), model=spec.get("model"),
                layer=spec.get("name"), layer_class=spec.get("class"), hardware=state.target.split("@mlx:")[-1],
                dtype=state.dtype, label=label, speedup=robust, run_id=state.run_id, target=state.target,
                captured_inputs=spec.get("captured_inputs"),
            )
            best.update({"robust_speedup": robust, "label": label, "path": str(path), "kernel_id": kid})
        slot.result = result
        log_event(state.run_id, "benchmark", target=state.target, label=label, speedup=round(robust, 4),
                  confirmed=result["confirmed"], improved=result["improved_over_best"])
        view = {k: v for k, v in result.items() if k != "source"}
        view["current_best"] = {k: best.get(k) for k in ("label", "robust_speedup")}
        return view

    return [layer_spec, check_idea, run_correctness, run_benchmark]
