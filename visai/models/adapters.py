"""Model adapters: one interface for Qwen3 (LLM), Parakeet (ASR), Nemotron-3 diarization.

Each adapter knows how to load the model under a deployment config, run the end-to-end workload on
fixed data points (latency / throughput), measure task quality, and run a short representative pass
for layer profiling. `cfg["patches"]` (layer-kernel replacements) is applied on top of any config.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx


def free() -> None:
    gc.collect()
    mx.clear_cache()


@dataclass
class Handle:
    model: Any
    extra: Any = None


class Adapter:
    key: str
    model_id: str
    task: str
    quality_name: str  # e.g. perplexity / WER / DER (lower is better; quality = -value)
    unit: str
    dtype: str = "bfloat16"
    search_space_doc: str = ""

    # -- to implement
    def load(self, cfg: dict[str, Any]) -> Handle: ...
    def workload(self, h: Handle, cfg: dict[str, Any], runs: int = 2) -> dict[str, Any]: ...
    def quality(self, h: Handle, cfg: dict[str, Any]) -> dict[str, Any]: ...
    def profile_run(self, h: Handle, cfg: dict[str, Any]) -> None: ...
    def validate(self, cfg: dict[str, Any]) -> list[str]: ...

    # -- shared
    def root(self, h: Handle):
        return h.model

    def model_cfg(self, cfg: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in (cfg or {}).items() if k != "patches"}

    def evaluate(self, cfg: dict[str, Any], *, with_quality: bool = True, runs: int = 2) -> dict[str, Any]:
        from visai.backends.module_patch import apply_patches

        h = self.load(self.model_cfg(cfg))
        patcher = None
        try:
            patcher = apply_patches(h, self, cfg.get("patches") or [])
            mx.reset_peak_memory()
            m = self.workload(h, cfg, runs=runs)
            m["peak_memory_gb"] = mx.get_peak_memory() / 2**30
            if with_quality:
                m.update(self.quality(h, cfg))
            m["n_errors"] = 0
            return m
        finally:
            if patcher:
                patcher.revert()
            del h
            free()

    def speed_of(self, cfg: dict[str, Any]) -> float:
        return self.evaluate(cfg, with_quality=False)["throughput"]

    def paired_speed(self, cfg_a: dict[str, Any], cfg_b: dict[str, Any], rounds: int = 3) -> dict[str, Any]:
        """Interleaved A/B/A/B throughput so both configs see the same machine load.

        Both models stay loaded; layer patches are class-level, so each side's patches are applied only
        around its own runs. Returns per-round throughputs, median B/A ratio, and how many rounds B won.
        """
        from visai.backends.module_patch import apply_patches

        ha = self.load(self.model_cfg(cfg_a))
        hb = ha if self.model_cfg(cfg_a) == self.model_cfg(cfg_b) else self.load(self.model_cfg(cfg_b))
        a_s, b_s = [], []
        try:
            for i in range(rounds + 1):  # round 0 warms both up
                for h, cfg, out in ((ha, cfg_a, a_s), (hb, cfg_b, b_s)):
                    ps = apply_patches(h, self, cfg.get("patches") or [])
                    try:
                        thr = self.workload(h, cfg, runs=1)["throughput"]
                    finally:
                        ps.revert()
                    if i:
                        out.append(thr)
        finally:
            del ha, hb
            free()
        ratios = sorted(b / a for a, b in zip(a_s, b_s))
        return {"a": a_s, "b": b_s, "ratio": ratios[len(ratios) // 2], "wins": sum(r > 1.0 for r in ratios),
                "rounds": rounds, "a_median": sorted(a_s)[len(a_s) // 2], "b_median": sorted(b_s)[len(b_s) // 2]}

    def paired_faster(self, cfg_a: dict[str, Any], cfg_b: dict[str, Any], rounds: int = 3) -> tuple[bool, dict]:
        p = self.paired_speed(cfg_a, cfg_b, rounds)
        return (p["ratio"] > 1.0 and p["wins"] * 2 > p["rounds"]), p

    def describe(self, m: dict[str, Any]) -> dict[str, Any]:
        keys = ("throughput", "latency_s", "unit", "peak_memory_gb", self.quality_name.lower(), "prefill_tok_s")
        return {k: (round(v, 5) if isinstance(v, float) else v) for k, v in m.items() if k in keys}


# ------------------------------------------------------------------ Qwen3 (LLM decode)


class QwenAdapter(Adapter):
    """Any mlx-lm causal LM (Qwen, Llama, Mistral, ...). `key` names the model in memory and the dashboard."""

    key = "qwen"
    task = "causal-lm"
    quality_name = "perplexity"
    unit = "decode tok/s"

    def __init__(self, model_id: str = "Qwen/Qwen3-0.6B", key: str | None = None):
        from visai.deploy.mlx_search import SEARCH_SPACE_DOC

        self.model_id = model_id
        if key:
            self.key = key
        self.search_space_doc = SEARCH_SPACE_DOC.replace(
            "- kernels: list of promoted kernel ops to patch in, e.g. [\"add_rmsnorm\"] (only ops with a promoted winner)\n", ""
        )

    def load(self, cfg):
        from visai.models.mlx_model import load

        quant = {"bits": cfg.get("bits"), "group_size": cfg.get("group_size", 64), "keep_high": cfg.get("keep_high")}
        model, tok = load(self.model_id, quant if cfg.get("bits") else None)
        return Handle(model, tok)

    def workload(self, h, cfg, runs=2):
        from visai.deploy.mlx_search import gen_kwargs
        from visai.models.mlx_model import measure_decode

        s = measure_decode(h.model, h.extra, input_tokens=256, output_tokens=128, runs=max(runs, 2),
                           gen_kwargs=gen_kwargs(cfg))
        tps = s["median_decode_tok_s"]
        return {"throughput": tps, "unit": self.unit, "latency_s": 1.0 / tps,
                "prefill_tok_s": s["median_prefill_tok_s"]}

    def quality(self, h, cfg):
        from visai.verify.perplexity import perplexity

        q = perplexity(h.model, h.extra)
        return {"perplexity": q["perplexity"], "quality": q["quality"]}

    def profile_run(self, h, cfg):
        from mlx_lm.models.cache import make_prompt_cache

        from visai.models.mlx_model import DEFAULT_PROMPT, _prompt_tokens

        cache = make_prompt_cache(h.model)
        ids = mx.array(_prompt_tokens(h.extra, DEFAULT_PROMPT, 128))[None]
        logits = h.model(ids, cache=cache)
        tok = mx.argmax(logits[:, -1, :], axis=-1)
        for _ in range(24):
            logits = h.model(tok[:, None], cache=cache)
            tok = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(tok)

    def validate(self, cfg):
        from visai.deploy.mlx_search import validate

        return validate(self.model_cfg(cfg))


# ------------------------------------------------------------------ Parakeet (ASR)


class ParakeetAdapter(Adapter):
    key = "parakeet"
    task = "automatic-speech-recognition"
    quality_name = "WER"
    unit = "x real-time (10-min meeting)"

    def __init__(self):
        from visai.speech import asr

        self.asr = asr
        self.model_id = asr.PARAKEET_ID
        self.search_space_doc = asr.SEARCH_SPACE_DOC

    def _cfg(self, cfg):
        return {**self.asr.STOCK, **self.model_cfg(cfg)}

    def load(self, cfg):
        return Handle(self.asr.load(self._cfg(cfg)))

    def workload(self, h, cfg, runs=2):
        rtfx = self.asr.speed(h.model, self._cfg(cfg), runs=runs)
        from visai.speech.data import load_manifest, prepare_ami

        dur = load_manifest(prepare_ami())[0]["duration"]
        return {"throughput": rtfx, "unit": self.unit, "latency_s": dur / rtfx}

    def quality(self, h, cfg):
        from visai.speech.data import load_manifest, prepare_librispeech
        from visai.speech.metrics import wer

        c = self._cfg(cfg)
        rows = load_manifest(prepare_librispeech())[:40]
        w = wer([r["text"] for r in rows], [self.asr._transcribe(h.model, r["path"], c) for r in rows])
        return {"wer": w, "quality": -w}

    def profile_run(self, h, cfg):
        from visai.speech.data import load_manifest, prepare_librispeech

        rows = load_manifest(prepare_librispeech())[:3]
        for r in rows:
            self.asr._transcribe(h.model, r["path"], self._cfg(cfg))

    def validate(self, cfg):
        return self.asr.validate(self.model_cfg(cfg))


# ------------------------------------------------------------------ Nemotron-3 diarization


class DiarAdapter(Adapter):
    key = "diar"
    task = "speaker-diarization"
    quality_name = "DER"
    unit = "x real-time (AMI meetings)"
    dtype = "float32"

    def __init__(self):
        from visai.speech import diar

        self.diar = diar
        self.model_id = diar.DIAR_ID
        self.search_space_doc = diar.SEARCH_SPACE_DOC

    def _cfg(self, cfg):
        return {**self.diar.STOCK, **self.model_cfg(cfg)}

    def load(self, cfg):
        from visai.speech.data import load_audio, load_manifest, prepare_ami

        meetings = load_manifest(prepare_ami())
        return Handle(self.diar.load(self._cfg(cfg)), (meetings, [load_audio(m["path"]) for m in meetings]))

    def workload(self, h, cfg, runs=2):
        meetings, audios = h.extra
        c = self._cfg(cfg)
        self.diar._diarize(h.model, audios[0][: 16000 * 60], c)
        best = None
        for _ in range(runs):
            t0 = time.perf_counter()
            for a in audios:
                self.diar._diarize(h.model, a, c)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        total = sum(m["duration"] for m in meetings)
        return {"throughput": total / best, "unit": self.unit, "latency_s": best}

    def quality(self, h, cfg):
        from visai.speech.metrics import der

        meetings, audios = h.extra
        c = self._cfg(cfg)
        ds = [der([tuple(s) for s in m["segments"]], self.diar._diarize(h.model, a, c), m["duration"])["der"]
              for m, a in zip(meetings, audios)]
        d = sum(ds) / len(ds)
        return {"der": d, "quality": -d}

    def profile_run(self, h, cfg):
        _, audios = h.extra
        self.diar._diarize(h.model, audios[0][: 16000 * 90], self._cfg(cfg))

    def validate(self, cfg):
        return self.diar.validate(self.model_cfg(cfg))


ADAPTERS = {"qwen": QwenAdapter, "parakeet": ParakeetAdapter, "diar": DiarAdapter}


def llm_key(model_id: str) -> str:
    """Stable memory/dashboard key for a custom mlx-lm model, e.g. Qwen/Qwen3-1.7B -> llm-qwen3-1-7b."""
    import re

    name = model_id.split("/")[-1].lower()
    return "llm-" + re.sub(r"[^a-z0-9]+", "-", name).strip("-")


def get_adapter(key: str, model_id: str | None = None) -> Adapter:
    if key in ADAPTERS and not (key == "qwen" and model_id and model_id != "Qwen/Qwen3-0.6B"):
        return ADAPTERS[key]()
    if key == "llm" or key.startswith("llm-") or model_id:
        if not model_id:
            raise KeyError(f"model {key} needs --model-id (a Hugging Face mlx-lm model id)")
        return QwenAdapter(model_id, key=key if key.startswith("llm-") else llm_key(model_id))
    raise KeyError(f"unknown model {key}; have {sorted(ADAPTERS)} or llm --model-id <hf id>")
