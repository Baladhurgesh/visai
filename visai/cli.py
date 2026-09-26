"""visai CLI. JSON on stdout, human output on stderr (kernel-forge tool convention)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from visai.jsonio import dump

logging.getLogger("strands").setLevel(logging.ERROR)

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Visai: self-improving model & kernel optimization agent")
memory_app = typer.Typer(no_args_is_help=True, help="Project memory (Atlas)")
skills_app = typer.Typer(no_args_is_help=True, help="Learned optimization skills")
db_app = typer.Typer(no_args_is_help=True, help="Database setup")
app.add_typer(memory_app, name="memory")
app.add_typer(skills_app, name="skills")
app.add_typer(db_app, name="db")


@app.command()
def doctor() -> None:
    """Check .env, Atlas, OpenRouter, and Metal."""
    from visai.config import settings
    from visai.memory.db import get_db
    from visai.profiler.hardware import probe

    s = settings()
    out: dict = {"hardware": probe(), "model": s.model, "reflect_model": s.reflect_model, "embed_model": s.embed_model}
    try:
        out["db"] = get_db().ping()
    except Exception as exc:  # noqa: BLE001
        out["db"] = {"error": str(exc)[:300]}
    try:
        from visai.llm.openrouter import chat

        out["openrouter"] = chat([{"role": "user", "content": "Reply with: ok"}], max_tokens=50)["text"][:20]
    except Exception as exc:  # noqa: BLE001
        out["openrouter"] = {"error": str(exc)[:300]}
    dump(out)


@db_app.command("init")
def db_init() -> None:
    """Create collections, indexes, and Atlas Vector Search indexes."""
    from visai.memory.db import get_db

    dump(get_db().ensure_indexes())


kernels_app = typer.Typer(no_args_is_help=True, help="Kernel registry (winning / integrated kernels in Atlas)")
app.add_typer(kernels_app, name="kernels")


@kernels_app.command("list")
def kernels_list(model: Optional[str] = None) -> None:
    from visai.memory.kernels import list_kernels

    dump(list_kernels(model))


@kernels_app.command("backfill")
def kernels_backfill() -> None:
    """Register winners recorded before the registry existed."""
    from visai.memory.kernels import backfill

    dump(backfill())


@app.command()
def export(model: str = typer.Option("qwen", help="qwen | parakeet | diar"), out: Optional[Path] = None) -> None:
    """Bundle the latest optimized config + its kernels (pulled from Atlas) into a portable folder."""
    import json as _json

    from visai.memory.db import get_db
    from visai.memory.kernels import kernel_id, materialize

    db = get_db()
    cmp_ = db.comparisons.find_one({"adapter": model}, sort=[("ts", -1)])
    if not cmp_:
        raise typer.BadParameter(f"no comparison stored for {model}; run the pipeline first")
    out = out or Path("out/export") / model
    (out / "kernels").mkdir(parents=True, exist_ok=True)
    cfg = dict(cmp_["final_config"])
    patches = []
    for p in cfg.get("patches") or []:
        kid = p.get("kernel_id")
        if not kid and p.get("candidate") and Path(p["candidate"]).exists():
            kid = kernel_id(Path(p["candidate"]).read_text())
        src_path = materialize(kid) if kid and db.kernels.find_one({"kernel_id": kid}) else p.get("candidate")
        dest = out / "kernels" / f"{p['class'].split(':')[-1]}_{kid}.py"
        dest.write_text(Path(src_path).read_text())
        patches.append({**p, "kernel_id": kid, "candidate": str(dest)})
    cfg["patches"] = patches
    manifest = {"model": cmp_["model"], "adapter": model, "hardware": cmp_.get("hardware"), "config": cfg,
                "unoptimized": cmp_["unoptimized"], "optimized": cmp_["optimized"], "speedup": cmp_["speedup"],
                "quality_metric": cmp_.get("quality_metric"),
                "apply": f"uv run visai compare --model {model} --config \"$(jq -c .config manifest.json)\""}
    (out / "manifest.json").write_text(_json.dumps(manifest, indent=2, default=str))
    dump({"status": "ok", "dir": str(out), "kernels": [p["kernel_id"] for p in patches], "speedup": cmp_["speedup"]})


@db_app.command("push-local")
def db_push_local() -> None:
    """Copy everything recorded in the local fallback (out/localdb) into Atlas (idempotent by _id)."""
    from visai.memory.db import COLLECTIONS, LocalCollection, get_db

    db = get_db()
    if db.backend != "atlas":
        dump({"status": "error", "error": "Atlas not reachable; fix MONGODB_URI first", "atlas_error": db.atlas_error})
        raise typer.Exit(1)
    counts = {}
    for name in COLLECTIONS:
        rows = LocalCollection(db._root / f"{name}.jsonl").find()
        n = 0
        for r in rows:
            db._db[name].replace_one({"_id": r["_id"]}, r, upsert=True)
            n += 1
        counts[name] = n
    dump({"status": "ok", "pushed": counts, "indexes": db.ensure_indexes()})


@app.command()
def gate(op: str, candidate: Path, dtype: str = "float16") -> None:
    """Gate 1: correctness of a candidate vs the reference (seeded + hidden inputs)."""
    from visai.backends.mlx_metal import MLXMetalBackend

    res = MLXMetalBackend().evaluate(op, candidate, dtype, mode="gate")
    dump(res)
    raise typer.Exit(0 if (res.get("correctness") or {}).get("status") == "pass" else 1)


@app.command()
def bench(op: str, candidate: Optional[Path] = None, dtype: str = "float16") -> None:
    """Microbenchmark reference, runtime baseline, and (optionally) a candidate."""
    from visai.backends.mlx_metal import MLXMetalBackend

    b = MLXMetalBackend()
    dump(b.evaluate(op, candidate, dtype, mode="all") if candidate else {"bench": b.baseline(op, dtype)})


@app.command()
def kernelbench(
    ops: str = typer.Option("add_rmsnorm", help="comma-separated op names"),
    budget: int = typer.Option(4, help="candidates per op (kernel-forge used 10)"),
    baseline: str = typer.Option("runtime", help="runtime (what mlx-lm runs) or reference (naive eager)"),
    dtype: str = "float16",
    rel: float = typer.Option(0.03, help="required speedup over stock (0.03 = 3%)"),
) -> None:
    """Run the self-improving kernel loop over KernelBench-style ops."""
    from rich.console import Console

    from visai.loop import optimize_op
    from visai.memory.store import GateConfig
    from visai.report import print_kernel_summary

    con = Console(stderr=True)

    def on_event(step: str, data: dict) -> None:
        con.print(f"[dim]{step}[/dim] {data}")

    results = []
    for op in [o.strip() for o in ops.split(",") if o.strip()]:
        summary = optimize_op(op, dtype=dtype, budget=budget, baseline_mode=baseline,
                              gate=GateConfig(rel=rel), on_event=on_event)
        print_kernel_summary(summary)
        results.append(summary)
    dump({"results": results})


@app.command()
def profile(model: str = "Qwen/Qwen3-0.6B") -> None:
    """ProfileBrief for an MLX model's decode step."""
    from visai.models.mlx_model import load
    from visai.profiler.brief import mlx_profile_brief

    m, tok = load(model)
    brief = mlx_profile_brief(m, tok, model)
    table = brief.pop("table", "")
    dump(brief, None, table)


@app.command("eval")
def eval_cmd(
    model: str = "Qwen/Qwen3-0.6B",
    dataset: str = "gsm8k",
    n: int = 10,
    shots: int = 4,
    base: Optional[str] = typer.Option(None, help="OpenAI-compatible server base; in-process if omitted"),
) -> None:
    """Gate 2 frozen eval (GSM8K / jsonl / HF dataset)."""
    from visai.verify.task_eval import format_table, run_eval_http, run_eval_inprocess

    if base:
        brief = run_eval_http(base, model, dataset=dataset, n=n, shots=shots)
    else:
        from visai.models.mlx_model import load

        m, tok = load(model)
        brief = run_eval_inprocess(m, tok, dataset=dataset, n=n, shots=shots)
    dump(brief, None, format_table(brief["metrics"], dataset))


@app.command()
def optimize(
    model: str = typer.Option("Qwen/Qwen3-0.6B"),
    hardware: str = typer.Option("", help="label only; detected automatically"),
    workload: str = typer.Option("decode"),
    quality_budget: str = typer.Option("0.5%", help="allowed relative quality drop, e.g. 0.5%"),
    budget: int = typer.Option(6, help="deployment candidates"),
    kernels: str = typer.Option("add_rmsnorm", help="ops whose promoted kernels are tried inside the model"),
    full_eval: bool = typer.Option(False, help="also run the GSM8K slice on baseline and winner"),
) -> None:
    """End-to-end: deployment-config search + promoted kernels, gated on task quality."""
    from visai.deploy.optimize import optimize_model
    from visai.report import print_optimize_report

    qb = float(quality_budget.strip().rstrip("%")) / 100.0
    rep = optimize_model(model, workload=workload, quality_budget_rel=qb, budget=budget,
                         kernel_ops=[k for k in kernels.split(",") if k], full_eval=full_eval, hardware_label=hardware)
    print_optimize_report(rep)
    dump(rep)


@app.command()
def run(
    model: str = typer.Option("qwen", help="qwen | parakeet | diar | all"),
    top_layers: int = typer.Option(2, help="bottleneck layers to optimize"),
    max_layer_iters: int = typer.Option(6, help="hard cap on iterations per layer"),
    layer_patience: int = typer.Option(3, help="stop a layer after N non-improving iterations"),
    layer_minutes: float = typer.Option(25.0, help="time cap per layer"),
    max_config_iters: int = typer.Option(6, help="hard cap on model-level candidates"),
    config_patience: int = typer.Option(3),
    config_minutes: float = typer.Option(30.0),
    target_speedup: Optional[float] = typer.Option(None, help="stop a layer / the model level once this speedup is reached, e.g. 2.0"),
    resume_from: Optional[str] = typer.Option(None, help="reuse profile, captures and layer results from a previous run id"),
) -> None:
    """Full pipeline: baseline -> layer profile -> layer kernel loops -> integrate -> model-level -> compare."""
    from rich.console import Console

    from visai.pipeline import optimize_pipeline
    from visai.report import print_pipeline_report

    con = Console(stderr=True)

    def on_event(step: str, data: dict) -> None:
        if step == "profile_table":
            con.print(data["table"])
        else:
            con.print(f"[bold cyan]{step}[/bold cyan] {data}")

    keys = ["parakeet", "diar", "qwen"] if model == "all" else [model]
    reports = []
    for k in keys:
        rep = optimize_pipeline(k, top_layers=top_layers, max_layer_iters=max_layer_iters, layer_patience=layer_patience,
                                layer_minutes=layer_minutes, max_config_iters=max_config_iters,
                                config_patience=config_patience, config_minutes=config_minutes,
                                target_speedup=target_speedup, resume_from=resume_from, on_event=on_event)
        print_pipeline_report(rep)
        reports.append(rep)
    dump({"reports": reports})


@app.command()
def status(all_runs: bool = typer.Option(False, help="include finished pipeline runs")) -> None:
    """Progress of pipeline runs: stage, current layer, best speedups, stop reasons."""
    from visai.memory.db import get_db

    db = get_db()
    flt = {"kind": "pipeline"} if all_runs else {"kind": "pipeline", "status": "running"}
    out = []
    for r in db.runs.find(flt, sort=[("run_id", 1)]):
        rid = r["run_id"]
        ev = db.events.find({"run_id": rid}, sort=[("ts", 1)])
        steps = [e for e in ev if e["step"] in ("baseline", "profile", "layer_baseline", "benchmark", "integration",
                                                  "deploy_eval", "saturated", "target_reached", "writer_error")]
        layers: dict = {}
        cur = None
        for e in steps:
            if e["step"] == "layer_baseline":
                cur = e.get("layer")
                layers[cur] = {"original_us": round(float(e.get("runtime_us") or 0), 2), "best": 1.0, "benchmarks": 0, "stop": None}
            elif e["step"] == "benchmark" and cur in layers and "layers" not in str(e.get("target", "")) and e.get("improved") is not None:
                layers[cur]["benchmarks"] += 1
                if str(e.get("improved")) == "True":
                    layers[cur]["best"] = max(layers[cur]["best"], float(e.get("speedup") or 1))
            elif e["step"] in ("saturated", "target_reached") and cur in layers and ":" + cur + "@" in str(e.get("target", "")):
                layers[cur]["stop"] = e.get("reason") or f"target reached {e.get('speedup')}"
        stage = "baseline"
        names = [e["step"] for e in steps]
        if "profile" in names:
            stage = "layer loops"
        if "integration" in names:
            stage = "integration"
        if "deploy_eval" in names:
            stage = "model level"
        base = next((e.get("stock") for e in steps if e["step"] == "baseline"), None)
        out.append({"run_id": rid, "status": r.get("status"), "stage": stage, "current_layer": cur, "baseline": base,
                    "layers": layers, "last_event": ev[-1]["ts"] if ev else None,
                    "writer_errors": sum(1 for e in steps if e["step"] == "writer_error"),
                    "report": {k: (r.get("report") or {}).get(k) for k in ("comparison", "integration", "model_level")} if r.get("report") else None})
    dump(out)


@app.command()
def compare(model: str = typer.Option("qwen"), config: str = typer.Option("{}", help="optimized config JSON (incl. patches)")) -> None:
    """Head-to-head: unoptimized vs optimized on the same data points on this machine."""
    import json as _json

    from visai.models.adapters import get_adapter
    from visai.pipeline import compare as _compare

    dump(_compare(get_adapter(model), _json.loads(config)))


@app.command("layer-profile")
def layer_profile(model: str = typer.Option("qwen")) -> None:
    """Exclusive time per layer class on a real forward pass (no optimization)."""
    from visai.models.adapters import get_adapter
    from visai.profiler.layers import format_table, profile_layers

    a = get_adapter(model)
    h = a.load({})
    prof = profile_layers(a.root(h), lambda: a.profile_run(h, {}))
    dump({"model": a.model_id, "layers": prof["layers"][:20], "wall_s": prof["wall_s"]}, None, format_table(prof))


speech_app = typer.Typer(no_args_is_help=True, help="Case study 2: Parakeet ASR + Nemotron-3 diarization on Mac")
app.add_typer(speech_app, name="speech")


@speech_app.command("prepare")
def speech_prepare(n_librispeech: int = 60, n_meetings: int = 2, crop_s: float = 600.0) -> None:
    """Cache LibriSpeech (WER) and AMI (DER + meeting workload) slices under out/data."""
    from visai.speech.data import load_manifest, prepare_ami, prepare_librispeech

    ls = prepare_librispeech(n_librispeech)
    ami = prepare_ami(n_meetings, crop_s)
    dump({"librispeech": str(ls), "n_librispeech": len(load_manifest(ls)), "ami": str(ami),
          "meetings": [{"duration": m["duration"], "n_segments": len(m["segments"])} for m in load_manifest(ami)]})


@speech_app.command("baseline")
def speech_baseline(target: str = typer.Option("both", help="asr | diar | both")) -> None:
    """Measure stock RTFx + WER/DER for Parakeet and/or Nemotron-3-Diarization."""
    from visai.speech import asr, diar

    out = {}
    if target in ("asr", "both"):
        out["asr"] = asr.describe(asr.evaluate({}))
    if target in ("diar", "both"):
        out["diar"] = diar.describe(diar.evaluate({}))
    dump(out)


@speech_app.command("optimize")
def speech_optimize(
    target: str = typer.Option("both", help="asr | diar | both"),
    budget: int = typer.Option(4, help="config candidates per model"),
    wer_budget: float = typer.Option(0.003, help="allowed absolute WER increase (0.003 = +0.3 pts)"),
    der_budget: float = typer.Option(0.005, help="allowed absolute DER increase"),
) -> None:
    """Gated deployment search for the meeting pipeline (shared agent loop + Atlas memory)."""
    from visai.speech.optimize import optimize_speech

    targets = ["asr", "diar"] if target == "both" else [target]
    dump(optimize_speech(targets, budget=budget, wer_budget_abs=wer_budget, der_budget_abs=der_budget))


@app.command()
def report(run_id: str) -> None:
    """Show a stored run."""
    from visai.memory.db import get_db, strip_ids

    db = get_db()
    run = db.runs.find_one({"run_id": run_id})
    trials = strip_ids(db.experiments.find({"run_id": run_id}, sort=[("ts", 1)]))
    for t in trials:
        (t.get("candidate") or {}).pop("source", None)
    dump({"run": strip_ids([run])[0] if run else None, "trials": trials})


@app.command("export-hermes")
def export_hermes() -> None:
    """Write Hermes/NemoClaw tool specs (integrations/hermes) from Visai's tool functions."""
    from visai.integrations_export import export

    dump(export())


@memory_app.command("show")
def memory_show(target: Optional[str] = None) -> None:
    from visai.memory.store import show

    dump(show(target))


@memory_app.command("allowed")
def memory_allowed(idea: str, target: Optional[str] = None) -> None:
    from visai.memory.store import idea_allowed

    v = idea_allowed(idea, scope=target)
    dump(v)
    raise typer.Exit(0 if v["allowed"] else 1)


@memory_app.command("refresh")
def memory_refresh(target: Optional[str] = None) -> None:
    from visai.memory.store import refresh_lessons

    text = refresh_lessons(target)
    dump({"status": "ok", "chars": len(text)})


@memory_app.command("sync-hermes")
def memory_sync_hermes() -> None:
    from visai.memory.store import sync_hermes

    dump({"status": "ok", **sync_hermes()})


@skills_app.command("list")
def skills_list() -> None:
    from visai.skills.update import list_skills

    rows = list_skills()
    for r in rows:
        r.pop("example_source", None)
    dump(rows)


if __name__ == "__main__":
    app()
