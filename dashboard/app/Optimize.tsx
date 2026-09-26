"use client";

import { useEffect, useState } from "react";

type Doc = Record<string, any>;

const MODELS = [
  { key: "qwen", label: "Qwen3-0.6B (LLM decode)", quality: "rel", qdefault: 0.01, qlabel: "perplexity increase (relative)" },
  { key: "parakeet", label: "Parakeet-TDT 0.6B v3 (ASR)", quality: "abs", qdefault: 0.003, qlabel: "WER increase (absolute)" },
  { key: "diar", label: "Nemotron-3 Diarization", quality: "abs", qdefault: 0.005, qlabel: "DER increase (absolute)" },
  { key: "llm", label: "Other mlx-lm LLM (Hugging Face id)…", quality: "rel", qdefault: 0.01, qlabel: "perplexity increase (relative)" },
];

const HARDWARE = [
  { key: "local", label: "This Mac — Apple M3 Pro, MLX / Metal", enabled: true },
  { key: "modal-cuda", label: "NVIDIA GPU via Modal — CUDA / Triton (not configured)", enabled: false },
  { key: "iphone", label: "iPhone / Core ML (not configured)", enabled: false },
];

function Field({ label, children, hint }: { label: string; children: any; hint?: string }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint ? <small className="muted">{hint}</small> : null}
    </label>
  );
}

export default function Optimize({ onOpenModel }: { onOpenModel: (key: string) => void }) {
  const [model, setModel] = useState("qwen");
  const [modelId, setModelId] = useState("Qwen/Qwen3-1.7B");
  const [hardware, setHardware] = useState("local");
  const [target, setTarget] = useState("2.0");
  const [quality, setQuality] = useState("0.01");
  const [topLayers, setTopLayers] = useState("3");
  const [layerIters, setLayerIters] = useState("6");
  const [layerMin, setLayerMin] = useState("20");
  const [cfgIters, setCfgIters] = useState("6");
  const [cfgMin, setCfgMin] = useState("20");
  const [jobs, setJobs] = useState<Doc[]>([]);
  const [msg, setMsg] = useState<{ kind: string; text: string } | null>(null);
  const [busy, setBusy] = useState(false);
  const spec = MODELS.find((m) => m.key === model)!;

  useEffect(() => setQuality(String(spec.qdefault)), [model]); // eslint-disable-line react-hooks/exhaustive-deps

  const load = () => fetch("/api/jobs").then((r) => r.json()).then((d) => setJobs(d.jobs || []));
  useEffect(() => {
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, []);

  async function submit(force = false) {
    setBusy(true);
    setMsg(null);
    const body: Doc = {
      model, model_id: modelId, hardware, target_speedup: target, top_layers: topLayers, max_layer_iters: layerIters,
      layer_minutes: layerMin, max_config_iters: cfgIters, config_minutes: cfgMin, force,
      [spec.quality === "rel" ? "quality_budget_rel" : "quality_budget_abs"]: quality,
    };
    const r = await fetch("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const d = await r.json();
    setBusy(false);
    if (r.status === 409 && d.running) {
      if (window.confirm(`${d.error}\n\nStart anyway?`)) return submit(true);
      return;
    }
    if (!r.ok) return setMsg({ kind: "bad", text: d.error || "failed to start" });
    setMsg({ kind: "good", text: `Started ${d.label} on ${d.hardware}: ${d.command}` });
    load();
  }

  async function stop(id: string) {
    await fetch(`/api/jobs?job_id=${encodeURIComponent(id)}`, { method: "DELETE" });
    load();
  }

  const worst = Number(topLayers) * Number(layerMin) + Number(cfgMin) + 10;

  return (
    <>
      <div className="panel">
        <h2>Optimize a model</h2>
        <p className="muted" style={{ marginTop: 0 }}>
          Runs the full Visai pipeline: baseline → layer profile → layer kernel loops → integrate (paired A/B) → model-level
          search → unoptimized vs optimized. Results appear live in the model's tab and in MongoDB Atlas.
        </p>
        <div className="form">
          <Field label="Model">
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              {MODELS.map((m) => <option key={m.key} value={m.key}>{m.label}</option>)}
            </select>
          </Field>
          {model === "llm" && (
            <Field label="Hugging Face model id" hint="any mlx-lm causal LM, e.g. Qwen/Qwen3-1.7B, mlx-community/Llama-3.2-1B-Instruct-4bit">
              <input value={modelId} onChange={(e) => setModelId(e.target.value)} placeholder="org/model" />
            </Field>
          )}
          <Field label="Hardware">
            <select value={hardware} onChange={(e) => setHardware(e.target.value)}>
              {HARDWARE.map((h) => <option key={h.key} value={h.key} disabled={!h.enabled}>{h.label}</option>)}
            </select>
          </Field>
          <Field label="Target speedup" hint="each layer / the model level stops once reached">
            <input type="number" step="0.1" min="1.05" value={target} onChange={(e) => setTarget(e.target.value)} />
          </Field>
          <Field label={`Quality budget: ${spec.qlabel}`} hint={spec.quality === "rel" ? "0.01 = +1%" : "0.003 = +0.3 points"}>
            <input type="number" step="0.001" min="0" value={quality} onChange={(e) => setQuality(e.target.value)} />
          </Field>
          <Field label="Bottleneck layers to optimize">
            <input type="number" min="1" max="8" value={topLayers} onChange={(e) => setTopLayers(e.target.value)} />
          </Field>
          <Field label="Max iterations per layer">
            <input type="number" min="1" max="20" value={layerIters} onChange={(e) => setLayerIters(e.target.value)} />
          </Field>
          <Field label="Minutes per layer (cap)">
            <input type="number" min="1" max="120" value={layerMin} onChange={(e) => setLayerMin(e.target.value)} />
          </Field>
          <Field label="Max model-level configs">
            <input type="number" min="1" max="20" value={cfgIters} onChange={(e) => setCfgIters(e.target.value)} />
          </Field>
          <Field label="Model-level minutes (cap)">
            <input type="number" min="1" max="120" value={cfgMin} onChange={(e) => setCfgMin(e.target.value)} />
          </Field>
        </div>
        <div style={{ display: "flex", gap: 12, alignItems: "center", marginTop: 12 }}>
          <button className="primary" disabled={busy || (model === "llm" && !modelId.includes("/"))} onClick={() => submit(false)}>
            {busy ? "Starting…" : "Start optimization"}
          </button>
          <span className="muted">worst case ≈ {worst} min (usually less: target, saturation and patience stop earlier)</span>
        </div>
        {msg && <div className={`insight ${msg.kind === "bad" ? "bad-insight" : ""}`} style={{ marginTop: 10 }}>{msg.text}</div>}
      </div>

      <div className="panel">
        <h2>Jobs</h2>
        {jobs.length === 0 && <div className="muted">No jobs started from the dashboard yet.</div>}
        {jobs.map((j) => (
          <div key={j.job_id} className="job">
            <div style={{ display: "flex", gap: 10, alignItems: "baseline", flexWrap: "wrap" }}>
              <b className="mono">{j.label}</b>
              <span className="muted">{j.hardware}</span>
              <span className={`pill ${j.status === "running" ? "warn" : j.status === "done" ? "good" : "bad"}`}>{j.status}</span>
              <span className="muted mono">{j.created?.slice(11, 19)}</span>
              <span style={{ marginLeft: "auto", display: "flex", gap: 8 }}>
                <button onClick={() => onOpenModel(j.adapter_key)}>Open model tab</button>
                {j.status === "running" && <button onClick={() => stop(j.job_id)}>Stop</button>}
              </span>
            </div>
            <div className="mono muted" style={{ fontSize: 11, margin: "4px 0" }}>{j.command}</div>
            <details open={j.status === "running"}>
              <summary>log</summary>
              <pre className="lessons" style={{ maxHeight: 220 }}>{(j.tail || []).join("\n") || "(starting…)"}</pre>
            </details>
          </div>
        ))}
      </div>
    </>
  );
}
