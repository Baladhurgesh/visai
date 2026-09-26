"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import Architecture from "./Architecture";
import ModelView from "./ModelView";
import Optimize from "./Optimize";

const TABS = [
  { key: "optimize", label: "+ Optimize a model" },
  { key: "architecture", label: "Architecture & tools" },
  { key: "overview", label: "Overview & memory" },
  { key: "qwen", label: "Qwen3-0.6B" },
  { key: "parakeet", label: "Parakeet ASR" },
  { key: "diar", label: "Nemotron diarization" },
];

type Doc = Record<string, any>;

const fmt = (x: any, d = 3) => (typeof x === "number" && isFinite(x) ? x.toFixed(d) : "-");

function outcomePill(t: Doc) {
  if (t.genuine) return <span className="pill good">KEEP</span>;
  const o = t.miss?.outcome || "?";
  const cls = ["broke", "compile_error", "quality_drop", "quality_and_noise"].includes(o) ? "bad" : "warn";
  return <span className={`pill ${cls}`}>{o}</span>;
}

function SpeedupChart({ trials, gate = 1.03 }: { trials: Doc[]; gate?: number }) {
  const W = 560, H = 180, pad = 28;
  const vals = trials.map((t) => (t.candidate?.speedup as number) || 0);
  const max = Math.max(1.15, gate + 0.05, ...vals) + 0.02;
  const lo = Math.min(0.7, ...vals.filter((v) => v > 0).map((v) => v - 0.05));
  const bw = (W - pad * 2) / Math.max(vals.length, 1);
  const y = (v: number) => H - pad - ((Math.max(v, lo) - lo) / (max - lo)) * (H - pad * 2);
  return (
    <svg width={W} height={H} style={{ maxWidth: "100%" }}>
      <line x1={pad} x2={W - pad} y1={y(1)} y2={y(1)} stroke="#8b98a5" strokeDasharray="3 3" />
      <line x1={pad} x2={W - pad} y1={y(gate)} y2={y(gate)} stroke="#3fb950" strokeDasharray="4 2" />
      <text x={W - pad} y={y(gate) - 4} fill="#3fb950" fontSize="10" textAnchor="end">gate {gate}x</text>
      <text x={W - pad} y={y(1) + 12} fill="#8b98a5" fontSize="10" textAnchor="end">stock 1.0x</text>
      {trials.map((t, i) => {
        const v = vals[i];
        const color = t.genuine ? "#3fb950" : v ? "#d29922" : "#f85149";
        return (
          <g key={i}>
            <rect x={pad + i * bw + 3} width={Math.max(bw - 6, 2)} y={v ? y(v) : H - pad - 3} height={v ? H - pad - y(v) : 3} fill={color} rx={2} />
            <text x={pad + i * bw + bw / 2} y={H - 8} fill="#8b98a5" fontSize="10" textAnchor="middle">{t.slot}</text>
            {v ? <text x={pad + i * bw + bw / 2} y={y(v) - 3} fill="#e6edf3" fontSize="10" textAnchor="middle">{v.toFixed(2)}</text> : null}
          </g>
        );
      })}
    </svg>
  );
}

function Compounding({ data }: { data: Record<string, Doc[]> }) {
  const keys = Object.keys(data);
  if (!keys.length) return <div className="muted">Run the same target more than once to see memory compounding.</div>;
  return (
    <div style={{ display: "grid", gap: 14 }}>
      {keys.map((k) => (
        <div key={k}>
          <div className="mono" style={{ marginBottom: 6 }}>{k}</div>
          <table>
            <thead><tr><th>run</th><th>trials</th><th>trials to first win</th><th>Gate 1 pass rate</th><th>best speedup</th></tr></thead>
            <tbody>
              {data[k].map((r, i) => (
                <tr key={r.run_id}>
                  <td className="mono">{i === 0 ? "cold " : "warm "}{r.run_id.slice(0, 16)}</td>
                  <td>{r.trials}</td>
                  <td>{r.trials_to_win ?? <span className="muted">no win</span>}</td>
                  <td><div className="bar" style={{ width: `${Math.round(r.gate1_pass_rate * 100)}px` }} /> {Math.round(r.gate1_pass_rate * 100)}%</td>
                  <td>{fmt(r.best_speedup)}x</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  );
}

function BeforeAfter({ rows }: { rows: Doc[] }) {
  if (!rows?.length) return <div className="muted">Run `visai run --model all` to produce unoptimized vs optimized comparisons.</div>;
  return (
    <table>
      <thead>
        <tr><th>model</th><th>unoptimized</th><th>optimized</th><th>speedup</th><th>latency</th><th>quality</th><th>memory</th></tr>
      </thead>
      <tbody>
        {rows.map((c) => {
          const q = (c.quality_metric || "").toLowerCase();
          return (
            <tr key={c._id}>
              <td className="mono">{c.model}<div className="muted">{c.hardware}</div></td>
              <td>{fmt(c.unoptimized?.throughput, 2)}<div className="muted">{c.unit}</div></td>
              <td><b>{fmt(c.optimized?.throughput, 2)}</b></td>
              <td><span className={`pill ${c.speedup > 1 ? "good" : "warn"}`}>{fmt(c.speedup)}x</span></td>
              <td>-{fmt(c.latency_reduction_pct, 1)}%</td>
              <td>{c.quality_metric} {fmt(c.unoptimized?.[q], 4)} → {fmt(c.optimized?.[q], 4)}</td>
              <td>{fmt(c.unoptimized?.peak_memory_gb, 2)} → {fmt(c.optimized?.peak_memory_gb, 2)} GB</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function PipelineView({ run }: { run: Doc }) {
  const layers: Doc[] = run?.profile?.layers || [];
  const report = run?.run?.report;
  const targets: string[] = run?.profile?.targets || [];
  const maxPct = Math.max(1, ...layers.map((l) => l.pct || 0));
  return (
    <div className="grid">
      <div className="panel">
        <h2>Layer profile (self time)</h2>
        <table>
          <thead><tr><th>layer</th><th>self %</th><th>calls</th><th>us/call</th></tr></thead>
          <tbody>
            {layers.slice(0, 12).map((l) => (
              <tr key={l.class}>
                <td className="mono">{l.name} {targets.includes(l.name) ? <span className="pill warn">target</span> : null}</td>
                <td><div className="bar" style={{ width: `${Math.round((l.pct / maxPct) * 140)}px` }} /> {fmt(l.pct, 1)}%</td>
                <td>{l.calls}</td>
                <td>{fmt(l.us_per_call, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="panel">
        <h2>Layer loops → integration → model level</h2>
        {(report?.layers || []).map((lr: Doc) => (
          <div key={lr.layer} style={{ marginBottom: 6 }}>
            <b>{lr.layer}</b>: {lr.best ? `best ${lr.best.label} ${fmt(lr.best.robust_speedup)}x` : "no improvement"}{" "}
            <span className="muted">[{lr.iterations} iters · {lr.stop_reason}]</span>
          </div>
        ))}
        {(report?.integration || []).map((it: Doc) => (
          <div key={it.layer} style={{ marginBottom: 6 }}>
            integrate {it.layer}: <span className={`pill ${it.kept ? "good" : "warn"}`}>{it.kept ? "kept" : "reverted"}</span>{" "}
            {it.paired_speedup_vs_current ? `paired ${fmt(it.paired_speedup_vs_current)}x (${it.paired_wins})` : ""}{" "}
            <span className="muted">{it.why}</span>
          </div>
        ))}
        {report?.model_level && (
          <div>model level: <b>{report.model_level.best_label}</b> <span className="mono">{JSON.stringify(report.model_level.best_config)}</span>{" "}
            <span className="muted">[{report.model_level.candidates} candidates · {report.model_level.stop_reason}]</span></div>
        )}
        {!report && <div className="muted">running…</div>}
      </div>
    </div>
  );
}

export default function Page() {
  const [tab, setTabState] = useState("qwen");
  const setTab = (t: string) => {
    setTabState(t);
    window.history.replaceState(null, "", `?tab=${t}`);
  };
  useEffect(() => {
    const t = new URLSearchParams(window.location.search).get("tab");
    if (t) setTabState(t);
  }, []);
  const [ov, setOv] = useState<Doc | null>(null);
  const [sel, setSel] = useState<string | null>(null);
  const [run, setRun] = useState<Doc | null>(null);
  const [feed, setFeed] = useState<Doc[]>([]);
  const feedRef = useRef<HTMLDivElement>(null);

  const loadOverview = () => fetch("/api/overview").then((r) => r.json()).then(setOv);
  useEffect(() => {
    loadOverview();
    const t = setInterval(loadOverview, 5000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => {
    if (!sel && ov?.runs?.length) setSel(ov.runs[0].run_id);
  }, [ov, sel]);
  useEffect(() => {
    if (!sel) return;
    const load = () => fetch(`/api/run/${sel}`).then((r) => r.json()).then(setRun);
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, [sel]);
  useEffect(() => {
    const since = new Date(Date.now() - 30 * 60 * 1000).toISOString().slice(0, 19) + "Z";
    const es = new EventSource(`/api/stream?since=${since}`);
    es.onmessage = (m) => setFeed((f) => [...f.slice(-300), JSON.parse(m.data)]);
    return () => es.close();
  }, []);
  useEffect(() => {
    feedRef.current?.scrollTo({ top: feedRef.current.scrollHeight });
  }, [feed]);

  const trials: Doc[] = run?.experiments || [];
  const isModel = run?.run?.kind === "model";
  const stock = trials[0]?.stock;
  const report = run?.run?.report;
  const lastReflection = useMemo(() => (run?.reflections || []).slice(-1)[0], [run]);

  return (
    <>
      <header>
        <h1>Visai</h1>
        <span className="sub">profile → optimize → verify → benchmark → learn</span>
        <span className="pill" style={{ marginLeft: "auto" }}>memory: {!ov ? "…" : ov.backend === "atlas" ? "MongoDB Atlas" : "local fallback"}</span>
      </header>
      <nav className="tabs">
        {[
          ...TABS,
          ...Array.from(
            new Map<string, Doc>(
              (ov?.runs || [])
                .filter((r: Doc) => String(r.adapter || "").startsWith("llm-"))
                .map((r: Doc): [string, Doc] => [r.adapter, { key: r.adapter, label: String(r.model || r.adapter).split("/").pop() }]),
            ).values(),
          ),
          ...(tab.startsWith("llm-") && !(ov?.runs || []).some((r: Doc) => r.adapter === tab) ? [{ key: tab, label: tab }] : []),
        ].map((t: Doc) => (
          <button key={t.key} className={tab === t.key ? "active" : ""} onClick={() => setTab(t.key)}>{t.label}</button>
        ))}
      </nav>
      {tab === "optimize" ? (
        <main><Optimize onOpenModel={(k) => setTab(k)} /></main>
      ) : tab === "architecture" ? (
        <main><Architecture /></main>
      ) : tab !== "overview" ? (
        <main><ModelView modelKey={tab} /></main>
      ) : (
      <main>
        <div className="panel stats">
          <div className="stat"><b>{ov?.counts?.runs ?? "-"}</b><span>runs</span></div>
          <div className="stat"><b>{ov?.counts?.experiments ?? "-"}</b><span>experiments</span></div>
          <div className="stat"><b>{ov?.counts?.skills ?? "-"}</b><span>learned skills</span></div>
          <div className="stat"><b>{ov?.counts?.do_not_repeat ?? "-"}</b><span>do-not-repeat fingerprints</span></div>
        </div>

        <div className="panel">
          <h2>Unoptimized vs optimized on this Mac</h2>
          <BeforeAfter rows={ov?.comparisons || []} />
        </div>

        <div className="grid">
          <div className="panel">
            <h2>Runs</h2>
            <table>
              <thead><tr><th>run</th><th>kind</th><th>target</th><th>status</th></tr></thead>
              <tbody>
                {(ov?.runs || []).map((r: Doc) => (
                  <tr key={r.run_id} className={`click ${sel === r.run_id ? "sel" : ""}`} onClick={() => setSel(r.run_id)}>
                    <td className="mono">{r.run_id}</td>
                    <td>{r.kind}</td>
                    <td className="mono">{r.target}</td>
                    <td>{r.status === "done" ? (r.won || r.report?.improvement_pct > 0 ? <span className="pill good">done</span> : <span className="pill">done</span>) : <span className="pill warn">{r.status}</span>}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="panel">
            <h2>Live agent feed</h2>
            <div className="feed mono" ref={feedRef}>
              {feed.map((e, i) => (
                <div key={i}>
                  <span className="muted">{e.ts?.slice(11, 19)}</span> <b>{e.step}</b> {e.agent ? <span className="muted">[{e.agent}]</span> : null}{" "}
                  {e.tool ? <span>{e.tool} </span> : null}
                  {e.label ? <span>{e.label} </span> : null}
                  {e.status ? <span className={e.status === "pass" || e.status === "success" ? "good" : "warn"}>{e.status} </span> : null}
                  {e.speedup ? <span>{Number(e.speedup).toFixed(3)}x </span> : null}
                  {e.blocked_by ? <span className="bad">blocked: {String(e.blocked_by).slice(0, 120)}</span> : null}
                  {e.outcome ? <span>{e.outcome}</span> : null}
                  {e.skills ? <span className="muted">skills: {JSON.stringify(e.skills)}</span> : null}
                </div>
              ))}
            </div>
          </div>
        </div>

        {run?.run?.kind === "pipeline" && <PipelineView run={run} />}

        {run?.run && (
          <div className="panel">
            <h2>Run {run.run.run_id}</h2>
            <div className="mono muted" style={{ marginBottom: 8 }}>
              {run.run.target} · baseline mode {run.run.baseline_mode || "-"} · budget {run.run.budget}
            </div>
            {stock && (
              <div style={{ marginBottom: 10 }}>
                stock: <b>{isModel ? `${fmt(stock.throughput, 1)} tok/s, ppl ${fmt(stock.perplexity)}` : `${fmt(stock.latency_us, 2)} us`}</b>
                {!isModel && <span className="muted"> (reference {fmt(stock.reference_us, 2)} us, runtime {fmt(stock.runtime_us, 2)} us)</span>}
              </div>
            )}
            {report && (
              <div style={{ marginBottom: 10 }}>
                <b>{fmt(report.baseline?.decode_tok_s, 1)} → {fmt(report.optimized?.decode_tok_s, 1)} tok/s</b> ({fmt(report.improvement_pct, 1)}%) · quality regression {fmt(report.quality_regression_pct, 2)}% (allowed {fmt(report.quality_budget_pct, 2)}%)
              </div>
            )}
            <SpeedupChart trials={trials} />
            <table>
              <thead>
                <tr><th>slot</th><th>label</th><th>class</th><th>Gate 1</th><th>{isModel ? "tok/s" : "speedup"}</th>{isModel && <th>ppl</th>}<th>confirmed</th><th>outcome</th><th>why</th></tr>
              </thead>
              <tbody>
                {trials.map((t: Doc) => (
                  <tr key={t._id}>
                    <td>{t.slot}</td>
                    <td className="mono">{t.candidate?.label}</td>
                    <td>{t.candidate?.cls}</td>
                    <td>{t.candidate?.correctness}</td>
                    <td>{isModel ? fmt(t.candidate?.throughput, 1) : t.candidate?.speedup ? `${fmt(t.candidate.speedup)}x` : "-"}</td>
                    {isModel && <td>{fmt(t.candidate?.perplexity)}</td>}
                    <td>{String(t.candidate?.confirmed ?? "-")}</td>
                    <td>{outcomePill(t)}</td>
                    <td className="muted">{t.miss?.why}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {lastReflection && (
              <div style={{ marginTop: 12 }}>
                <h2>Latest reflection</h2>
                <div><b>Why it missed:</b> {lastReflection.explain_miss}</div>
                <div><b>Next try:</b> {lastReflection.next_try?.title} <span className="muted">({lastReflection.next_try?.cls})</span> — {lastReflection.next_try?.how}</div>
              </div>
            )}
          </div>
        )}

        <div className="grid">
          <div className="panel">
            <h2>Memory compounding (cold vs warm)</h2>
            <Compounding data={ov?.compounding || {}} />
          </div>
          <div className="panel">
            <h2>Skill library (Atlas Vector Search)</h2>
            <table>
              <thead><tr><th>skill</th><th>op</th><th>hw</th><th>trials</th><th>wins</th><th>mean</th><th>confidence</th></tr></thead>
              <tbody>
                {(ov?.skills || []).map((s: Doc) => (
                  <tr key={s._id}>
                    <td className="mono">{s.name}<div className="muted">{(s.strategy || []).join(", ")}</div></td>
                    <td>{s.operator}</td>
                    <td>{s.hardware}</td>
                    <td>{s.evidence?.trials}</td>
                    <td>{s.evidence?.positive_speedups}</td>
                    <td>{fmt(s.evidence?.mean_speedup)}x</td>
                    <td><div className="bar" style={{ width: `${Math.round((s.confidence || 0) * 100)}px` }} /> {fmt(s.confidence, 2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="grid">
          <div className="panel">
            <h2>Do-not-repeat (persistent)</h2>
            <table>
              <tbody>
                {(ov?.do_not_repeat || []).map((d: Doc) => (
                  <tr key={d._id}><td>{d.text}</td><td className="mono muted">{d.scope}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="panel">
            <h2>LESSONS.md (rendered from Atlas)</h2>
            <pre className="lessons">{ov?.lessons_md || "(run a loop to populate)"}</pre>
          </div>
        </div>
      </main>
      )}
    </>
  );
}
