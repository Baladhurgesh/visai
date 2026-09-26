"use client";

import { useEffect, useState } from "react";

type Doc = Record<string, any>;
const fmt = (x: any, d = 2) => (typeof x === "number" && isFinite(x) ? x.toFixed(d) : "-");
const pct = (x: any, d = 2) => (typeof x === "number" && isFinite(x) ? `${(x * 100).toFixed(d)}%` : "-");

const WHAT: Record<string, { produced: string; note: string }> = {
  qwen: {
    produced: "8-bit weights (group size 64) + 4096-token prefill step, found by the model-level agent",
    note: "The RMSNorm and RoPE layer kernels are in the final config but measured ~0.94x together vs stock; the speedup comes from quantization.",
  },
  parakeet: {
    produced: "5 s chunk overlap instead of 15 s, found by the model-level agent",
    note: "Layer kernels excluded: LSTM + Conv1d together measured 0.94x vs stock in paired runs, even though LSTM alone hit 4.61x at the layer level.",
  },
  diar: {
    produced: "speaker-activity threshold 0.25 + 1.2 s segment merging (harness sweep, agent budget exhausted)",
    note: "Speed was already saturated (~480x real-time), so the win is quality: missed speech was the dominant error.",
  },
};

function Card({ m }: { m: Doc }) {
  const q = (m.quality_metric || "").toLowerCase();
  const u = m.unoptimized || {}, o = m.optimized || {};
  const qBefore = u[q], qAfter = o[q];
  const isRate = q === "wer" || q === "der";
  const w = WHAT[m.key];
  return (
    <div className="panel result-card">
      <h2 style={{ color: "var(--text)", fontSize: 16, textTransform: "none", letterSpacing: 0 }}>{m.label}</h2>
      <div className="stats">
        <div className="stat"><b>{fmt(u.throughput, 1)} → <span style={{ color: m.speedup > 1.01 ? "var(--good)" : "var(--text)" }}>{fmt(o.throughput, 1)}</span></b><span>{m.unit}</span></div>
        <div className="stat"><b>{fmt(m.speedup)}x</b><span>speed (paired)</span></div>
      </div>
      <div className="stats" style={{ marginTop: 10 }}>
        <div className="stat"><b>{isRate ? pct(qBefore) : fmt(qBefore, 2)} → <span style={{ color: isRate && qAfter < qBefore - 0.001 ? "var(--good)" : "var(--text)" }}>{isRate ? pct(qAfter) : fmt(qAfter, 2)}</span></b><span>{m.quality_metric}</span></div>
        <div className="stat"><b>{fmt(u.peak_memory_gb)} → {fmt(o.peak_memory_gb)}</b><span>peak memory GB</span></div>
        <div className="stat"><b>{m.latency_reduction_pct > 0 ? "-" : "+"}{fmt(Math.abs(m.latency_reduction_pct), 1)}%</b><span>latency</span></div>
      </div>
      {w && (
        <>
          <div style={{ marginTop: 10 }}><b>What produced it:</b> {w.produced}</div>
          <div className="muted" style={{ marginTop: 4 }}>{w.note}</div>
        </>
      )}
    </div>
  );
}

export default function Results() {
  const [d, setD] = useState<Doc | null>(null);
  useEffect(() => {
    fetch("/api/results").then((r) => r.json()).then(setD);
  }, []);
  if (!d) return <div className="panel muted">loading results…</div>;
  const v = (d.validations || []).find((x: Doc) => x.adapter === "diar");
  const cold = d.compounding?.[0], warm = d.compounding?.[1];
  return (
    <>
      <div className="panel">
        <h2>Results on this Mac (Apple M3 Pro, 18 GB)</h2>
        <p className="muted" style={{ margin: 0 }}>
          Unoptimized vs optimized on the same data points; speed from interleaved A/B runs. Live from MongoDB Atlas:
          {" "}{d.counts.pipeline_runs} pipeline runs, {d.counts.experiments} experiments, {d.counts.skills} skills,
          {" "}{d.counts.do_not_repeat} do-not-repeat fingerprints, {d.counts.kernels} registered kernels.
        </p>
      </div>
      <div className="grid">{d.models.map((m: Doc) => <Card key={m.key} m={m} />)}</div>

      {v && (
        <div className="panel">
          <h2>Diarization quality: validated on meetings it never saw</h2>
          <table>
            <thead><tr><th>data</th><th>stock DER</th><th>optimized DER</th><th>missed speech</th><th>false alarm</th><th>confusion</th></tr></thead>
            <tbody>
              <tr>
                <td>Tuning set: AMI test meetings 1–2 (20 min)</td>
                <td>{pct(d.models.find((m: Doc) => m.key === "diar")?.unoptimized?.der)}</td>
                <td><b>{pct(d.models.find((m: Doc) => m.key === "diar")?.optimized?.der)}</b></td>
                <td colSpan={3} className="muted">used to choose threshold + merge gap</td>
              </tr>
              <tr>
                <td>Held-out: {v.dataset}</td>
                <td>{pct(v.stock.der)}</td>
                <td><b style={{ color: "var(--good)" }}>{pct(v.optimized.der)}</b></td>
                <td>{pct(v.stock.missed)} → {pct(v.optimized.missed)}</td>
                <td>{pct(v.stock.false_alarm)} → {pct(v.optimized.false_alarm)}</td>
                <td>{pct(v.stock.confusion)} → {pct(v.optimized.confusion)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      <div className="grid">
        <div className="panel">
          <h2>Layer kernels: isolated speedup vs whole model</h2>
          <table>
            <thead><tr><th>model</th><th>layer</th><th>best isolated</th><th>in whole model (paired)</th><th>decision</th></tr></thead>
            <tbody>
              {d.layers.sort((a: Doc, b: Doc) => b.best - a.best).map((l: Doc) => (
                <tr key={`${l.adapter}${l.layer}`}>
                  <td>{l.model}</td>
                  <td className="mono">{l.layer}</td>
                  <td><b>{fmt(l.best)}x</b></td>
                  <td>{l.e2e != null ? `${fmt(l.e2e, 3)}x vs previous state` : <span className="muted">not integrated</span>}</td>
                  <td>{l.status === "integrated" ? <span className="pill good">kept</span> : l.status === "reverted" ? <span className="pill bad">reverted</span> : <span className="pill">layer only</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: 10 }}>
            <b>All kept layer kernels together vs stock:</b>{" "}
            {Object.entries(d.allKept).map(([k, v]: [string, any]) => <span key={k} className="chip">{k}: {fmt(v, 3)}x</span>)}
          </div>
          <div className="insight">
            Isolated layer wins (up to 4.6x) did not survive end to end: step-by-step gains of ~5–10% sit inside run-to-run noise,
            and together the kept kernels measured ~0.94x. Model-level changes (quantization, chunking, post-processing) delivered
            the real improvements. This is why Visai gates every kernel on whole-model paired measurements.
          </div>
        </div>
        <div className="panel">
          <h2>Memory compounding (Parakeet decoder LSTM)</h2>
          {cold && warm ? (
            <>
              <table>
                <thead><tr><th>run</th><th>1st candidate</th><th>best</th><th>candidates to best</th></tr></thead>
                <tbody>
                  {d.compounding.map((r: Doc, i: number) => (
                    <tr key={r.run_id}>
                      <td className="mono">{i === 0 ? "cold" : "warm"} {r.run_id.slice(9, 15)}</td>
                      <td>{fmt(r.first)}x</td>
                      <td><b>{fmt(r.best)}x</b></td>
                      <td>{r.bestAt ?? "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="insight">
                Cold run: first candidate {fmt(cold.first)}x, best {fmt(cold.best)}x after {cold.bestAt} candidates. The next run retrieved LSTM
                skills from Atlas: first candidate already {fmt(warm.first)}x, best {fmt(warm.best)}x by candidate {warm.bestAt}.
                (The third run stopped as soon as it hit the 2x target.)
              </div>
            </>
          ) : <div className="muted">not enough runs</div>}
        </div>
      </div>
    </>
  );
}
