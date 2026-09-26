"use client";

import { useEffect, useState } from "react";

type Doc = Record<string, any>;
const fmt = (x: any, d = 3) => (typeof x === "number" && isFinite(x) ? x.toFixed(d) : "-");
const isTrue = (v: any) => v === true || String(v).toLowerCase() === "true";
const short = (rid: string) => (rid || "").replace("20260926T", "").replace(/Z_.*/, "");

function BestSoFar({ attempts, target }: { attempts: Doc[]; target?: number }) {
  const W = 620, H = 170, pad = 30;
  if (!attempts.length) return <div className="muted">No attempts recorded.</div>;
  const pts = attempts.map((a) => (a.speedup == null ? null : Number(a.speedup)));
  let best = 1;
  const bestLine = pts.map((p, i) => {
    if (p != null && attempts[i].improved) best = Math.max(best, p);
    return best;
  });
  const max = Math.max(1.2, target || 0, ...pts.filter((p): p is number => p != null), ...bestLine) * 1.08;
  const lo = Math.min(0.6, ...pts.filter((p): p is number => p != null)) * 0.95;
  const x = (i: number) => pad + (i * (W - 2 * pad)) / Math.max(attempts.length - 1, 1);
  const y = (v: number) => H - pad - ((v - lo) / (max - lo)) * (H - 2 * pad);
  const runBreaks = attempts.map((a, i) => (i > 0 && a.run_id !== attempts[i - 1].run_id ? i : -1)).filter((i) => i > 0);
  return (
    <svg width={W} height={H} style={{ maxWidth: "100%" }}>
      <line x1={pad} x2={W - pad} y1={y(1)} y2={y(1)} stroke="#8b98a5" strokeDasharray="3 3" />
      <text x={W - pad} y={y(1) + 12} fill="#8b98a5" fontSize="10" textAnchor="end">original 1.0x</text>
      {target ? (
        <>
          <line x1={pad} x2={W - pad} y1={y(target)} y2={y(target)} stroke="#58a6ff" strokeDasharray="4 2" />
          <text x={W - pad} y={y(target) - 4} fill="#58a6ff" fontSize="10" textAnchor="end">target {target}x</text>
        </>
      ) : null}
      {runBreaks.map((i) => (
        <g key={i}>
          <line x1={x(i) - 4} x2={x(i) - 4} y1={pad - 10} y2={H - pad} stroke="#30363d" />
          <text x={x(i)} y={pad - 12} fill="#8b98a5" fontSize="9">run {short(attempts[i].run_id)}</text>
        </g>
      ))}
      <polyline fill="none" stroke="#3fb950" strokeWidth={2} points={bestLine.map((b, i) => `${x(i)},${y(b)}`).join(" ")} />
      {attempts.map((a, i) => {
        const p = pts[i];
        const passed = a.gate1?.some((g: Doc) => g.status === "pass");
        const color = p == null ? "#f85149" : a.improved ? "#3fb950" : "#d29922";
        return (
          <g key={i}>
            <circle cx={x(i)} cy={p == null ? y(lo) + 4 : y(p)} r={4} fill={color}>
              <title>{`${a.label}\n${a.cls || ""}  ${p == null ? (passed ? "passed Gate 1, not benchmarked" : "failed Gate 1") : p.toFixed(3) + "x"}${a.improved ? "  NEW BEST" : ""}`}</title>
            </circle>
          </g>
        );
      })}
      <text x={pad} y={H - 8} fill="#8b98a5" fontSize="10">attempts in order (green = new best, amber = passed but not better, red = failed Gate 1)</text>
    </svg>
  );
}

function gateSummary(g: Doc[] | undefined) {
  if (!g?.length) return "-";
  const last = g[g.length - 1];
  const trail = g.map((x) => (x.status === "pass" ? "✓" : "✗")).join("");
  return `${trail} ${last.n_pass}/${last.n}${g.length > 1 ? ` (${g.length} tries)` : ""}`;
}

function LayerCard({ lay, integ, target }: { lay: Doc; integ?: Doc; target?: number }) {
  const runs: Doc[] = lay.runs || [];
  const cold = runs[0];
  const warm = runs.slice(1);
  return (
    <div className="panel" style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", gap: 12, alignItems: "baseline", flexWrap: "wrap" }}>
        <h2 style={{ margin: 0 }}>{lay.name}</h2>
        {lay.pct != null && <span className="muted">{fmt(lay.pct, 1)}% of self time · {lay.calls} calls</span>}
        <span className={`pill ${lay.best > 1.03 ? "good" : "warn"}`}>best {fmt(lay.best)}x layer speedup</span>
        {integ && (
          <span className={`pill ${isTrue(integ.kept) ? "good" : "bad"}`}>
            {isTrue(integ.kept) ? "kept in model" : "reverted in model"} · paired {fmt(Number(integ.paired_speedup))}x e2e
          </span>
        )}
      </div>
      <BestSoFar attempts={lay.attempts} target={target} />

      <h3 className="sub">How previous runs helped</h3>
      <table>
        <thead>
          <tr><th>run</th><th>skills retrieved</th><th>ideas blocked by memory</th><th>1st candidate</th><th>best</th><th>attempts to best</th><th>stop reason</th></tr>
        </thead>
        <tbody>
          {runs.map((r, i) => (
            <tr key={r.run_id}>
              <td className="mono">{i === 0 ? "cold " : "warm "}{short(r.run_id)}</td>
              <td>{r.skills?.length ? r.skills.slice(0, 4).map((s: string) => <span key={s} className="chip">{s}</span>) : <span className="muted">none</span>}</td>
              <td>{r.blocked || 0}</td>
              <td>{r.first != null ? `${fmt(r.first)}x` : "-"}</td>
              <td><b>{fmt(r.best)}x</b></td>
              <td>{r.bestAt ?? "-"}</td>
              <td className="muted" style={{ maxWidth: 360 }}>{r.stop ? String(r.stop).slice(0, 160) : "-"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {cold && warm.length > 0 && (
        <div className="insight">
          Cold run reached {fmt(cold.best)}x after {cold.bestAt ?? cold.attempts} benchmarked candidates; the next run started with{" "}
          {warm[0].skills?.length || 0} retrieved skills and its first candidate was already {fmt(warm[0].first)}x
          {warm[0].bestAt ? `, best ${fmt(warm[0].best)}x by candidate ${warm[0].bestAt}` : ""}.
        </div>
      )}
      {lay.blocked?.length > 0 && (
        <details>
          <summary>{lay.blocked.length} ideas blocked by do-not-repeat memory</summary>
          <ul>{lay.blocked.map((b: Doc, i: number) => <li key={i}><span className="mono">{b.label}</span> <span className="muted">← {String(b.blocked_by).slice(0, 200)}</span></li>)}</ul>
        </details>
      )}
      <details>
        <summary>All {lay.attempts.length} candidates the agent tried</summary>
        <table>
          <thead><tr><th>run</th><th>candidate</th><th>class</th><th>Gate 1</th><th>layer speedup</th><th></th><th>idea / why</th></tr></thead>
          <tbody>
            {lay.attempts.map((a: Doc, i: number) => (
              <tr key={i}>
                <td className="mono muted">{short(a.run_id)}</td>
                <td className="mono">{a.label}</td>
                <td>{a.cls || "-"}</td>
                <td className="mono">{gateSummary(a.gate1)}</td>
                <td>{a.speedup != null ? `${fmt(a.speedup)}x` : "-"}</td>
                <td>{a.improved ? <span className="pill good">new best</span> : null}</td>
                <td className="muted" style={{ maxWidth: 420 }}>{String(a.idea || a.why || "").slice(0, 220)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}

export default function ModelView({ modelKey }: { modelKey: string }) {
  const [d, setD] = useState<Doc | null>(null);
  useEffect(() => {
    setD(null);
    const load = () => fetch(`/api/model/${modelKey}`).then((r) => r.json()).then(setD);
    load();
    const t = setInterval(load, 8000);
    return () => clearInterval(t);
  }, [modelKey]);
  if (!d) return <div className="panel muted">loading…</div>;

  const c = d.comparison;
  const q = (c?.quality_metric || d.model.quality || "").toLowerCase();
  const target = d.runs.map((r: Doc) => r.target_speedup).filter(Boolean).slice(-1)[0];
  const latestInteg: Record<string, Doc> = {};
  for (const e of d.integration) latestInteg[e.layer] = e;
  const layersTouched = d.layers.filter((l: Doc) => l.attempts?.length);
  const totalAttempts = layersTouched.reduce((s: number, l: Doc) => s + l.attempts.length, 0);
  const keptLayers = Object.values(latestInteg).filter((e: Doc) => isTrue(e.kept)).length;
  const latestModelRun = d.latest?.run_id;
  const ml = d.modelLevel.filter((m: Doc) => !latestModelRun || m.run_id === latestModelRun);

  return (
    <>
      <div className="panel">
        <div style={{ display: "flex", gap: 16, alignItems: "baseline", flexWrap: "wrap" }}>
          <h2 style={{ margin: 0, fontSize: 18, color: "var(--text)", textTransform: "none", letterSpacing: 0 }}>{d.model.label}</h2>
          <span className="mono muted">{d.model.id}</span>
          <span className="muted">{d.runs.length} runs in memory</span>
        </div>
        {c ? (
          <div className="stats" style={{ marginTop: 12 }}>
            <div className="stat"><b>{fmt(c.unoptimized?.throughput, 1)}</b><span>unoptimized {c.unit}</span></div>
            <div className="stat"><b style={{ color: "var(--good)" }}>{fmt(c.optimized?.throughput, 1)}</b><span>optimized {c.unit}</span></div>
            <div className="stat"><b>{fmt(c.speedup, 2)}x</b><span>end-to-end speedup (paired)</span></div>
            <div className="stat"><b>-{fmt(c.latency_reduction_pct, 1)}%</b><span>latency</span></div>
            <div className="stat"><b>{fmt(c.unoptimized?.[q], 4)} → {fmt(c.optimized?.[q], 4)}</b><span>{c.quality_metric}</span></div>
            <div className="stat"><b>{fmt(c.unoptimized?.peak_memory_gb, 2)} → {fmt(c.optimized?.peak_memory_gb, 2)}</b><span>peak memory GB</span></div>
          </div>
        ) : (
          <div className="muted" style={{ marginTop: 10 }}>No completed comparison yet — pipeline still running.</div>
        )}
        {c?.final_config && <div className="mono muted" style={{ marginTop: 8 }}>final config: {JSON.stringify({ ...c.final_config, patches: (c.final_config.patches || []).map((p: Doc) => p.class?.split(":")[1]) })}</div>}
        <div className="flow">
          <span>Baseline</span>→<span>Profile ({d.profile?.layers?.length || 0} layers)</span>→
          <span>Layer loops ({layersTouched.length} layers, {totalAttempts} candidates)</span>→
          <span>Integration ({keptLayers} kept)</span>→<span>Model level ({ml.length} candidates)</span>→<span>Compare</span>
        </div>
      </div>

      <div className="grid">
        <div className="panel">
          <h2>Where the time goes (layer self time)</h2>
          <table>
            <tbody>
              {(d.profile?.layers || []).slice(0, 10).map((l: Doc) => {
                const lay = d.layers.find((x: Doc) => x.name === l.name);
                return (
                  <tr key={l.class}>
                    <td className="mono">{l.name}{d.profile?.targets?.includes(l.name) ? <span className="pill warn" style={{ marginLeft: 6 }}>target</span> : null}</td>
                    <td style={{ width: 180 }}><div className="bar" style={{ width: `${Math.round(l.pct * 2.5)}px` }} /></td>
                    <td>{fmt(l.pct, 1)}%</td>
                    <td>{lay?.attempts?.length ? `best ${fmt(lay.best)}x` : ["Linear", "Embedding"].includes(l.name) ? <span className="muted">vendor GEMM → model level</span> : ""}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <h2>Integration into the whole model (paired A/B)</h2>
          <table>
            <thead><tr><th>layer</th><th>layer speedup</th><th>e2e paired</th><th>rounds won</th><th>decision</th></tr></thead>
            <tbody>
              {Object.values(latestInteg).map((e: Doc) => {
                const lay = d.layers.find((x: Doc) => x.name === e.layer);
                return (
                  <tr key={e.layer}>
                    <td className="mono">{e.layer}</td>
                    <td>{fmt(lay?.best)}x</td>
                    <td>{fmt(Number(e.paired_speedup))}x</td>
                    <td>{e.wins}/3</td>
                    <td>{isTrue(e.kept) ? <span className="pill good">kept</span> : <span className="pill bad">reverted</span>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <div className="muted" style={{ marginTop: 8 }}>A layer kernel is kept only if the whole model gets faster (interleaved A/B runs) and quality stays within budget.</div>
        </div>
      </div>

      <h2 className="section">Layer optimization: what the agent tried</h2>
      {layersTouched.map((lay: Doc) => <LayerCard key={lay.name} lay={lay} integ={latestInteg[lay.name]} target={target} />)}

      <div className="panel">
        <h2>Model-level optimization {latestModelRun ? `(run ${short(latestModelRun)})` : ""}</h2>
        {d.e2eBest.length > 0 && (
          <div style={{ marginBottom: 8 }}>
            end-to-end best over time:{" "}
            {d.e2eBest.map((e: Doc, i: number) => <span key={i} className="chip">{short(e.run_id)} {e.label}: {fmt(e.e2e, 2)}x</span>)}
          </div>
        )}
        <table>
          <thead><tr><th>#</th><th>config</th><th>class</th><th>paired vs best</th><th>{d.model.quality}</th><th>outcome</th><th>why</th></tr></thead>
          <tbody>
            {ml.map((m: Doc, i: number) => (
              <tr key={i}>
                <td>{m.slot}</td>
                <td className="mono">{m.label}<div className="muted">{JSON.stringify(m.config || {}).slice(0, 120)}</div></td>
                <td>{m.cls}</td>
                <td>{m.paired ? `${fmt(m.paired)}x` : m.speedup_vs_best ? `${fmt(m.speedup_vs_best)}x` : "-"}</td>
                <td>{fmt(m.quality, 4)}</td>
                <td>{m.keep ? <span className="pill good">kept</span> : <span className="pill warn">{m.outcome}</span>}</td>
                <td className="muted" style={{ maxWidth: 360 }}>{String(m.why || "").slice(0, 160)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {d.modelBlocked.length > 0 && (
          <details><summary>{d.modelBlocked.length} model-level ideas blocked by memory</summary>
            <ul>{d.modelBlocked.map((b: Doc, i: number) => <li key={i}><span className="mono">{b.label}</span> <span className="muted">← {String(b.blocked_by).slice(0, 200)}</span></li>)}</ul>
          </details>
        )}
        {d.saturations.length > 0 && (
          <div className="muted" style={{ marginTop: 8 }}>
            {d.saturations.slice(-3).map((s: Doc, i: number) => <div key={i}>{s.step}: {String(s.reason || "").slice(0, 220)}</div>)}
          </div>
        )}
      </div>
    </>
  );
}
