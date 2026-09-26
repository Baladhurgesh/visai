"use client";

import { useEffect, useState } from "react";

type Doc = Record<string, any>;
const fmt = (x: any, d = 2) => (typeof x === "number" && isFinite(x) ? x.toFixed(d) : "-");
const pct = (x: any, d = 2) => (typeof x === "number" && isFinite(x) ? `${(x * 100).toFixed(d)}%` : "-");
const ratio = (a: any, b: any) => (typeof a === "number" && typeof b === "number" && b > 0 ? a / b : null);

const ROWS: { key: string; label: string; unit?: string; digits?: number; lower?: boolean; rate?: boolean }[] = [
  { key: "e2e_s", label: "End to end", unit: "s", lower: true },
  { key: "diarize_s", label: "Diarization", unit: "s", lower: true },
  { key: "asr_s", label: "Transcription", unit: "s", lower: true },
  { key: "asr_rtfx", label: "Transcription speed", unit: "x real-time" },
  { key: "notes_s", label: "Notes", unit: "s", lower: true },
  { key: "decode_tok_s", label: "Notes decode", unit: "tok/s" },
  { key: "prefill_tok_s", label: "Notes prefill", unit: "tok/s", digits: 0 },
  { key: "asr_peak_gb", label: "Transcription memory", unit: "GB", lower: true },
  { key: "llm_peak_gb", label: "Notes memory", unit: "GB", lower: true },
  { key: "wer_vs_reference", label: "Word error vs reference", rate: true, lower: true },
  { key: "der", label: "Diarization error", rate: true, lower: true },
  { key: "wder", label: "Word diarization error", rate: true, lower: true },
  { key: "fact_recall", label: "Facts recalled", rate: true },
  { key: "transcript_ppl", label: "Transcript perplexity", lower: true },
];

function Speed({ x }: { x: number | null }) {
  if (x == null) return <span className="muted">-</span>;
  const cls = x >= 1.03 ? "good" : x < 0.97 ? "warn" : "";
  return <span className={`pill ${cls}`}>{fmt(x)}x</span>;
}

function Item({ it }: { it: Doc }) {
  const b = it.baseline, o = it.optimized;
  return (
    <div className="panel">
      <h2>{it.name} · {fmt(b.audio_s, 0)}s of audio · {o.speakers_detected} speakers</h2>
      <div className="stats">
        <div className="stat"><b>{fmt(b.e2e_s)}s → <span style={{ color: "var(--good)" }}>{fmt(o.e2e_s)}s</span></b><span>end to end</span></div>
        <div className="stat"><b>{fmt(ratio(b.e2e_s, o.e2e_s))}x</b><span>faster</span></div>
        <div className="stat"><b>{fmt(b.decode_tok_s, 1)} → {fmt(o.decode_tok_s, 1)}</b><span>notes tok/s</span></div>
      </div>
      <table style={{ marginTop: 12 }}>
        <thead><tr><th>stage</th><th>unoptimized</th><th>optimized</th><th></th></tr></thead>
        <tbody>
          {ROWS.filter((r) => b[r.key] != null || o[r.key] != null).map((r) => {
            const show = (v: any) => r.rate ? pct(v) : `${fmt(v, r.digits ?? 2)}${r.unit ? ` ${r.unit}` : ""}`;
            const x = ratio(b[r.key], o[r.key]);
            return (
              <tr key={r.key}>
                <td>{r.label}</td>
                <td>{show(b[r.key])}</td>
                <td><b>{show(o[r.key])}</b></td>
                <td>{r.rate || r.key === "transcript_ppl" ? <span className="muted">quality</span> : <Speed x={x == null ? null : r.lower ? x : 1 / x} />}</td>
              </tr>
            );
          })}
          {o.wer_vs_baseline != null && (
            <tr>
              <td>Transcript difference vs unoptimized</td>
              <td className="muted">reference</td>
              <td>{pct(o.wer_vs_baseline)}</td>
              <td className="muted">chunk boundaries, see below</td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

export default function Minutes() {
  const [d, setD] = useState<Doc | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    fetch("/api/minutes").then(async (r) => {
      if (!r.ok) { setErr("No benchmark results found. Run the Minutes benchmark first."); return; }
      setD(await r.json());
    });
  }, []);
  if (err) return <div className="panel muted">{err}</div>;
  if (!d) return <div className="panel muted">loading benchmark…</div>;

  const items: Doc[] = d.items || [];
  const geos = items.map((it) => ratio(it.baseline.e2e_s, it.optimized.e2e_s)).filter((x): x is number => x != null);
  const geomean = geos.length ? Math.exp(geos.reduce((s, x) => s + Math.log(x), 0) / geos.length) : null;
  const hw = d.meta?.hardware || {};

  return (
    <>
      <div className="panel">
        <h2>Minutes app · full meeting, unoptimized vs optimized</h2>
        <div className="stats">
          <div className="stat"><b style={{ color: "var(--good)" }}>{fmt(geomean)}x</b><span>end to end, geometric mean</span></div>
          <div className="stat"><b>{items.length}</b><span>recordings, same audio both ways</span></div>
          <div className="stat"><b>{d.meta?.runs}</b><span>runs, median reported</span></div>
        </div>
        <p className="muted" style={{ marginBottom: 0 }}>
          {hw.chip}, {hw.memory_gb} GB · macOS {hw.macos} · MLX {hw.mlx} · measured {d.meta?.date}.
          {" "}Unoptimized is bf16 Parakeet + bf16 Qwen3-0.6B with a 15 s chunk overlap.
          Optimized is the visai profile: Qwen 8-bit (group 64, 4096-token prefill) and Parakeet 6-bit with a 5 s overlap.
          Diarization is the same Nemotron 3 model in both, so its time does not change.
        </p>
      </div>

      {items.map((it) => <Item key={it.name} it={it} />)}

      <div className="panel">
        <h2>Where the transcription speed comes from</h2>
        <p style={{ marginTop: 0 }}>
          The 21.7% transcript difference on the 10 minute meeting is words duplicated or dropped at the
          2 minute chunk seams, not a different transcription of the speech. Word error against a known
          transcript, on a 6.5 minute recording doubled so it crosses three seams:
        </p>
        <table>
          <thead><tr><th>configuration</th><th>195 s, one seam</th><th>391 s, three seams</th></tr></thead>
          <tbody>
            <tr><td>Unoptimized (bf16, 15 s overlap)</td><td>1.73%</td><td>3.94%</td></tr>
            <tr><td>5 s overlap only</td><td>1.73%</td><td><b style={{ color: "var(--good)" }}>1.73%</b></td></tr>
            <tr><td>6-bit weights, 15 s overlap</td><td>1.73%</td><td>4.62%</td></tr>
            <tr><td>Optimized (6-bit, 5 s overlap)</td><td>1.73%</td><td>1.83%</td></tr>
          </tbody>
        </table>
        <div className="insight">
          The 5 second overlap is the real transcription win: faster, and it stops the model dropping words
          where chunks meet. The 6-bit weights add about a tenth of a point of word error and save roughly 1 GB.
        </div>
      </div>
    </>
  );
}
