import { NextResponse } from "next/server";
import { backend, find, findOne } from "@/lib/db";

export const dynamic = "force-dynamic";

export async function GET() {
  const [runs, skills, dnr, lessons, experiments, comparisons] = await Promise.all([
    find("runs", {}, { sort: { started: -1 }, limit: 40 }),
    find("skills", {}, { sort: { confidence: -1 }, limit: 100, project: { example_source: 0 } }),
    find("do_not_repeat", {}, { sort: { ts: -1 }, limit: 200 }),
    findOne("lessons", { kind: "lessons_md", scope: "global" }),
    find("experiments", {}, { sort: { ts: 1 }, project: { "candidate.source": 0 } }),
    find("comparisons", {}, { sort: { ts: -1 }, limit: 30 }),
  ]);
  const latestComparison: Record<string, any> = {};
  for (const c of comparisons) if (!latestComparison[c.adapter]) latestComparison[c.adapter] = c;

  // Compounding: per target, how many candidates each run needed before its first confirmed win.
  const byRun = new Map<string, any[]>();
  for (const e of experiments) {
    if (!byRun.has(e.run_id)) byRun.set(e.run_id, []);
    byRun.get(e.run_id)!.push(e);
  }
  const compounding: Record<string, any[]> = {};
  for (const r of [...runs].reverse()) {
    const ex = byRun.get(r.run_id) || [];
    if (!ex.length) continue;
    const key = r.op || r.model || r.target;
    const firstWin = ex.findIndex((e) => e.genuine);
    const passes = ex.filter((e) => ["pass", "skipped"].includes(e.candidate?.correctness));
    const best = Math.max(0, ...ex.map((e) => (e.candidate?.speedup as number) || 0));
    (compounding[key] ||= []).push({
      run_id: r.run_id,
      started: r.started,
      trials: ex.length,
      trials_to_win: firstWin >= 0 ? firstWin + 1 : null,
      gate1_pass_rate: ex.length ? passes.length / ex.length : 0,
      best_speedup: best,
    });
  }

  return NextResponse.json({
    backend,
    counts: { runs: runs.length, experiments: experiments.length, skills: skills.length, do_not_repeat: dnr.length },
    runs,
    skills,
    do_not_repeat: dnr,
    lessons_md: lessons?.text_md || "",
    compounding,
    comparisons: Object.values(latestComparison),
  });
}
