import { NextResponse } from "next/server";
import { find, findOne } from "@/lib/db";

export const dynamic = "force-dynamic";

export async function GET(_req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;
  const [run, experiments, events, reflections, profile] = await Promise.all([
    findOne("runs", { run_id: id }),
    find("experiments", { run_id: id }, { sort: { ts: 1 } }),
    find("events", { run_id: id }, { sort: { ts: 1 } }),
    find("reflections", { run_id: id }, { sort: { ts: 1 } }),
    findOne("profiles", { run_id: id }),
  ]);
  return NextResponse.json({ run, experiments, events: events.slice(-400), reflections, profile });
}
