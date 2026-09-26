import { readFileSync } from "fs";
import { NextResponse } from "next/server";
import { join } from "path";

export const dynamic = "force-dynamic";

// The Minutes benchmark writes its latest run next to the app repo.
const CANDIDATES = [
  process.env.MINUTES_BENCH,
  join(process.cwd(), "../../minutes/bench/results/latest/results.json"),
  "/Users/bala/Documents/minutes/bench/results/latest/results.json",
].filter(Boolean) as string[];

export async function GET() {
  for (const path of CANDIDATES) {
    try {
      return NextResponse.json(JSON.parse(readFileSync(path, "utf8")));
    } catch {
      // try the next location
    }
  }
  return NextResponse.json({ error: "no benchmark results found" }, { status: 404 });
}
