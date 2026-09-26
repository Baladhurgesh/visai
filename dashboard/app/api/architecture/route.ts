import { NextResponse } from "next/server";
import { backend, count, envValue } from "@/lib/db";

export const dynamic = "force-dynamic";

const COLLECTIONS = [
  "runs", "experiments", "events", "skills", "do_not_repeat", "lessons", "reflections", "kernels",
  "profiles", "comparisons", "sessions", "session_messages",
];

export async function GET() {
  const counts: Record<string, number> = {};
  await Promise.all(COLLECTIONS.map(async (c) => (counts[c] = await count(c).catch(() => 0))));
  return NextResponse.json({
    backend,
    writer_model: envValue("OPENROUTER_MODEL"),
    reflect_model: envValue("OPENROUTER_REFLECT_MODEL") || envValue("OPENROUTER_MODEL"),
    embed_model: envValue("OPENROUTER_EMBED_MODEL") || "openai/text-embedding-3-small",
    counts,
  });
}
