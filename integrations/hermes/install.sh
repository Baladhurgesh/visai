#!/usr/bin/env bash
# Install the visai-opt skill for Hermes or NemoClaw/OpenClaw.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
target="${1:-hermes}"
case "$target" in
  hermes) dest="$HOME/.hermes/skills/visai-opt" ;;
  nemoclaw|openclaw) dest="${OPENCLAW_SKILLS_DIR:-$HOME/.openclaw/skills}/visai-opt" ;;
  *) echo "usage: $0 [hermes|nemoclaw]" >&2; exit 2 ;;
esac
mkdir -p "$dest"
cp "$HERE/SKILL.md" "$HERE/launch_query.txt" "$HERE/tools.json" "$dest/"
cp "$HERE"/tool_spec_*.json "$dest/" 2>/dev/null || true
echo "installed visai-opt -> $dest"
