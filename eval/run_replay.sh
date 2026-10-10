#!/usr/bin/env bash
# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
#
# The engine replay and the wall-clock break-even.
#
#   eval/run_replay.sh --bin DIR --out DIR [--eval_dir DIR] [--rg PATH] [--lock]
#          [--max_calls_per_snapshot N] [--snapshot SHA ...]
#
# DIR (--bin) holds the binaries being measured: ferret, ferret_bench and
# zoekt-bench. Stages, in order (each stage's wall time goes to OUT/stages.log):
#
#   1. build    harness.systems build: OUT/data with fresh indexes, built twice
#   2. replay   harness.replay on runs/main-k3-r2, Zoekt indexes discarded
#   3. systems  harness.systems measure: OUT/systems.json
#   4. analyze  analysis.analyze --systems --replay into OUT/results
#
# Run it on a quiet machine, after the final benchmarks. It never touches the
# evaluation directories (read only). Pass --lock to hold the heavy lock
# (~/ferret-data/bench-work/heavy.lock) for the whole run. EVAL_MAIN and RG
# override the data directory and the ripgrep binary.
#
# Snapshots go XL, L, then S, so Chromium's Zoekt index is built, used and
# discarded before Linux's is built. Peak disk is about 14 GB (20 GB worst case), during Chromium's Zoekt phase.
set -euo pipefail

if [ -z "${REPLAY_CAFFEINATED:-}" ]; then
  export REPLAY_CAFFEINATED=1
  exec caffeinate -is "$0" "$@"
fi

EVAL_MAIN="${EVAL_MAIN:-$HOME/ferret-data/eval-main}"
TASKS="$EVAL_MAIN/analysis-r2/tasks.jsonl"
COMBINED="$EVAL_MAIN/analysis-r2/combined"
REPLAY_RUN="$EVAL_MAIN/runs/main-k3-r2"
SNAPSHOTS_SOURCE="$EVAL_MAIN/snapshots"
RG="${RG:-$HOME/ferret-data/bench-work/rgb/ripgrep-15.2.0-aarch64-apple-darwin/rg}"
HERE="$(cd "$(dirname "$0")" && pwd)"
EVAL_DIR="$HERE"
MAX_CALLS=800
BIN="" OUT="" LOCK_HELD=0
ONLY=()

while [ $# -gt 0 ]; do
  case "$1" in
    --bin) BIN="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --eval_dir) EVAL_DIR="$2"; shift 2 ;;
    --rg) RG="$2"; shift 2 ;;
    --max_calls_per_snapshot) MAX_CALLS="$2"; shift 2 ;;
    --snapshot) ONLY+=("$2"); shift 2 ;;
    --lock) LOCK_HELD=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$BIN" ] && [ -n "$OUT" ] || { echo "usage: eval/run_replay.sh --bin DIR --out DIR" >&2; exit 2; }
for b in ferret ferret_bench zoekt-bench; do
  [ -x "$BIN/$b" ] || { echo "missing $BIN/$b" >&2; exit 2; }
done
[ -x "$RG" ] || { echo "missing ripgrep at $RG" >&2; exit 2; }
PY="$EVAL_DIR/.venv/bin/python"
[ -x "$PY" ] || { echo "missing $PY; run 'uv sync' in eval/" >&2; exit 2; }
[ ! -e "$OUT/replay/replay.csv" ] || { echo "$OUT already has a replay; use a new directory" >&2; exit 2; }

if [ "$LOCK_HELD" = 1 ]; then
  LOCK="$HOME/ferret-data/bench-work/heavy.lock"
  while ! mkdir "$LOCK" 2>/dev/null; do
    echo "heavy lock held by $(cat "$LOCK/owner" 2>/dev/null); waiting 30 s" >&2
    sleep 30
  done
  echo replay > "$LOCK/owner"
  trap 'rm -f "$LOCK/owner"; rmdir "$LOCK"' EXIT
fi

mkdir -p "$OUT"
cd "$EVAL_DIR"

# XL, then L, then S (sorted); restricted to the --snapshot arguments if any.
SNAPSHOT_ARGS=()
while read -r sha; do
  SNAPSHOT_ARGS+=("--snapshot=$sha")
done < <("$PY" - "$TASKS" ${ONLY[@]+"${ONLY[@]}"} <<'PYEOF'
import json, sys
tasks = [json.loads(line) for line in open(sys.argv[1])]
only = set(sys.argv[2:])
seen = []
for bucket in ("XL", "L", "S"):
    for sha in sorted({t["snapshot"] for t in tasks if t["bucket"] == bucket}):
        if (not only or sha in only) and sha not in seen:
            seen.append(sha)
print("\n".join(seen))
PYEOF
)
[ ${#SNAPSHOT_ARGS[@]} -gt 0 ] || { echo "no snapshots selected" >&2; exit 2; }

stage() {  # stage NAME COMMAND...
  local name="$1"; shift
  local start; start=$(date +%s)
  echo "== $name: $(date)" | tee -a "$OUT/stages.log"
  "$@"
  echo "== $name done in $(( $(date +%s) - start )) s" | tee -a "$OUT/stages.log"
}

stage build "$PY" -m harness.systems build \
  --ferret "$BIN/ferret" --run "$REPLAY_RUN" \
  --snapshots "$SNAPSHOTS_SOURCE" --out "$OUT/data" "${SNAPSHOT_ARGS[@]}"

mkdir -p "$OUT/replay"
stage replay "$PY" -m harness.replay \
  --run "$REPLAY_RUN" --data "$OUT/data" \
  --ferret "$BIN/ferret" --rg "$RG" \
  --zoekt "$BIN/zoekt-bench" --ferret_bench "$BIN/ferret_bench" \
  --max_calls_per_snapshot "$MAX_CALLS" --seed 0 --repeats 3 \
  --discard_zoekt_indexes "${SNAPSHOT_ARGS[@]}" \
  --out "$OUT/replay/replay.csv"

stage systems "$PY" -m harness.systems measure \
  --replay "$OUT/replay/replay.csv" --builds "$OUT/data/builds.json" \
  --tasks "$TASKS" --out "$OUT/systems.json"

stage analyze "$PY" -m analysis.analyze \
  --tasks "$TASKS" --run "$COMBINED" --out "$OUT/results" \
  --systems "$OUT/systems.json" --replay "$OUT/replay/replay.csv"

echo "done: $OUT/results" | tee -a "$OUT/stages.log"
