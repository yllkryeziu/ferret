#!/usr/bin/env bash
# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Usage: bench/run.sh CORPUS QUERIES RESULTS_DIR WORK_DIR [options]
#
# Runs the benchmarks of bench/README.md on one corpus, a git checkout, and
# writes their raw output to RESULTS_DIR. bench/report.py renders it.
#   speed      every match of each query: Ferret, ripgrep and Zoekt
#   agents     1, 2, 4, ... agents searching at once, each in its own
#              worktree: Ferret and ripgrep
#   worktrees  worktrees 1..K, each at a commit with changed files, made
#              searchable: Ferret and Zoekt (bench/run_m3.sh g3)
#   edits      an edit until a search finds it: Ferret and Zoekt
#              (bench/run_m3.sh freshness)
#
# Options:
#   --steps LIST     comma-separated steps (default speed,agents,worktrees,edits)
#   --agents N       most agents at once (default 8)
#   --worktrees K    worktrees (1..10, default 8)
#
# WORK_DIR must be disposable, outside CORPUS, and on the corpus's volume, so
# that cp -c clones the corpus instead of copying it. The corpus is never
# written. Run it on AC power, which it checks, and nothing else meanwhile.
# The machine is kept awake, but closing the lid can still sleep it: a step
# that spans a sleep writes STEP.SLEPT and stops the run.
set -euo pipefail
usage() { sed -n '4,26p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
[[ $# -ge 4 && -d "$1" && -f "$2" ]] || usage
corpus=$(cd "$1" && pwd -P)
queries=$(cd "$(dirname "$2")" && pwd -P)/$(basename "$2")
mkdir -p "$3" "$4"
results=$(cd "$3" && pwd -P)
work=$(cd "$4" && pwd -P)
shift 4
steps=speed,agents,worktrees,edits agents=8 worktrees=8
while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps) steps=$2; shift 2 ;;
    --agents) agents=$2; shift 2 ;;
    --worktrees) worktrees=$2; shift 2 ;;
    *) usage ;;
  esac
done
if [[ "$work" == "$corpus" || "$work" == "$corpus/"* ]]; then
  echo 'WORK_DIR must not be inside the corpus' >&2
  exit 2
fi
[[ $agents =~ ^[1-9][0-9]*$ ]] || { echo '--agents must be positive' >&2; exit 2; }
# caffeinate cannot keep a Mac on battery awake with its lid closed.
if ! pmset -g batt | head -1 | grep -q "'AC Power'"; then
  echo 'on battery: plug in the charger first' >&2
  exit 2
fi

root=$(cd "$(dirname "$0")/.." && pwd -P)
(cd "$root" && bazel build --config=opt //bench:search_agents //bench/ferret:ferret_bench)
bin=$(cd "$root" && bazel info --config=opt bazel-bin 2>/dev/null)
scratch=$(mktemp -d "$work/run.XXXXXX")
cleanup() {
  git -C "$scratch/corpus" worktree prune 2>/dev/null || true
  rm -rf "$scratch"
}
trap cleanup EXIT
(cd "$root/bench/zoekt" && go build -o "$scratch/zoekt-bench" .)
rg=$(command -v rg)

# Keep the machine awake for as long as this script runs.
caffeinate -dimsu -w $$ &

{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "ferret: $(git -C "$root" rev-parse --short HEAD)$(git -C "$root" diff --quiet HEAD -- ferret bench ':!*.md' || echo ' (modified)')"
  echo "corpus: $(basename "$corpus") at $(git -C "$corpus" rev-parse --short HEAD)"
  echo "machine: $(sysctl -n hw.model), $(($(sysctl -n hw.memsize) >> 30)) GiB, $(sysctl -n hw.ncpu) cores, macOS $(sw_vers -productVersion)"
  echo "power: $(pmset -g batt | head -1 | sed "s/.*'\(.*\)'.*/\1/")"
  echo "ripgrep: $("$rg" --version | head -1)"
  echo "zoekt: $(cd "$root/bench/zoekt" && go list -m -f '{{.Version}}' github.com/sourcegraph/zoekt)"
} >"$results/machine.txt"

clone() {
  [[ -d "$scratch/corpus" ]] || cp -cR "$corpus" "$scratch/corpus"
}

step_speed() {
  clone
  "$bin/bench/search_agents" --mode=speed --index_dir="$scratch/ferret-index" \
    --queries="$queries" --rg="$rg" "$scratch/corpus" >"$results/speed.jsonl"
  du -sk "$scratch/ferret-index" | awk '{print $1 * 1024}' >"$results/speed-ferret-index-bytes"
  rm -rf "$scratch/ferret-index"
  "$bin/bench/ferret/ferret_bench" files --root="$scratch/corpus" --out="$scratch/files" >/dev/null
  "$scratch/zoekt-bench" build --root "$scratch/corpus" --files "$scratch/files" \
    --out "$scratch/zoekt-index" --parallelism 8 >"$results/speed-zoekt-build.json"
  "$scratch/zoekt-bench" search --index "$scratch/zoekt-index" --queries "$queries" \
    --mode exhaustive --context 0 --gomaxprocs 8 --warmup 1 --runs 3 \
    >"$results/speed-zoekt.jsonl"
  rm -rf "$scratch/zoekt-index"
}

step_agents() {
  clone
  local trees=("$scratch/corpus") counts=1 n
  for ((n = 2; n <= agents; n++)); do
    git -C "$scratch/corpus" worktree add --quiet --detach "$scratch/agent$n" HEAD
    trees+=("$scratch/agent$n")
  done
  for ((n = 2; n < agents; n *= 2)); do counts+=",$n"; done
  ((agents == 1)) || counts+=",$agents"
  "$bin/bench/search_agents" --mode=agents --engines=ferret,ripgrep \
    --agents="$counts" --per_agent=30 --index_dir="$scratch/agents-index" \
    --queries="$queries" --rg="$rg" "${trees[@]}" >"$results/agents.jsonl"
  for ((n = 2; n <= agents; n++)); do
    git -C "$scratch/corpus" worktree remove --force "$scratch/agent$n"
  done
  rm -rf "$scratch/agents-index"
}

slept() {  # start end: sleeps recorded by pmset in between
  pmset -g log | awk -v a="$1" -v b="$2" '$4 == "Sleep" && $1 " " $2 >= a && $1 " " $2 <= b'
}

IFS=, read -ra wanted <<<"$steps"
for s in "${wanted[@]}"; do
  echo "== $s start $(date)"
  rm -f "$results/$s.SLEPT"
  t0=$(date "+%Y-%m-%d %H:%M:%S")
  case "$s" in
    speed) step_speed ;;
    agents) step_agents ;;
    worktrees) "$root/bench/run_m3.sh" "$corpus" "$results" "$work" --steps g3 --k "$worktrees" ;;
    edits) "$root/bench/run_m3.sh" "$corpus" "$results" "$work" --steps freshness --trials 100 ;;
    *) echo "unknown step $s" >&2; exit 2 ;;
  esac
  sleeps=$(slept "$t0" "$(date "+%Y-%m-%d %H:%M:%S")")
  if [[ -n "$sleeps" ]]; then
    printf '%s\n' "$sleeps" >"$results/$s.SLEPT"
    echo "== $s INVALID: the machine slept during it; see $results/$s.SLEPT" >&2
    exit 1
  fi
  echo "== $s done $(date)"
done
