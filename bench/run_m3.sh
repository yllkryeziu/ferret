#!/usr/bin/env bash
# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Usage: bench/run_m3.sh CORPUS RESULTS_DIR WORK_DIR [options]
#
# Benchmarks 3 and 4 of bench/README.md, each a step (bench/run.sh calls it):
#   g3         sharing for k = 1..K worktrees at synthetic commits, against
#              Zoekt multi-branch (g3.md)
#   freshness  edit to searchable: Ferret Notify and watcher, Zoekt commit +
#              delta build + reload (freshness.md, .csv, .png)
#
# Options:
#   --steps LIST         comma-separated steps (default g3,freshness)
#   --k N                worktrees for g3 (1..10, default 10)
#   --diff-fraction F    share of the eligible files each worktree adds to the
#                        diff from the previous one (default 0.005)
#   --trials N           freshness trials per path (default 200), after
#                        --warmup N more that are run and left out (default 1)
#   --seed N             seed of the commits (default 1)
#   --keep               keep the disposable copy of the corpus
#
# WORK_DIR must be disposable, outside CORPUS, and on the corpus's volume if
# cp -c is to clone. The corpus is never written: a copy-on-write clone in
# WORK_DIR gets the synthetic commits and the git worktrees. Run it alone. A
# step that spans a machine sleep is marked INVALID. Each step removes its
# worktrees and indexes when it is done.
set -euo pipefail

usage() { sed -n '4,27p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
[[ $# -ge 3 ]] || usage
[[ -d "$1" ]] || { echo "corpus $1 not found (the runner never mounts volumes)" >&2; exit 2; }
corpus=$(cd "$1" && pwd -P)
results=$2
mkdir -p "$3" "$results"
work=$(cd "$3" && pwd -P)
results=$(cd "$results" && pwd -P)
shift 3
if [[ "$work" == "$corpus" || "$work" == "$corpus/"* ]]; then
  echo 'WORK_DIR must not be inside the corpus' >&2
  exit 2
fi
steps=g3,freshness k=10 fraction=0.005 trials=200 warmup=1 seed=1 keep=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps) steps=$2; shift 2 ;;
    --k) k=$2; shift 2 ;;
    --diff-fraction) fraction=$2; shift 2 ;;
    --trials) trials=$2; shift 2 ;;
    --warmup) warmup=$2; shift 2 ;;
    --seed) seed=$2; shift 2 ;;
    --keep) keep=1; shift ;;
    *) usage ;;
  esac
done
[[ $k =~ ^([1-9]|10)$ ]] || { echo '--k must be 1..10' >&2; exit 2; }
[[ $trials =~ ^[1-9][0-9]*$ && $warmup =~ ^[0-9]+$ ]] ||
  { echo '--trials must be positive and --warmup at least 0' >&2; exit 2; }
root=$(cd "$(dirname "$0")/.." && pwd -P)
python=${PYTHON:-$root/eval/.venv/bin/python}
# A linked worktree has no venv of its own; use the main checkout's.
[[ -x "$python" ]] ||
  python=$(dirname "$(git -C "$root" rev-parse --path-format=absolute --git-common-dir)")/eval/.venv/bin/python

scratch=$(mktemp -d "$work/m3.XXXXXX")
cleanup() {
  if [[ $keep -eq 0 ]]; then
    git -C "$scratch/corpus" worktree prune 2>/dev/null || true
    rm -rf "$scratch"
  else
    echo "kept $scratch"
  fi
}
trap cleanup EXIT

(cd "$root" && bazel build --config=opt //ferret/workspace:workspace_benchmark //bench/ferret:ferret_bench)
bazel_bin=$(cd "$root" && bazel info --config=opt bazel-bin 2>/dev/null)
bench=$bazel_bin/ferret/workspace/workspace_benchmark
ferret_bench=$bazel_bin/bench/ferret/ferret_bench
(cd "$root/bench/zoekt" && go build -o "$scratch/zoekt-bench" .)
zoekt=$scratch/zoekt-bench

"$ferret_bench" files --root="$corpus" --out="$scratch/files" >"$scratch/files.json"
cp -cR "$corpus" "$scratch/corpus"
repo=$scratch/corpus
base=$(git -C "$repo" rev-parse HEAD)

slept() {  # start end: sleeps recorded by pmset in between
  pmset -g log | awk -v a="$1" -v b="$2" '$4 == "Sleep" && $1 " " $2 >= a && $1 " " $2 <= b'
}
timed() {  # name command...: runs a step and marks it INVALID across a sleep
  local name=$1; shift
  local t0 t1 sleeps
  echo "== $name start $(date)"
  t0=$(date "+%Y-%m-%d %H:%M:%S")
  "$@"
  t1=$(date "+%Y-%m-%d %H:%M:%S")
  sleeps=$(slept "$t0" "$t1")
  if [[ -n "$sleeps" ]]; then
    printf '%s\n' "$sleeps" >"$results/$name.SLEPT"
    echo "== $name INVALID: the machine slept during it; see $results/$name.SLEPT"
  fi
  echo "== $name done $(date)"
}

worktree() {  # path rev
  git -C "$repo" worktree add --quiet --detach "$1" "$2"
}

step_g3() {
  "$python" -m bench.synth_commits chain "$repo" --files "$scratch/files" \
    --seed "$seed" --steps $((k - 1)) --diff-fraction "$fraction" --out "$results/g3-chain.json"
  local args=() branches=()
  for ((n = 1; n <= k; n++)); do
    worktree "$scratch/wt$n" "m3/k$n"
    args+=("k$n=$scratch/wt$n")
    branches+=("m3/k$n")
  done
  worktree "$scratch/wt-same" "m3/k1"
  args+=("same=$scratch/wt-same")
  "$bench" --g3 "$scratch/index-g3" "${args[@]}" >"$results/g3-ferret.log" 2>&1
  rm -rf "$scratch/index-g3"
  for ((n = 1; n <= k; n++)); do git -C "$repo" worktree remove --force "$scratch/wt$n"; done
  git -C "$repo" worktree remove --force "$scratch/wt-same"
  local joined
  joined=$(IFS=,; echo "${branches[*]}")
  "$zoekt" branches --repo "$repo" --files "$scratch/files" --branches "$joined" \
    --out "$scratch/zoekt-branches" >"$results/g3-zoekt.jsonl" 2>"$results/g3-zoekt.err"
  rm -rf "$scratch/zoekt-branches"
  (cd "$root" && "$python" -m bench.m3_tables g3 --ferret-log "$results/g3-ferret.log" \
    --zoekt "$results/g3-zoekt.jsonl" --chain "$results/g3-chain.json" --out "$results/g3.md")
}

step_freshness() {
  # Ferret edits and restores a worktree. Zoekt commits every trial, so it
  # gets its own clone, which is removed afterwards.
  worktree "$scratch/wt-fresh" "$base"
  "$bench" --freshness "$scratch/index-fresh" "$scratch/wt-fresh" "$((trials + warmup))" \
    >"$results/freshness-ferret.log" 2>&1
  git -C "$repo" worktree remove --force "$scratch/wt-fresh"
  rm -rf "$scratch/index-fresh"
  cp -cR "$corpus" "$scratch/zoekt-repo"
  "$zoekt" freshness --repo "$scratch/zoekt-repo" --files "$scratch/files" \
    --index "$scratch/zoekt-fresh" --trials "$((trials + warmup))" --seed "$seed" \
    >"$results/freshness-zoekt.jsonl" 2>"$results/freshness-zoekt.err"
  rm -rf "$scratch/zoekt-repo" "$scratch/zoekt-fresh"
  (cd "$root" && "$python" -m bench.m3_tables freshness \
    --ferret-log "$results/freshness-ferret.log" --zoekt "$results/freshness-zoekt.jsonl" \
    --out "$results/freshness.md" --csv "$results/freshness-cdf.csv" --png "$results/freshness-cdf.png" \
    --warmup "$warmup")
}

IFS=, read -ra wanted <<<"$steps"
for s in "${wanted[@]}"; do
  case "$s" in
    g3) timed g3 step_g3 ;;
    freshness) timed freshness step_freshness ;;
    *) echo "unknown step $s" >&2; exit 2 ;;
  esac
done
