<!-- Copyright 2026 The Ferret Authors. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Benchmarks

Four questions about code search for coding agents. Each compares Ferret
with what an agent would otherwise use: **ripgrep**, which coding agents run
today, and **Zoekt**, the most widely used open-source code search index.

| | Question | Compared | Repositories |
|---|---|---|---|
| 1 | How long does a search take? | Ferret, Zoekt, ripgrep | Linux, Chromium |
| 2 | Does search slow down when several agents search at once? | Ferret, ripgrep | Linux |
| 3 | What does another worktree cost? | Ferret, Zoekt | Linux, Chromium |
| 4 | How soon can an agent find its own edit? | Ferret, Zoekt | Linux |

The latest results are in [results/2026-10-06](results/2026-10-06/README.md),
with [notes](results/2026-10-06/NOTES.md) on that run.

## Method

**Repositories.** Linux v7.3-rc1 (`cee9395acd80`, about 96,000 files) and
Chromium `d7bd2de3cbd6` (about 508,000 files), on a case-sensitive APFS
volume: Linux has files whose names differ only in case.

**Queries.** 200 per repository ([queries/](queries)): 100 identifiers, 50
pieces of text, 50 regular expressions. They were picked in a seeded shuffle
from the 1,000-query mixes that `queries/gen_mix.py` draws from each
repository's own code.

**1. Speed.** Every search asks for every matching line, without context, as
`rg -n` prints them. Ferret is timed through the engine its daemon runs, in
process. Zoekt is timed through its own searcher, in process, on all eight
cores. ripgrep runs as a process.
- Runs: Ferret and Zoekt run each query three times after one warm-up pass,
  and the median counts. ripgrep runs each query once, because it takes
  seconds and varies little. Ferret and ripgrep take turns, query by query.
- Matches: compared with ripgrep's by a digest of the (path, line) pairs.
  Zoekt indexes what its defaults allow and skips files that contain a NUL
  byte or more than 20,000 distinct trigrams, such as Linux's `MAINTAINERS`
  or minified JavaScript, which Ferret and ripgrep search: 49 files on Linux
  and 45,205 on Chromium, most of them binary test data.
- The wait for edits: by default, Ferret's server first waits until every
  earlier write to the worktree is visible, about 12 ms of macOS file-event
  latency. Zoekt does not wait and cannot see uncommitted edits. The table
  therefore reports Ferret's search without the wait, and adds a row with it.

**2. Agents at once.** Each agent searches in its own git worktree of the
same commit. Each agent runs its own sample of 30 queries back to back, and
all agents start together, in groups of 1, 2, 4 and 8.
- Ferret: all worktrees share one index, and Ferret searches as its daemon
  does by default, waiting for each worktree's recent edits.
- ripgrep: each agent reads its own worktree.

**3. Another worktree.** Worktree k is at a commit that changes 0.5% of the
files of worktree k - 1 (`synth_commits.py`; 479 files on Linux).
- Ferret: the time to register the worktree into the shared index until it
  is searchable, and how much the index grows.
- Zoekt: indexes branches together. A new branch means rebuilding that
  index, so each row is a full build of branches 1..k.
- A last row adds a worktree at the same commit as the first.

**4. Your own edit.** 100 edits, after one warm-up edit. Each appends a
unique line to a file, then times how long until a search finds it.
- Ferret: notices the write through its file watcher.
- Zoekt: needs a commit, an index update (a delta build) and a reload.
- ripgrep reads the files at every search, so its answer is always current.

**Machine.** A MacBook Air M3 (16 GB, 8 cores) on AC power, kept awake,
with no other benchmark running. A step during which the machine slept is
thrown away. Each results directory's `machine.txt`
records the commit and the tool versions.

## Running

```sh
bench/run.sh CORPUS bench/queries/linux.jsonl bench/results/DATE/linux WORK
bench/run.sh CORPUS bench/queries/chromium.jsonl bench/results/DATE/chromium WORK \
    --steps speed,worktrees --worktrees 3
eval/.venv/bin/python -m bench.report bench/results/DATE
```

WORK must be on the corpus's volume, so that `cp -c` clones the corpus
rather than copying it. The corpus itself is never written. Linux takes
about an hour; Chromium about an hour and a half.

| File | Role |
|---|---|
| `run.sh` | Runs the benchmarks on one corpus. |
| `search_agents.cc` | Benchmarks 1 and 2: Ferret and ripgrep. |
| `run_m3.sh`, `synth_commits.py`, `../ferret/workspace/workspace_benchmark.cc` | Benchmarks 3 and 4 for Ferret. |
| `zoekt/` | Zoekt's side of benchmarks 1, 3 and 4. |
| `ferret/` | `ferret_bench files`: the file list Zoekt indexes. |
| `report.py`, `m3_tables.py` | The tables and charts. |
