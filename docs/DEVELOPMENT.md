# Development

Ferret targets macOS arm64 with Bazelisk (Bazel 9.2 from `.bazelversion`),
Apple clang, Go, Python 3.13, uv, and Bash 4 or later. From the repository root:

```sh
bazel build //ferret/... //bench/...
bazel test //ferret/... //bench/...
bazel test --config=asan //ferret/...
bazel test --config=tsan //ferret/...
tools/lint.sh                         # Use --fix to apply formatting.
tools/compile_commands.sh             # Build and generate compile_commands.json.
```

`tools/coverage.sh` runs the non-manual `//ferret/...` tests under Apple
clang source-based coverage (`--config=coverage`), leaves the combined LCOV
report in `bazel-out/_coverage/_coverage_report.dat`, and prints line coverage
per `ferret/` package and overall (the target is at least 85%). Only sources of
non-testonly `//ferret` targets count: tests, test-only libraries, generated
files, and external repositories do not, and a source that no test links is
named instead of counted. Arguments go to `bazel coverage`, for example
`--local_test_jobs=4`. An uncached run takes about 23 CPU-minutes.

For lint and benchmark Python tests, install the locked Python dependencies
with `cd eval && uv sync --locked`, then run
`eval/.venv/bin/python -m pytest bench/tests` from the repository root.
The compilation database uses Bazel's execution root for external
headers (including Abseil, RE2, googletest, and tree-sitter). Regenerate it
after changing Bazel dependencies or targets; it is ignored by git.
For one-off clangd checks, use `clangd --check=ferret/index/index.cc --tweaks=`;
`--tweaks=` disables experimental refactoring checks that may log failures
unrelated to compilation.

Run `tools/tidy.sh` to check all C and C++ compilation units with pinned
clang-tidy 21.1.0. It generates `compile_commands.json` if missing; regenerate
the database with `tools/compile_commands.sh` after changing Bazel deps. Pass
file or directory paths to check a subset, for example
`tools/tidy.sh ferret/index/`. The temporary exclusions in
`tools/tidy_exclude.txt` apply to both full and scoped runs.

See [README.md](../README.md) for a quickstart and
[bench/README.md](../bench/README.md) for the benchmarks.
