#!/usr/bin/env bash
# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
# Run the non-manual //ferret tests under Apple clang source-based coverage and
# print line coverage of first-party production sources per package and
# overall. The combined LCOV report stays in
# bazel-out/_coverage/_coverage_report.dat. Arguments are extra flags for
# `bazel coverage`, for example --local_test_jobs=4.
# Usage: tools/coverage.sh [bazel-flag ...]

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# collect_cc_coverage.sh turns each test's profiles into LCOV with these.
llvm_cov="$(xcrun --find llvm-cov)"
llvm_profdata="$(xcrun --find llvm-profdata)"
bazel coverage --config=coverage \
  --test_env=LLVM_COV="$llvm_cov" --test_env=LLVM_PROFDATA="$llvm_profdata" \
  "$@" -- //ferret/...

# The source files of every //ferret library and binary that is not testonly.
production='kind("cc_library|cc_binary", //ferret/...)
  except attr(testonly, 1, //ferret/...)'
sources="$(mktemp)"
trap 'rm -f "$sources"' EXIT
bazel query --output=label \
  "kind('source file', labels(srcs, $production) + labels(hdrs, $production))" \
  > "$sources"
report="$(bazel info output_path)/_coverage/_coverage_report.dat"
python3 tools/coverage_report.py "$report" "$sources"
