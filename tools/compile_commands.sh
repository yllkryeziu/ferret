#!/usr/bin/env bash
# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
# Generate clangd's compilation database from Bazel's actual compile actions.
# Build first so generated headers exist. Re-run after changing Bazel targets or
# dependencies; compile_commands.json is local to this workspace and ignored.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

bazel build //ferret/... //bench/...
aquery="$(mktemp)"
trap 'rm -f "$aquery"' EXIT
bazel aquery 'mnemonic("CppCompile", //ferret/... + //bench/...)' \
  --output=jsonproto > "$aquery"
python3 tools/compile_commands.py "$aquery" "$PWD" \
  "$(bazel info execution_root 2>/dev/null)" "$(xcrun --find clang++)"
