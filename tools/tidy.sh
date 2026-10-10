#!/usr/bin/env bash
# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
# Run pinned clang-tidy over the compilation database, generating it if absent.
# Files under tools/tidy_exclude.txt are temporarily excluded. Pass files or
# directories to check only those paths; without arguments check all sources.
# Usage: tools/tidy.sh [file-or-directory ...]

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

if [[ ! -f compile_commands.json ]]; then
  tools/compile_commands.sh
fi

sdk="$(xcrun --show-sdk-path)"
logs="$(mktemp -d "$PWD/.tidy.XXXXXX")"
trap 'rm -rf "$logs"' EXIT

# Select from the Bazel database rather than walking the filesystem, and
# derive a header filter that cannot report findings in excluded packages.
python3 tools/tidy_files.py "$PWD" "$logs/files" "$logs/header_filter" "$@"
header_filter="$(cat "$logs/header_filter")"

mapfile -d '' -t files < "$logs/files"
if ((${#files[@]})); then
  # The PyPI binary lacks Apple's SDK configuration. Explicitly supply the
  # sysroot (including its libc++ headers) for every translation unit.
  export sdk logs header_filter
  printf '%s\0' "${files[@]}" | xargs -0 -n 1 -P 4 bash -c '
    file="$1"
    log="$logs/$(basename "$file").$$"
    if ! uvx --quiet --from clang-tidy==21.1.0 clang-tidy -p . \
        --header-filter="$header_filter" \
        --extra-arg=-isysroot --extra-arg="$sdk" "$file" > "$log" 2>&1; then
      cat "$log"
      touch "$logs/failed.$$"
    fi
  ' _
fi
failures=("$logs"/failed.*)
if [[ -e "${failures[0]}" ]]; then
  echo "tidy: ${#files[@]} files checked; findings or compile errors (failed: ${#failures[@]})" >&2
  exit 1
fi
echo "tidy: ${#files[@]} files checked; 0 findings, 0 compile errors"
