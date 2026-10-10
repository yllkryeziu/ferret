#!/usr/bin/env bash
# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Lints the tracked sources:
#   * clang-format, pinned, on C and C++;
#   * grep bans on constructs the conventions forbid;
#   * ruff on eval/ and bench/;
#   * gofmt on bench/zoekt.
# clang-format runs through uvx, so the pinned version needs no system
# install. With --fix, formatters rewrite files instead of failing.
#
# Usage: tools/lint.sh [--fix]

set -euo pipefail

readonly CLANG_FORMAT_VERSION="21.1.0"

fix=false
case "${1:-}" in
  "") ;;
  --fix) fix=true ;;
  *) echo "usage: tools/lint.sh [--fix]" >&2; exit 2 ;;
esac

cd "$(git rev-parse --show-toplevel)"
status=0

fail() {
  echo "lint: $*" >&2
  status=1
}

clang_format() {
  uvx --quiet --from "clang-format==${CLANG_FORMAT_VERSION}" clang-format "$@"
}

mapfile -t cpp_files < <(git ls-files -- '*.h' '*.cc')
if ((${#cpp_files[@]})); then
  if $fix; then
    clang_format -i "${cpp_files[@]}"
  elif ! clang_format --dry-run --Werror "${cpp_files[@]}"; then
    fail "clang-format: run tools/lint.sh --fix"
  fi
fi

# "pattern ## reason[ ## exempt file]"; each pattern is an extended regex
# matched against the C++ sources other than the exempt one.
readonly BANS=(
  '\.dump\( ## serialize JSON with SerializeJson ## ferret/rpc/json_codec.cc'
  'MutexLock [A-Za-z_][A-Za-z0-9_]*\(& ## pass the mutex by reference: MutexLock lock(mu_)'
  'std::function< ## use absl::AnyInvocable or absl::FunctionRef'
  'std::mutex ## use absl::Mutex'
  '#pragma once ## use #ifndef include guards'
  '(^|[^_A-Z])RETURN_IF_ERROR\( ## use ABSL_RETURN_IF_ERROR'
)
for ban in "${BANS[@]}"; do
  pattern="${ban%% ## *}"
  rest="${ban#* ## }"
  reason="${rest%% ## *}"
  exempt=""
  [[ "${rest}" == *" ## "* ]] && exempt="${rest#* ## }"
  files=()
  for file in "${cpp_files[@]}"; do
    [[ "${file}" == "${exempt}" ]] || files+=("${file}")
  done
  if ((${#files[@]})) && matches="$(grep -nE -- "${pattern}" "${files[@]}")"; then
    echo "${matches}" >&2
    fail "banned construct: ${reason}"
  fi
done

readonly RUFF=eval/.venv/bin/ruff
python_dirs=()
for dir in eval bench; do
  [[ -d "${dir}" ]] && python_dirs+=("${dir}")
done
if [[ -x "${RUFF}" ]]; then
  if $fix; then
    "${RUFF}" format --quiet "${python_dirs[@]}"
    "${RUFF}" check --quiet --fix "${python_dirs[@]}" || fail "ruff check"
  else
    "${RUFF}" format --quiet --check "${python_dirs[@]}" || fail "ruff format"
    "${RUFF}" check --quiet "${python_dirs[@]}" || fail "ruff check"
  fi
else
  fail "ruff not found; run 'uv sync' in eval/"
fi

mapfile -t go_files < <(git ls-files -- 'bench/*.go')
if ((${#go_files[@]})); then
  if $fix; then
    gofmt -w "${go_files[@]}"
  elif unformatted="$(gofmt -l "${go_files[@]}")" && [[ -n "${unformatted}" ]]; then
    echo "${unformatted}" >&2
    fail "gofmt: run tools/lint.sh --fix"
  fi
fi

exit "${status}"
