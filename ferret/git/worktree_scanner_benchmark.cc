// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Prints incremental registration phase times and file counts for a checkout.

#include "ferret/git/worktree_scanner.h"

#include <iostream>

#include "absl/status/statusor.h"
#include "absl/time/time.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: worktree_scanner_benchmark CHECKOUT\n";
    return 2;
  }
  const absl::StatusOr<ferret::WorktreeScan> scan =
      ferret::ScanWorktree(argv[1], {.verify_hint_fraction = 0.0});
  if (!scan.ok()) {
    std::cerr << scan.status() << '\n';
    return 1;
  }
  const ferret::ScanReport& r = scan->report;
  std::cout << "files=" << r.files << " hinted=" << r.hinted
            << " dirty=" << r.dirty << " untracked=" << r.untracked
            << " unmerged=" << r.unmerged
            << " assume_unchanged=" << r.assume_unchanged
            << " conversion=" << r.conversion
            << " skip_worktree=" << r.skip_worktree << " symlink=" << r.symlink
            << " not_regular=" << r.not_regular << " gitlink=" << r.gitlink
            << " attribute_candidates=" << r.attribute_candidates
            << " case_collisions=" << r.case_collisions << '\n'
            << "git_commands=" << absl::ToDoubleMilliseconds(r.git_commands)
            << "ms attributes=" << absl::ToDoubleMilliseconds(r.attributes)
            << "ms lstat=" << absl::ToDoubleMilliseconds(r.lstat) << "ms\n";
}
