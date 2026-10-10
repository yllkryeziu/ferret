// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Deterministic random inputs for the posting-list tests.
//
// Draws are written by hand on top of std::mt19937_64, whose output sequence
// the standard fixes, because the std distributions differ across standard
// libraries and would make a seed unreproducible elsewhere. Seeds come from
// ferret_testing::TestSeed (ferret/testing/test_seed.h).

#ifndef FERRET_POSTINGS_TESTING_RANDOM_POSTING_LISTS_H_
#define FERRET_POSTINGS_TESTING_RANDOM_POSTING_LISTS_H_

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "ferret/postings/ids.h"

namespace ferret_testing {

// Returns an integer in [0, bound). Requires bound > 0. The modulo bias is
// irrelevant for tests.
uint64_t UniformBelow(std::mt19937_64& rng, uint64_t bound);

// Returns `size` strictly increasing ids starting at `first`. Each later gap
// minus one is drawn from [0, 2^gap_bits), clamped so that the list fits in
// 32 bits: gap_bits = 0 gives a run of consecutive ids, and gap_bits = 32
// spends most of the id space on the first draws. Requires
// 0 <= gap_bits <= 32 and first + size - 1 <= UINT32_MAX.
std::vector<ferret::ChunkId> RandomPostingList(std::mt19937_64& rng,
                                               size_t size, int gap_bits,
                                               ferret::ChunkId first);

}  // namespace ferret_testing

#endif  // FERRET_POSTINGS_TESTING_RANDOM_POSTING_LISTS_H_
