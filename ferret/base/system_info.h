// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Facts about the machine that size thread pools and align memory advice.
//
// The query pool runs one thread per performance core: on an M3 that is 4 of
// 8 cores, and scheduling verification onto efficiency cores would put a slow
// worker on the critical path of every query. Segment sections are aligned to
// the page size so that madvise can target each one.

#ifndef FERRET_BASE_SYSTEM_INFO_H_
#define FERRET_BASE_SYSTEM_INFO_H_

#include <cstdint>

namespace ferret {

// Returns the number of performance cores: sysctl hw.perflevel0.physicalcpu
// on Apple silicon, falling back to hw.physicalcpu and then to
// std::thread::hardware_concurrency(). Always at least 1. Computed once.
// Thread-safe.
int PerformanceCoreCount();

// Returns the virtual memory page size in bytes (16 KiB on arm64 macOS).
// Computed once. Thread-safe.
int64_t PageSize();

}  // namespace ferret

#endif  // FERRET_BASE_SYSTEM_INFO_H_
