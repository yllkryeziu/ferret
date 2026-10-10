// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/base/system_info.h"

#include <unistd.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include <cstddef>
#include <cstdint>
#include <thread>

#include "absl/log/check.h"

namespace ferret {
namespace {

#if defined(__APPLE__)
// Returns the integer sysctl `name`, or 0 if it does not exist (for example
// hw.perflevel0.* on Intel Macs).
int SysctlInt(const char* name) {
  int value = 0;
  size_t size = sizeof(value);
  if (sysctlbyname(name, &value, &size, nullptr, 0) != 0) return 0;
  return value;
}
#endif

int ComputePerformanceCoreCount() {
#if defined(__APPLE__)
  for (const char* name : {"hw.perflevel0.physicalcpu", "hw.physicalcpu"}) {
    if (const int count = SysctlInt(name); count > 0) return count;
  }
#endif
  const unsigned int hardware = std::thread::hardware_concurrency();
  return hardware > 0 ? static_cast<int>(hardware) : 1;
}

}  // namespace

int PerformanceCoreCount() {
  static const int count = ComputePerformanceCoreCount();
  return count;
}

int64_t PageSize() {
  static const int64_t page_size = [] {
    const int64_t size = sysconf(_SC_PAGESIZE);
    CHECK_GT(size, 0) << "sysconf(_SC_PAGESIZE)";
    return size;
  }();
  return page_size;
}

}  // namespace ferret
