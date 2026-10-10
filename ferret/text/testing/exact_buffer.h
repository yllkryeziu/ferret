// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// A heap copy of a string with no slack after its last byte.
//
// SIMD kernels load 16 bytes at a time, and an off-by-one in their tail
// handling reads past the end of the input. A std::string usually has spare
// capacity after its size, where such a read goes unnoticed. Kernel tests run
// their inputs from an ExactBuffer so that ASan reports it instead.

#ifndef FERRET_TEXT_TESTING_EXACT_BUFFER_H_
#define FERRET_TEXT_TESTING_EXACT_BUFFER_H_

#include <cstddef>
#include <memory>

#include "absl/strings/string_view.h"

namespace ferret_testing {

// Not thread-safe.
class ExactBuffer final {
 public:
  explicit ExactBuffer(absl::string_view text)
      : size_(text.size()), bytes_(std::make_unique<char[]>(text.size())) {
    text.copy(bytes_.get(), text.size());
  }
  ExactBuffer(const ExactBuffer&) = delete;
  ExactBuffer& operator=(const ExactBuffer&) = delete;

  // The copy. Valid while *this is alive.
  absl::string_view view() const {
    return absl::string_view(bytes_.get(), size_);
  }

 private:
  size_t size_;
  std::unique_ptr<char[]> bytes_;
};

}  // namespace ferret_testing

#endif  // FERRET_TEXT_TESTING_EXACT_BUFFER_H_
