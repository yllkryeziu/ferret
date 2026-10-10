// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/text/trigram.h"

#include <string>

#include "absl/strings/escaping.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace ferret {

std::string TrigramToString(Trigram trigram) {
  const char bytes[3] = {static_cast<char>(TrigramByte(trigram, 0)),
                         static_cast<char>(TrigramByte(trigram, 1)),
                         static_cast<char>(TrigramByte(trigram, 2))};
  return absl::StrCat("\"", absl::CHexEscape(absl::string_view(bytes, 3)),
                      "\"");
}

}  // namespace ferret
