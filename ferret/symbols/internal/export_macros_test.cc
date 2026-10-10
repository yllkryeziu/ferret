// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Tests offset-preserving blanking of C++ class export annotations.

#include "ferret/symbols/internal/export_macros.h"

#include <cstddef>
#include <string>

#include "absl/strings/string_view.h"
#include "gtest/gtest.h"

namespace ferret {
namespace {

TEST(ExportMacrosTest, BlanksAnnotationsWithoutMovingAnyByte) {
  constexpr absl::string_view kSource =
      "class CONTENT_EXPORT WebContents {};\n"
      "struct BASE_EXPORT Foo {};\r\n"
      "union COMPONENT_EXPORT(NET) Data {};\n"
      "class FIRST_EXPORT\n SECOND_EXPORT Widget {};\n"
      "class __attribute__((visibility(\"default\"))) Visible {};\n";
  constexpr absl::string_view kExpected =
      "class                WebContents {};\n"
      "struct             Foo {};\r\n"
      "union                       Data {};\n"
      "class             \n               Widget {};\n"
      "class                                        Visible {};\n";
  std::string buffer;
  const absl::string_view result =
      symbols_internal::BlankExportMacros(kSource, buffer);
  ASSERT_EQ(result.size(), kSource.size());
  ASSERT_EQ(result, kExpected);
  for (size_t i = 0; i < kSource.size(); ++i) {
    if (kSource[i] == '\n') EXPECT_EQ(result[i], '\n') << i;
    if (result[i] != kSource[i]) EXPECT_EQ(result[i], ' ') << i;
  }
  std::string again;
  EXPECT_EQ(symbols_internal::BlankExportMacros(result, again), result);
}

TEST(ExportMacrosTest, LeavesNamesStringsCommentsAndTemplatesAlone) {
  constexpr absl::string_view kSource =
      "class FOO {}; template <class T> struct BAR {};\n"
      "const char* text = \"class CONTENT_EXPORT Foo\";\n"
      "// class CONTENT_EXPORT Foo\n"
      "/* struct BASE_EXPORT Foo */\n"
      "auto raw = R\"tag(class CONTENT_EXPORT Foo)tag\";\n"
      "struct NORMAL field;\n";
  std::string buffer;
  EXPECT_EQ(symbols_internal::BlankExportMacros(kSource, buffer), kSource);
}

}  // namespace
}  // namespace ferret
