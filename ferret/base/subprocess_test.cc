// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/base/subprocess.h"

#include <cstddef>
#include <string>

#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include "ferret/testing/status_matchers.h"

namespace ferret {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(SubprocessTest, CapturesStdoutAndExitCode) {
  FERRET_ASSERT_OK_AND_ASSIGN(SubprocessResult result,
                              RunSubprocess({"printf", "hello %s", "world"}));
  EXPECT_EQ(result.exit_code, 0);
  EXPECT_EQ(result.stdout_data, "hello world");
}

TEST(SubprocessTest, NonZeroExitIsNotAnError) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      SubprocessResult result,
      RunSubprocess({"sh", "-c", "echo oops >&2; exit 3"}));
  EXPECT_EQ(result.exit_code, 3);
  EXPECT_EQ(result.stderr_data, "oops\n");
}

TEST(SubprocessTest, ArgumentsAreNotInterpretedByAShell) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      SubprocessResult result,
      RunSubprocess({"printf", "%s", "$(echo pwned); ls"}));
  EXPECT_EQ(result.stdout_data, "$(echo pwned); ls");
}

TEST(SubprocessTest, RunsInRequestedDirectory) {
  FERRET_ASSERT_OK_AND_ASSIGN(SubprocessResult result,
                              RunSubprocess({"pwd"}, {.cwd = "/"}));
  EXPECT_EQ(result.stdout_data, "/\n");
}

TEST(SubprocessTest, AppliesEnvironmentOverrides) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      SubprocessResult result,
      RunSubprocess({"sh", "-c", "printf '%s|%s' \"$FERRET_A\" \"$HOME\""},
                    {.env_overrides = {"FERRET_A=1", "HOME=/nowhere"}}));
  EXPECT_EQ(result.stdout_data, "1|/nowhere");
}

TEST(SubprocessTest, FeedsStdin) {
  FERRET_ASSERT_OK_AND_ASSIGN(
      SubprocessResult result,
      RunSubprocess({"tr", "a-z", "A-Z"}, {.stdin_data = "shout"}));
  EXPECT_EQ(result.stdout_data, "SHOUT");
}

TEST(SubprocessTest, DrainsLargeOutputWithoutDeadlock) {
  // 4 MiB on stdout and stderr each: far beyond a pipe buffer.
  FERRET_ASSERT_OK_AND_ASSIGN(
      SubprocessResult result,
      RunSubprocess(
          {"sh", "-c",
           "head -c 4194304 /dev/zero; head -c 4194304 /dev/zero >&2"}));
  EXPECT_EQ(result.stdout_data.size(), 4194304u);
  EXPECT_EQ(result.stderr_data.size(), 4194304u);
}

TEST(SubprocessTest, LargeStdinWithEchoedOutputDoesNotDeadlock) {
  // `cat` fills its stdout pipe long before it has read 4 MiB of stdin, so
  // stdin must be written only as fast as the pipe accepts it.
  const std::string input(size_t{4} << 20, 'x');
  FERRET_ASSERT_OK_AND_ASSIGN(
      SubprocessResult result,
      RunSubprocess({"cat"},
                    {.stdin_data = input, .timeout = absl::Seconds(30)}));
  EXPECT_EQ(result.stdout_data.size(), input.size());
  EXPECT_EQ(result.stdout_data, input);
}

TEST(SubprocessTest, ChildThatIgnoresStdinIsNotAnError) {
  // The child exits without reading; the remaining stdin is dropped.
  FERRET_ASSERT_OK_AND_ASSIGN(
      SubprocessResult result,
      RunSubprocess({"true"}, {.stdin_data = std::string(size_t{4} << 20, 'x'),
                               .timeout = absl::Seconds(30)}));
  EXPECT_EQ(result.exit_code, 0);
}

TEST(SubprocessTest, EnforcesOutputCap) {
  EXPECT_THAT(
      RunSubprocess({"sh", "-c", "head -c 1048576 /dev/zero"},
                    {.max_output_bytes = 1024}),
      StatusIs(absl::StatusCode::kResourceExhausted, HasSubstr("output cap")));
}

TEST(SubprocessTest, EnforcesTimeout) {
  EXPECT_THAT(
      RunSubprocess({"sleep", "10"}, {.timeout = absl::Milliseconds(50)}),
      StatusIs(absl::StatusCode::kDeadlineExceeded));
}

TEST(SubprocessTest, MissingBinaryIsAnError) {
  EXPECT_FALSE(RunSubprocess({"ferret-no-such-binary-xyz"}).ok());
}

TEST(SubprocessTest, EmptyArgvIsInvalid) {
  EXPECT_THAT(RunSubprocess({}), StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace ferret
