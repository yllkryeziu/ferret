// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/rpc/framing.h"

#include <sys/_types/_ssize_t.h>
#include <unistd.h>

#include <cstddef>
#include <string>
#include <thread>

#include "absl/status/status.h"
#include "gtest/gtest.h"

#include "ferret/rpc/json_codec.h"
#include "ferret/rpc/testing/test_pipe.h"

namespace ferret {
namespace {

using ferret_testing::TestPipe;

std::string ReadFrame(RpcFrameReader& reader) {
  auto frame = reader.Read();
  EXPECT_TRUE(frame.ok()) << frame.status();
  return frame.ok() ? *frame : "";
}

TEST(RpcFrameReaderTest, ReadsEverySplitAndSeveralFramesPerRead) {
  const std::string frames = "first\n\nsecond\r\nthird\n";
  for (size_t split = 0; split <= frames.size(); ++split) {
    SCOPED_TRACE(split);
    TestPipe pipe;
    RpcFrameReader reader(pipe.read_fd());
    ASSERT_EQ(write(pipe.write_fd(), frames.data(), split),
              static_cast<ssize_t>(split));
    std::thread remainder([&] {
      EXPECT_EQ(
          write(pipe.write_fd(), frames.data() + split, frames.size() - split),
          static_cast<ssize_t>(frames.size() - split));
      pipe.CloseWrite();
    });
    ASSERT_EQ(ReadFrame(reader), "first");
    ASSERT_EQ(ReadFrame(reader), "");
    // CR is data; only LF delimits a frame.
    ASSERT_EQ(ReadFrame(reader), "second\r");
    ASSERT_EQ(ReadFrame(reader), "third");
    EXPECT_EQ(reader.Read().status().code(), absl::StatusCode::kOutOfRange);
    remainder.join();
  }
}

TEST(RpcFrameReaderTest, SkipsOversizedFramesIncludingAcrossReads) {
  TestPipe pipe;
  RpcFrameReader reader(pipe.read_fd(), {.max_frame_bytes = 3});
  ASSERT_EQ(write(pipe.write_fd(), "abc\nabcd\nxy", 11), 11);
  ASSERT_EQ(ReadFrame(reader), "abc");
  EXPECT_EQ(reader.Read().status().code(),
            absl::StatusCode::kResourceExhausted);
  ASSERT_EQ(write(pipe.write_fd(), "yz\nok\n", 6), 6);
  pipe.CloseWrite();
  EXPECT_EQ(reader.Read().status().code(),
            absl::StatusCode::kResourceExhausted);
  ASSERT_EQ(ReadFrame(reader), "ok");
  EXPECT_EQ(reader.Read().status().code(), absl::StatusCode::kOutOfRange);
}

TEST(RpcFrameReaderTest, DropsTruncatedLastFrameOnlyOnce) {
  TestPipe pipe;
  RpcFrameReader reader(pipe.read_fd());
  ASSERT_EQ(write(pipe.write_fd(), "ok\nunfinished", 13), 13);
  pipe.CloseWrite();
  EXPECT_EQ(ReadFrame(reader), "ok");
  EXPECT_EQ(reader.Read().status().code(), absl::StatusCode::kDataLoss);
  EXPECT_EQ(reader.Read().status().code(), absl::StatusCode::kOutOfRange);
}

TEST(RpcFrameReaderTest, PreservesNonUtf8Bytes) {
  TestPipe pipe;
  RpcFrameReader reader(pipe.read_fd());
  ASSERT_EQ(write(pipe.write_fd(), "\xff\x80\n", 3), 3);
  pipe.CloseWrite();
  EXPECT_EQ(ReadFrame(reader), std::string("\xff\x80", 2));
}

TEST(RpcFrameWriterTest, RejectsEmbeddedNewlineWithoutBreakingStream) {
  TestPipe pipe;
  RpcFrameWriter writer(pipe.write_fd());
  EXPECT_EQ(writer.Write("a\nb").code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(writer.Write("hello\r").ok());
  pipe.CloseWrite();
  EXPECT_EQ(pipe.ReadAll(), "hello\r\n");
  EXPECT_TRUE(writer.status().ok());
}

TEST(RpcFrameReaderTest, JsonNestingLimitIsIndependentOfFraming) {
  TestPipe pipe;
  RpcFrameReader reader(pipe.read_fd());
  RpcFrameWriter writer(pipe.write_fd());
  EXPECT_TRUE(writer.Write("[[0]]").ok());
  EXPECT_TRUE(writer.Write("[[[0]]]").ok());
  pipe.CloseWrite();
  auto within_limit = reader.Read();
  ASSERT_TRUE(within_limit.ok());
  EXPECT_TRUE(ParseJson(*within_limit, {.max_depth = 2}).ok());
  auto too_deep = reader.Read();
  ASSERT_TRUE(too_deep.ok());
  EXPECT_EQ(ParseJson(*too_deep, {.max_depth = 2}).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(RpcFrameWriterTest, ConcurrentWritersNeverInterleaveFrames) {
  TestPipe pipe;
  RpcFrameWriter writer(pipe.write_fd());
  RpcFrameReader reader(pipe.read_fd());
  const std::string a(2000, 'a');
  const std::string b(2000, 'b');
  std::thread first([&] { EXPECT_TRUE(writer.Write(a).ok()); });
  std::thread second([&] { EXPECT_TRUE(writer.Write(b).ok()); });
  const std::string one = ReadFrame(reader);
  const std::string two = ReadFrame(reader);
  EXPECT_TRUE((one == a && two == b) || (one == b && two == a));
  first.join();
  second.join();
}

TEST(RpcFrameWriterTest, NonBlockingWriteRetriesUntilReaderDrainsPipe) {
  TestPipe pipe;
  pipe.SetWriteNonBlocking();
  RpcFrameWriter writer(pipe.write_fd());
  RpcFrameReader reader(pipe.read_fd());
  const std::string frame(200000, 'x');
  std::thread consumer([&] { EXPECT_EQ(ReadFrame(reader), frame); });
  EXPECT_TRUE(writer.Write(frame).ok());
  consumer.join();
}

TEST(RpcFrameWriterTest, FailedWriteBreaksTheStream) {
  TestPipe pipe;
  RpcFrameWriter writer(pipe.write_fd());
  pipe.CloseWrite();
  const absl::Status failure = writer.Write("first");
  EXPECT_FALSE(failure.ok());
  EXPECT_EQ(writer.status(), failure);
  EXPECT_EQ(writer.Write("second").code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(RpcFrameReaderTest, NonBlockingReadWaitsForInput) {
  TestPipe pipe;
  pipe.SetReadNonBlocking();
  RpcFrameReader reader(pipe.read_fd());
  std::thread producer([&] {
    RpcFrameWriter writer(pipe.write_fd());
    EXPECT_TRUE(writer.Write("ready").ok());
    pipe.CloseWrite();
  });
  EXPECT_EQ(ReadFrame(reader), "ready");
  EXPECT_EQ(reader.Read().status().code(), absl::StatusCode::kOutOfRange);
  producer.join();
}

}  // namespace
}  // namespace ferret
