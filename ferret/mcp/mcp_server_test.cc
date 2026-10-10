// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Golden transcripts are JSONL. Each line has exactly one of "send" (a JSON
// client message), "send_raw" (literal malformed input), or "expect" (the
// next server response). Expectations compare parsed JSON, except that
// "PLACEHOLDER" accepts a parser diagnostic whose wording may change.
// A transcript is driven incrementally, so expect also checks response order.

#include "ferret/mcp/mcp_server.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "nlohmann/json_fwd.hpp"

#include "ferret/mcp/mcp_tool.h"
#include "ferret/mcp/testing/fake_mcp_tools.h"
#include "ferret/rpc/framing.h"
#include "ferret/rpc/json_codec.h"
#include "ferret/rpc/json_rpc.h"
#include "ferret/rpc/testing/test_pipe.h"
#include "ferret/testing/test_seed.h"

namespace ferret {
namespace {

using Json = nlohmann::json;
using ferret_testing::BlockingMcpTool;
using ferret_testing::TestPipe;

McpToolRegistry MakeTools(BlockingMcpTool** blocker) {
  std::vector<std::unique_ptr<McpTool>> tools;
  tools.push_back(std::make_unique<ferret_testing::EchoMcpTool>());
  auto blocking = std::make_unique<BlockingMcpTool>();
  *blocker = blocking.get();
  tools.push_back(std::move(blocking));
  tools.push_back(std::make_unique<ferret_testing::FailingMcpTool>());
  auto registry = McpToolRegistry::FromTools(std::move(tools));
  EXPECT_TRUE(registry.ok()) << registry.status();
  return std::move(*registry);
}

// Owns the descriptors, server, and reader thread for one session. The output
// pipe is closed only after Serve returns, so Read() can detect stray lines.
class ServerSession final {
 public:
  explicit ServerSession(McpServerOptions options = {})
      : tools_(MakeTools(&blocker_)),
        server_(tools_, std::move(options)),
        input_writer_(input_.write_fd()),
        output_reader_(output_.read_fd()),
        worker_([this] {
          status_ = server_.Serve(input_.read_fd(), output_.write_fd());
        }) {}
  ServerSession(const ServerSession&) = delete;
  ServerSession& operator=(const ServerSession&) = delete;
  ~ServerSession() {
    ReleaseBlocker();
    Finish();
  }

  absl::Status SendRaw(absl::string_view text) {
    return input_writer_.Write(text);
  }
  absl::Status Send(const Json& message) {
    return SendRaw(SerializeJson(message));
  }
  absl::StatusOr<std::string> Read() { return output_reader_.Read(); }
  BlockingMcpTool* blocker() const { return blocker_; }
  void ReleaseBlocker() {
    if (!released_) {
      blocker_->Release();
      released_ = true;
    }
  }

  void Finish() {
    if (worker_.joinable()) {
      input_.CloseWrite();
      worker_.join();
      output_.CloseWrite();
      EXPECT_TRUE(status_.ok()) << status_;
    }
  }

 private:
  TestPipe input_;
  TestPipe output_;
  BlockingMcpTool* blocker_ = nullptr;
  bool released_ = false;
  McpToolRegistry tools_;
  McpServer server_;
  RpcFrameWriter input_writer_;
  RpcFrameReader output_reader_;
  absl::Status status_;
  std::thread worker_;
};

Json Request(int id, absl::string_view method, Json params = nullptr) {
  Json request = {{"jsonrpc", "2.0"}, {"id", id}, {"method", method}};
  if (!params.is_null()) request["params"] = std::move(params);
  return request;
}

Json Cancel(int id) {
  return {{"jsonrpc", "2.0"},
          {"method", "notifications/cancelled"},
          {"params", {{"requestId", id}}}};
}

void ExpectResponse(ServerSession& session, int id, const Json& expected) {
  const auto frame = session.Read();
  ASSERT_TRUE(frame.ok()) << frame.status();
  const auto document = ParseJson(*frame);
  ASSERT_TRUE(document.ok()) << document.status();
  EXPECT_EQ(*document, expected);
  EXPECT_EQ((*document)["id"], id);
}

class BadStructuredMcpTool final : public McpTool {
 public:
  absl::string_view name() const override { return "bad"; }
  absl::string_view description() const override {
    return "Returns an invalid result.";
  }
  Json input_schema() const override { return {{"type", "object"}}; }
  Json output_schema() const override { return {{"type", "object"}}; }
  absl::StatusOr<McpToolResult> Call(JsonView,
                                     const McpToolContext&) const override {
    return McpToolResult{.text = "bad", .structured = Json::array({1})};
  }
};

TEST(McpServerTest, InvalidToolOutputIsAnInternalErrorNotACrash) {
  std::vector<std::unique_ptr<McpTool>> tools;
  tools.push_back(std::make_unique<BadStructuredMcpTool>());
  auto registry = McpToolRegistry::FromTools(std::move(tools));
  ASSERT_TRUE(registry.ok());
  McpServer server(*registry, {});
  TestPipe input;
  TestPipe output;
  RpcFrameWriter writer(input.write_fd());
  RpcFrameReader reader(output.read_fd());
  absl::Status status;
  std::thread worker(
      [&] { status = server.Serve(input.read_fd(), output.write_fd()); });
  EXPECT_TRUE(writer
                  .Write(SerializeJson(Request(
                      1, "initialize", {{"protocolVersion", "2025-11-25"}})))
                  .ok());
  ASSERT_TRUE(reader.Read().ok());
  EXPECT_TRUE(
      writer.Write(SerializeJson(Request(2, "tools/call", {{"name", "bad"}})))
          .ok());
  auto frame = reader.Read();
  ASSERT_TRUE(frame.ok()) << frame.status();
  auto response = ParseJson(*frame);
  ASSERT_TRUE(response.ok());
  EXPECT_EQ((*response)["error"]["code"], kJsonRpcInternalError);
  input.CloseWrite();
  worker.join();
  output.CloseWrite();
  EXPECT_TRUE(status.ok()) << status;
}

TEST(McpServerTest, ReplaysEveryGoldenTranscript) {
  const char* runfiles = std::getenv("TEST_SRCDIR");
  ASSERT_NE(runfiles, nullptr);
  for (const char* file :
       {"cancellation", "legacy_2025_03_26", "legacy_2025_06_18",
        "legacy_2025_11_25", "legacy_negotiation", "malformed",
        "modern_2026_07_28", "unsupported_version"}) {
    SCOPED_TRACE(file);
    const std::string path =
        absl::StrCat(runfiles, "/_main/ferret/mcp/testdata/", file, ".jsonl");
    std::ifstream transcript(path);
    ASSERT_TRUE(transcript.good()) << path;
    ServerSession session({.server_version = "0.0.0-test"});
    std::string line;
    int number = 0;
    while (std::getline(transcript, line)) {
      SCOPED_TRACE(++number);
      auto step = ParseJson(line);
      ASSERT_TRUE(step.ok()) << step.status();
      if (step->contains("send")) {
        EXPECT_TRUE(session.Send((*step)["send"]).ok());
      } else if (step->contains("send_raw")) {
        EXPECT_TRUE(
            session.SendRaw((*step)["send_raw"].get<std::string>()).ok());
      } else {
        ASSERT_TRUE(step->contains("expect"));
        auto frame = session.Read();
        ASSERT_TRUE(frame.ok()) << frame.status();
        auto actual = ParseJson(*frame);
        ASSERT_TRUE(actual.ok()) << actual.status();
        Json expected = (*step)["expect"];
        if (expected.contains("error") &&
            expected["error"].value("message", "") == "PLACEHOLDER") {
          EXPECT_EQ((*actual)["error"]["code"], expected["error"]["code"]);
          EXPECT_EQ((*actual)["id"], nullptr);
          EXPECT_TRUE((*actual)["error"]["message"].is_string());
        } else {
          EXPECT_EQ(*actual, expected);
        }
      }
    }
    session.ReleaseBlocker();
    session.Finish();
    EXPECT_EQ(session.Read().status().code(), absl::StatusCode::kOutOfRange);
  }
}

TEST(McpServerTest,
     ReaderRunsWhileCallsAreBoundedAndCancellationSuppressesResults) {
  ServerSession session({.max_concurrent_calls = 2,
                         .max_pending_calls = 3,
                         .shutdown_grace = absl::Milliseconds(100)});
  EXPECT_TRUE(
      session
          .Send(Request(1, "initialize", {{"protocolVersion", "2025-11-25"}}))
          .ok());
  auto initialized = session.Read();
  ASSERT_TRUE(initialized.ok());
  for (int id : {1, 2, 3, 4}) {
    EXPECT_TRUE(
        session.Send(Request(id + 10, "tools/call", {{"name", "block"}})).ok());
  }
  session.blocker()->WaitForStartedCalls(2);
  EXPECT_EQ(session.blocker()->peak_running_calls(), 2);
  ExpectResponse(
      session, 14,
      {{"jsonrpc", "2.0"},
       {"id", 14},
       {"error",
        {{"code", -32603},
         {"message", "Internal error: 3 tool calls are already pending"}}}});
  EXPECT_TRUE(session.Send(Cancel(13)).ok());
  EXPECT_TRUE(session.Send(Request(20, "ping")).ok());
  ExpectResponse(session, 20,
                 {{"jsonrpc", "2.0"}, {"id", 20}, {"result", Json::object()}});
  for (int id : {11, 12}) EXPECT_TRUE(session.Send(Cancel(id)).ok());
  session.Finish();
  EXPECT_EQ(session.Read().status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(session.blocker()->cancelled_calls(), 2);
  EXPECT_EQ(session.blocker()->peak_running_calls(), 2);
}

TEST(McpServerTest, WritesWholeLinesInCompletionOrder) {
  ServerSession session({.max_concurrent_calls = 2});
  EXPECT_TRUE(
      session
          .Send(Request(1, "initialize", {{"protocolVersion", "2025-11-25"}}))
          .ok());
  ASSERT_TRUE(session.Read().ok());
  EXPECT_TRUE(session.Send(Request(2, "tools/call", {{"name", "block"}})).ok());
  session.blocker()->WaitForStartedCalls(1);
  EXPECT_TRUE(
      session
          .Send(Request(3, "tools/call",
                        {{"name", "echo"}, {"arguments", {{"message", "ok"}}}}))
          .ok());
  ExpectResponse(session, 3,
                 {{"jsonrpc", "2.0"},
                  {"id", 3},
                  {"result",
                   {{"content", {{{"type", "text"}, {"text", "ok"}}}},
                    {"isError", false},
                    {"structuredContent", {{"message", "ok"}}}}}});
  session.ReleaseBlocker();
  ExpectResponse(session, 2,
                 {{"jsonrpc", "2.0"},
                  {"id", 2},
                  {"result",
                   {{"content", {{{"type", "text"}, {"text", "released"}}}},
                    {"isError", false},
                    {"structuredContent", {{"released", true}}}}}});
  session.Finish();
  EXPECT_EQ(session.Read().status().code(), absl::StatusCode::kOutOfRange);
}

TEST(McpServerTest, SeededUntrustedMessagesStayBoundedAndWellFormed) {
  const uint64_t seed = ferret_testing::TestSeed(0x5eed1234);
  SCOPED_TRACE(seed);
  BlockingMcpTool* blocker = nullptr;
  McpToolRegistry tools = MakeTools(&blocker);
  McpServer server(tools, {.max_message_bytes = 512,
                           .max_json_depth = 4,
                           .max_concurrent_calls = 2,
                           .max_pending_calls = 2,
                           .shutdown_grace = absl::Milliseconds(100)});
  TestPipe input;
  TestPipe output;
  RpcFrameWriter writer(input.write_fd());
  absl::Status status;
  std::thread serve([&] {
    status = server.Serve(input.read_fd(), output.write_fd());
    output.CloseWrite();
  });
  std::vector<std::string> responses;
  std::thread drain([&] {
    RpcFrameReader reader(output.read_fd());
    while (true) {
      auto frame = reader.Read();
      if (!frame.ok()) {
        EXPECT_EQ(frame.status().code(), absl::StatusCode::kOutOfRange);
        break;
      }
      responses.push_back(*std::move(frame));
    }
  });
  std::mt19937_64 random(seed);
  const std::vector<std::string> cases = {
      R"json({"jsonrpc":"2.0","id":1,"method":"ping"})json",
      R"json({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"block"}})json",
      R"json({"jsonrpc":"2.0","id":true,"method":"tools/list"})json",
      R"json({"jsonrpc":"2.0","id":1e9999,"method":"ping"})json",
      R"json({"jsonrpc":"2.0","id":1,"method":"unknown"})json",
      R"json({"jsonrpc":"2.0","id":2,"method":"tools/list","params":[]})json",
      R"json({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":false}})json",
      R"json({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo","arguments":{"message":7}}})json",
      R"json({"jsonrpc":"2.0","id":3,"method":"tools/list","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":false}}})json",
      R"json({"jsonrpc":0,"id":1,"method":"ping"})json",
      R"json({"jsonrpc":"2.0","id":[],"method":"ping"})json",
      R"json({"jsonrpc":"2.0","id":"s","method":false})json",
      R"json({"jsonrpc":"2.0","id":3,"method":"tools/list","params":true})json",
      R"json({"jsonrpc":"2.0","id":3,"method":"tools/list","params":{"cursor":4}})json",
      R"json({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo","arguments":false}})json",
      R"json({"jsonrpc":"2.0","id":3,"method":"tools/list","params":{"_meta":"wrong"}})json",
      R"json({"jsonrpc":"2.0","id":3,"method":"tools/list","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":true}}})json",
      R"json({"jsonrpc":"2.0","id":3,"method":"tools/list","params":{"_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{},"io.modelcontextprotocol/clientInfo":1}}})json",
      R"json({"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":false}})json",
      std::string(600, 'x'),
      std::string(70, '[') + "0" + std::string(70, ']'),
  };
  for (int i = 0; i < 300; ++i) {
    std::string text = cases[random() % cases.size()];
    switch (random() % 4) {
      case 0:
        text.resize(random() % (text.size() + 1));
        break;
      case 1:
        if (!text.empty()) text[random() % text.size()] = '\xff';
        break;
      case 2:
        text += "  garbage";
        break;
      default:
        break;
    }
    EXPECT_TRUE(writer.Write(text).ok());
  }
  input.CloseWrite();
  serve.join();
  drain.join();
  EXPECT_TRUE(status.ok()) << status;
  ASSERT_FALSE(responses.empty());
  for (const std::string& frame : responses) {
    EXPECT_LT(frame.size(), size_t{4096});
    auto parsed = ParseJson(frame);
    ASSERT_TRUE(parsed.ok()) << parsed.status();
    ASSERT_TRUE(parsed->is_object());
    EXPECT_EQ((*parsed)["jsonrpc"], "2.0");
    EXPECT_TRUE(parsed->contains("id"));
    EXPECT_TRUE(parsed->contains("result") || parsed->contains("error"));
  }
}

}  // namespace
}  // namespace ferret
