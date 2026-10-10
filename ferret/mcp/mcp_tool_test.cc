// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/mcp/mcp_tool.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/notification.h"
#include "gtest/gtest.h"
#include "nlohmann/json_fwd.hpp"

#include "ferret/mcp/testing/fake_mcp_tools.h"
#include "ferret/rpc/json_codec.h"

namespace ferret {
namespace {

using Json = nlohmann::json;
using ferret_testing::EchoMcpTool;
using ferret_testing::FailingMcpTool;

class ConfigurableMcpTool final : public McpTool {
 public:
  ConfigurableMcpTool(std::string name, Json input, Json output)
      : name_(std::move(name)),
        input_(std::move(input)),
        output_(std::move(output)) {}
  absl::string_view name() const override { return name_; }
  absl::string_view description() const override { return "For validation."; }
  Json input_schema() const override { return input_; }
  Json output_schema() const override { return output_; }
  absl::StatusOr<McpToolResult> Call(JsonView,
                                     const McpToolContext&) const override {
    return McpToolResult{};
  }

 private:
  std::string name_;
  Json input_;
  Json output_;
};

TEST(McpToolTest, RegistryKeepsRegistrationOrderAndFindsTools) {
  std::vector<std::unique_ptr<McpTool>> tools;
  tools.push_back(std::make_unique<EchoMcpTool>());
  tools.push_back(std::make_unique<FailingMcpTool>());
  auto registry = McpToolRegistry::FromTools(std::move(tools));
  ASSERT_TRUE(registry.ok()) << registry.status();
  ASSERT_EQ(registry->tools().size(), size_t{2});
  EXPECT_EQ(registry->tools()[0]->name(), "echo");
  EXPECT_EQ(registry->tools()[1]->name(), "fail");
  EXPECT_EQ(registry->Find("echo"), registry->tools()[0].get());
  EXPECT_EQ(registry->Find("absent"), nullptr);
}

TEST(McpToolTest, RejectsDuplicateNames) {
  std::vector<std::unique_ptr<McpTool>> tools;
  tools.push_back(std::make_unique<EchoMcpTool>());
  tools.push_back(std::make_unique<EchoMcpTool>());
  EXPECT_EQ(McpToolRegistry::FromTools(std::move(tools)).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(McpToolTest, ValidatesNamesAndObjectSchemas) {
  const Json object = {{"type", "object"}};
  const Json array = {{"type", "array"}};
  for (const std::string& name : std::vector<std::string>{
           "", "with space", "a/b", std::string(129, 'x')}) {
    std::vector<std::unique_ptr<McpTool>> tools;
    tools.push_back(
        std::make_unique<ConfigurableMcpTool>(name, object, nullptr));
    EXPECT_EQ(McpToolRegistry::FromTools(std::move(tools)).status().code(),
              absl::StatusCode::kInvalidArgument)
        << name;
  }
  for (const auto& [input, output] :
       {std::pair<Json, Json>{array, nullptr}, {object, array}}) {
    std::vector<std::unique_ptr<McpTool>> tools;
    tools.push_back(
        std::make_unique<ConfigurableMcpTool>("valid", input, output));
    EXPECT_EQ(McpToolRegistry::FromTools(std::move(tools)).status().code(),
              absl::StatusCode::kInvalidArgument);
  }
  std::vector<std::unique_ptr<McpTool>> tools;
  tools.push_back(std::make_unique<ConfigurableMcpTool>(std::string(128, 'x'),
                                                        object, nullptr));
  EXPECT_TRUE(McpToolRegistry::FromTools(std::move(tools)).ok());
}

TEST(McpToolTest, ChecksArgumentsAndDistinguishesToolErrors) {
  EchoMcpTool echo;
  FailingMcpTool fail;
  absl::Notification cancellation;
  McpToolContext context(cancellation);
  EXPECT_FALSE(context.cancelled());
  const Json good = {{"message", "hello"}};
  auto result = echo.Call(JsonView(good), context);
  ASSERT_TRUE(result.ok());
  EXPECT_EQ(result->text, "hello");
  EXPECT_EQ(result->structured, good);
  const Json bad = {{"message", 7}};
  EXPECT_EQ(echo.Call(JsonView(bad), context).status().code(),
            absl::StatusCode::kInvalidArgument);
  const Json empty = Json::object();
  EXPECT_EQ(echo.Call(JsonView(empty), context).status().code(),
            absl::StatusCode::kInvalidArgument);
  result = fail.Call(JsonView(empty), context);
  ASSERT_TRUE(result.ok());
  EXPECT_TRUE(result->is_error);
  const Json internal = {{"internal", true}};
  EXPECT_EQ(fail.Call(JsonView(internal), context).status().code(),
            absl::StatusCode::kInternal);
  const Json wrong_type = {{"internal", "yes"}};
  EXPECT_EQ(fail.Call(JsonView(wrong_type), context).status().code(),
            absl::StatusCode::kInvalidArgument);
  cancellation.Notify();
  EXPECT_TRUE(context.cancelled());
}

}  // namespace
}  // namespace ferret
