// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Real-binary daemon lifecycle, reconnection, and concurrent client tests.
#include "ferret/rpc/rpc_client.h"

#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

#include "ferret/base/subprocess.h"
#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/rpc/daemon.h"
#include "ferret/rpc/daemon_test_guard.h"
#include "ferret/rpc/framing.h"
#include "ferret/rpc/json_codec.h"
#include "ferret/rpc/json_rpc.h"
#include "ferret/search/search_types.h"
#include "ferret/testing/temp_dir.h"
#include "ferret/workspace/workspace.h"

extern char** environ;

// The Apple SDK exposes F_OK through unistd.h, but include-cleaner misses it.
// NOLINTBEGIN(misc-include-cleaner)
namespace ferret {
namespace {

constexpr char kBinary[] = "ferret/cli/ferret";

// Keep each test's daemon and index isolated; always drain on teardown.
class RpcClientTest : public ::testing::Test {
 protected:
  void SetUp() override {
    signal(SIGPIPE, SIG_IGN);  // NOLINT(misc-include-cleaner)
    ferret_testing::IsolateGitEnvironment(temp_.Join("home"));
    root_ = temp_.Join("root");
    ASSERT_EQ(mkdir(root_.c_str(), 0700), 0);
    index_ = temp_.Join("index");
    client_ = std::make_unique<RpcClient>(kBinary, index_);
    daemon_.Watch(root_);
  }
  void TearDown() override { daemon_.Stop(kBinary, index_); }
  void Register() {
    absl::Status status = client_->Register(root_);
    ASSERT_TRUE(status.ok()) << status;
  }
  pid_t StartServer(const std::string& flag) {
    const std::string index_flag = absl::StrCat("--index_dir=", index_);
    char* argv[] = {const_cast<char*>(kBinary),
                    const_cast<char*>("serve"),
                    root_.data(),
                    const_cast<char*>(index_flag.c_str()),
                    const_cast<char*>(flag.c_str()),
                    nullptr};
    pid_t pid = -1;
    EXPECT_EQ(posix_spawn(&pid, kBinary, nullptr, nullptr, argv, environ), 0);
    daemon_.WatchPid(pid);
    auto socket = DaemonSocketPath(root_);
    EXPECT_TRUE(socket.ok());
    if (socket.ok()) {
      for (int i = 0; i < 3000 && access(socket->c_str(), F_OK) != 0; ++i) {
        usleep(10000);
      }
      EXPECT_EQ(access(socket->c_str(), F_OK), 0);
    }
    return pid;
  }
  ferret_testing::TempDir temp_;
  std::string root_;
  std::string index_;
  std::unique_ptr<RpcClient> client_;
  DaemonTestGuard daemon_;
};

TEST_F(RpcClientTest, ReplacesStaleSocketAndSearchesWithSync) {
  const std::string file = absl::StrCat(root_, "/file.cc");
  FILE* output = fopen(file.c_str(), "w");
  ASSERT_NE(output, nullptr);
  fputs("before\n", output);
  fclose(output);
  auto socket = DaemonSocketPath(root_);
  ASSERT_TRUE(socket.ok()) << socket.status();
  FILE* stale = fopen(socket->c_str(), "w");
  ASSERT_NE(stale, nullptr);
  fclose(stale);
  Register();
  SearchRequest request;
  request.pattern = "before";
  SearchConsistency sync{.mode = SearchConsistencyMode::kSync};
  auto first = client_->Search(root_, request, sync);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ(first->response.num_matching_files, 1);
  output = fopen(file.c_str(), "w");
  ASSERT_NE(output, nullptr);
  fputs("after needle\n", output);
  fclose(output);
  request.pattern = "after needle";
  auto edited = client_->Search(root_, request, sync);
  ASSERT_TRUE(edited.ok()) << edited.status();
  EXPECT_EQ(edited->response.num_matching_files, 1);
  EXPECT_FALSE(edited->stale);
}

TEST_F(RpcClientTest, EightClientsShareOneDaemonAndConcurrentCalls) {
  constexpr int kClients = 8;
  std::vector<std::thread> threads;
  std::atomic<int> successes = 0;
  std::vector<std::unique_ptr<RpcClient>> clients;
  clients.reserve(kClients);
  threads.reserve(kClients);
  for (int i = 0; i < kClients; ++i) {
    clients.push_back(std::make_unique<RpcClient>(kBinary, index_));
  }
  for (const auto& client : clients) {
    threads.emplace_back([&, client = client.get()] {
      absl::Status status = client->Register(root_);
      if (!status.ok()) {
        ADD_FAILURE() << status;
        return;
      }
      for (int i = 0; i < 5; ++i) {
        auto search = client->Search(root_, SearchRequest{.pattern = "absent"},
                                     SearchConsistency());
        if (!search.ok()) {
          ADD_FAILURE() << search.status();
          return;
        }
      }
      ++successes;
    });
  }
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(successes.load(), kClients);
  Register();
  auto statuses = client_->IndexStatus();
  ASSERT_TRUE(statuses.ok()) << statuses.status();
  EXPECT_EQ(statuses->size(), size_t{1});
  auto key = RepositoryKeyForRoot(root_);
  ASSERT_TRUE(key.ok());
  std::ifstream log(absl::StrCat(index_, "/", *key, "/daemon.log"));
  ASSERT_TRUE(log.good());
  int starts = 0;
  std::string line;
  while (std::getline(log, line)) {
    if (line.find("daemon start pid=") == 0) ++starts;
  }
  EXPECT_EQ(starts, 1);
}

// A daemon runs `git rev-parse --git-common-dir` before it takes its lock, so
// a daemon that loses the race never reaches daemon.log. Making that call take
// a second in daemons slows startup; the client must wait for its one daemon
// instead of spawning another each retry (which once started hundreds and
// froze the machine).
TEST_F(RpcClientTest, SlowDaemonStartupSpawnsOneDaemon) {
  const std::string bin = temp_.Join("slow-git");
  ASSERT_EQ(mkdir(bin.c_str(), 0700), 0);
  const std::string callers = temp_.Join("git-callers");
  const char* path = getenv("PATH");
  const std::string original_path = path != nullptr ? path : "/usr/bin:/bin";
  {
    std::ofstream script(absl::StrCat(bin, "/git"));
    script << "#!/bin/sh\n"
           << "case \"$PPID $*\" in\n"
           << "  " << getpid() << "\\ *) ;;\n"
           << "  *--git-common-dir*) echo \"$PPID\" >> '" << callers << "'\n"
           << "    sleep 1 ;;\n"
           << "esac\n"
           << "PATH='" << original_path << "' exec git \"$@\"\n";
  }
  ASSERT_EQ(chmod(absl::StrCat(bin, "/git").c_str(), 0700), 0);
  ASSERT_EQ(setenv("PATH", absl::StrCat(bin, ":", original_path).c_str(), 1),
            0);
  const absl::Status status = client_->Register(root_);
  ASSERT_EQ(setenv("PATH", original_path.c_str(), 1), 0);
  ASSERT_TRUE(status.ok()) << status;
  std::ifstream lines(callers);
  std::set<std::string> daemons;
  for (std::string line; std::getline(lines, line);) daemons.insert(line);
  EXPECT_EQ(daemons.size(), size_t{1});
}

TEST_F(RpcClientTest, SearchesAndNotifiesDoNotRunGitAfterRegistration) {
  const std::string bin = temp_.Join("logged-git");
  ASSERT_EQ(mkdir(bin.c_str(), 0700), 0);
  const std::string calls = temp_.Join("git-calls");
  const char* path = getenv("PATH");
  const std::string original_path = path != nullptr ? path : "/usr/bin:/bin";
  {
    std::ofstream script(absl::StrCat(bin, "/git"));
    script << "#!/bin/sh\n"
           << "echo \"$PPID $*\" >> '" << calls << "'\n"
           << "PATH='" << original_path << "' exec git \"$@\"\n";
  }
  ASSERT_EQ(chmod(absl::StrCat(bin, "/git").c_str(), 0700), 0);
  ASSERT_EQ(setenv("PATH", absl::StrCat(bin, ":", original_path).c_str(), 1),
            0);
  // The daemon inherits this PATH when registration starts it.
  const absl::Status registered = client_->Register(root_);
  ASSERT_TRUE(registered.ok()) << registered;
  auto count_calls = [&] {
    std::ifstream input(calls);
    size_t count = 0;
    for (std::string line; std::getline(input, line);) ++count;
    return count;
  };
  const size_t baseline = count_calls();
  for (int i = 0; i < 20; ++i) {
    auto result = client_->Search(root_, SearchRequest{.pattern = "missing"},
                                  SearchConsistency());
    EXPECT_TRUE(result.ok()) << result.status();
  }
  for (int i = 0; i < 5; ++i) {
    auto generation = client_->Notify(root_, {"missing.cc"});
    EXPECT_TRUE(generation.ok()) << generation.status();
  }
  EXPECT_EQ(count_calls(), baseline);
  EXPECT_EQ(setenv("PATH", original_path.c_str(), 1), 0);
}

TEST_F(RpcClientTest, RejectsUnregisteredAndForeignRootsOnSearchAndNotify) {
  Register();
  const std::string unregistered = temp_.Join("unregistered");
  const std::string foreign = temp_.Join("foreign");
  ASSERT_EQ(mkdir(unregistered.c_str(), 0700), 0);
  ASSERT_EQ(mkdir(foreign.c_str(), 0700), 0);
  for (const std::string& root : {unregistered, foreign}) {
    EXPECT_EQ(client_
                  ->Search(root, SearchRequest{.pattern = "missing"},
                           SearchConsistency())
                  .status()
                  .code(),
              absl::StatusCode::kNotFound);
    EXPECT_EQ(client_->Notify(root, {}).status().code(),
              absl::StatusCode::kNotFound);
  }
}

TEST_F(RpcClientTest, ReusesRegisteredCanonicalRootForAliases) {
  Register();
  const std::string alias = absl::StrCat(root_, "/.");
  EXPECT_TRUE(client_
                  ->Search(alias, SearchRequest{.pattern = "missing"},
                           SearchConsistency())
                  .ok());
  EXPECT_TRUE(client_->Notify(alias, {}).ok());
  EXPECT_TRUE(client_
                  ->Search(alias, SearchRequest{.pattern = "missing"},
                           SearchConsistency())
                  .ok());
  EXPECT_TRUE(client_->Register(alias).ok());
  EXPECT_EQ(client_->Roots(), std::vector<std::string>{root_});
}

TEST_F(RpcClientTest, SearchesNeverLogQueryText) {
  Register();
  const std::string secret = "query-text-must-not-appear-in-daemon-log";
  auto response = client_->Search(root_, SearchRequest{.pattern = secret},
                                  SearchConsistency());
  ASSERT_TRUE(response.ok()) << response.status();
  auto key = RepositoryKeyForRoot(root_);
  ASSERT_TRUE(key.ok());
  std::ifstream log(absl::StrCat(index_, "/", *key, "/daemon.log"));
  ASSERT_TRUE(log.good());
  std::string content((std::istreambuf_iterator<char>(log)),
                      std::istreambuf_iterator<char>());
  EXPECT_EQ(content.find(secret), std::string::npos);
  EXPECT_EQ(content.find("daemon search"), std::string::npos);
}

TEST_F(RpcClientTest, ServeMeasuresLatencyWithoutLoggingPatterns) {
  StartServer("--measure_latency");
  Register();
  const std::string pattern = "pattern-not-in-measured-log";
  auto response = client_->Search(root_, SearchRequest{.pattern = pattern},
                                  SearchConsistency());
  ASSERT_TRUE(response.ok()) << response.status();
  auto key = RepositoryKeyForRoot(root_);
  ASSERT_TRUE(key.ok());
  std::ifstream log(absl::StrCat(index_, "/", *key, "/daemon.log"));
  ASSERT_TRUE(log.good());
  std::string content((std::istreambuf_iterator<char>(log)),
                      std::istreambuf_iterator<char>());
  EXPECT_THAT(content, ::testing::HasSubstr("daemon search elapsed_us="));
  EXPECT_EQ(content.find(pattern), std::string::npos);
}

TEST_F(RpcClientTest, RetainsFramesBufferedDuringHello) {
  auto path = DaemonSocketPath(root_);
  ASSERT_TRUE(path.ok()) << path.status();
  const int listener = socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(listener, 0);
  sockaddr_un address = {};
  address.sun_family = AF_UNIX;
  path->copy(address.sun_path, path->size());
  ASSERT_EQ(
      bind(listener, reinterpret_cast<sockaddr*>(&address), sizeof(address)),
      0);
  ASSERT_EQ(listen(listener, 1), 0);
  std::thread server([&] {
    const int fd = accept(listener, nullptr, nullptr);
    if (fd < 0) return;
    RpcFrameReader reader(fd);
    auto hello = reader.Read();
    EXPECT_TRUE(hello.ok());
    // Both responses arrive in the same read. A new reader after hello
    // discards the buffered register response and sees EOF instead.
    const std::string replies =
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{\"compatible\":true}}\n"
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":{}}\n";
    EXPECT_EQ(write(fd, replies.data(), replies.size()),
              static_cast<ssize_t>(replies.size()));
    auto registration = reader.Read();
    EXPECT_TRUE(registration.ok());
    usleep(100000);
    close(fd);
  });
  EXPECT_TRUE(client_->Register(root_).ok());
  server.join();
  close(listener);
  unlink(path->c_str());
}

TEST_F(RpcClientTest, RejectsRootsOutsideRepository) {
  Register();
  const std::string foreign = temp_.Join("foreign");
  ASSERT_EQ(mkdir(foreign.c_str(), 0700), 0);
  auto socket = DaemonSocketPath(root_);
  ASSERT_TRUE(socket.ok());
  const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(fd, 0);
  sockaddr_un address = {};
  address.sun_family = AF_UNIX;
  socket->copy(address.sun_path, socket->size());
  ASSERT_EQ(connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)),
            0);
  RpcFrameReader reader(fd);
  RpcFrameWriter writer(fd);
  auto call = [&](int id, const std::string& method, nlohmann::json params) {
    EXPECT_TRUE(writer
                    .Write(SerializeJsonRpcMessage(
                        JsonRpcRequest{.id = int64_t{id},
                                       .method = method,
                                       .params = std::move(params)}))
                    .ok());
    auto frame = reader.Read();
    EXPECT_TRUE(frame.ok()) << frame.status();
    return frame.ok() ? nlohmann::json::parse(*frame) : nlohmann::json();
  };
  auto hello =
      call(1, "hello", {{"protocol_version", 1}, {"build_id", "test"}});
  EXPECT_TRUE(hello.contains("result"));
  auto registered = call(2, "register", {{"root", foreign}});
  EXPECT_EQ(registered["error"]["data"],
            static_cast<int>(absl::StatusCode::kInvalidArgument));
  auto foreign_key = RepositoryKeyForRoot(foreign);
  ASSERT_TRUE(foreign_key.ok());
  auto searched = call(3, "search", {{"root", foreign}, {"key", *foreign_key}});
  EXPECT_EQ(searched["error"]["data"],
            static_cast<int>(absl::StatusCode::kInvalidArgument));
  auto notified = call(4, "notify", {{"root", foreign}, {"key", *foreign_key}});
  EXPECT_EQ(notified["error"]["data"],
            static_cast<int>(absl::StatusCode::kInvalidArgument));
  EXPECT_EQ(searched["error"]["data"],
            static_cast<int>(absl::StatusCode::kInvalidArgument));
  close(fd);
}

TEST_F(RpcClientTest, AnExistingRootRegistersWhileSiblingIsCold) {
  auto repo = ferret_testing::GitRepoBuilder::Init(temp_, "repo");
  ASSERT_TRUE(repo.ok()) << repo.status();
  ASSERT_TRUE(repo->WriteFile("file.cc", "needle\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("initial").ok());
  auto sibling = repo->AddWorktree("sibling");
  ASSERT_TRUE(sibling.ok()) << sibling.status();
  root_ = repo->root();
  daemon_.Watch(root_);
  std::atomic<bool> entered = false;
  std::atomic<bool> release = false;
  absl::AnyInvocable<void(absl::string_view)> hook =
      [&](absl::string_view root) {
        if (root != sibling->root()) return;
        entered.store(true);
        while (!release.load()) usleep(1000);
      };
  std::thread daemon([&] {
    EXPECT_TRUE(
        RunDaemon(root_, index_, "test", absl::Seconds(10), false, &hook).ok());
  });
  // The test binary has a different build id. Use a direct matching hello
  // instead of RpcClient, which would correctly drain the test daemon.
  auto path = DaemonSocketPath(root_);
  ASSERT_TRUE(path.ok());
  for (int i = 0; i < 200 && access(path->c_str(), F_OK) != 0; ++i) {
    usleep(10000);
  }
  auto register_root = [&](const std::string& root, bool drain = false) {
    const int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    sockaddr_un address = {};
    address.sun_family = AF_UNIX;
    path->copy(address.sun_path, path->size());
    if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) !=
        0) {
      close(fd);
      return false;
    }
    RpcFrameWriter writer(fd);
    RpcFrameReader reader(fd);
    auto call = [&](const std::string& method, nlohmann::json params) {
      if (!writer
               .Write(SerializeJsonRpcMessage(
                   JsonRpcRequest{.id = int64_t{1},
                                  .method = method,
                                  .params = std::move(params)}))
               .ok()) {
        return false;
      }
      auto frame = reader.Read();
      return frame.ok() && frame->find("\"result\"") != std::string::npos;
    };
    bool ok = call("hello", {{"protocol_version", 1}, {"build_id", "test"}}) &&
              call("register", {{"root", root}});
    if (drain) ok = ok && call("drain", nlohmann::json::object());
    close(fd);
    return ok;
  };
  ASSERT_TRUE(register_root(root_));
  std::atomic<bool> sibling_ok = false;
  std::thread cold([&] { sibling_ok.store(register_root(sibling->root())); });
  for (int i = 0; i < 200 && !entered.load(); ++i) usleep(10000);
  EXPECT_TRUE(entered.load());
  std::atomic<bool> warm_done = false;
  std::atomic<bool> warm_ok = false;
  std::thread warm([&] {
    warm_ok.store(register_root(root_));
    warm_done.store(true);
  });
  for (int i = 0; i < 20 && !warm_done.load(); ++i) usleep(10000);
  EXPECT_TRUE(warm_done.load());
  release.store(true);
  warm.join();
  cold.join();
  EXPECT_TRUE(warm_ok.load());
  EXPECT_TRUE(sibling_ok.load());
  EXPECT_TRUE(register_root(root_, true));
  daemon.join();
}

TEST_F(RpcClientTest, ConcurrentRegistrationOfSameRootIsIdempotent) {
  constexpr int kCalls = 8;
  std::vector<std::thread> threads;
  std::atomic<int> successes = 0;
  threads.reserve(kCalls);
  for (int i = 0; i < kCalls; ++i) {
    threads.emplace_back([&] {
      if (client_->Register(root_).ok()) ++successes;
    });
  }
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(successes.load(), kCalls);
  EXPECT_EQ(client_->Roots().size(), size_t{1});
}

TEST_F(RpcClientTest, SerializesConcurrentCallsOnOneConnection) {
  Register();
  constexpr int kCalls = 16;
  std::atomic<int> successes = 0;
  std::vector<std::thread> threads;
  threads.reserve(kCalls);
  for (int i = 0; i < kCalls; ++i) {
    threads.emplace_back([&] {
      auto search = client_->Search(root_, SearchRequest{.pattern = "missing"},
                                    SearchConsistency());
      if (search.ok() && search->response.num_matching_files == 0) {
        ++successes;
      }
    });
  }
  for (auto& thread : threads) thread.join();
  EXPECT_EQ(successes.load(), kCalls);
}

TEST_F(RpcClientTest, NoSpawnWithoutDaemonOrAfterDeath) {
  RpcClient passive(kBinary, index_, {.spawn_daemon = false});
  ASSERT_TRUE(passive.Register(root_).ok());
  auto search = [&] {
    return passive.Search(root_, SearchRequest{.pattern = "absent"},
                          SearchConsistency());
  };
  EXPECT_FALSE(search().ok());
  auto socket_path = DaemonSocketPath(root_);
  ASSERT_TRUE(socket_path.ok());
  EXPECT_NE(access(socket_path->c_str(), F_OK), 0);

  const pid_t pid = StartServer("--idle_timeout=30m");
  ASSERT_TRUE(search().ok());
  ASSERT_EQ(kill(pid, SIGKILL), 0);  // NOLINT(misc-include-cleaner)
  int status;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  EXPECT_FALSE(search().ok());
  EXPECT_FALSE(search().ok());
  const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(fd, 0);
  sockaddr_un address = {};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, socket_path->c_str(), socket_path->size() + 1);
  EXPECT_NE(connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)),
            0);
  close(fd);
}

TEST_F(RpcClientTest, ReconnectsAfterDaemonIsKilled) {
  const pid_t pid = StartServer("--idle_timeout=30m");
  Register();
  ASSERT_EQ(kill(pid, SIGKILL), 0);  // NOLINT(misc-include-cleaner)
  int exit_status;
  ASSERT_EQ(waitpid(pid, &exit_status, 0), pid);
  auto socket = DaemonSocketPath(root_);
  ASSERT_TRUE(socket.ok());
  for (int i = 0; i < 100 && access(socket->c_str(), F_OK) == 0; ++i) {
    usleep(10000);
  }
  auto result = client_->Search(root_, SearchRequest{.pattern = "absent"},
                                SearchConsistency());
  // The first call detects the broken connection; the next reconnects.
  if (!result.ok()) {
    result = client_->Search(root_, SearchRequest{.pattern = "absent"},
                             SearchConsistency());
  }
  ASSERT_TRUE(result.ok()) << result.status();
}

TEST_F(RpcClientTest, BuildMismatchDrainsAndRestarts) {
  const pid_t pid = StartServer("--test_build_id=old-binary");
  Register();
  int exit_status = 0;
  ASSERT_EQ(waitpid(pid, &exit_status, 0), pid);
  EXPECT_TRUE(WIFEXITED(exit_status));
  auto statuses = client_->IndexStatus();
  ASSERT_TRUE(statuses.ok()) << statuses.status();
  EXPECT_EQ(statuses->size(), size_t{1});
}

TEST_F(RpcClientTest, IdleDaemonExitsWhenNoConnectionsRemain) {
  const pid_t pid = StartServer("--idle_timeout=3s");
  int exit_status = 0;
  ASSERT_EQ(waitpid(pid, &exit_status, 0), pid);
  EXPECT_TRUE(WIFEXITED(exit_status));
  EXPECT_EQ(WEXITSTATUS(exit_status), 0);
}

TEST_F(RpcClientTest, RequiresHelloBeforeAnyOtherMethod) {
  StartServer("--idle_timeout=30m");
  auto path = DaemonSocketPath(root_);
  ASSERT_TRUE(path.ok()) << path.status();
  const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  ASSERT_GE(fd, 0);
  sockaddr_un address = {};
  address.sun_family = AF_UNIX;
  ASSERT_LT(path->size(), sizeof(address.sun_path));
  path->copy(address.sun_path, path->size());
  ASSERT_EQ(connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)),
            0);
  RpcFrameWriter writer(fd);
  RpcFrameReader reader(fd);
  ASSERT_TRUE(
      writer
          .Write(
              R"({"jsonrpc":"2.0","id":1,"method":"index_status","params":{}})")
          .ok());
  auto reply = reader.Read();
  ASSERT_TRUE(reply.ok()) << reply.status();
  auto parsed = ParseJson(*reply);
  ASSERT_TRUE(parsed.ok()) << parsed.status();
  EXPECT_EQ((*parsed)["error"]["message"], "hello must be first");
  close(fd);
}

TEST_F(RpcClientTest, NotifyReturnsGenerationForConsistentSearch) {
  Register();
  const std::string file = absl::StrCat(root_, "/notified.cc");
  FILE* output = fopen(file.c_str(), "w");
  ASSERT_NE(output, nullptr);
  fputs("notified needle\n", output);
  fclose(output);
  auto generation = client_->Notify(root_, {"notified.cc"});
  ASSERT_TRUE(generation.ok()) << generation.status();
  EXPECT_GT(*generation, uint64_t{0});
  SearchConsistency consistency{.mode = SearchConsistencyMode::kNotify,
                                .min_generation = *generation};
  auto result = client_->Search(
      root_, SearchRequest{.pattern = "notified needle"}, consistency);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_EQ(result->response.num_matching_files, 1);
  EXPECT_FALSE(result->stale);
}

TEST_F(RpcClientTest, FindSymbolForwardsThroughDaemon) {
  Register();
  const std::string file = absl::StrCat(root_, "/definition.cc");
  FILE* output = fopen(file.c_str(), "w");
  ASSERT_NE(output, nullptr);
  fputs("int RemoteDefinition() { return 1; }\n", output);
  fclose(output);
  auto generation = client_->Notify(root_, {"definition.cc"});
  ASSERT_TRUE(generation.ok()) << generation.status();
  FindSymbolRequest request{.name = "RemoteDefinition"};
  SearchConsistency consistency{.mode = SearchConsistencyMode::kNotify,
                                .min_generation = *generation,
                                .timeout = absl::Seconds(10)};
  auto found = client_->FindSymbol(root_, request, consistency);
  ASSERT_TRUE(found.ok()) << found.status();
  EXPECT_FALSE(found->stale);
  const absl::Time deadline = absl::Now() + absl::Seconds(10);
  while (found->pending_files != 0 && absl::Now() < deadline) {
    absl::SleepFor(absl::Milliseconds(2));
    found = client_->FindSymbol(root_, request, {});
    ASSERT_TRUE(found.ok()) << found.status();
  }
  ASSERT_EQ(found->occurrences.size(), 1u);
  EXPECT_EQ(found->occurrences[0].path, "definition.cc");
  EXPECT_EQ(found->occurrences[0].source_line,
            "int RemoteDefinition() { return 1; }");
  EXPECT_TRUE(found->occurrences[0].is_definition);
}

TEST_F(RpcClientTest, TwoMcpSessionsShareDaemonButNotWorktreeViews) {
  auto repo = ferret_testing::GitRepoBuilder::Init(temp_, "repo");
  ASSERT_TRUE(repo.ok()) << repo.status();
  ASSERT_TRUE(repo->WriteFile("file.cc", "common needle\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("initial").ok());
  auto sibling = repo->AddWorktree("sibling");
  ASSERT_TRUE(sibling.ok()) << sibling.status();
  ASSERT_TRUE(repo->WriteFile("only.cc", "unique first needle\n").ok());
  ASSERT_TRUE(sibling->WriteFile("other.cc", "unique second needle\n").ok());
  root_ = repo->root();
  daemon_.Watch(root_);
  daemon_.Watch(sibling->root());
  const std::string index_flag = absl::StrCat("--index_dir=", index_);
  auto search_session = [&](const std::string& root,
                            const std::string& pattern) {
    const std::string input = absl::StrCat(
        "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\","
        "\"params\":{\"protocolVersion\":\"2025-11-25\","
        "\"capabilities\":{},\"clientInfo\":{\"name\":\"test\","
        "\"version\":\"1\"}}}\n",
        "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\","
        "\"params\":{\"name\":\"search\",\"arguments\":{\"pattern\":\"",
        pattern, "\"}}}\n");
    return RunSubprocess(
        std::vector<std::string>{kBinary, "mcp", root, index_flag},
        {.stdin_data = input});
  };
  auto first = search_session(root_, "unique first needle");
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ(first->exit_code, 0) << first->stderr_data;
  EXPECT_THAT(first->stdout_data, ::testing::HasSubstr("\"matching_files\":1"));
  auto other = search_session(sibling->root(), "unique first needle");
  ASSERT_TRUE(other.ok()) << other.status();
  EXPECT_EQ(other->exit_code, 0) << other->stderr_data;
  EXPECT_THAT(other->stdout_data, ::testing::HasSubstr("\"matching_files\":0"));
  auto second = search_session(sibling->root(), "unique second needle");
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_THAT(second->stdout_data,
              ::testing::HasSubstr("\"matching_files\":1"));
  const auto first_socket = DaemonSocketPath(root_);
  const auto second_socket = DaemonSocketPath(sibling->root());
  ASSERT_TRUE(first_socket.ok() && second_socket.ok());
  EXPECT_EQ(*first_socket, *second_socket);
  Register();
  ASSERT_TRUE(client_->Register(sibling->root()).ok());
  const auto statuses = client_->IndexStatus();
  ASSERT_TRUE(statuses.ok()) << statuses.status();
  EXPECT_EQ(statuses->size(), size_t{2});
}

TEST_F(RpcClientTest, TwoWorktreesShareOneIndexWithSeparateViews) {
  // The git builder creates the repository at a distinct location; replace
  // the default plain root before starting the client.
  auto repo = ferret_testing::GitRepoBuilder::Init(temp_, "repo");
  ASSERT_TRUE(repo.ok()) << repo.status();
  ASSERT_TRUE(repo->WriteFile("file.cc", "common needle\n").ok());
  ASSERT_TRUE(repo->Add().ok());
  ASSERT_TRUE(repo->Commit("initial").ok());
  auto sibling = repo->AddWorktree("sibling");
  ASSERT_TRUE(sibling.ok()) << sibling.status();
  ASSERT_TRUE(repo->WriteFile("only.cc", "first view needle\n").ok());
  ASSERT_TRUE(sibling->WriteFile("other.cc", "second view needle\n").ok());
  root_ = repo->root();
  daemon_.Watch(root_);
  Register();
  ASSERT_TRUE(client_->Register(sibling->root()).ok());
  const auto socket_a = DaemonSocketPath(root_);
  const auto socket_b = DaemonSocketPath(sibling->root());
  ASSERT_TRUE(socket_a.ok() && socket_b.ok());
  EXPECT_EQ(*socket_a, *socket_b);
  SearchConsistency consistency{.mode = SearchConsistencyMode::kSync};
  auto first = client_->Search(
      root_, SearchRequest{.pattern = "first view needle"}, consistency);
  auto second = client_->Search(sibling->root(),
                                SearchRequest{.pattern = "first view needle"},
                                consistency);
  ASSERT_TRUE(first.ok() && second.ok());
  EXPECT_EQ(first->response.num_matching_files, 1);
  EXPECT_EQ(second->response.num_matching_files, 0);
  auto statuses = client_->IndexStatus();
  ASSERT_TRUE(statuses.ok());
  EXPECT_EQ(statuses->size(), size_t{2});
}

}  // namespace
}  // namespace ferret
// NOLINTEND(misc-include-cleaner)
