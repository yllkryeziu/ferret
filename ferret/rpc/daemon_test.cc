// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Filesystem permissions, peer credentials, lock arbitration and stale sockets.
#include "ferret/rpc/daemon.h"

#include <dirent.h>
#if defined(__APPLE__)
#include <mach-o/loader.h>
#endif
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <string>
#include <thread>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "nlohmann/json.hpp"

#include "ferret/engine/engine.h"
#include "ferret/git/testing/git_repo_builder.h"
#include "ferret/rpc/framing.h"
#include "ferret/rpc/json_codec.h"
#include "ferret/rpc/json_rpc.h"
#include "ferret/testing/temp_dir.h"

// The Apple SDK exposes F_OK through unistd.h, but include-cleaner misses it.
// NOLINTBEGIN(misc-include-cleaner)
namespace ferret {
namespace {

void WriteBuildIdFixture(const std::string& path, const void* bytes,
                         size_t size) {
  FILE* file = fopen(path.c_str(), "wb");
  ASSERT_NE(file, nullptr);
  EXPECT_EQ(fwrite(bytes, 1, size, file), size);
  EXPECT_EQ(fclose(file), 0);
}

TEST(DaemonBuildIdTest, FallsBackToSha256WithoutLinkerIdentity) {
  ferret_testing::TempDir temp;
  const std::string path = temp.Join("blob");
  WriteBuildIdFixture(path, "abc", 3);
  const auto first = DaemonBuildId(path);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ(*first,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(DaemonBuildId(path), first);
  WriteBuildIdFixture(path, "abd", 3);
  const auto changed = DaemonBuildId(path);
  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_NE(*first, *changed);
}

#if defined(__APPLE__)
TEST(DaemonBuildIdTest, RealCliHasMachOUuid) {
  const char* runfiles = getenv("TEST_SRCDIR");
  ASSERT_NE(runfiles, nullptr);
  const std::string executable =
      std::string(runfiles) + "/_main/ferret/cli/ferret";
  const auto id = DaemonBuildId(executable);
  ASSERT_TRUE(id.ok()) << id.status();
  EXPECT_EQ(id->substr(0, 6), "macho:");
}

TEST(DaemonBuildIdTest, UsesMachOUuid) {
  ferret_testing::TempDir temp;
  const std::string path = temp.Join("macho");
  struct Fixture {
    mach_header_64 header;
    uuid_command uuid;
  } fixture = {};
  fixture.header.magic = MH_MAGIC_64;
  fixture.header.ncmds = 1;
  fixture.header.sizeofcmds = sizeof(uuid_command);
  fixture.uuid.cmd = LC_UUID;
  fixture.uuid.cmdsize = sizeof(uuid_command);
  fixture.uuid.uuid[0] = 0x42;
  WriteBuildIdFixture(path, &fixture, sizeof(fixture));
  const auto first = DaemonBuildId(path);
  ASSERT_TRUE(first.ok()) << first.status();
  EXPECT_EQ(*first, "macho:42000000000000000000000000000000");
  EXPECT_EQ(DaemonBuildId(path), first);
  fixture.uuid.uuid[0] = 0x43;
  WriteBuildIdFixture(path, &fixture, sizeof(fixture));
  const auto changed = DaemonBuildId(path);
  ASSERT_TRUE(changed.ok()) << changed.status();
  EXPECT_NE(*first, *changed);
}
#endif

int OpenFds() {
  DIR* directory = opendir("/dev/fd");
  if (directory == nullptr) return -1;
  int count = 0;
  while (readdir(directory) != nullptr) ++count;
  closedir(directory);
  return count;
}

int ConnectSocket(const std::string& path) {
  const int fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (fd < 0) return fd;
  sockaddr_un address = {};
  address.sun_family = AF_UNIX;
  path.copy(address.sun_path, path.size());
  if (connect(fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) ==
      0) {
    return fd;
  }
  close(fd);
  return -1;
}

TEST(DaemonTest, SearchAndNotifyRequireRegistrationAndRepository) {
  ferret_testing::TempDir temp;
  auto repository = ferret_testing::GitRepoBuilder::Init(temp, "root");
  ASSERT_TRUE(repository.ok()) << repository.status();
  const std::string root = repository->root();
  const std::string unregistered = temp.Join("root/subdir");
  const std::string other = temp.Join("other");
  ASSERT_EQ(mkdir(unregistered.c_str(), 0700), 0);
  ASSERT_EQ(mkdir(other.c_str(), 0700), 0);
  auto key = RepositoryKeyForRoot(unregistered);
  auto other_key = RepositoryKeyForRoot(other);
  ASSERT_TRUE(key.ok() && other_key.ok());
  std::thread owner([&] {
    EXPECT_TRUE(
        RunDaemon(root, temp.Join("index"), "test", absl::Seconds(5)).ok());
  });
  auto path = DaemonSocketPath(root);
  ASSERT_TRUE(path.ok());
  for (int i = 0; i < 500 && access(path->c_str(), F_OK) != 0; ++i) {
    usleep(10000);
  }
  const int fd = ConnectSocket(*path);
  ASSERT_GE(fd, 0);
  RpcFrameWriter writer(fd);
  RpcFrameReader reader(fd);
  auto call = [&](absl::string_view method, const nlohmann::json& params) {
    EXPECT_TRUE(writer
                    .Write(SerializeJsonRpcMessage(
                        JsonRpcRequest{.id = int64_t{1},
                                       .method = std::string(method),
                                       .params = params}))
                    .ok());
    auto frame = reader.Read();
    EXPECT_TRUE(frame.ok()) << frame.status();
    return frame.ok() ? ParseJson(*frame)
                      : absl::StatusOr<nlohmann::json>(frame.status());
  };
  ASSERT_TRUE(
      call("hello", {{"protocol_version", 1}, {"build_id", "test"}}).ok());
  for (const auto& [candidate_root, candidate_key, code] :
       std::vector<std::tuple<std::string, std::string, int>>{
           {unregistered, *key, 5}, {other, *other_key, 3}}) {
    for (absl::string_view method : {"search", "notify"}) {
      auto response =
          call(method, {{"root", candidate_root}, {"key", candidate_key}});
      ASSERT_TRUE(response.ok()) << response.status();
      EXPECT_EQ((*response)["error"]["data"], code);
    }
  }
  EXPECT_TRUE(call("drain", nlohmann::json::object()).ok());
  close(fd);
  owner.join();
}

TEST(DaemonTest, ReapsClosedConnectionsAndRejectsBeyondLimit) {
  ferret_testing::TempDir temp;
  const std::string root = temp.Join("root");
  ASSERT_EQ(mkdir(root.c_str(), 0700), 0);
  std::thread owner([&] {
    EXPECT_TRUE(
        RunDaemon(root, temp.Join("index"), "test", absl::Seconds(3)).ok());
  });
  auto path = DaemonSocketPath(root);
  ASSERT_TRUE(path.ok());
  for (int i = 0; i < 500 && access(path->c_str(), F_OK) != 0; ++i) {
    usleep(10000);
  }
  ASSERT_EQ(access(path->c_str(), F_OK), 0);
  const int baseline = OpenFds();
  ASSERT_GT(baseline, 0);
  for (int i = 0; i < 200; ++i) {
    const int fd = ConnectSocket(*path);
    ASSERT_GE(fd, 0);
    RpcFrameWriter writer(fd);
    RpcFrameReader reader(fd);
    ASSERT_TRUE(
        writer
            .Write(SerializeJsonRpcMessage(JsonRpcRequest{
                .id = int64_t{1},
                .method = "hello",
                .params = {{"protocol_version", 1}, {"build_id", "test"}}}))
            .ok());
    ASSERT_TRUE(reader.Read().ok());
    close(fd);
  }
  // The wake pipe reaps finished threads without needing another accept.
  int after = OpenFds();
  for (int i = 0; i < 200 && after > baseline + 8; ++i) {
    usleep(10000);
    after = OpenFds();
  }
  EXPECT_LE(after, baseline + 8);
  std::vector<int> peers;
  for (int i = 0; i < 64; ++i) {
    const int fd = ConnectSocket(*path);
    ASSERT_GE(fd, 0);
    RpcFrameWriter writer(fd);
    RpcFrameReader reader(fd);
    ASSERT_TRUE(
        writer
            .Write(SerializeJsonRpcMessage(JsonRpcRequest{
                .id = int64_t{1},
                .method = "hello",
                .params = {{"protocol_version", 1}, {"build_id", "test"}}}))
            .ok());
    ASSERT_TRUE(reader.Read().ok());
    peers.push_back(fd);
  }
  const int excess = ConnectSocket(*path);
  ASSERT_GE(excess, 0);
  RpcFrameReader reader(excess);
  auto rejected = reader.Read();
  ASSERT_TRUE(rejected.ok()) << rejected.status();
  EXPECT_NE(rejected->find("connection limit reached"), std::string::npos);
  close(excess);
  for (int fd : peers) close(fd);
  owner.join();
}

TEST(DaemonTest, RepliesToMalformedAndNonRequestFrames) {
  ferret_testing::TempDir temp;
  const std::string root = temp.Join("root");
  ASSERT_EQ(mkdir(root.c_str(), 0700), 0);
  std::thread owner([&] {
    EXPECT_TRUE(
        RunDaemon(root, temp.Join("index"), "test", absl::Milliseconds(600))
            .ok());
  });
  auto path = DaemonSocketPath(root);
  ASSERT_TRUE(path.ok());
  for (int i = 0; i < 500 && access(path->c_str(), F_OK) != 0; ++i) {
    usleep(10000);
  }
  for (const auto& [frame, code] :
       std::vector<std::pair<std::string, std::string>>{
           {"{bad json", "-32700"},
           {"{\"jsonrpc\":\"2.0\",\"method\":\"hello\"}", "-32600"}}) {
    const int fd = ConnectSocket(*path);
    ASSERT_GE(fd, 0);
    RpcFrameWriter writer(fd);
    RpcFrameReader reader(fd);
    ASSERT_TRUE(writer.Write(frame).ok());
    auto response = reader.Read();
    ASSERT_TRUE(response.ok()) << response.status();
    EXPECT_NE(response->find(code), std::string::npos);
    close(fd);
  }
  owner.join();
}

TEST(DaemonTest, RejectsInsecureDirectoryAndSymlink) {
  ferret_testing::TempDir temp;
  const std::string path = temp.Join("socket");
  ASSERT_TRUE(CheckSocketDirectory(path).ok());
  struct stat st;
  ASSERT_EQ(lstat(path.c_str(), &st), 0);
  EXPECT_EQ(st.st_mode & 0777, 0700);
  ASSERT_EQ(chmod(path.c_str(), 0755), 0);
  EXPECT_EQ(CheckSocketDirectory(path).code(),
            absl::StatusCode::kPermissionDenied);
  const std::string alias = temp.Join("alias");
  ASSERT_EQ(symlink(path.c_str(), alias.c_str()), 0);
  EXPECT_EQ(CheckSocketDirectory(alias).code(),
            absl::StatusCode::kPermissionDenied);
}

TEST(DaemonTest, OnlySameUidPeerIsAccepted) {
  int fds[2];
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM, 0, fds), 0);
  EXPECT_TRUE(CheckDaemonPeer(fds[0]).ok());
  close(fds[0]);
  close(fds[1]);
  EXPECT_FALSE(CheckDaemonPeer(-1).ok());
}

TEST(DaemonTest, ASecondOwnerCannotUnlinkSocket) {
  ferret_testing::TempDir temp;
  const std::string root = temp.Join("root");
  ASSERT_EQ(mkdir(root.c_str(), 0700), 0);
  const std::string index = temp.Join("index");
  std::thread owner([&] {
    EXPECT_TRUE(RunDaemon(root, index, "test", absl::Seconds(5)).ok());
  });
  auto path = DaemonSocketPath(root);
  if (path.ok()) {
    for (int i = 0; i < 500 && access(path->c_str(), F_OK) != 0;
         ++i) {  // NOLINT(misc-include-cleaner)
      usleep(10000);
    }
    const bool started = access(path->c_str(), F_OK) == 0;
    EXPECT_TRUE(started);
    if (started) {
      EXPECT_EQ(RunDaemon(root, index, "test", absl::Milliseconds(5)).code(),
                absl::StatusCode::kAlreadyExists);
      EXPECT_EQ(access(path->c_str(), F_OK), 0);
    }
  } else {
    ADD_FAILURE() << path.status();
  }
  owner.join();
  if (path.ok()) EXPECT_NE(access(path->c_str(), F_OK), 0);
}

TEST(DaemonTest, StartsWithPersistedRootsAndReregistersWithoutRescanning) {
  ferret_testing::TempDir temp;
  auto repo = ferret_testing::GitRepoBuilder::Init(temp, "repo");
  ASSERT_TRUE(repo.ok()) << repo.status();
  ASSERT_TRUE(repo->WriteFile("edit.cc", "daemonrestore needle\n").ok());
  const std::string index = temp.Join("index");
  {
    Engine initial(*Env::Default(), index);
    ASSERT_TRUE(initial.Register(repo->root()).ok());
  }
  std::thread server([&] {
    EXPECT_TRUE(RunDaemon(repo->root(), index, "test", absl::Seconds(10)).ok());
  });
  auto socket_path = DaemonSocketPath(repo->root());
  ASSERT_TRUE(socket_path.ok()) << socket_path.status();
  for (int i = 0; i < 500 && access(socket_path->c_str(), F_OK) != 0; ++i) {
    usleep(10000);
  }
  const int fd = ConnectSocket(*socket_path);
  ASSERT_GE(fd, 0);
  RpcFrameWriter writer(fd);
  RpcFrameReader reader(fd);
  auto call = [&](absl::string_view method, const nlohmann::json& params) {
    EXPECT_TRUE(writer
                    .Write(SerializeJsonRpcMessage(
                        JsonRpcRequest{.id = int64_t{1},
                                       .method = std::string(method),
                                       .params = params}))
                    .ok());
    auto frame = reader.Read();
    EXPECT_TRUE(frame.ok()) << frame.status();
    return frame.ok() ? ParseJson(*frame)
                      : absl::StatusOr<nlohmann::json>(frame.status());
  };
  EXPECT_TRUE(
      call("hello", {{"protocol_version", 1}, {"build_id", "test"}}).ok());
  bool published = false;
  for (int i = 0; i < 500; ++i) {
    auto status = call("index_status", nlohmann::json::object());
    ASSERT_TRUE(status.ok()) << status.status();
    if ((*status)["result"].size() == 1) {
      EXPECT_TRUE((*status)["result"][0].contains("root_hex"));
      EXPECT_EQ((*status)["result"][0]["file_count"], 1);
      published = true;
      break;
    }
    usleep(10000);
  }
  EXPECT_TRUE(published);
  auto registration = call("register", {{"root", repo->root()}});
  ASSERT_TRUE(registration.ok()) << registration.status();
  EXPECT_FALSE(registration->contains("error"));
  auto status = call("index_status", nlohmann::json::object());
  ASSERT_TRUE(status.ok()) << status.status();
  EXPECT_EQ((*status)["result"].size(), size_t{1});
  EXPECT_TRUE(call("drain", nlohmann::json::object()).ok());
  close(fd);
  server.join();
}

}  // namespace
}  // namespace ferret
// NOLINTEND(misc-include-cleaner)
