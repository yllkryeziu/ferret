// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Experiment: is FSEventStreamFlushSync a barrier on its own? Usage:
//   fsevents_flush_experiment DIR THREADS WRITES_PER_THREAD NOISE_THREADS MODE
//       [LATENCY_SECONDS [NODEFER(0/1) [GAP_US]]]
// MODE is one of:
//   flush        write a file, call FlushSync, count the writes whose event
//                was not delivered when FlushSync returned;
//   flush_serial the same with FlushSync calls serialized by a mutex;
//   cookie       create a cookie file, call FlushSync, then wait for it;
//   cookie_wait  create a cookie file and wait for it with no FlushSync.
// For the cookie modes, "late" is the time from creation to delivery.
// DIR must be an empty scratch directory. Misses are counted, never fatal.

#include <CoreFoundation/CFArray.h>
#include <CoreFoundation/CFBase.h>
#include <CoreFoundation/CFString.h>
#include <CoreServices/CoreServices.h>  // IWYU pragma: keep
#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <sys/fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/synchronization/mutex.h"

namespace {

ABSL_CONST_INIT absl::Mutex g_mu(absl::kConstInit);
std::unordered_set<std::string> g_seen;
std::unordered_map<std::string, uint64_t> g_ids;
std::atomic<int64_t> g_events{0};

void OnStream(ConstFSEventStreamRef, void*, size_t count, void* paths,
              const FSEventStreamEventFlags[],
              const FSEventStreamEventId ids[]) {
  absl::MutexLock lock(g_mu);
  for (size_t i = 0; i < count; ++i) {
    g_seen.insert(static_cast<char**>(paths)[i]);
    g_ids[static_cast<char**>(paths)[i]] = ids[i];
  }
  g_events += static_cast<int64_t>(count);
}

bool Seen(const std::string& path) {
  absl::MutexLock lock(g_mu);
  return g_seen.count(path) != 0;
}

void Touch(const std::string& path, const char* bytes) {
  const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
  if (fd < 0) std::abort();
  if (write(fd, bytes, 8) != 8) std::abort();
  close(fd);
}

int64_t Nanos() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 6 || argc > 9) {
    std::fprintf(stderr, "usage: %s DIR THREADS WRITES NOISE MODE\n", argv[0]);
    return 2;
  }
  std::error_code error;
  const std::string dir = std::filesystem::canonical(argv[1], error).string();
  if (error) return 2;
  const int threads = std::atoi(argv[2]);
  const int writes = std::atoi(argv[3]);
  const int noise = std::atoi(argv[4]);
  const std::string mode = argv[5];
  const double latency = argc > 6 ? std::atof(argv[6]) : 0.01;
  const bool nodefer = argc > 7 ? std::atoi(argv[7]) != 0 : true;
  const int gap_us = argc > 8 ? std::atoi(argv[8]) : 0;
  for (int t = 0; t < threads + noise; ++t) {
    mkdir((dir + "/t" + std::to_string(t)).c_str(), 0700);
  }
  CFStringRef root = CFStringCreateWithCString(kCFAllocatorDefault, dir.c_str(),
                                               kCFStringEncodingUTF8);
  const void* values[] = {root};
  CFArrayRef paths =
      CFArrayCreate(kCFAllocatorDefault, values, 1, &kCFTypeArrayCallBacks);
  CFRelease(root);
  FSEventStreamRef stream =
      FSEventStreamCreate(kCFAllocatorDefault, &OnStream, nullptr, paths,
                          kFSEventStreamEventIdSinceNow, latency,
                          kFSEventStreamCreateFlagFileEvents |
                              (nodefer ? kFSEventStreamCreateFlagNoDefer : 0) |
                              kFSEventStreamCreateFlagWatchRoot);
  CFRelease(paths);
  dispatch_queue_t queue =
      dispatch_queue_create("flush.experiment", DISPATCH_QUEUE_SERIAL);
  FSEventStreamSetDispatchQueue(stream, queue);
  if (!FSEventStreamStart(stream)) return 3;
  usleep(500000);

  std::atomic<bool> stop{false};
  std::vector<std::thread> noisy;
  noisy.reserve(static_cast<size_t>(noise));
  for (int n = 0; n < noise; ++n) {
    noisy.emplace_back([&, n] {
      const std::string base = dir + "/t" + std::to_string(threads + n);
      for (int i = 0; !stop.load(); ++i) {
        Touch(base + "/n" + std::to_string(i % 200), "noise!!!");
        if (i % 50 == 0) usleep(1000);
      }
    });
  }

  absl::Mutex serial;
  std::atomic<int64_t> misses{0};
  std::atomic<int64_t> never{0};
  std::atomic<int64_t> total{0};
  std::atomic<int64_t> id_covered{0};
  std::atomic<int64_t> id_after{0};
  absl::Mutex stats_mu;
  std::vector<int64_t> flush_ns;
  std::vector<int64_t> late_ns;
  const int64_t begin = Nanos();
  std::vector<std::thread> workers;
  workers.reserve(static_cast<size_t>(threads));
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&, t] {
      std::vector<int64_t> mine;
      std::vector<int64_t> late;
      for (int i = 0; i < writes; ++i) {
        const std::string path =
            dir + "/t" + std::to_string(t) +
            (mode.rfind("cookie", 0) == 0 ? "/cookie-" : "/f") +
            std::to_string(i);
        if (gap_us > 0) usleep(static_cast<useconds_t>(gap_us));
        Touch(path, "payload!");
        const int64_t start = Nanos();
        if (mode == "flush") {
          FSEventStreamFlushSync(stream);
        } else if (mode == "flush_after_id") {
          // Does asking for the current event id make the service read the
          // kernel's queue?
          (void)FSEventsGetCurrentEventId();
          FSEventStreamFlushSync(stream);
        } else if (mode.rfind("delay", 0) == 0) {
          usleep(static_cast<useconds_t>(std::atoi(mode.c_str() + 5)));
          FSEventStreamFlushSync(stream);
        } else if (mode == "ids") {
          const uint64_t current = FSEventsGetCurrentEventId();
          bool got = false;
          for (int w = 0; w < 100000 && !got; ++w) {
            absl::MutexLock lock(g_mu);
            const auto it = g_ids.find(path);
            if (it != g_ids.end()) {
              got = true;
              if (it->second <= current)
                ++id_covered;
              else
                ++id_after;
            }
            if (!got) usleep(50);
          }
        } else if (mode == "flush_serial" || mode == "cookie") {
          absl::MutexLock lock(serial);
          FSEventStreamFlushSync(stream);
        }
        bool seen = Seen(path);
        if (mode == "cookie_wait" || mode == "cookie") {
          while (!seen && Nanos() - start < 2000000000) {
            usleep(20);
            seen = Seen(path);
          }
          late.push_back(Nanos() - start);
          seen = true;
        }
        mine.push_back(Nanos() - start);
        if (!seen) {
          ++misses;
          const int64_t waited = Nanos();
          while (!(seen = Seen(path)) && Nanos() - waited < 2000000000) {
            usleep(100);
          }
          if (seen) {
            late.push_back(Nanos() - waited);
          } else {
            ++never;
          }
        }
        ++total;
      }
      absl::MutexLock lock(stats_mu);
      flush_ns.insert(flush_ns.end(), mine.begin(), mine.end());
      late_ns.insert(late_ns.end(), late.begin(), late.end());
    });
  }
  for (std::thread& worker : workers) worker.join();
  stop.store(true);
  for (std::thread& n : noisy) n.join();
  const double seconds = static_cast<double>(Nanos() - begin) / 1e9;
  std::sort(flush_ns.begin(), flush_ns.end());
  std::sort(late_ns.begin(), late_ns.end());
  auto pct = [](const std::vector<int64_t>& v, size_t p) -> int64_t {
    return v.empty() ? -1
                     : v[std::min(v.size() - 1, v.size() * p / 100)] / 1000;
  };
  std::printf(
      "mode=%s threads=%d writes=%lld noise=%d seconds=%.1f events=%lld "
      "misses=%lld never_delivered=%lld flush_or_check_us p50=%lld p99=%lld "
      "late_us p50=%lld p99=%lld max=%lld\n",
      mode.c_str(), threads, static_cast<long long>(total.load()), noise,
      seconds, static_cast<long long>(g_events.load()),
      static_cast<long long>(misses.load()),
      static_cast<long long>(never.load()),
      static_cast<long long>(pct(flush_ns, 50)),
      static_cast<long long>(pct(flush_ns, 99)),
      static_cast<long long>(pct(late_ns, 50)),
      static_cast<long long>(pct(late_ns, 99)),
      static_cast<long long>(pct(late_ns, 100)));
  if (mode == "ids") {
    std::printf("event id <= current id at write return: %lld, later: %lld\n",
                static_cast<long long>(id_covered.load()),
                static_cast<long long>(id_after.load()));
  }
  FSEventStreamStop(stream);
  FSEventStreamInvalidate(stream);
  FSEventStreamRelease(stream);
  dispatch_release(queue);
  return 0;
}
