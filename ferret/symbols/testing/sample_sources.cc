// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

#include "ferret/symbols/testing/sample_sources.h"

#include "absl/log/log.h"
#include "absl/strings/string_view.h"

#include "ferret/text/language.h"

namespace ferret_testing {
namespace {

constexpr absl::string_view kCSample = R"sample(/*
 * ring.c - a fixed-size byte ring shared by the capture drivers.
 */
#include <stddef.h>
#include "ring.h"

#define RING_MIN_SIZE 16
#define ring_is_empty(r) ((r)->head == (r)->tail)

struct ring;

typedef void (*ring_cb)(struct ring *ring, void *data);

enum ring_state {
	RING_IDLE,
	RING_BUSY = 4,
};

struct ring {
	size_t head;
	size_t tail;
	enum ring_state state;
	ring_cb on_full;
	union {
		unsigned char *bytes;
		unsigned long *words;
	} buf;
};

typedef struct ring ring_t;

extern int ring_debug;
static unsigned int ring_count;
static const char *const ring_state_names[] = { "idle", "busy" };

int ring_init(struct ring *ring, size_t size);
static void ring_notify(struct ring *ring);

int ring_init(struct ring *ring, size_t size)
{
	struct ring_stats { int pushes; } stats = { 0 };

	if (size < RING_MIN_SIZE)
		return -1;
	ring->head = ring->tail = 0;
	ring_count++;
	return stats.pushes;
}

static void ring_notify(struct ring *ring)
{
#ifdef CONFIG_RING_DEBUG
	int depth = ring_debug;
#endif
	if (ring->on_full)
		ring->on_full(ring, NULL);
}

/* Kept in K&R form for the old toolchain. */
int ring_legacy_size(ring, scale)
	struct ring *ring;
	int scale;
{
	return (int)(ring->tail - ring->head) * scale;
}

void (*ring_handler(int signo))(int)
{
	return NULL;
}

char **ring_names(void);
)sample";

constexpr absl::string_view kCppSample = R"sample(// Disk cache backend.

#include <memory>
#include <string>

namespace net::cache {

class Backend;

using EntryId = uint64_t;

enum class CacheType : uint8_t {
  kDisk = 0,
  kMemory = 1,
};

inline constexpr int kMaxEntries = 1024;
extern const char kCacheDirName[];

template <typename T>
class ScopedRef {
 public:
  explicit ScopedRef(T* ptr) : ptr_(ptr) {}
  ~ScopedRef();

  T* get() const { return ptr_; }
  T& operator*() const;

 private:
  T* ptr_;
};

class NET_EXPORT DiskCache : public Backend {
 public:
  struct Options {
    int max_bytes = 0;
  };

  static std::unique_ptr<DiskCache> Create(const Options& options);

  DiskCache(const DiskCache&) = delete;
  DiskCache& operator=(const DiskCache&) = delete;
  ~DiskCache() override;

  int OpenEntry(const std::string& key, EntryId* id);
  virtual void Flush() = 0;
  explicit operator bool() const;

  friend bool operator==(const DiskCache& a, const DiskCache& b);

 private:
  DiskCache();

  int entry_count_ = 0;
};

namespace {

constexpr char kIndexFile[] = "index";

struct EntryRecord {
  EntryId id;
  int size;
};

int Clamp(int value, int limit) { return value < limit ? value : limit; }

}  // namespace

const char kCacheDirName[] = "Cache";

template <typename T>
ScopedRef<T>::~ScopedRef() = default;

template <typename T>
T& ScopedRef<T>::operator*() const {
  return *ptr_;
}

DiskCache::DiskCache() = default;
DiskCache::~DiskCache() = default;

// static
std::unique_ptr<DiskCache> DiskCache::Create(const Options& options) {
  auto cache = std::unique_ptr<DiskCache>(new DiskCache());
  cache->entry_count_ = Clamp(options.max_bytes, kMaxEntries);
  return cache;
}

int DiskCache::OpenEntry(const std::string& key, EntryId* id) {
  struct Probe {
    int attempts = 0;
  } probe;
  *id = static_cast<EntryId>(key.size());
  return probe.attempts;
}

DiskCache::operator bool() const { return entry_count_ > 0; }

bool operator==(const DiskCache& a, const DiskCache& b) {
  return a.entry_count_ == b.entry_count_;
}

}  // namespace net::cache
)sample";

constexpr absl::string_view kPythonSample =
    R"sample("""Retry helpers for flaky remote operations."""

from __future__ import annotations

import asyncio
import functools
import logging
from typing import Callable, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)
DEFAULT_ATTEMPTS: int = 3
_registry: dict[str, Callable]

type Predicate = Callable[[Exception], bool]


class RetryError(Exception):
    """Raised when every attempt fails."""

    def __init__(self, attempts: int, last: Exception) -> None:
        super().__init__(f"failed after {attempts} attempts")
        self.attempts = attempts
        self.last = last


def retry(attempts: int = DEFAULT_ATTEMPTS):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last = None
            for attempt in range(attempts):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    logger.warning("attempt %d failed", attempt)
                    last = e
            raise RetryError(attempts, last)

        return wrapper

    return decorator


class Backoff:
    base_delay = 0.1

    class Policy:
        def next_delay(self, attempt):
            return 2**attempt

    @classmethod
    def default(cls) -> Backoff:
        return cls()

    async def sleep(self, attempt: int) -> None:
        await asyncio.sleep(self.base_delay * attempt)

    @property
    def jitter(self):
        return 0


if __name__ == "__main__":
    result = retry()(print)("hello")
)sample";

}  // namespace

absl::string_view SampleSource(ferret::Language language) {
  switch (language) {
    case ferret::Language::kC:
      return kCSample;
    case ferret::Language::kCpp:
      return kCppSample;
    case ferret::Language::kPython:
      return kPythonSample;
    default:
      LOG(FATAL) << "no sample source for " << ferret::LanguageName(language);
  }
}

}  // namespace ferret_testing
