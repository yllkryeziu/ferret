// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0
//
// Times searches the way coding agents run them, for benchmarks 1 and 2 of
// bench/README.md.
//
// Usage: search_agents --mode=speed --index_dir=DIR --queries=FILE
//        [--rg=RG] [--engines=...] [--runs=N] WT
//        search_agents --mode=agents --index_dir=DIR --queries=FILE
//        [--rg=RG] [--engines=...] [--agents=1,2,4,8] [--per_agent=N]
//        WT1 ... WTN
//
// Ferret registers every worktree into one index, as its daemon does, and
// ripgrep searches the same worktrees. Every search asks for every matching
// line without context, as `rg -n` prints them. `ferret` searches with the
// daemon's default consistency (kSync: a search first waits until every
// earlier write to its worktree is visible); `ferret_nowait` skips that
// wait.
//
// speed: one agent runs every query; the engines take turns query by query.
// The Ferret engines run each query --runs times and ripgrep once, since it
// takes seconds and varies little.
// agents: for each agent count n, agents 1..n search at once, agent i in
// worktree i, each over its own seeded sample of --per_agent queries (the
// same sample at every n). Each engine runs alone, every search once.
//
// Prints JSON lines: one per registration, then one per timed search with
// the engine, agent count, agent, query id and class, microseconds, and the
// matched lines with their digest (bench/ferret/result_set.h, the digest
// Zoekt's harness writes).

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "bench/ferret/result_set.h"
#include "nlohmann/json.hpp"

#include "ferret/base/env.h"
#include "ferret/engine/engine.h"
#include "ferret/engine/ripgrep_backend.h"
#include "ferret/search/search_types.h"
#include "ferret/workspace/workspace.h"

ABSL_FLAG(std::string, mode, "speed", "speed or agents.");
ABSL_FLAG(std::string, index_dir, "", "Directory for Ferret's index.");
ABSL_FLAG(std::string, queries, "", "JSONL query set.");
ABSL_FLAG(std::string, rg, "",
          "ripgrep executable (default: FERRET_RG, PATH).");
ABSL_FLAG(std::string, engines, "ferret,ferret_nowait,ripgrep",
          "Engines to time: ferret, ferret_nowait, ripgrep.");
ABSL_FLAG(std::string, agents, "1", "Agent counts, comma-separated.");
ABSL_FLAG(int, per_agent, 30, "agents: queries per agent.");
ABSL_FLAG(int, runs, 3, "speed: timed runs of each Ferret search.");
ABSL_FLAG(uint64_t, seed, 1, "Seed of the agents' query samples.");

namespace {

struct Query {
  std::string id;
  std::string query_class;
  ferret::SearchRequest request;
};

struct EngineUnderTest {
  std::string name;
  const ferret::SearchBackend* backend = nullptr;
  ferret::SearchConsistency consistency;
};

absl::StatusOr<std::vector<Query>> LoadQueries(const std::string& path) {
  std::ifstream input(path);
  if (!input) return absl::NotFoundError("cannot open " + path);
  std::vector<Query> queries;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const auto item = nlohmann::json::parse(line, nullptr, false);
    if (item.is_discarded()) return absl::InvalidArgumentError("bad " + line);
    Query query;
    query.id = item["id"].get<std::string>();
    query.query_class = item["class"].get<std::string>();
    ferret::SearchRequest& request = query.request;
    request.pattern = item["pattern"].get<std::string>();
    request.regex = item["kind"] == "regex";
    request.case_sensitive = item["case_sensitive"].get<bool>();
    request.max_files = std::numeric_limits<int64_t>::max();
    request.max_lines_per_file = std::numeric_limits<int>::max();
    request.max_lines = std::numeric_limits<int64_t>::max();
    request.max_verify_bytes = std::numeric_limits<int64_t>::max();
    request.exact_counts = true;
    request.context_lines = 0;
    request.text_only = true;
    queries.push_back(std::move(query));
  }
  return queries;
}

// Agent i's seeded sample of the queries.
std::vector<const Query*> Sample(const std::vector<Query>& queries,
                                 int per_agent, uint64_t seed, int agent) {
  std::vector<const Query*> sample;
  sample.reserve(queries.size());
  for (const Query& query : queries) sample.push_back(&query);
  std::mt19937_64 random(seed * 1000 + agent);
  std::shuffle(sample.begin(), sample.end(), random);
  sample.resize(std::min<size_t>(sample.size(), per_agent));
  return sample;
}

// Runs one search and returns its JSON line, or an error line.
nlohmann::json Time(const EngineUnderTest& engine, const std::string& root,
                    const Query& query) {
  const absl::Time start = absl::Now();
  auto result = engine.backend->Search(root, query.request, engine.consistency);
  const int64_t us = absl::ToInt64Microseconds(absl::Now() - start);
  nlohmann::json line = {{"engine", engine.name},
                         {"query", query.id},
                         {"class", query.query_class},
                         {"us", us}};
  if (!result.ok()) {
    line["error"] = result.status().ToString();
    return line;
  }
  const std::vector<ferret_bench::LineRef> lines =
      ferret_bench::MatchedLines(result->response);
  line["lines"] = lines.size();
  line["digest"] = ferret_bench::LineSetDigest(lines);
  line["stale"] = result->stale;
  return line;
}

}  // namespace

int main(int argc, char** argv) {
  std::vector<char*> args = absl::ParseCommandLine(argc, argv);
  std::vector<std::string> roots(args.begin() + 1, args.end());
  const std::string mode = absl::GetFlag(FLAGS_mode);
  if ((mode != "speed" && mode != "agents") || roots.empty() ||
      absl::GetFlag(FLAGS_index_dir).empty()) {
    std::cerr << "usage: search_agents --index_dir=DIR --queries=FILE "
                 "[flags] WT1 [WT2 ...]\n";
    return 2;
  }
  auto queries = LoadQueries(absl::GetFlag(FLAGS_queries));
  if (!queries.ok()) {
    std::cerr << queries.status() << '\n';
    return 2;
  }
  std::vector<int> agent_counts;
  for (absl::string_view part :
       absl::StrSplit(absl::GetFlag(FLAGS_agents), ',')) {
    int n = 0;
    if (!absl::SimpleAtoi(part, &n) || n < 1 ||
        n > static_cast<int>(roots.size())) {
      std::cerr << "--agents: each count must be 1.." << roots.size() << '\n';
      return 2;
    }
    agent_counts.push_back(n);
  }

  ferret::Engine index(*ferret::Env::Default(), absl::GetFlag(FLAGS_index_dir));
  ferret::EngineBackend ferret_backend(index);
  ferret::RipgrepBackend ripgrep(absl::GetFlag(FLAGS_rg));
  std::vector<EngineUnderTest> engines;
  for (absl::string_view name :
       absl::StrSplit(absl::GetFlag(FLAGS_engines), ',')) {
    if (name == "ferret") {
      engines.push_back(
          {.name = "ferret",
           .backend = &ferret_backend,
           .consistency = {.mode = ferret::SearchConsistencyMode::kSync,
                           .timeout = absl::Seconds(30)}});
    } else if (name == "ferret_nowait") {
      engines.push_back({.name = "ferret_nowait", .backend = &ferret_backend});
    } else if (name == "ripgrep") {
      engines.push_back({.name = "ripgrep", .backend = &ripgrep});
    } else {
      std::cerr << "unknown engine " << name << '\n';
      return 2;
    }
  }

  for (const std::string& root : roots) {
    const absl::Time start = absl::Now();
    if (const absl::Status s = index.Register(root); !s.ok()) {
      std::cerr << root << ": " << s << '\n';
      return 2;
    }
    const int64_t ms = absl::ToInt64Milliseconds(absl::Now() - start);
    if (const absl::Status s = ripgrep.Register(root); !s.ok()) {
      std::cerr << root << ": " << s << '\n';
      return 2;
    }
    std::cout << nlohmann::json{{"register", root}, {"ms", ms}} << '\n'
              << std::flush;
  }

  // Warm-up: Ferret runs every query once, ripgrep one query per worktree.
  const int per_agent = absl::GetFlag(FLAGS_per_agent);
  const uint64_t seed = absl::GetFlag(FLAGS_seed);
  for (const EngineUnderTest& engine : engines) {
    if (engine.backend == &ripgrep) {
      for (const std::string& root : roots) Time(engine, root, (*queries)[0]);
    } else {
      for (const Query& query : *queries) Time(engine, roots[0], query);
    }
  }

  if (mode == "speed") {
    for (const Query& query : *queries) {
      for (const EngineUnderTest& engine : engines) {
        const int runs =
            engine.backend == &ripgrep ? 1 : absl::GetFlag(FLAGS_runs);
        for (int run = 0; run < runs; ++run) {
          nlohmann::json line = Time(engine, roots[0], query);
          line["agents"] = 1;
          line["agent"] = 0;
          line["run"] = run;
          std::cout << line << '\n' << std::flush;
        }
      }
    }
    return 0;
  }
  absl::Mutex out_mu;
  for (const int n : agent_counts) {
    for (const EngineUnderTest& engine : engines) {
      std::vector<std::thread> threads;
      threads.reserve(n);
      for (int agent = 0; agent < n; ++agent) {
        threads.emplace_back([&, agent] {
          for (const Query* query : Sample(*queries, per_agent, seed, agent)) {
            nlohmann::json line = Time(engine, roots[agent], *query);
            line["agents"] = n;
            line["agent"] = agent;
            line["run"] = 0;
            absl::MutexLock lock(out_mu);
            std::cout << line << '\n' << std::flush;
          }
        });
      }
      for (std::thread& thread : threads) thread.join();
    }
  }
  return 0;
}
