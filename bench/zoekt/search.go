// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"math/rand/v2"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/sourcegraph/zoekt"
	"github.com/sourcegraph/zoekt/query"
	"github.com/sourcegraph/zoekt/search"
)

// Search modes: every match, or the top K files.
const (
	modeExhaustive = "exhaustive"
	modeTopK       = "topk"
)

// searchConfig holds the per-query search parameters of one invocation.
type searchConfig struct {
	mode         string
	k            int
	contextLines int
	maxMatches   int
}

// options returns the SearchOptions for a query. numDocs is Zoekt's
// EstimateDocCount for the query; it is used in top-K mode only.
//
// Exhaustive mode lifts Zoekt's default match limits to maxMatches, the
// memory safety cap; a result that reaches it is reported as truncated and
// excluded from parity. Top-K mode applies the limits that Zoekt's own JSON
// API derives from MaxDocDisplayCount, so Zoekt stops early exactly as its
// server would.
func (c searchConfig) options(numDocs int) *zoekt.SearchOptions {
	opts := &zoekt.SearchOptions{NumContextLines: c.contextLines}
	if c.mode == modeExhaustive {
		opts.ShardMaxMatchCount = c.maxMatches
		opts.TotalMaxMatchCount = c.maxMatches
		return opts
	}
	opts.MaxDocDisplayCount = c.k
	opts.ShardMaxMatchCount, opts.TotalMaxMatchCount = topKMatchLimits(numDocs, c.k)
	return opts
}

// topKMatchLimits reproduces internal/json.CalculateDefaultSearchLimits at
// the pinned version: the per-shard and total match limits (0 = unlimited)
// that Zoekt's JSON API uses for MaxDocDisplayCount = k when the query's
// shards hold numDocs documents.
func topKMatchLimits(numDocs, k int) (shardMax, totalMax int) {
	if numDocs > 10000 {
		return k*5 + (5*k)/(numDocs/1000), 0
	}
	n := numDocs + k*100
	return n, n
}

// searchRecord is one line of `zoekt-bench search` output. The schema is
// shared with Ferret's harness and consumed by bench/report.py.
type searchRecord struct {
	ID           string           `json:"id"`
	Class        string           `json:"class"`
	Engine       string           `json:"engine"`
	Mode         string           `json:"mode"`
	K            int              `json:"k,omitempty"`
	Context      int              `json:"context"`
	Threads      int              `json:"threads"`
	Clients      int              `json:"clients"`
	LatenciesNS  []int64          `json:"latencies_ns"`
	Truncated    bool             `json:"truncated"`
	FileCount    int              `json:"file_count"`
	ResultCount  int              `json:"result_count"`
	ResultDigest string           `json:"result_digest"`
	DigestStable bool             `json:"digest_stable"`
	EngineStats  map[string]int64 `json:"engine_stats"`
	Results      *[]lineRef       `json:"results,omitempty"`
	// Error is why the query could not be built (for example a regex that
	// Go's regexp/syntax rejects). Such a query is not run, and the other
	// queries are unaffected.
	Error string `json:"error,omitempty"`
}

// querySlot is one query and the measurements collected for it. Clients
// running concurrently share slots, so the measurements are guarded.
type querySlot struct {
	spec querySpec
	q    query.Q
	opts *zoekt.SearchOptions

	mu     sync.Mutex
	timed  bool // Whether a timed run has been recorded.
	record searchRecord
}

// searchHarness runs a query set against one open searcher.
type searchHarness struct {
	searcher    zoekt.Searcher
	slots       []*querySlot
	clients     int
	seed        uint64
	fullResults bool
}

func runSearch(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("search", flag.ContinueOnError)
	indexDir := fs.String("index", "", "index directory written by `zoekt-bench build`")
	queriesPath := fs.String("queries", "", "query set (JSONL)")
	mode := fs.String("mode", modeTopK, `"exhaustive" or "topk"`)
	k := fs.Int("k", 20, "files per result in topk mode (MaxDocDisplayCount)")
	contextLines := fs.Int("context", 2, "context lines around each match (NumContextLines)")
	maxMatches := fs.Int("max_matches", 1_000_000, "exhaustive-mode match cap; results reaching it are marked truncated")
	gomaxprocs := fs.Int("gomaxprocs", 4, "GOMAXPROCS, which also sizes Zoekt's shard search parallelism")
	clients := fs.Int("clients", 1, "concurrent clients, each running every pass over all queries")
	warmup := fs.Int("warmup", 3, "untimed passes over all queries before timing")
	runs := fs.Int("runs", 10, "timed passes over all queries")
	seed := fs.Uint64("seed", 1, "seed for the per-pass query order")
	fullResults := fs.Bool("full_results", false, "also emit each query's (path, line) set, not only its digest")
	if err := parseFlags(fs, args); err != nil {
		return err
	}
	if err := requireFlags(fs, "index", "queries"); err != nil {
		return err
	}
	cfg := searchConfig{mode: *mode, contextLines: *contextLines, maxMatches: *maxMatches}
	switch {
	case *mode == modeTopK:
		cfg.k = *k
		if *k < 1 {
			return usageError{fmt.Errorf("--k must be positive, got %d", *k)}
		}
	case *mode != modeExhaustive:
		return usageError{fmt.Errorf("--mode must be %q or %q, got %q", modeExhaustive, modeTopK, *mode)}
	}
	for _, f := range []struct {
		name       string
		value, min int
	}{
		{"context", *contextLines, 0},
		{"max_matches", *maxMatches, 1},
		{"gomaxprocs", *gomaxprocs, 1},
		{"clients", *clients, 1},
		{"warmup", *warmup, 0},
		{"runs", *runs, 1},
	} {
		if f.value < f.min {
			return usageError{fmt.Errorf("--%s must be at least %d, got %d", f.name, f.min, f.value)}
		}
	}

	specs, err := readQueries(*queriesPath)
	if err != nil {
		return err
	}
	// Zoekt sizes its shard scheduler from GOMAXPROCS when the searcher is
	// created, so this must come first.
	runtime.GOMAXPROCS(*gomaxprocs)
	searcher, err := openSearcher(*indexDir)
	if err != nil {
		return err
	}
	defer searcher.Close()

	h := &searchHarness{searcher: searcher, clients: *clients, seed: *seed, fullResults: *fullResults}
	ctx := context.Background()
	sets := newFileSetCache()
	for _, spec := range specs {
		slot, err := newQuerySlot(ctx, searcher, spec, cfg, sets)
		if err != nil {
			return err
		}
		slot.record.Threads = *gomaxprocs
		slot.record.Clients = *clients
		h.slots = append(h.slots, slot)
	}
	pass := 0
	for ; pass < *warmup; pass++ {
		if err := h.runPass(ctx, pass, false); err != nil {
			return err
		}
	}
	for ; pass < *warmup+*runs; pass++ {
		if err := h.runPass(ctx, pass, true); err != nil {
			return err
		}
	}

	w := bufio.NewWriter(stdout)
	for _, slot := range h.slots {
		if err := writeJSON(w, &slot.record); err != nil {
			return err
		}
	}
	return w.Flush()
}

// openSearcher opens every shard in dir once. It fails on a directory
// without shards, which Zoekt would otherwise serve as an empty index.
func openSearcher(dir string) (zoekt.Streamer, error) {
	shards, err := filepath.Glob(filepath.Join(dir, "*.zoekt"))
	if err != nil {
		return nil, err
	}
	if len(shards) == 0 {
		return nil, fmt.Errorf("%s: no *.zoekt shards", dir)
	}
	return search.NewDirectorySearcher(dir)
}

// newQuerySlot compiles spec and fixes its search options. In top-K mode it
// runs Zoekt's document-count estimate once, outside every timed region,
// although Zoekt's JSON API pays for it on each request.
//
// A query that cannot be built becomes a slot with only an Error, which no
// pass runs; the error return is for failures of the searcher itself.
func newQuerySlot(ctx context.Context, s zoekt.Searcher, spec querySpec, cfg searchConfig, sets *fileSetCache) (*querySlot, error) {
	q, err := zoektQueryWith(spec, sets)
	if err != nil {
		return &querySlot{
			spec: spec,
			record: searchRecord{
				ID:     spec.ID,
				Class:  spec.Class,
				Engine: "zoekt",
				Mode:   cfg.mode,
				Error:  err.Error(),
			},
		}, nil
	}
	numDocs := 0
	if cfg.mode == modeTopK {
		est, err := s.Search(ctx, q, &zoekt.SearchOptions{EstimateDocCount: true})
		if err != nil {
			return nil, fmt.Errorf("query %s: estimating document count: %w", spec.ID, err)
		}
		numDocs = est.ShardFilesConsidered
	}
	return &querySlot{
		spec: spec,
		q:    q,
		opts: cfg.options(numDocs),
		record: searchRecord{
			ID:      spec.ID,
			Class:   spec.Class,
			Engine:  "zoekt",
			Mode:    cfg.mode,
			K:       cfg.k,
			Context: cfg.contextLines,
		},
	}, nil
}

// runPass runs every query once per client. Each (pass, client) pair gets
// its own deterministic query order, so no query is always measured right
// after the same predecessor.
func (h *searchHarness) runPass(ctx context.Context, pass int, timed bool) error {
	errs := make([]error, h.clients)
	var wg sync.WaitGroup
	for c := range h.clients {
		order := rand.New(rand.NewPCG(h.seed, uint64(pass*h.clients+c))).Perm(len(h.slots))
		wg.Add(1)
		go func() {
			defer wg.Done()
			for _, i := range order {
				if h.slots[i].record.Error != "" {
					continue
				}
				if err := h.runQuery(ctx, h.slots[i], timed); err != nil {
					errs[c] = err
					return
				}
			}
		}()
	}
	wg.Wait()
	return errors.Join(errs...)
}

// runQuery runs one search and, if timed, records its latency and result
// set. Only the Search call is timed; Zoekt has already copied the results
// out of the shards when it returns.
func (h *searchHarness) runQuery(ctx context.Context, slot *querySlot, timed bool) error {
	start := time.Now()
	res, err := h.searcher.Search(ctx, slot.q, slot.opts)
	elapsed := time.Since(start)
	if err != nil {
		return fmt.Errorf("query %s: %w", slot.spec.ID, err)
	}
	if res.Crashes > 0 {
		return fmt.Errorf("query %s: %d shard(s) crashed", slot.spec.ID, res.Crashes)
	}
	if !timed {
		return nil
	}
	lines := matchedLines(res.Files)
	digest := lineSetDigest(lines)

	slot.mu.Lock()
	defer slot.mu.Unlock()
	r := &slot.record
	r.LatenciesNS = append(r.LatenciesNS, elapsed.Nanoseconds())
	r.Truncated = r.Truncated || truncated(res.Stats, slot.opts)
	if slot.timed {
		r.DigestStable = r.DigestStable && digest == r.ResultDigest
		return nil
	}
	slot.timed = true
	r.FileCount = len(res.Files)
	r.ResultCount = len(lines)
	r.ResultDigest = digest
	r.DigestStable = true
	r.EngineStats = engineStats(res.Stats)
	if h.fullResults {
		r.Results = &lines
	}
	return nil
}

// truncated reports whether a search may have returned a partial result: one
// that skipped files or shards, or that reached TotalMaxMatchCount. Reaching
// the total limit cancels the shards still running without counting them as
// skipped, so which matches survive depends on timing. A query is truncated
// if any of its timed runs is.
func truncated(s zoekt.Stats, opts *zoekt.SearchOptions) bool {
	return s.FilesSkipped > 0 || s.ShardsSkipped > 0 ||
		(opts.TotalMaxMatchCount > 0 && s.MatchCount >= opts.TotalMaxMatchCount)
}

// engineStats returns the Zoekt counters recorded next to each query's
// latency: how much of the index and content the search touched.
func engineStats(s zoekt.Stats) map[string]int64 {
	return map[string]int64{
		"content_bytes_loaded":       s.ContentBytesLoaded,
		"index_bytes_loaded":         s.IndexBytesLoaded,
		"file_count":                 int64(s.FileCount),
		"files_considered":           int64(s.FilesConsidered),
		"files_loaded":               int64(s.FilesLoaded),
		"files_skipped":              int64(s.FilesSkipped),
		"shard_files_considered":     int64(s.ShardFilesConsidered),
		"shards_scanned":             int64(s.ShardsScanned),
		"shards_skipped":             int64(s.ShardsSkipped),
		"shards_skipped_filter":      int64(s.ShardsSkippedFilter),
		"match_count":                int64(s.MatchCount),
		"ngram_matches":              int64(s.NgramMatches),
		"ngram_lookups":              int64(s.NgramLookups),
		"regexps_considered":         int64(s.RegexpsConsidered),
		"match_tree_construction_ns": s.MatchTreeConstruction.Nanoseconds(),
		"match_tree_search_ns":       s.MatchTreeSearch.Nanoseconds(),
	}
}
