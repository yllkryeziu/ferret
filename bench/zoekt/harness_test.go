// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

// End-to-end tests: each builds a real index in a temporary directory and
// drives a subcommand through run(), as a benchmark script would.

package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/sourcegraph/zoekt"
)

// writeTree writes files (path -> content) under root and returns a file
// list naming them in sorted order.
func writeTree(t *testing.T, root string, files map[string]string) string {
	t.Helper()
	var paths []string
	for p, content := range files {
		full := filepath.Join(root, filepath.FromSlash(p))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		paths = append(paths, p)
	}
	slices.Sort(paths)
	list := filepath.Join(t.TempDir(), "files")
	if err := os.WriteFile(list, []byte(strings.Join(paths, "\n")+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return list
}

// runOK runs a subcommand, fails the test unless it exits 0, and returns its
// stdout.
func runOK(t *testing.T, args ...string) []byte {
	t.Helper()
	var stdout, stderr bytes.Buffer
	if code := run(args, &stdout, &stderr); code != 0 {
		t.Fatalf("run(%q) = %d, stderr:\n%s", args, code, stderr.String())
	}
	return stdout.Bytes()
}

// decodeLines decodes one JSON value of type T per line of out.
func decodeLines[T any](t *testing.T, out []byte) []T {
	t.Helper()
	var values []T
	scanner := bufio.NewScanner(bytes.NewReader(out))
	for scanner.Scan() {
		var v T
		if err := json.Unmarshal(scanner.Bytes(), &v); err != nil {
			t.Fatalf("decoding %q: %v", scanner.Text(), err)
		}
		values = append(values, v)
	}
	return values
}

// decodedRecord is a search record as bench/report.py reads it.
type decodedRecord struct {
	ID           string          `json:"id"`
	Mode         string          `json:"mode"`
	K            int             `json:"k"`
	Threads      int             `json:"threads"`
	Clients      int             `json:"clients"`
	LatenciesNS  []int64         `json:"latencies_ns"`
	Truncated    bool            `json:"truncated"`
	FileCount    int             `json:"file_count"`
	ResultCount  int             `json:"result_count"`
	ResultDigest string          `json:"result_digest"`
	DigestStable bool            `json:"digest_stable"`
	RawResults   json.RawMessage `json:"results"`
	Error        string          `json:"error"`
}

// lines returns the record's full result set.
func (r decodedRecord) lines(t *testing.T) []lineRef {
	t.Helper()
	var pairs [][]any
	if err := json.Unmarshal(r.RawResults, &pairs); err != nil {
		t.Fatalf("query %s: decoding results: %v", r.ID, err)
	}
	lines := []lineRef{}
	for _, p := range pairs {
		lines = append(lines, lineRef{Path: p[0].(string), Line: int(p[1].(float64))})
	}
	return lines
}

var searchCorpus = map[string]string{
	"src/a.c":  "int foo(void)\n{\n\treturn bar(1);\n}\n",
	"src/b.py": "def bar(x):\n    return foo(x)\n",
	"README":   "Foo and foo\nnothing here\n",
}

const searchQueries = `{"id":"sensitive","class":"identifier","kind":"literal","pattern":"foo","case_sensitive":true}
{"id":"insensitive","class":"identifier","kind":"literal","pattern":"FOO","case_sensitive":false}
{"id":"call","class":"regex","kind":"regex","pattern":"\\bbar\\(","case_sensitive":true}
{"id":"short","class":"short","kind":"literal","pattern":"(","case_sensitive":true}
{"id":"zero","class":"zero","kind":"literal","pattern":"no_such_thing","case_sensitive":true}
`

func TestBuildAndSearch(t *testing.T) {
	root := t.TempDir()
	list := writeTree(t, root, searchCorpus)
	indexDir := filepath.Join(t.TempDir(), "index")

	build := decodeLines[buildReport](t, runOK(t, "build", "--root="+root, "--files="+list, "--out="+indexDir))[0]
	if build.Engine != "zoekt" || build.EngineVersion != "v0.0.0-20260911061844-153817f643cd" {
		t.Errorf("build engine = %s %s, want the pinned zoekt", build.Engine, build.EngineVersion)
	}
	wantBytes := int64(0)
	for _, content := range searchCorpus {
		wantBytes += int64(len(content))
	}
	if build.FilesListed != 3 || build.FilesIndexed != 3 || build.BytesIndexed != wantBytes || len(build.Skipped) != 0 {
		t.Errorf("build = %+v, want all 3 files (%d bytes) indexed", build, wantBytes)
	}
	if build.Shards != 1 || build.IndexBytes <= 0 || build.WallSeconds <= 0 || build.PeakRSSBytes <= 0 {
		t.Errorf("build cost = %+v, want one shard and positive measurements", build)
	}
	if build.Options.SizeMax != 1<<20 || build.Options.TrigramMax != 20000 || build.Options.CTags {
		t.Errorf("build options = %+v, want -file_limit 1048576 -max_trigram_count 20000 without ctags", build.Options)
	}

	queries := writeQueries(t, searchQueries)
	out := runOK(t, "search", "--index="+indexDir, "--queries="+queries, "--mode=exhaustive",
		"--context=0", "--warmup=1", "--runs=3", "--clients=2", "--full_results")
	records := decodeLines[decodedRecord](t, out)
	want := map[string][]lineRef{
		"sensitive":   {{"README", 1}, {"src/a.c", 1}, {"src/b.py", 2}},
		"insensitive": {{"README", 1}, {"src/a.c", 1}, {"src/b.py", 2}},
		"call":        {{"src/a.c", 3}, {"src/b.py", 1}},
		"short":       {{"src/a.c", 1}, {"src/a.c", 3}, {"src/b.py", 1}, {"src/b.py", 2}},
		"zero":        {},
	}
	if len(records) != len(want) {
		t.Fatalf("got %d records, want %d", len(records), len(want))
	}
	for _, r := range records {
		got := r.lines(t)
		if !reflect.DeepEqual(got, want[r.ID]) {
			t.Errorf("query %s: results = %v, want %v", r.ID, got, want[r.ID])
		}
		if r.Mode != modeExhaustive || r.Threads != 4 || r.Clients != 2 || len(r.LatenciesNS) != 6 {
			t.Errorf("query %s: mode %s, threads %d, clients %d, %d latencies; want exhaustive, 4, 2, 6",
				r.ID, r.Mode, r.Threads, r.Clients, len(r.LatenciesNS))
		}
		if r.ResultCount != len(got) || r.ResultDigest != lineSetDigest(got) || !r.DigestStable || r.Truncated {
			t.Errorf("query %s: count %d, digest %s, stable %v, truncated %v disagree with results %v",
				r.ID, r.ResultCount, r.ResultDigest, r.DigestStable, r.Truncated, got)
		}
	}

	// Top-K returns at most K files, all of them from the exhaustive set.
	out = runOK(t, "search", "--index="+indexDir, "--queries="+queries, "--mode=topk", "--k=1",
		"--warmup=0", "--runs=1", "--full_results")
	for _, r := range decodeLines[decodedRecord](t, out) {
		if r.K != 1 || r.FileCount > 1 {
			t.Errorf("query %s: k %d, %d files; want at most 1", r.ID, r.K, r.FileCount)
		}
		for _, l := range r.lines(t) {
			if !slices.Contains(want[r.ID], l) {
				t.Errorf("query %s: top-K result %v is not in the exhaustive set", r.ID, l)
			}
		}
	}
}

func TestSearchScopesQueriesToTheirFiles(t *testing.T) {
	root := t.TempDir()
	list := writeTree(t, root, searchCorpus)
	indexDir := filepath.Join(t.TempDir(), "index")
	runOK(t, "build", "--root="+root, "--files="+list, "--out="+indexDir)
	queries := writeQueries(t, `{"id":"unscoped","class":"c","kind":"literal","pattern":"foo","case_sensitive":true}
{"id":"scoped","class":"c","kind":"literal","pattern":"foo","case_sensitive":true,"files":["src/b.py","missing.c"]}
{"id":"regex","class":"c","kind":"regex","pattern":"\\bbar\\(","case_sensitive":true,"files":["src/a.c"]}
{"id":"none","class":"c","kind":"literal","pattern":"foo","case_sensitive":true,"files":[]}
`)
	out := runOK(t, "search", "--index="+indexDir, "--queries="+queries, "--mode=exhaustive",
		"--context=0", "--warmup=0", "--runs=1", "--full_results")
	want := map[string][]lineRef{
		"unscoped": {{"README", 1}, {"src/a.c", 1}, {"src/b.py", 2}},
		"scoped":   {{"src/b.py", 2}},
		"regex":    {{"src/a.c", 3}},
		"none":     {},
	}
	for _, r := range decodeLines[decodedRecord](t, out) {
		if got := r.lines(t); !reflect.DeepEqual(got, want[r.ID]) {
			t.Errorf("query %s: results = %v, want %v", r.ID, got, want[r.ID])
		}
	}
}

func TestSearchReadsFilesFromAFileOnce(t *testing.T) {
	root := t.TempDir()
	list := writeTree(t, root, searchCorpus)
	indexDir := filepath.Join(t.TempDir(), "index")
	runOK(t, "build", "--root="+root, "--files="+list, "--out="+indexDir)
	scopeDir := t.TempDir()
	scope := filepath.Join(scopeDir, "scope.txt")
	if err := os.WriteFile(scope, []byte("src/b.py\nmissing.c\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	empty := filepath.Join(scopeDir, "empty.txt")
	if err := os.WriteFile(empty, nil, 0o644); err != nil {
		t.Fatal(err)
	}
	queries := writeQueries(t, `{"id":"unscoped","class":"c","kind":"literal","pattern":"foo","case_sensitive":true}
{"id":"scoped","class":"c","kind":"literal","pattern":"foo","case_sensitive":true,"files_from":"`+scope+`"}
{"id":"again","class":"c","kind":"literal","pattern":"foo","case_sensitive":false,"files_from":"`+scope+`"}
{"id":"none","class":"c","kind":"literal","pattern":"foo","case_sensitive":true,"files_from":"`+empty+`"}
{"id":"gone","class":"c","kind":"literal","pattern":"foo","case_sensitive":true,"files_from":"`+filepath.Join(scopeDir, "gone.txt")+`"}
`)
	// "again" shares "scoped"'s path; TestFileSetCacheReadsEachPathOnce
	// covers the single read.
	out := runOK(t, "search", "--index="+indexDir, "--queries="+queries, "--mode=exhaustive",
		"--context=0", "--warmup=0", "--runs=2", "--full_results")
	want := map[string][]lineRef{
		"unscoped": {{"README", 1}, {"src/a.c", 1}, {"src/b.py", 2}},
		"scoped":   {{"src/b.py", 2}},
		"again":    {{"src/b.py", 2}},
		"none":     {},
	}
	for _, r := range decodeLines[decodedRecord](t, out) {
		if r.ID == "gone" {
			if r.Error == "" || len(r.LatenciesNS) != 0 {
				t.Errorf("query gone: error %q, latencies %v; want an error and no timing", r.Error, r.LatenciesNS)
			}
			continue
		}
		if r.Error != "" {
			t.Errorf("query %s: unexpected error %q", r.ID, r.Error)
		}
		if got := r.lines(t); !reflect.DeepEqual(got, want[r.ID]) {
			t.Errorf("query %s: results = %v, want %v", r.ID, got, want[r.ID])
		}
	}
}

func TestFileSetCacheReadsEachPathOnce(t *testing.T) {
	path := filepath.Join(t.TempDir(), "scope.txt")
	if err := os.WriteFile(path, []byte("a\nb/c\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	cache := newFileSetCache()
	first, err := cache.load(path)
	if err != nil || len(first) != 2 {
		t.Fatalf("load() = %v, %v; want 2 paths", first, err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	second, err := cache.load(path)
	if err != nil || len(second) != 2 {
		t.Errorf("second load() = %v, %v; want the cached set", second, err)
	}
}

func TestSearchReportsAQueryThatDoesNotBuildAndRunsTheRest(t *testing.T) {
	root := t.TempDir()
	list := writeTree(t, root, searchCorpus)
	indexDir := filepath.Join(t.TempDir(), "index")
	runOK(t, "build", "--root="+root, "--files="+list, "--out="+indexDir)
	// Go's regexp/syntax rejects the nested repetition and the \C escape.
	queries := writeQueries(t, `{"id":"ok1","class":"c","kind":"literal","pattern":"foo","case_sensitive":true}
{"id":"nested","class":"c","kind":"regex","pattern":"a**","case_sensitive":true}
{"id":"escape","class":"c","kind":"regex","pattern":"fo\\C","case_sensitive":true}
{"id":"ok2","class":"c","kind":"regex","pattern":"fo+","case_sensitive":true}
`)
	for _, mode := range []string{"exhaustive", "topk"} {
		out := runOK(t, "search", "--index="+indexDir, "--queries="+queries, "--mode="+mode,
			"--warmup=1", "--runs=2", "--full_results")
		records := decodeLines[decodedRecord](t, out)
		if len(records) != 4 {
			t.Fatalf("%s: %d records, want 4", mode, len(records))
		}
		for _, r := range records {
			switch r.ID {
			case "nested", "escape":
				if !strings.Contains(r.Error, "query "+r.ID) || len(r.LatenciesNS) != 0 {
					t.Errorf("%s: query %s: error %q, latencies %v", mode, r.ID, r.Error, r.LatenciesNS)
				}
			default:
				if r.Error != "" || len(r.LatenciesNS) != 2 || r.ResultCount == 0 {
					t.Errorf("%s: query %s: %+v, want a normal result", mode, r.ID, r)
				}
			}
		}
	}
}

func TestSearchMarksTruncatedResults(t *testing.T) {
	root := t.TempDir()
	list := writeTree(t, root, searchCorpus)
	indexDir := filepath.Join(t.TempDir(), "index")
	runOK(t, "build", "--root="+root, "--files="+list, "--out="+indexDir)
	queries := writeQueries(t, `{"id":"short","class":"short","kind":"literal","pattern":"(","case_sensitive":true}`+"\n")

	out := runOK(t, "search", "--index="+indexDir, "--queries="+queries, "--mode=exhaustive",
		"--max_matches=1", "--warmup=0", "--runs=1")
	if r := decodeLines[decodedRecord](t, out)[0]; !r.Truncated {
		t.Errorf("record = %+v, want truncated at --max_matches=1", r)
	}
}

func TestBuildReportsSkippedFiles(t *testing.T) {
	// More than 20000 distinct trigrams without a NUL byte.
	rng := rand.New(rand.NewPCG(1, 2))
	const alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	manyTrigrams := make([]byte, 40000)
	for i := range manyTrigrams {
		manyTrigrams[i] = alphabet[rng.IntN(len(alphabet))]
	}
	root := t.TempDir()
	list := writeTree(t, root, map[string]string{
		"big":    strings.Repeat("x", zoektSizeMax+1),
		"binary": "abc\x00def",
		"many":   string(manyTrigrams),
		"ok.c":   "int ok;\n",
		// A real file whose content equals a placeholder is not a skip.
		"placeholder.txt": notIndexedMarker + "contains binary content",
		"tiny":            "ab",
	})

	var stdout, stderr bytes.Buffer
	code := run([]string{"build", "--root=" + root, "--files=" + list, "--out=" + filepath.Join(t.TempDir(), "index")},
		&stdout, &stderr)
	if code != 1 || !strings.Contains(stderr.String(), "zoekt skipped 4 of 6 listed files") {
		t.Errorf("run(build) = %d, stderr %q; want 1 and a skip error", code, stderr.String())
	}
	report := decodeLines[buildReport](t, stdout.Bytes())[0]
	wantSkipped := map[string][]string{
		"binary":            {"binary"},
		"too_large":         {"big"},
		"too_many_trigrams": {"many"},
		"too_small":         {"tiny"},
	}
	if !reflect.DeepEqual(report.Skipped, wantSkipped) {
		t.Errorf("skipped = %v, want %v", report.Skipped, wantSkipped)
	}
	wantBytes := int64(len("int ok;\n") + len(notIndexedMarker+"contains binary content"))
	if report.FilesIndexed != 2 || report.BytesIndexed != wantBytes {
		t.Errorf("indexed %d files, %d bytes; want 2 files, %d bytes", report.FilesIndexed, report.BytesIndexed, wantBytes)
	}
}

func TestBuildRejectsNonEmptyIndexDir(t *testing.T) {
	root := t.TempDir()
	list := writeTree(t, root, searchCorpus)
	indexDir := t.TempDir()
	if err := os.WriteFile(filepath.Join(indexDir, "stale.zoekt"), nil, 0o644); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	if code := run([]string{"build", "--root=" + root, "--files=" + list, "--out=" + indexDir}, &stdout, &stderr); code != 1 ||
		!strings.Contains(stderr.String(), "not empty") {
		t.Errorf("run(build) = %d, stderr %q; want 1 and \"not empty\"", code, stderr.String())
	}
}

func git(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", dir, "-c", "user.name=t", "-c", "user.email=t@localhost"}, args...)...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %q: %v\n%s", args, err, out)
	}
	return strings.TrimSpace(string(out))
}

func TestFreshness(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is not installed")
	}
	repo := t.TempDir()
	list := writeTree(t, repo, searchCorpus)
	git(t, repo, "init", "--quiet")
	git(t, repo, "add", ".")
	git(t, repo, "commit", "--quiet", "--message", "base")

	const trials = 3
	out := runOK(t, "freshness", "--repo="+repo, "--files="+list, "--index="+filepath.Join(t.TempDir(), "index"),
		"--trials=3", "--trial_timeout=1m")
	records := decodeLines[freshnessRecord](t, out)
	if len(records) != trials {
		t.Fatalf("got %d records, want %d", len(records), trials)
	}
	for i, r := range records {
		if r.Trial != i || r.Engine != "zoekt" || searchCorpus[r.Path] == "" {
			t.Errorf("record %d = %+v", i, r)
		}
		if r.CommitNS <= 0 || r.BuildNS <= 0 || r.ReloadNS < 0 || r.TotalNS != r.CommitNS+r.BuildNS+r.ReloadNS {
			t.Errorf("record %d stages = %+v, want positive stages that sum to the total", i, r)
		}
		if r.Shards != i+2 {
			t.Errorf("record %d has %d shards, want %d (one base shard plus one per trial)", i, r.Shards, i+2)
		}
	}
	if got := git(t, repo, "rev-list", "--count", "HEAD"); got != "4" {
		t.Errorf("repository has %s commits, want 4 (base plus one per trial)", got)
	}
	if status := git(t, repo, "status", "--porcelain"); status != "" {
		t.Errorf("worktree is dirty after the trials:\n%s", status)
	}
}

func TestRunRejectsBadInvocations(t *testing.T) {
	tests := []struct {
		args []string
		want int
	}{
		{args: nil, want: 2},
		{args: []string{"nope"}, want: 2},
		{args: []string{"search", "--queries=q"}, want: 2},
		{args: []string{"search", "--index=i", "--queries=q", "--mode=fast"}, want: 2},
		{args: []string{"search", "--index=i", "--queries=q", "--runs=0"}, want: 2},
		{args: []string{"build", "--root=r", "--files=f", "--out=o", "extra"}, want: 2},
		{args: []string{"build", "-help"}, want: 0},
	}
	for _, tc := range tests {
		var stdout, stderr bytes.Buffer
		if got := run(tc.args, &stdout, &stderr); got != tc.want {
			t.Errorf("run(%q) = %d, want %d; stderr:\n%s", tc.args, got, tc.want, stderr.String())
		}
	}
}

func TestTruncatedWhenTotalMatchLimitIsReached(t *testing.T) {
	opts := &zoekt.SearchOptions{TotalMaxMatchCount: 10}
	for _, tc := range []struct {
		stats zoekt.Stats
		want  bool
	}{
		{zoekt.Stats{MatchCount: 9}, false},
		{zoekt.Stats{MatchCount: 10}, true},
		{zoekt.Stats{MatchCount: 42}, true},
		{zoekt.Stats{MatchCount: 1, FilesSkipped: 1}, true},
		{zoekt.Stats{MatchCount: 1, ShardsSkipped: 1}, true},
	} {
		if got := truncated(tc.stats, opts); got != tc.want {
			t.Errorf("truncated(%+v) = %v, want %v", tc.stats, got, tc.want)
		}
	}
	if truncated(zoekt.Stats{MatchCount: 1 << 30}, &zoekt.SearchOptions{}) {
		t.Error("a search without a total limit is never truncated by its match count")
	}
}
