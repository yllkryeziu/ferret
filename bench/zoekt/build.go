// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"syscall"
	"time"

	"github.com/sourcegraph/zoekt"
	"github.com/sourcegraph/zoekt/index"
	"github.com/sourcegraph/zoekt/query"
	"github.com/sourcegraph/zoekt/search"
)

// Build limits equivalent to `zoekt-index -file_limit 1048576
// -max_trigram_count 20000`. Ferret's `zoekt` eligibility policy applies the
// same rules, so a correctly generated file list has no Zoekt skips.
const (
	zoektSizeMax    = 1 << 20
	zoektTrigramMax = 20000
)

// notIndexedMarker is the content Zoekt stores in place of a skipped file,
// followed by the reason's explanation (index/shard_builder.go and
// index/document.go at the pinned version). The strings are unexported, so
// they are repeated here; TestBuildReportsSkippedFiles catches drift.
const notIndexedMarker = "NOT-INDEXED: "

// skipExplanations maps each explanation Zoekt writes after
// notIndexedMarker to the reason name used in reports.
var skipExplanations = map[string]string{
	"exceeds the maximum size limit": "too_large",
	"contains too few trigrams":      "too_small",
	"contains binary content":        "binary",
	"contains too many trigrams":     "too_many_trigrams",
}

// buildReport is the JSON document written by `zoekt-bench build`. The field
// names are consumed by bench/report.py.
type buildReport struct {
	Engine        string `json:"engine"`
	EngineVersion string `json:"engine_version"`
	Name          string `json:"name"`
	FilesListed   int    `json:"files_listed"`
	FilesIndexed  int    `json:"files_indexed"`
	BytesIndexed  int64  `json:"bytes_indexed"`
	// Skipped maps a skip reason to the listed paths Zoekt did not index.
	Skipped      map[string][]string `json:"skipped"`
	Shards       int                 `json:"shards"`
	IndexBytes   int64               `json:"index_bytes"`
	WallSeconds  float64             `json:"wall_seconds"`
	UserSeconds  float64             `json:"user_seconds"`
	SysSeconds   float64             `json:"sys_seconds"`
	CPUSeconds   float64             `json:"cpu_seconds"`
	PeakRSSBytes int64               `json:"peak_rss_bytes"`
	Options      buildOptionsReport  `json:"options"`
}

// buildOptionsReport records the effective builder options.
type buildOptionsReport struct {
	SizeMax     int  `json:"size_max"`
	TrigramMax  int  `json:"trigram_max"`
	ShardMax    int  `json:"shard_max"`
	Parallelism int  `json:"parallelism"`
	CTags       bool `json:"ctags"`
	GOMAXPROCS  int  `json:"gomaxprocs"`
}

func runBuild(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("build", flag.ContinueOnError)
	root := fs.String("root", "", "corpus root that the paths in --files are relative to")
	files := fs.String("files", "", "file list: one repo-relative path per line")
	out := fs.String("out", "", "index directory to create; must be absent or empty")
	name := fs.String("name", "", "repository name stored in the shards (default: base name of --root)")
	parallelism := fs.Int("parallelism", 8, "shards built concurrently, as zoekt-index -parallelism")
	if err := parseFlags(fs, args); err != nil {
		return err
	}
	if err := requireFlags(fs, "root", "files", "out"); err != nil {
		return err
	}
	if *parallelism < 1 {
		return usageError{fmt.Errorf("--parallelism must be positive, got %d", *parallelism)}
	}
	paths, err := readFileList(*files)
	if err != nil {
		return err
	}
	report, err := buildIndex(*root, paths, *out, repoName(*name, *root), *parallelism)
	if err != nil {
		return err
	}
	if err := writeJSON(stdout, report); err != nil {
		return err
	}
	return checkNoSkips(report)
}

// repoName returns name, or the base name of root when name is empty.
func repoName(name, root string) string {
	if name != "" {
		return name
	}
	return filepath.Base(filepath.Clean(root))
}

// checkNoSkips fails if Zoekt skipped any listed file: the list was then not
// generated with the `zoekt` skip policy, and the engines index different
// file sets.
func checkNoSkips(report *buildReport) error {
	n := 0
	for _, paths := range report.Skipped {
		n += len(paths)
	}
	if n > 0 {
		return fmt.Errorf("zoekt skipped %d of %d listed files (see \"skipped\"); the list is not zoekt-eligible", n, report.FilesListed)
	}
	return nil
}

// builderOptions returns the options for every build of the index named name
// in indexDir, full or delta. Delta builds require the options that enter
// Options.GetHash to match the full build's.
func builderOptions(name, indexDir string, parallelism int) index.Options {
	return index.Options{
		IndexDir:    indexDir,
		SizeMax:     zoektSizeMax,
		TrigramMax:  zoektTrigramMax,
		Parallelism: parallelism,
		// Ferret's text index has no symbols, and ctags would make the build
		// depend on whichever ctags binary is on PATH.
		DisableCTags:          true,
		RepositoryDescription: zoekt.Repository{Name: name},
	}
}

// buildIndex indexes paths (relative to root) into the new directory
// indexDir the way zoekt-index does: files are read serially in list order
// and handed to one index.Builder, which builds shards in parallel. Wall
// time, CPU time, and peak RSS cover reading, indexing, and writing.
func buildIndex(root string, paths []string, indexDir, name string, parallelism int) (*buildReport, error) {
	if err := createEmptyDir(indexDir); err != nil {
		return nil, err
	}
	opts := builderOptions(name, indexDir, parallelism)
	sizes := make(map[string]int64, len(paths))

	before, err := currentUsage()
	if err != nil {
		return nil, err
	}
	start := time.Now()
	builder, err := index.NewBuilder(opts)
	if err != nil {
		return nil, err
	}
	// Finish is idempotent and must also run on failure to remove temporary
	// shards; on success the explicit call below reports its error.
	defer func() { _ = builder.Finish() }()
	for _, p := range paths {
		content, err := readRegularFile(root, p)
		if err != nil {
			return nil, err
		}
		sizes[p] = int64(len(content))
		if err := builder.AddFile(p, content); err != nil {
			return nil, fmt.Errorf("adding %s: %w", p, err)
		}
	}
	if err := builder.Finish(); err != nil {
		return nil, fmt.Errorf("writing shards: %w", err)
	}
	wall := time.Since(start)
	after, err := currentUsage()
	if err != nil {
		return nil, err
	}

	skipped, err := findSkippedFiles(indexDir, sizes)
	if err != nil {
		return nil, err
	}
	shards, indexBytes, err := indexSize(indexDir)
	if err != nil {
		return nil, err
	}
	report := &buildReport{
		Engine:        "zoekt",
		EngineVersion: zoektVersion(),
		Name:          name,
		FilesListed:   len(paths),
		Skipped:       skipped,
		Shards:        shards,
		IndexBytes:    indexBytes,
		WallSeconds:   wall.Seconds(),
		UserSeconds:   (after.user - before.user).Seconds(),
		SysSeconds:    (after.sys - before.sys).Seconds(),
		PeakRSSBytes:  after.maxRSSBytes,
	}
	report.CPUSeconds = report.UserSeconds + report.SysSeconds
	skippedPaths := make(map[string]bool)
	for _, ps := range skipped {
		for _, p := range ps {
			skippedPaths[p] = true
		}
	}
	for p, size := range sizes {
		if !skippedPaths[p] {
			report.FilesIndexed++
			report.BytesIndexed += size
		}
	}
	effective := opts
	effective.SetDefaults()
	report.Options = buildOptionsReport{
		SizeMax:     effective.SizeMax,
		TrigramMax:  effective.TrigramMax,
		ShardMax:    effective.ShardMax,
		Parallelism: effective.Parallelism,
		CTags:       !effective.DisableCTags,
		GOMAXPROCS:  runtime.GOMAXPROCS(0),
	}
	return report, nil
}

// findSkippedFiles returns, by reason, the listed files that Zoekt stored as
// a skip placeholder instead of indexing. It reads the shards rather than
// re-running Zoekt's checks, so it reports what was actually stored and adds
// nothing to the measured build. sizes holds each listed file's original
// size; a real file whose content happens to equal a placeholder keeps its
// own size and is not reported.
func findSkippedFiles(indexDir string, sizes map[string]int64) (map[string][]string, error) {
	searcher, err := search.NewDirectorySearcher(indexDir)
	if err != nil {
		return nil, err
	}
	defer searcher.Close()
	q := &query.Substring{Pattern: notIndexedMarker, CaseSensitive: true, Content: true}
	all := len(sizes) + 1
	res, err := searcher.Search(context.Background(), q,
		&zoekt.SearchOptions{ShardMaxMatchCount: all, TotalMaxMatchCount: all})
	if err != nil {
		return nil, err
	}
	skipped := make(map[string][]string)
	for _, file := range res.Files {
		for _, lm := range file.LineMatches {
			explanation, ok := strings.CutPrefix(string(lm.Line), notIndexedMarker)
			if !ok || lm.LineNumber != 1 || sizes[file.FileName] == int64(len(lm.Line)) {
				continue
			}
			if reason, ok := skipExplanations[explanation]; ok {
				skipped[reason] = append(skipped[reason], file.FileName)
			}
		}
	}
	for _, paths := range skipped {
		sort.Strings(paths)
	}
	return skipped, nil
}

// createEmptyDir creates dir if needed and fails if it holds any entry, so
// that a report measures exactly one build.
func createEmptyDir(dir string) error {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	if len(entries) > 0 {
		return fmt.Errorf("%s: index directory is not empty", dir)
	}
	return nil
}

// indexSize returns the number of shards in dir and the total size of its
// regular files (shards and their .meta sidecars).
func indexSize(dir string) (shards int, bytes int64, err error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, 0, err
	}
	for _, e := range entries {
		if !e.Type().IsRegular() {
			continue
		}
		info, err := e.Info()
		if err != nil {
			return 0, 0, err
		}
		bytes += info.Size()
		if strings.HasSuffix(e.Name(), ".zoekt") {
			shards++
		}
	}
	return shards, bytes, nil
}

// resourceUsage is this process's CPU time and peak resident set size.
type resourceUsage struct {
	user, sys   time.Duration
	maxRSSBytes int64
}

func currentUsage() (resourceUsage, error) {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_SELF, &ru); err != nil {
		return resourceUsage{}, fmt.Errorf("getrusage: %w", err)
	}
	maxRSS := int64(ru.Maxrss)
	if runtime.GOOS != "darwin" {
		// ru_maxrss is in bytes on macOS and in KiB on Linux.
		maxRSS *= 1024
	}
	return resourceUsage{
		user:        time.Duration(ru.Utime.Nano()),
		sys:         time.Duration(ru.Stime.Nano()),
		maxRSSBytes: maxRSS,
	}, nil
}
