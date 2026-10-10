// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"io"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"regexp/syntax"
	"runtime"
	"strconv"
	"time"

	"github.com/sourcegraph/zoekt"
	"github.com/sourcegraph/zoekt/index"
	"github.com/sourcegraph/zoekt/query"
)

// freshnessRecord is one line of `zoekt-bench freshness` output: the stages
// of one edit becoming searchable. The clock starts once the edit is in the
// worktree, where Ferret's Notify path also starts.
type freshnessRecord struct {
	Trial  int    `json:"trial"`
	Engine string `json:"engine"`
	Path   string `json:"path"`
	// CommitNS is `git commit` of the edited file, which Zoekt's delta
	// indexer requires.
	CommitNS int64 `json:"commit_ns"`
	// BuildNS is writing the delta shard and the tombstones.
	BuildNS int64 `json:"build_ns"`
	// ReloadNS is the open searcher noticing the new files, reloading, and
	// answering with only the new version.
	ReloadNS int64 `json:"reload_ns"`
	TotalNS  int64 `json:"total_ns"`
	// Shards is the shard count after the trial; each trial adds one.
	Shards int `json:"shards"`
}

// freshnessExperiment edits, commits, delta-indexes, and polls one index.
type freshnessExperiment struct {
	repo         string
	paths        []string
	opts         index.Options
	searcher     zoekt.Searcher
	rng          *rand.Rand
	pollInterval time.Duration
	timeout      time.Duration
}

func runFreshness(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("freshness", flag.ContinueOnError)
	repo := fs.String("repo", "", "disposable git clone of the corpus; every trial commits an edit to it")
	files := fs.String("files", "", "file list: one repo-relative path per line")
	indexDir := fs.String("index", "", "index directory to create; must be absent or empty")
	name := fs.String("name", "", "repository name stored in the shards (default: base name of --repo)")
	trials := fs.Int("trials", 20, "edits to time")
	seed := fs.Uint64("seed", 1, "seed for choosing edited files and markers")
	gomaxprocs := fs.Int("gomaxprocs", 4, "GOMAXPROCS for the searcher and the builder")
	parallelism := fs.Int("parallelism", 8, "shards built concurrently by the initial full build")
	pollInterval := fs.Duration("poll_interval", time.Millisecond, "interval between searchability probes")
	timeout := fs.Duration("trial_timeout", 5*time.Minute, "fail a trial that is not searchable after this long")
	if err := parseFlags(fs, args); err != nil {
		return err
	}
	if err := requireFlags(fs, "repo", "files", "index"); err != nil {
		return err
	}
	if *trials < 1 || *gomaxprocs < 1 || *parallelism < 1 || *pollInterval <= 0 || *timeout <= 0 {
		return usageError{fmt.Errorf("--trials, --gomaxprocs, --parallelism, --poll_interval, and --trial_timeout must be positive")}
	}
	paths, err := readFileList(*files)
	if err != nil {
		return err
	}
	runtime.GOMAXPROCS(*gomaxprocs)
	repoName := repoName(*name, *repo)
	report, err := buildIndex(*repo, paths, *indexDir, repoName, *parallelism)
	if err != nil {
		return err
	}
	if err := checkNoSkips(report); err != nil {
		return err
	}
	searcher, err := openSearcher(*indexDir)
	if err != nil {
		return err
	}
	defer searcher.Close()

	e := &freshnessExperiment{
		repo:         *repo,
		paths:        paths,
		opts:         builderOptions(repoName, *indexDir, *parallelism),
		searcher:     searcher,
		rng:          rand.New(rand.NewPCG(*seed, 0)),
		pollInterval: *pollInterval,
		timeout:      *timeout,
	}
	ctx := context.Background()
	for trial := range *trials {
		record, err := e.runTrial(ctx, trial)
		if err != nil {
			return fmt.Errorf("trial %d: %w", trial, err)
		}
		// Written per trial so that completed trials survive a later failure.
		if err := writeJSON(stdout, record); err != nil {
			return err
		}
	}
	return nil
}

// runTrial appends a unique marker line to a listed file, commits it,
// delta-indexes it, and waits until the open searcher serves only the new
// version.
func (e *freshnessExperiment) runTrial(ctx context.Context, trial int) (*freshnessRecord, error) {
	marker := fmt.Sprintf("ferret_freshness_%d_%016x", trial, e.rng.Uint64())
	path, edited, err := e.pickEditTarget(marker)
	if err != nil {
		return nil, err
	}
	if err := os.WriteFile(filepath.Join(e.repo, filepath.FromSlash(path)), edited, 0o644); err != nil {
		return nil, err
	}

	start := time.Now()
	if err := commitFile(e.repo, path, "zoekt-bench freshness trial "+strconv.Itoa(trial)); err != nil {
		return nil, err
	}
	committed := time.Now()
	if err := deltaBuild(e.opts, path, edited); err != nil {
		return nil, fmt.Errorf("delta build of %s: %w", path, err)
	}
	built := time.Now()
	if err := waitSearchable(ctx, e.searcher, path, marker, e.pollInterval, e.timeout); err != nil {
		return nil, err
	}
	searchable := time.Now()

	shards, _, err := indexSize(e.opts.IndexDir)
	if err != nil {
		return nil, err
	}
	return &freshnessRecord{
		Trial:    trial,
		Engine:   "zoekt",
		Path:     path,
		CommitNS: committed.Sub(start).Nanoseconds(),
		BuildNS:  built.Sub(committed).Nanoseconds(),
		ReloadNS: searchable.Sub(built).Nanoseconds(),
		TotalNS:  searchable.Sub(start).Nanoseconds(),
		Shards:   shards,
	}, nil
}

// pickEditTarget chooses a listed file and returns its path and its content
// with the marker line appended. Appending can push a file over Zoekt's size
// or trigram limit, and a skipped file would never become searchable, so
// candidates are drawn until the edited content passes Zoekt's own checks.
func (e *freshnessExperiment) pickEditTarget(marker string) (string, []byte, error) {
	const attempts = 100
	var checker index.DocChecker
	for range attempts {
		p := e.paths[e.rng.IntN(len(e.paths))]
		content, err := readRegularFile(e.repo, p)
		if err != nil {
			return "", nil, err
		}
		edited := appendLine(content, marker)
		if len(edited) <= zoektSizeMax && checker.Check(edited, zoektTrigramMax, false) == index.SkipReasonNone {
			return p, edited, nil
		}
	}
	return "", nil, fmt.Errorf("no listed file stays indexable after an edit (%d attempts)", attempts)
}

// appendLine returns content followed by line and a newline, first
// terminating content's last line if it is unterminated.
func appendLine(content []byte, line string) []byte {
	out := make([]byte, 0, len(content)+len(line)+2)
	out = append(out, content...)
	if len(out) > 0 && out[len(out)-1] != '\n' {
		out = append(out, '\n')
	}
	out = append(out, line...)
	return append(out, '\n')
}

// commitFile commits the worktree version of path, and only path, in repo.
// Hooks, signing, and automatic gc are disabled so that the measured time is
// the commit itself; the identity is fixed so that no user config is needed.
func commitFile(repo, path, message string) error {
	cmd := exec.Command("git", "-C", repo,
		"-c", "core.hooksPath=/dev/null",
		"-c", "commit.gpgSign=false",
		"-c", "gc.auto=0",
		"-c", "user.name=zoekt-bench",
		"-c", "user.email=zoekt-bench@localhost",
		"commit", "--quiet", "--no-verify", "--message", message, "--", path)
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("git commit %s: %w: %s", path, err, bytes.TrimSpace(out))
	}
	return nil
}

// deltaBuild writes one delta shard holding content as the new version of
// path and tombstones path in every older shard: the index update that
// `zoekt-git-index -delta` performs for a one-file commit. Calling the
// builder directly skips zoekt-git-index's repository open and tree diff,
// so the measurement is a lower bound on Zoekt's cost.
func deltaBuild(opts index.Options, path string, content []byte) error {
	opts.IsDelta = true
	b, err := index.NewBuilder(opts)
	if err != nil {
		return err
	}
	b.MarkFileAsChangedOrRemoved(path)
	if err := b.AddFile(path, content); err != nil {
		_ = b.Finish() // Removes temporary files; the Add error is the one to report.
		return err
	}
	return b.Finish()
}

// waitSearchable polls s until the edit to path is visible and consistent:
// the marker is found in path, and path names exactly one live document,
// which means the delta shard is loaded and the old version's tombstone has
// taken effect.
func waitSearchable(ctx context.Context, s zoekt.Searcher, path, marker string, interval, timeout time.Duration) error {
	re, err := syntax.Parse("^"+regexp.QuoteMeta(path)+"$", zoektRegexpFlags)
	if err != nil {
		return err
	}
	byName := &query.Regexp{Regexp: re, FileName: true, CaseSensitive: true}
	withMarker := query.NewAnd(byName, &query.Substring{Pattern: marker, CaseSensitive: true, Content: true})
	deadline := time.Now().Add(timeout)
	for {
		found, err := countFiles(ctx, s, withMarker, path)
		if err != nil {
			return err
		}
		live, err := countFiles(ctx, s, byName, path)
		if err != nil {
			return err
		}
		if found == 1 && live == 1 {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("%s: not searchable after %v (%d copies with the marker, %d live copies)",
				path, timeout, found, live)
		}
		time.Sleep(interval)
	}
}

// countFiles returns how many documents named path match q.
func countFiles(ctx context.Context, s zoekt.Searcher, q query.Q, path string) (int, error) {
	res, err := s.Search(ctx, q, &zoekt.SearchOptions{})
	if err != nil {
		return 0, err
	}
	n := 0
	for _, f := range res.Files {
		if f.FileName == path {
			n++
		}
	}
	return n, nil
}
