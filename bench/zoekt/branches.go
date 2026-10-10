// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/sourcegraph/zoekt"
	"github.com/sourcegraph/zoekt/index"
)

// branchesRecord is one line of `zoekt-bench branches` output: the cost of
// indexing the first K branches of a repository into one fresh index.
type branchesRecord struct {
	Engine        string   `json:"engine"`
	EngineVersion string   `json:"engine_version"`
	K             int      `json:"k"`
	Branches      []string `json:"branches"`
	// Docs is the documents Zoekt stored: one per distinct (path, blob), however
	// many of the K branches contain it. DocBranchPairs is the (path, branch)
	// pairs, which is what K full indexes would store.
	Docs           int   `json:"docs"`
	DocBranchPairs int   `json:"doc_branch_pairs"`
	BytesIndexed   int64 `json:"bytes_indexed"`
	// BytesIfUnshared is the bytes of the (path, branch) pairs.
	BytesIfUnshared int64   `json:"bytes_if_unshared"`
	Shards          int     `json:"shards"`
	IndexBytes      int64   `json:"index_bytes"`
	WallSeconds     float64 `json:"wall_seconds"`
	UserSeconds     float64 `json:"user_seconds"`
	SysSeconds      float64 `json:"sys_seconds"`
	CPUSeconds      float64 `json:"cpu_seconds"`
	PeakRSSBytes    int64   `json:"peak_rss_bytes"`
}

// branchFile is one (path, blob) of a branch.
type branchFile struct {
	path string
	oid  string
}

// docKey is what Zoekt's git indexer (gitindex.IndexGitRepo) deduplicates
// by: a path with a blob. Two branches that hold the same blob at the same
// path share one document, whose Branches lists both.
type docKey struct {
	path string
	oid  string
}

// runBranches builds, for each k in --ks, a fresh index of the first k
// branches with one Builder, the way `zoekt-git-index -branches b1,...,bk`
// does. gitindex itself needs go-git, which the harness does not vendor, so
// this reads the same inputs with git and applies the same sharing rule.
func runBranches(args []string, stdout io.Writer) error {
	fs := flag.NewFlagSet("branches", flag.ContinueOnError)
	repo := fs.String("repo", "", "git repository holding the branches")
	files := fs.String("files", "", "file list: one repo-relative path per line; other paths are not indexed")
	branches := fs.String("branches", "", "comma-separated branch names or commits, in order")
	out := fs.String("out", "", "directory that receives one index directory per k; must be absent or empty")
	ks := fs.String("ks", "", "comma-separated k values (default: 1..number of branches)")
	name := fs.String("name", "", "repository name stored in the shards (default: base name of --repo)")
	parallelism := fs.Int("parallelism", 8, "shards built concurrently")
	keep := fs.Bool("keep", false, "keep the index directories (default: remove each after it is measured)")
	if err := parseFlags(fs, args); err != nil {
		return err
	}
	if err := requireFlags(fs, "repo", "files", "branches", "out"); err != nil {
		return err
	}
	paths, err := readFileList(*files)
	if err != nil {
		return err
	}
	allowed := make(map[string]bool, len(paths))
	for _, p := range paths {
		allowed[p] = true
	}
	names := strings.Split(*branches, ",")
	kList, err := parseKs(*ks, len(names))
	if err != nil {
		return usageError{err}
	}
	if *parallelism < 1 {
		return usageError{fmt.Errorf("--parallelism must be positive")}
	}
	if err := createEmptyDir(*out); err != nil {
		return err
	}
	versions := make([]string, len(names))
	trees := make([][]branchFile, len(names))
	for i, b := range names {
		if versions[i], err = gitOutput(*repo, "rev-parse", "--verify", b+"^{commit}"); err != nil {
			return err
		}
	}
	repoNameValue := repoName(*name, *repo)
	for _, k := range kList {
		for i := 0; i < k; i++ {
			if trees[i] == nil {
				if trees[i], err = listBranchFiles(*repo, versions[i], allowed); err != nil {
					return err
				}
			}
		}
		record, err := buildBranches(*repo, filepath.Join(*out, "k"+strconv.Itoa(k)), repoNameValue,
			names[:k], versions[:k], trees[:k], *parallelism)
		if err != nil {
			return fmt.Errorf("k=%d: %w", k, err)
		}
		record.K = k
		if err := writeJSON(stdout, record); err != nil {
			return err
		}
		if !*keep {
			if err := os.RemoveAll(filepath.Join(*out, "k"+strconv.Itoa(k))); err != nil {
				return err
			}
		}
	}
	return nil
}

func parseKs(ks string, n int) ([]int, error) {
	if ks == "" {
		out := make([]int, n)
		for i := range out {
			out[i] = i + 1
		}
		return out, nil
	}
	var out []int
	for _, s := range strings.Split(ks, ",") {
		k, err := strconv.Atoi(s)
		if err != nil || k < 1 || k > n {
			return nil, fmt.Errorf("--ks entry %q is not in 1..%d", s, n)
		}
		out = append(out, k)
	}
	return out, nil
}

func gitOutput(repo string, args ...string) (string, error) {
	cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("git %s: %w: %s", strings.Join(args, " "), err, bytes.TrimSpace(stderr.Bytes()))
	}
	return strings.TrimSpace(string(out)), nil
}

// listBranchFiles returns the regular files of commit that are in allowed.
func listBranchFiles(repo, commit string, allowed map[string]bool) ([]branchFile, error) {
	cmd := exec.Command("git", "-C", repo, "ls-tree", "-r", "-z", "--full-tree", commit)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git ls-tree %s: %w: %s", commit, err, bytes.TrimSpace(stderr.Bytes()))
	}
	var files []branchFile
	for _, entry := range bytes.Split(out, []byte{0}) {
		if len(entry) == 0 {
			continue
		}
		// "<mode> blob <oid>\t<path>"
		meta, path, ok := bytes.Cut(entry, []byte{'\t'})
		fields := strings.Fields(string(meta))
		if !ok || len(fields) != 3 || fields[1] != "blob" || (fields[0] != "100644" && fields[0] != "100755") {
			continue
		}
		if allowed[string(path)] {
			files = append(files, branchFile{path: string(path), oid: fields[2]})
		}
	}
	return files, nil
}

// catFile reads blobs through one `git cat-file --batch`.
type catFile struct {
	cmd *exec.Cmd
	in  io.WriteCloser
	out *bufio.Reader
}

func newCatFile(repo string) (*catFile, error) {
	cmd := exec.Command("git", "-C", repo, "cat-file", "--batch")
	in, err := cmd.StdinPipe()
	if err != nil {
		return nil, err
	}
	out, err := cmd.StdoutPipe()
	if err != nil {
		return nil, err
	}
	if err := cmd.Start(); err != nil {
		return nil, err
	}
	return &catFile{cmd: cmd, in: in, out: bufio.NewReaderSize(out, 1<<20)}, nil
}

func (c *catFile) read(oid string) ([]byte, error) {
	if _, err := io.WriteString(c.in, oid+"\n"); err != nil {
		return nil, err
	}
	header, err := c.out.ReadString('\n')
	if err != nil {
		return nil, err
	}
	fields := strings.Fields(header)
	if len(fields) != 3 || fields[1] != "blob" {
		return nil, fmt.Errorf("cat-file %s: %q", oid, strings.TrimSpace(header))
	}
	size, err := strconv.Atoi(fields[2])
	if err != nil {
		return nil, err
	}
	data := make([]byte, size+1)
	if _, err := io.ReadFull(c.out, data); err != nil {
		return nil, err
	}
	return data[:size], nil
}

func (c *catFile) close() {
	_ = c.in.Close()
	_ = c.cmd.Wait()
}

func childUsage() (user, sys time.Duration, err error) {
	var ru syscall.Rusage
	if err := syscall.Getrusage(syscall.RUSAGE_CHILDREN, &ru); err != nil {
		return 0, 0, err
	}
	return time.Duration(ru.Utime.Nano()), time.Duration(ru.Stime.Nano()), nil
}

// buildBranches indexes trees (one per branch, in order) into the new
// directory indexDir. Time and CPU cover the git reads, the indexing, and the
// shard writes, like buildIndex.
func buildBranches(repo, indexDir, name string, names, versions []string, trees [][]branchFile, parallelism int) (*branchesRecord, error) {
	if err := createEmptyDir(indexDir); err != nil {
		return nil, err
	}
	// Documents in first-seen order, with the branches that hold each.
	var order []docKey
	held := make(map[docKey][]string)
	record := &branchesRecord{Engine: "zoekt", EngineVersion: zoektVersion(), Branches: names}
	for i, tree := range trees {
		for _, f := range tree {
			key := docKey{f.path, f.oid}
			if _, seen := held[key]; !seen {
				order = append(order, key)
			}
			held[key] = append(held[key], names[i])
			record.DocBranchPairs++
		}
	}
	opts := builderOptions(name, indexDir, parallelism)
	for i, b := range names {
		opts.RepositoryDescription.Branches = append(opts.RepositoryDescription.Branches,
			zoekt.RepositoryBranch{Name: b, Version: versions[i]})
	}

	before, err := currentUsage()
	if err != nil {
		return nil, err
	}
	childUserBefore, childSysBefore, err := childUsage()
	if err != nil {
		return nil, err
	}
	start := time.Now()
	cat, err := newCatFile(repo)
	if err != nil {
		return nil, err
	}
	defer cat.close()
	builder, err := index.NewBuilder(opts)
	if err != nil {
		return nil, err
	}
	defer func() { _ = builder.Finish() }()
	for _, key := range order {
		content, err := cat.read(key.oid)
		if err != nil {
			return nil, err
		}
		n := int64(len(content))
		record.BytesIndexed += n
		record.BytesIfUnshared += n * int64(len(held[key]))
		record.Docs++
		if err := builder.Add(index.Document{Name: key.path, Content: content, Branches: held[key]}); err != nil {
			return nil, fmt.Errorf("adding %s: %w", key.path, err)
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
	childUserAfter, childSysAfter, err := childUsage()
	if err != nil {
		return nil, err
	}
	record.Shards, record.IndexBytes, err = indexSize(indexDir)
	if err != nil {
		return nil, err
	}
	record.WallSeconds = wall.Seconds()
	record.UserSeconds = (after.user - before.user + childUserAfter - childUserBefore).Seconds()
	record.SysSeconds = (after.sys - before.sys + childSysAfter - childSysBefore).Seconds()
	record.CPUSeconds = record.UserSeconds + record.SysSeconds
	record.PeakRSSBytes = after.maxRSSBytes
	return record, nil
}
