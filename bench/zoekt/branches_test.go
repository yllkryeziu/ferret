// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"testing"

	"github.com/sourcegraph/zoekt"
	"github.com/sourcegraph/zoekt/query"
)

var branchCorpus = map[string]string{
	"src/a.c":  "int shared_one(void)\n{\n\treturn bar(1);\n}\n",
	"src/b.c":  "int shared_two(void)\n{\n\treturn bar(2);\n}\n",
	"src/c.py": "def shared_three(x):\n    return x + 3\n",
}

func TestBranchesShareIdenticalBlobsAcrossBranches(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git is not installed")
	}
	repo := t.TempDir()
	list := writeTree(t, repo, branchCorpus)
	// A tracked file that the list does not name is not indexed.
	if err := os.WriteFile(filepath.Join(repo, "unlisted.txt"), []byte("unlisted content\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git(t, repo, "init", "--quiet", "--initial-branch=b1")
	git(t, repo, "add", ".")
	git(t, repo, "commit", "--quiet", "--message", "base")
	// b2 edits src/a.c; b3 edits src/a.c again and src/b.c.
	edit := func(branch, path, content string) {
		git(t, repo, "checkout", "--quiet", "-B", branch, "b1")
		if branch == "b3" {
			git(t, repo, "checkout", "--quiet", "b2", "--", ".")
			git(t, repo, "checkout", "--quiet", "-B", "b3")
		}
		if err := os.WriteFile(filepath.Join(repo, path), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
		git(t, repo, "commit", "--quiet", "-am", branch)
	}
	edit("b2", "src/a.c", "int shared_one(void)\n{\n\treturn bar(10);\n}\n")
	edit("b3", "src/b.c", "int shared_two(void)\n{\n\treturn bar(20);\n}\n")
	git(t, repo, "checkout", "--quiet", "b1")

	out := filepath.Join(t.TempDir(), "idx")
	records := decodeLines[branchesRecord](t, runOK(t, "branches", "--repo="+repo, "--files="+list,
		"--branches=b1,b2,b3", "--out="+out, "--keep"))
	if len(records) != 3 {
		t.Fatalf("got %d records, want 3", len(records))
	}
	// Documents are distinct (path, blob) pairs: 3, then +1 (a.c), then +1 (b.c).
	wantDocs := []int{3, 4, 5}
	wantPairs := []int{3, 6, 9}
	for i, r := range records {
		if r.K != i+1 || r.Docs != wantDocs[i] || r.DocBranchPairs != wantPairs[i] {
			t.Errorf("record %d = k %d docs %d pairs %d, want %d, %d, %d",
				i, r.K, r.Docs, r.DocBranchPairs, i+1, wantDocs[i], wantPairs[i])
		}
		if r.BytesIfUnshared < r.BytesIndexed || r.IndexBytes <= 0 || r.WallSeconds <= 0 {
			t.Errorf("record %d = %+v", i, r)
		}
	}
	if records[2].BytesIfUnshared <= records[2].BytesIndexed {
		t.Errorf("k=3 stored every file once per branch: %+v", records[2])
	}

	// In the k=3 index an unchanged file names all three branches, and an
	// edited one only the branches that hold that version.
	searcher, err := openSearcher(filepath.Join(out, "k3"))
	if err != nil {
		t.Fatal(err)
	}
	defer searcher.Close()
	branchesOf := func(pattern string) map[string][]string {
		res, err := searcher.Search(context.Background(),
			&query.Substring{Pattern: pattern, CaseSensitive: true, Content: true}, &zoekt.SearchOptions{})
		if err != nil {
			t.Fatal(err)
		}
		got := map[string][]string{}
		for _, f := range res.Files {
			b := slices.Clone(f.Branches)
			slices.Sort(b)
			got[f.FileName] = b
		}
		return got
	}
	if got, want := branchesOf("shared_three"), map[string][]string{"src/c.py": {"b1", "b2", "b3"}}; !reflect.DeepEqual(got, want) {
		t.Errorf("unchanged file: %v, want %v", got, want)
	}
	if got, want := branchesOf("bar(10)"), map[string][]string{"src/a.c": {"b2", "b3"}}; !reflect.DeepEqual(got, want) {
		t.Errorf("edited once: %v, want %v", got, want)
	}
	if got, want := branchesOf("bar(2)"), map[string][]string{"src/b.c": {"b1", "b2"}}; !reflect.DeepEqual(got, want) {
		t.Errorf("edited in the last branch only: %v, want %v", got, want)
	}
}

func TestBranchesRejectsBadK(t *testing.T) {
	if _, err := parseKs("0", 3); err == nil {
		t.Error("k=0 accepted")
	}
	if _, err := parseKs("4", 3); err == nil {
		t.Error("k beyond the branch count accepted")
	}
	if got, _ := parseKs("", 3); !reflect.DeepEqual(got, []int{1, 2, 3}) {
		t.Errorf("default ks = %v", got)
	}
}
