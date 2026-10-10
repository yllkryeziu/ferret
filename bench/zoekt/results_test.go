// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/json"
	"slices"
	"testing"

	"github.com/sourcegraph/zoekt"
)

func TestMatchedLines(t *testing.T) {
	files := []zoekt.FileMatch{
		{FileName: "a/b.c", LineMatches: []zoekt.LineMatch{
			{LineNumber: 12}, {LineNumber: 3}, {LineNumber: 12},
		}},
		{FileName: "a.c", LineMatches: []zoekt.LineMatch{
			{LineNumber: 1},
			// A filename match is not a content line.
			{LineNumber: 0, FileName: true},
		}},
		{FileName: "empty.c"},
	}
	got := matchedLines(files)
	// Paths sort by bytes ('.' < '/'); lines sort numerically (3 < 12).
	want := []lineRef{{"a.c", 1}, {"a/b.c", 3}, {"a/b.c", 12}}
	if !slices.Equal(got, want) {
		t.Errorf("matchedLines() = %v, want %v", got, want)
	}
}

func TestLineSetDigest(t *testing.T) {
	// The same vector is pinned in bench/ferret/result_set_test.cc, so that
	// the Go and C++ harnesses agree on the encoding.
	lines := []lineRef{{"a.c", 1}, {"a/b.c", 3}, {"a/b.c", 12}}
	if got, want := lineSetDigest(lines), "240b57ffcb3e75c91fb996a0f7e1c3ecb3609b68aabbdb12ff6a292d0e4b4adf"; got != want {
		t.Errorf("lineSetDigest() = %s, want %s", got, want)
	}
	if got, want := lineSetDigest(nil), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"; got != want {
		t.Errorf("lineSetDigest(nil) = %s, want the SHA-256 of nothing %s", got, want)
	}
}

func TestLineRefMarshalJSON(t *testing.T) {
	got, err := json.Marshal([]lineRef{{"dir/f\"x.c", 7}})
	if err != nil {
		t.Fatal(err)
	}
	if want := `[["dir/f\"x.c",7]]`; string(got) != want {
		t.Errorf("json.Marshal() = %s, want %s", got, want)
	}
}
