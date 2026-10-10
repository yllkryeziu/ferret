// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"slices"
	"strconv"

	"github.com/sourcegraph/zoekt"
)

// lineRef names one matching line: a repo-relative path and a 1-based line
// number.
type lineRef struct {
	Path string
	Line int
}

// MarshalJSON encodes r as ["path", line], which keeps full result sets
// compact.
func (r lineRef) MarshalJSON() ([]byte, error) {
	return json.Marshal([]any{r.Path, r.Line})
}

// compareLineRefs orders by path bytes, then by line number.
func compareLineRefs(a, b lineRef) int {
	if c := cmp.Compare(a.Path, b.Path); c != 0 {
		return c
	}
	return cmp.Compare(a.Line, b.Line)
}

// matchedLines returns the distinct (path, line) pairs of the content
// matches in files, sorted by compareLineRefs. Zoekt reports a match that
// spans lines once, at its first line; the query set contains no pattern that
// can match a newline, so every match is one line. Filename matches are
// ignored because benchmark queries search content only.
func matchedLines(files []zoekt.FileMatch) []lineRef {
	var lines []lineRef
	for _, f := range files {
		for _, lm := range f.LineMatches {
			if !lm.FileName {
				lines = append(lines, lineRef{Path: f.FileName, Line: lm.LineNumber})
			}
		}
	}
	slices.SortFunc(lines, compareLineRefs)
	return slices.Compact(lines)
}

// lineSetDigest returns the canonical digest of a sorted, distinct line set:
// the lowercase hex SHA-256 of, for each pair in order, the path bytes, a NUL
// byte, the decimal line number, and a newline. Ferret's harnesses
// (bench/ferret/result_set.cc) compute the same digest, which is how
// bench/report.py compares result sets too large to exchange in full.
func lineSetDigest(lines []lineRef) string {
	h := sha256.New()
	var buf []byte
	for _, l := range lines {
		buf = append(buf[:0], l.Path...)
		buf = append(buf, 0)
		buf = strconv.AppendInt(buf, int64(l.Line), 10)
		buf = append(buf, '\n')
		h.Write(buf)
	}
	return hex.EncodeToString(h.Sum(nil))
}
