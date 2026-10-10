// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"regexp/syntax"
	"strings"

	"github.com/sourcegraph/zoekt/query"
)

// zoektRegexpFlags are the regexp/syntax flags Zoekt's own query parser uses
// for regex atoms (query/parse.go at the pinned version), so a regex means
// here what it means to Zoekt users.
const zoektRegexpFlags = syntax.ClassNL | syntax.PerlX | syntax.UnicodeGroups

// querySpec is one line of a query set (bench/queries/*.jsonl). Only the
// fields the harness needs are decoded; the full schema is in
// bench/queries/gen_mix.py.
type querySpec struct {
	ID      string `json:"id"`
	Class   string `json:"class"`
	Kind    string `json:"kind"`
	Pattern string `json:"pattern"`
	// CaseSensitive is a pointer so that a missing field is an error rather
	// than a silent case-insensitive default.
	CaseSensitive *bool `json:"case_sensitive"`
	// Files, when present (even if empty), restricts the query to exactly
	// these repo-relative paths. The replay (eval/harness/replay.py) fills it
	// from `ferret_bench files --path_glob`, so Zoekt is scoped by Ferret's
	// own glob matcher and the matching stays outside Zoekt's timing.
	Files []string `json:"files"`
	// FilesFrom is the path of a file with one repo-relative path per line,
	// used as Files would be. It is for scopes too large to repeat in every
	// query: each distinct path is read once per process and cached. It is
	// exclusive with Files.
	FilesFrom string `json:"files_from"`
}

// readQueries reads and validates a query set: one JSON object per line,
// with unique non-empty ids.
func readQueries(path string) ([]querySpec, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var specs []querySpec
	seen := make(map[string]int)
	scanner := bufio.NewScanner(f)
	scanner.Buffer(make([]byte, 64<<10), 16<<20)
	for lineNo := 1; scanner.Scan(); lineNo++ {
		var spec querySpec
		if err := json.Unmarshal(scanner.Bytes(), &spec); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
		if err := spec.validate(); err != nil {
			return nil, fmt.Errorf("%s:%d: %w", path, lineNo, err)
		}
		if first, ok := seen[spec.ID]; ok {
			return nil, fmt.Errorf("%s:%d: id %q duplicates line %d", path, lineNo, spec.ID, first)
		}
		seen[spec.ID] = lineNo
		specs = append(specs, spec)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	if len(specs) == 0 {
		return nil, fmt.Errorf("%s: no queries", path)
	}
	return specs, nil
}

func (s querySpec) validate() error {
	switch {
	case s.ID == "":
		return fmt.Errorf("missing id")
	case s.Class == "":
		return fmt.Errorf("query %s: missing class", s.ID)
	case s.Kind != "literal" && s.Kind != "regex":
		return fmt.Errorf("query %s: kind %q is neither \"literal\" nor \"regex\"", s.ID, s.Kind)
	case s.Pattern == "":
		return fmt.Errorf("query %s: empty pattern", s.ID)
	case s.CaseSensitive == nil:
		return fmt.Errorf("query %s: missing case_sensitive", s.ID)
	case s.Files != nil && s.FilesFrom != "":
		return fmt.Errorf("query %s: files and files_from are exclusive", s.ID)
	}
	return nil
}

// zoektQuery builds the Zoekt query for spec, which must be valid. Literals
// become a content Substring and regexes a content Regexp, parsed and
// simplified exactly as Zoekt's parser would; case sensitivity is always
// explicit, never Zoekt's case:auto. A spec with a Files list is the And of
// that content query and a FileNameSet, which is built here, once, and not
// while the query is timed.
func zoektQuery(spec querySpec) (query.Q, error) {
	return zoektQueryWith(spec, newFileSetCache())
}

// zoektQueryWith is zoektQuery with the file sets of `files_from` read
// through sets, so queries sharing a scope share one read and one set.
func zoektQueryWith(spec querySpec, sets *fileSetCache) (query.Q, error) {
	content, err := contentQuery(spec)
	if err != nil {
		return content, err
	}
	switch {
	case spec.FilesFrom != "":
		set, err := sets.load(spec.FilesFrom)
		if err != nil {
			return nil, fmt.Errorf("query %s: %w", spec.ID, err)
		}
		return query.NewAnd(content, &query.FileNameSet{Set: set}), nil
	case spec.Files != nil:
		set := make(map[string]struct{}, len(spec.Files))
		for _, path := range spec.Files {
			set[path] = struct{}{}
		}
		return query.NewAnd(content, &query.FileNameSet{Set: set}), nil
	}
	return content, nil
}

// fileSetCache reads each file list (one path per line) once. An empty list
// is an empty set, which matches nothing. The sets are shared between
// queries and never modified.
type fileSetCache struct {
	sets map[string]map[string]struct{}
}

func newFileSetCache() *fileSetCache {
	return &fileSetCache{sets: make(map[string]map[string]struct{})}
}

func (c *fileSetCache) load(path string) (map[string]struct{}, error) {
	if set, ok := c.sets[path]; ok {
		return set, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("files_from: %w", err)
	}
	set := make(map[string]struct{})
	for _, line := range strings.Split(string(data), "\n") {
		if line != "" {
			set[line] = struct{}{}
		}
	}
	c.sets[path] = set
	return set, nil
}

// contentQuery builds the content match of spec, without any file scope.
func contentQuery(spec querySpec) (query.Q, error) {
	switch spec.Kind {
	case "literal":
		return &query.Substring{Pattern: spec.Pattern, CaseSensitive: *spec.CaseSensitive, Content: true}, nil
	case "regex":
		re, err := syntax.Parse(spec.Pattern, zoektRegexpFlags)
		if err != nil {
			return nil, fmt.Errorf("query %s: %w", spec.ID, err)
		}
		return &query.Regexp{
			Regexp:        query.OptimizeRegexp(re, zoektRegexpFlags),
			CaseSensitive: *spec.CaseSensitive,
			Content:       true,
		}, nil
	default:
		return nil, fmt.Errorf("query %s: unknown kind %q", spec.ID, spec.Kind)
	}
}
