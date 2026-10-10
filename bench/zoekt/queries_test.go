// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"path/filepath"
	"regexp/syntax"
	"strings"
	"testing"

	"github.com/sourcegraph/zoekt/query"
)

func writeQueries(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "queries.jsonl")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReadQueries(t *testing.T) {
	path := writeQueries(t, `{"id":"q1","class":"identifier","kind":"literal","pattern":"foo","case_sensitive":true,"tercile":1}
{"id":"q2","class":"regex","kind":"regex","pattern":"foo\\w+","case_sensitive":false}
`)
	specs, err := readQueries(path)
	if err != nil {
		t.Fatalf("readQueries() error = %v", err)
	}
	if len(specs) != 2 {
		t.Fatalf("readQueries() returned %d queries, want 2", len(specs))
	}
	if s := specs[1]; s.ID != "q2" || s.Class != "regex" || s.Kind != "regex" ||
		s.Pattern != `foo\w+` || *s.CaseSensitive {
		t.Errorf("specs[1] = %+v", s)
	}
}

func TestReadQueriesRejectsInvalidSets(t *testing.T) {
	const valid = `{"id":"q1","class":"c","kind":"literal","pattern":"p","case_sensitive":true}`
	tests := []struct {
		name    string
		content string
		wantErr string
	}{
		{name: "empty", content: "", wantErr: "no queries"},
		{name: "not json", content: "{\n", wantErr: ":1:"},
		{name: "missing id", content: `{"class":"c","kind":"literal","pattern":"p","case_sensitive":true}`, wantErr: "missing id"},
		{name: "missing class", content: `{"id":"q1","kind":"literal","pattern":"p","case_sensitive":true}`, wantErr: "missing class"},
		{name: "unknown kind", content: `{"id":"q1","class":"c","kind":"glob","pattern":"p","case_sensitive":true}`, wantErr: `kind "glob"`},
		{name: "empty pattern", content: `{"id":"q1","class":"c","kind":"literal","pattern":"","case_sensitive":true}`, wantErr: "empty pattern"},
		{name: "missing case_sensitive", content: `{"id":"q1","class":"c","kind":"literal","pattern":"p"}`, wantErr: "missing case_sensitive"},
		{name: "duplicate id", content: valid + "\n" + valid + "\n", wantErr: `:2: id "q1" duplicates line 1`},
		{name: "files and files_from", content: `{"id":"q1","class":"c","kind":"literal","pattern":"p","case_sensitive":true,"files":[],"files_from":"x.txt"}`, wantErr: "exclusive"},
		{name: "blank line", content: valid + "\n\n", wantErr: ":2:"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, err := readQueries(writeQueries(t, tc.content))
			if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("readQueries() error = %v, want it to contain %q", err, tc.wantErr)
			}
		})
	}
}

func spec(kind, pattern string, caseSensitive bool) querySpec {
	return querySpec{ID: "q", Class: "c", Kind: kind, Pattern: pattern, CaseSensitive: &caseSensitive}
}

func TestZoektQueryLiteral(t *testing.T) {
	for _, caseSensitive := range []bool{true, false} {
		q, err := zoektQuery(spec("literal", "a (b", caseSensitive))
		if err != nil {
			t.Fatalf("zoektQuery() error = %v", err)
		}
		s, ok := q.(*query.Substring)
		if !ok {
			t.Fatalf("zoektQuery() = %T, want *query.Substring", q)
		}
		// Literals are never split on whitespace or parsed as regexes, and
		// case sensitivity is explicit rather than case:auto.
		want := query.Substring{Pattern: "a (b", CaseSensitive: caseSensitive, Content: true}
		if *s != want {
			t.Errorf("zoektQuery() = %+v, want %+v", *s, want)
		}
	}
}

func TestZoektQueryWithFilesIsScopedToTheirSet(t *testing.T) {
	s := spec("literal", "foo", true)
	s.Files = []string{"src/a.c", "README"}
	q, err := zoektQuery(s)
	if err != nil {
		t.Fatalf("zoektQuery() error = %v", err)
	}
	and, ok := q.(*query.And)
	if !ok || len(and.Children) != 2 {
		t.Fatalf("zoektQuery() = %#v, want an And of two queries", q)
	}
	if _, ok := and.Children[0].(*query.Substring); !ok {
		t.Errorf("first child = %T, want *query.Substring", and.Children[0])
	}
	set, ok := and.Children[1].(*query.FileNameSet)
	if !ok || len(set.Set) != 2 {
		t.Fatalf("second child = %#v, want a FileNameSet of 2 paths", and.Children[1])
	}
	for _, path := range s.Files {
		if _, ok := set.Set[path]; !ok {
			t.Errorf("FileNameSet lacks %q", path)
		}
	}
}

func TestZoektQueryWithEmptyFilesMatchesNothing(t *testing.T) {
	// A scoped call whose glob admits no file must not become unscoped.
	s := spec("literal", "foo", true)
	s.Files = []string{}
	q, err := zoektQuery(s)
	if err != nil {
		t.Fatalf("zoektQuery() error = %v", err)
	}
	if and, ok := q.(*query.And); !ok || len(and.Children) != 2 {
		t.Errorf("zoektQuery() = %#v, want an And with an empty FileNameSet", q)
	}
}

func TestZoektQueryRegexp(t *testing.T) {
	q, err := zoektQuery(spec("regex", `(foo|bar)[ \t]*\(`, false))
	if err != nil {
		t.Fatalf("zoektQuery() error = %v", err)
	}
	re, ok := q.(*query.Regexp)
	if !ok {
		t.Fatalf("zoektQuery() = %T, want *query.Regexp", q)
	}
	if !re.Content || re.FileName || re.CaseSensitive {
		t.Errorf("zoektQuery() = %+v, want a case-insensitive content query", re)
	}
	// Zoekt's parser turns capture groups into non-capturing ones.
	if got, want := re.RegexpString(), `(?:foo|bar)[\t ]*\(`; got != want {
		t.Errorf("RegexpString() = %q, want %q", got, want)
	}
}

func TestZoektRegexpFlagsKeepLineSemantics(t *testing.T) {
	// Without OneLine, ^ and $ are line anchors, as in Ferret's (?m) mode;
	// without DotNL, '.' never crosses a line.
	re, err := syntax.Parse(`^a.b$`, zoektRegexpFlags)
	if err != nil {
		t.Fatal(err)
	}
	want := []syntax.Op{syntax.OpBeginLine, syntax.OpLiteral, syntax.OpAnyCharNotNL, syntax.OpLiteral, syntax.OpEndLine}
	if re.Op != syntax.OpConcat || len(re.Sub) != len(want) {
		t.Fatalf("parse tree = %v, want a concatenation of %v", re, want)
	}
	for i, sub := range re.Sub {
		if sub.Op != want[i] {
			t.Errorf("Sub[%d].Op = %v, want %v", i, sub.Op, want[i])
		}
	}
}

func TestZoektQueryRejectsInvalidRegexp(t *testing.T) {
	s := spec("regex", "(", true)
	s.ID = "broken"
	if _, err := zoektQuery(s); err == nil || !strings.Contains(err.Error(), "query broken") {
		t.Errorf("zoektQuery() error = %v, want an error naming the query", err)
	}
}

func TestTopKMatchLimits(t *testing.T) {
	tests := []struct {
		numDocs, k           int
		wantShard, wantTotal int
	}{
		{numDocs: 200, k: 20, wantShard: 2200, wantTotal: 2200},
		{numDocs: 10000, k: 20, wantShard: 12000, wantTotal: 12000},
		{numDocs: 95000, k: 20, wantShard: 101, wantTotal: 0},
		{numDocs: 463000, k: 20, wantShard: 100, wantTotal: 0},
	}
	for _, tc := range tests {
		shard, total := topKMatchLimits(tc.numDocs, tc.k)
		if shard != tc.wantShard || total != tc.wantTotal {
			t.Errorf("topKMatchLimits(%d, %d) = %d, %d; want %d, %d",
				tc.numDocs, tc.k, shard, total, tc.wantShard, tc.wantTotal)
		}
	}
}

func TestSearchConfigOptions(t *testing.T) {
	exhaustive := searchConfig{mode: modeExhaustive, contextLines: 0, maxMatches: 500}.options(0)
	if exhaustive.MaxDocDisplayCount != 0 || exhaustive.ShardMaxMatchCount != 500 ||
		exhaustive.TotalMaxMatchCount != 500 || exhaustive.NumContextLines != 0 {
		t.Errorf("exhaustive options = %+v", exhaustive)
	}
	topK := searchConfig{mode: modeTopK, k: 20, contextLines: 2}.options(95000)
	if topK.MaxDocDisplayCount != 20 || topK.NumContextLines != 2 ||
		topK.ShardMaxMatchCount != 101 || topK.TotalMaxMatchCount != 0 {
		t.Errorf("top-K options = %+v", topK)
	}
	if topK.ChunkMatches || topK.Whole {
		t.Errorf("top-K options = %+v, want line matches without whole files", topK)
	}
}
