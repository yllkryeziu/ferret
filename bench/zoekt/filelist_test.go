// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestReadFileList(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    []string
		wantErr string
	}{
		{name: "trailing newline", content: "a.c\nb/c.h\n", want: []string{"a.c", "b/c.h"}},
		{name: "no trailing newline", content: "a.c\nb/c.h", want: []string{"a.c", "b/c.h"}},
		{name: "keeps list order", content: "z.c\na.c\n", want: []string{"z.c", "a.c"}},
		{name: "empty", content: "", wantErr: "empty file list"},
		{name: "blank line", content: "a.c\n\nb.c\n", wantErr: ":2: empty path"},
		{name: "absolute", content: "/etc/passwd\n", wantErr: "not a clean relative path"},
		{name: "parent", content: "../x.c\n", wantErr: "not a clean relative path"},
		{name: "dot segment", content: "a/./b.c\n", wantErr: "not a clean relative path"},
		{name: "trailing slash", content: "a/\n", wantErr: "not a clean relative path"},
		{name: "carriage return", content: "a.c\r\n", wantErr: "contains CR or NUL"},
		{name: "duplicate", content: "a.c\nb.c\na.c\n", wantErr: ":3: \"a.c\" duplicates line 1"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			listPath := filepath.Join(t.TempDir(), "files")
			if err := os.WriteFile(listPath, []byte(tc.content), 0o644); err != nil {
				t.Fatal(err)
			}
			got, err := readFileList(listPath)
			if tc.wantErr != "" {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("readFileList() error = %v, want it to contain %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("readFileList() error = %v", err)
			}
			if !slices.Equal(got, tc.want) {
				t.Errorf("readFileList() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestReadRegularFileRejectsSymlink(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "target.c"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink("target.c", filepath.Join(root, "link.c")); err != nil {
		t.Fatal(err)
	}
	if got, err := readRegularFile(root, "target.c"); err != nil || string(got) != "x" {
		t.Errorf("readRegularFile(target.c) = %q, %v; want \"x\", nil", got, err)
	}
	if _, err := readRegularFile(root, "link.c"); err == nil || !strings.Contains(err.Error(), "not a regular file") {
		t.Errorf("readRegularFile(link.c) error = %v, want \"not a regular file\"", err)
	}
}
