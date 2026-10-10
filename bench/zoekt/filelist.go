// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// readFileList reads a benchmark file list: one repo-relative,
// '/'-separated path per line, optionally ending in a newline. Both engines
// index exactly this list (bench/README.md), so any entry that
// could resolve differently for them is an error: empty or unclean paths,
// paths that leave the root, carriage returns, and duplicates.
func readFileList(listPath string) ([]string, error) {
	data, err := os.ReadFile(listPath)
	if err != nil {
		return nil, err
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("%s: empty file list", listPath)
	}
	lines := strings.Split(strings.TrimSuffix(string(data), "\n"), "\n")
	seen := make(map[string]int, len(lines))
	for i, p := range lines {
		lineNo := i + 1
		switch {
		case p == "":
			return nil, fmt.Errorf("%s:%d: empty path", listPath, lineNo)
		case strings.ContainsAny(p, "\r\x00"):
			return nil, fmt.Errorf("%s:%d: path %q contains CR or NUL", listPath, lineNo, p)
		case path.IsAbs(p) || path.Clean(p) != p || p == ".." || strings.HasPrefix(p, "../"):
			return nil, fmt.Errorf("%s:%d: %q is not a clean relative path", listPath, lineNo, p)
		}
		if first, ok := seen[p]; ok {
			return nil, fmt.Errorf("%s:%d: %q duplicates line %d", listPath, lineNo, p, first)
		}
		seen[p] = lineNo
	}
	return lines, nil
}

// readRegularFile returns the content of the listed path rel under root.
// Lists name regular files only; following a symlink would index bytes that
// the other engine does not see.
func readRegularFile(root, rel string) ([]byte, error) {
	full := filepath.Join(root, filepath.FromSlash(rel))
	info, err := os.Lstat(full)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("%s: not a regular file (mode %v)", full, info.Mode())
	}
	return os.ReadFile(full)
}
