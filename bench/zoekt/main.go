// Copyright 2026 The Ferret Authors.
// SPDX-License-Identifier: Apache-2.0

// Command zoekt-bench drives the pinned Zoekt library in-process for Ferret's
// benchmarks; bench/README.md describes how its output is used.
//
// The harness exists because the zoekt CLI reopens every shard on each call.
// On Linux that costs 70-410 ms per query against about 2 ms of search time
// inside the process, so the CLI is not a latency baseline. zoekt-bench opens
// the shard directory once and builds queries programmatically, never through
// query.Parse, whose case:auto default and whitespace splitting would change
// the query being measured.
//
// Subcommands:
//
//	build      index a file list into shards and report the build cost
//	search     time a query set; emit per-query latencies and result sets
//	freshness  time edit-to-searchable through commit, delta build, reload
//	branches   index the first k branches of a repository, for k = 1..K
//
// Run "zoekt-bench <subcommand> -help" for flags. Reports go to stdout as
// JSON; logs go to stderr.
package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"runtime/debug"
	"sort"
)

// zoektModule is the module path of the benchmarked library.
const zoektModule = "github.com/sourcegraph/zoekt"

// subcommands maps each subcommand name to its implementation. An
// implementation parses its own flags from args and writes its report to
// stdout.
var subcommands = map[string]func(args []string, stdout io.Writer) error{
	"build":     runBuild,
	"search":    runSearch,
	"freshness": runFreshness,
	"branches":  runBranches,
}

// usageError marks an invalid invocation, which exits with status 2.
type usageError struct{ err error }

func (e usageError) Error() string { return e.err.Error() }
func (e usageError) Unwrap() error { return e.err }

func main() {
	os.Exit(run(os.Args[1:], os.Stdout, os.Stderr))
}

// run executes one subcommand and returns the process exit status.
func run(args []string, stdout, stderr io.Writer) int {
	if len(args) == 0 {
		printUsage(stderr)
		return 2
	}
	cmd, ok := subcommands[args[0]]
	if !ok {
		fmt.Fprintf(stderr, "zoekt-bench: unknown subcommand %q\n", args[0])
		printUsage(stderr)
		return 2
	}
	err := cmd(args[1:], stdout)
	switch {
	case err == nil:
		return 0
	case errors.Is(err, flag.ErrHelp):
		return 0
	case errors.As(err, new(usageError)):
		fmt.Fprintf(stderr, "zoekt-bench %s: %v\n", args[0], err)
		return 2
	default:
		fmt.Fprintf(stderr, "zoekt-bench %s: %v\n", args[0], err)
		return 1
	}
}

func printUsage(w io.Writer) {
	names := make([]string, 0, len(subcommands))
	for name := range subcommands {
		names = append(names, name)
	}
	sort.Strings(names)
	fmt.Fprintln(w, "usage: zoekt-bench <subcommand> [flags]")
	fmt.Fprintf(w, "subcommands: %q\n", names)
}

// parseFlags parses args into fs and rejects positional arguments, which no
// subcommand takes.
func parseFlags(fs *flag.FlagSet, args []string) error {
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return err
		}
		return usageError{err}
	}
	if fs.NArg() != 0 {
		return usageError{fmt.Errorf("unexpected arguments %q", fs.Args())}
	}
	return nil
}

// requireFlags returns a usage error naming the first flag in names whose
// value is empty.
func requireFlags(fs *flag.FlagSet, names ...string) error {
	for _, name := range names {
		if fs.Lookup(name).Value.String() == "" {
			return usageError{fmt.Errorf("--%s is required", name)}
		}
	}
	return nil
}

// writeJSON writes v to w as one line of JSON.
func writeJSON(w io.Writer, v any) error {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	return enc.Encode(v)
}

// zoektVersion returns the version of the linked Zoekt module, which go.mod
// pins. Every report records it so that results name the exact baseline.
func zoektVersion() string {
	info, ok := debug.ReadBuildInfo()
	if !ok {
		return "unknown"
	}
	for _, dep := range info.Deps {
		if dep.Path != zoektModule {
			continue
		}
		if dep.Replace != nil {
			return dep.Replace.Path + "@" + dep.Replace.Version
		}
		return dep.Version
	}
	return "unknown"
}
