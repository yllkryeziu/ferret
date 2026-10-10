# Notes on this run

Linux ran from 00:35 to 01:33 and Chromium from 01:33 to 02:58 CEST on
2026-10-06, on AC power. No step spanned a sleep.

* `chromium/machine.txt` says `4969570 (modified)`: when the Chromium run
  started, `bench/README.md` had an uncommitted edit (a link to these
  results). No code had changed, and Bazel rebuilt nothing (every action was
  a cache hit). `run.sh` now leaves Markdown out of that check.
* One Chromium ripgrep search, `chromium-0337` (one matching line), took
  108 s, against a median of 15.5 s. Ferret's searches of the same query that
  wait for recent edits took 25 to 95 ms instead of about 13, so the machine
  was briefly stalled; the medians and 95th percentiles are unaffected.
* Zoekt's matches differ from ripgrep's on 11 Linux and 43 Chromium queries.
  Its default policy skips files that contain a NUL byte or more than 20,000
  distinct trigrams: 49 files that ripgrep searches on Linux and 45,205 on
  Chromium (43,842 of them binary). For `chromium-0101` and `chromium-0927`
  the full line sets were compared: every line only ripgrep returns is in a
  skipped file, and Zoekt returns no line that ripgrep does not.
