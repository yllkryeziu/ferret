# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Task pipelines for the agent experiment.

Every task source ends in the same record, `tasks.schema.Task`. LocBench
arrives through `tasks.benchmarks`; each task's commit is fetched into a
`tasks.fetch.CommitCache` and exported as a history-free snapshot by
`tasks.snapshot`, the only repository an agent ever sees.
"""
