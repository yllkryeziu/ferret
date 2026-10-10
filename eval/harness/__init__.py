# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Harness for the agent experiment.

mini-swe-agent's `DefaultAgent` drives a model against `FerretEnvironment`,
which offers five tools and nothing else: `search`, served by the `ferret`
command-line tool over a prebuilt index, `read`, `ls`, `find`, and `submit`.
Arms differ only in the output shape of search, so everything the model sees
is identical across arms except the shape of search results.
"""
