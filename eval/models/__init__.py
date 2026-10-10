# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Models for the agent experiment, all behind mini-swe-agent's Model protocol.

* `models.azure`: gpt-6-luna through the Azure OpenAI v1 endpoint and
  litellm's Responses API, the model of record.
* `models.scripted`: fixed tool calls, for tests and offline smoke runs.

Each module provides a factory whose `build(arm, episode_id)` returns a fresh
model for one episode and whose `provenance()` describes it for the
trajectory. Every model reports tool calls as the same action dicts and
records the same per-turn metrics, so the harness never sees which API, if
any, is behind it.
"""
