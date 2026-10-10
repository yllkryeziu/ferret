# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Local and replay models never need an API or a provider secret."""

import pytest
from minisweagent.exceptions import FormatError

from harness.tools import SUBMIT
from models.local import LocalModel
from models.replay import ReplayModel

_MESSAGE = {
    "role": "assistant",
    "tool_calls": [
        {
            "id": "a",
            "type": "function",
            "function": {"name": "submit", "arguments": '{"files":["x"]}'},
        }
    ],
}


def test_local_uses_injected_inference_and_parses_tools():
    model = LocalModel(lambda messages: dict(_MESSAGE), (SUBMIT,))
    assert model.query([])["extra"]["actions"][0]["tool"] == "submit"
    with pytest.raises(FormatError):
        LocalModel(lambda messages: {"role": "assistant"}, (SUBMIT,)).query([])


def test_replay_consumes_a_bounded_transcript():
    replay = ReplayModel([_MESSAGE], (SUBMIT,))
    assert replay.query([])["extra"]["actions"][0]["tool"] == "submit"
    with pytest.raises(RuntimeError, match="no more recorded turns"):
        replay.query([])


def test_restored_local_factory_uses_injected_tools(monkeypatch):
    from harness.arms import ARMS
    from models import local
    from models.local import LocalModelFactory

    seen = {}

    def fake_chat_model(**kwargs):
        seen.update(kwargs)
        return "constructed; no request"

    monkeypatch.setattr(local, "ToolCallingChatModel", fake_chat_model)
    factory = LocalModelFactory("offline")
    assert factory.build(ARMS["A1"], "episode", tools=(SUBMIT,)) == (
        "constructed; no request"
    )
    assert seen["tools"] == (SUBMIT,)
    assert factory.provenance()["model_id"] == "openai/offline"


def test_restored_replay_factory_checks_trajectory_tool_hash(tmp_path):
    from harness.arms import ARMS
    from harness.tools import tool_set_sha256
    from harness.trajectory import episode_path, write_episode
    from models.replay import ReplayModelFactory

    path = episode_path(tmp_path, "one")
    write_episode(
        path,
        {
            "episode_id": "one",
            "format_version": 1,
            "tool_schema_sha256": tool_set_sha256((SUBMIT,)),
        },
        [_MESSAGE],
    )
    factory = ReplayModelFactory(tmp_path)
    with pytest.raises(ValueError, match="different tool schema"):
        factory.build(ARMS["A1"], "one")


def test_replay_retains_recorded_usage_and_format_error():
    from models.toolcalls import ApiFlavor

    recorded = {
        **_MESSAGE,
        "extra": {
            "usage": {"input_tokens": 30, "output_tokens": 3},
            "actions": [{"tool": "submit", "arguments": {"files": ["x"]}}],
        },
    }
    replay = ReplayModel(
        [recorded, {"role": "assistant", "extra": {"interrupt_type": "FormatError"}}],
        (SUBMIT,),
        api_flavor=ApiFlavor.CHAT,
    )
    first = replay.query([])
    assert first["extra"]["usage"] == recorded["extra"]["usage"]
    first["extra"]["usage"]["input_tokens"] = 0
    assert recorded["extra"]["usage"]["input_tokens"] == 30
    with pytest.raises(FormatError):
        replay.query([])


def test_replay_factory_round_trips_usage_and_tool_hash(tmp_path):
    from harness.arms import ARMS
    from harness.tools import tool_set_sha256
    from harness.trajectory import episode_path, write_episode
    from models.replay import ReplayModelFactory

    path = episode_path(tmp_path, "one")
    message = {
        **_MESSAGE,
        "extra": {
            "actions": [{"tool": "submit", "arguments": {"files": ["x"]}}],
            "usage": {"input_tokens": 7, "output_tokens": 2},
        },
    }
    write_episode(
        path,
        {
            "episode_id": "one",
            "format_version": 1,
            "api_flavor": "chat",
            "tool_schema_sha256": tool_set_sha256(ARMS["A0"].tools),
        },
        [message],
    )
    replay = ReplayModelFactory(tmp_path).build(ARMS["A0"], "one")
    assert replay.query([])["extra"]["usage"]["input_tokens"] == 7
