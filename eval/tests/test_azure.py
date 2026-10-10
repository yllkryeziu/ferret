# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for models.azure, with litellm replaced by a fake."""

import json

import pytest

from harness.arms import ARMS
from models import azure


def test_requires_credentials():
    with pytest.raises(ValueError, match="AZURE_API_KEY, AZURE_API_BASE") as error:
        azure.AzureModelFactory.from_environment(environ={})
    with pytest.raises(ValueError, match="AZURE_API_KEY"):
        azure.AzureModelFactory.from_environment(environ={"AZURE_API_BASE": "x"})
    assert "secret" not in str(error.value)
    factory = azure.AzureModelFactory.from_environment(
        environ={"AZURE_API_KEY": "secret", "AZURE_API_BASE": "https://x/openai/v1"}
    )
    assert factory.model_id == "openai/gpt-6-luna"


def test_sends_credentials_per_request_and_never_records_them(monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "super-secret-key")
    monkeypatch.setenv("AZURE_API_BASE", "https://example.invalid/openai/v1")
    sent = {}

    def fake_responses(**kwargs):
        sent.update(kwargs)
        return {
            "object": "response",
            "model": "gpt-6-luna",
            "output": [
                {
                    "type": "function_call",
                    "name": "search",
                    "arguments": '{"pattern": "x"}',
                    "call_id": "c1",
                }
            ],
            "usage": {"input_tokens": 12, "output_tokens": 3},
        }

    monkeypatch.setattr("models.litellm_models.litellm.responses", fake_responses)
    factory = azure.AzureModelFactory()
    model = factory.build(ARMS["A3"], "episode")
    message = model.query([{"role": "user", "content": "find x"}])

    assert sent["api_key"] == "super-secret-key"
    assert sent["api_base"] == "https://example.invalid/openai/v1"
    assert sent["timeout"] == azure.REQUEST_TIMEOUT_S
    assert sent["model"] == "openai/gpt-6-luna"
    assert sent["reasoning"] == {"effort": "medium"}
    assert sent["temperature"] == 1.0
    assert [tool["name"] for tool in sent["tools"]] == [
        "search",
        "read",
        "ls",
        "find",
        "submit",
    ]
    assert message["extra"]["actions"] == [
        {"tool": "search", "arguments": {"pattern": "x"}, "tool_call_id": "c1"}
    ]
    assert message["extra"]["usage"]["input_tokens"] == 12
    recorded = json.dumps([message, model.serialize(), factory.provenance()])
    assert "super-secret-key" not in recorded
