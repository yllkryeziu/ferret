# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""Tests for models.rewriter, with litellm replaced by a fake."""

import pytest

from models.rewriter import AzureRewriter


def test_requires_credentials_and_a_model_other_than_the_agent():
    with pytest.raises(ValueError, match="AZURE_API_KEY, AZURE_API_BASE"):
        AzureRewriter.from_environment(environ={})
    with pytest.raises(ValueError, match="agent model"):
        AzureRewriter("gpt-6-luna")
    rewriter = AzureRewriter.from_environment(
        environ={"AZURE_API_KEY": "secret", "AZURE_API_BASE": "https://x/openai/v1"}
    )
    assert rewriter.rewriter_id.startswith("azure/gpt-6-sol/effort=low/prompt=")
    assert "secret" not in repr(rewriter)


def test_returns_the_message_text_and_counts_tokens(monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "super-secret-key")
    monkeypatch.setenv("AZURE_API_BASE", "https://example.invalid/openai/v1")
    sent = []

    def fake_responses(**kwargs):
        sent.append(kwargs)
        return {
            "output": [
                {"type": "reasoning", "summary": []},
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": " The screen stays dark."},
                        {"type": "output_text", "text": " Nothing is logged. "},
                    ],
                },
            ],
            "usage": {"input_tokens": 200, "output_tokens": 50},
        }

    monkeypatch.setattr("models.rewriter.litellm.responses", fake_responses)
    rewriter = AzureRewriter()
    assert rewriter("rewrite this") == "The screen stays dark. Nothing is logged."
    assert rewriter("rewrite that") == "The screen stays dark. Nothing is logged."

    assert [call["input"] for call in sent] == ["rewrite this", "rewrite that"]
    assert sent[0]["model"] == "openai/gpt-6-sol"
    assert sent[0]["reasoning"] == {"effort": "low"}
    assert sent[0]["api_key"] == "super-secret-key"
    assert sent[0]["api_base"] == "https://example.invalid/openai/v1"
    assert (rewriter.input_tokens, rewriter.output_tokens) == (400, 100)


def test_a_reply_without_text_is_empty(monkeypatch):
    monkeypatch.setenv("AZURE_API_KEY", "k")
    monkeypatch.setenv("AZURE_API_BASE", "https://example.invalid/openai/v1")
    monkeypatch.setattr(
        "models.rewriter.litellm.responses",
        lambda **_: {"output": [{"type": "reasoning", "summary": []}]},
    )
    assert AzureRewriter()("rewrite this") == ""
