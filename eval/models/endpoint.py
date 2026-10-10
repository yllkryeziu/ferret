# Copyright 2026 The Ferret Authors.
# SPDX-License-Identifier: Apache-2.0
"""The Azure OpenAI v1 endpoint shared by the agent and the rewriter.

Kept free of mini-swe-agent imports, so task preparation can use it without
loading the agent framework.
"""

import os
from collections.abc import Mapping

AZURE_ENVIRONMENT_VARIABLES = ("AZURE_API_KEY", "AZURE_API_BASE")

AGENT_MODEL = "gpt-6-luna"


def check_credentials(environ: Mapping[str, str] = os.environ) -> None:
    """Checks that the endpoint credentials are set.

    Raises:
        ValueError: naming, but never echoing, the missing variables.
    """
    missing = [name for name in AZURE_ENVIRONMENT_VARIABLES if not environ.get(name)]
    if missing:
        raise ValueError(f"set the environment variables {', '.join(missing)}")
