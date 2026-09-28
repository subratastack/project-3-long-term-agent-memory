"""Hugging Face Hub credentials for the local embedding and reranking models.

Both default models are public, so a token is optional: without one,
downloads still work but are rate-limited and huggingface_hub logs an
"unauthenticated requests" warning. Kept free of heavy imports so
`cross_encoder.py` can use it without pulling in torch at module level.
"""

from __future__ import annotations

import os

HUGGINGFACE_API_KEY_ENV = "HUGGINGFACE_API_KEY"


def huggingface_token() -> str | None:
    """Return `HUGGINGFACE_API_KEY` from the environment, or None if unset/blank.

    None lets huggingface_hub fall back to its own defaults (`HF_TOKEN` or a
    token saved by `hf auth login`).
    """
    return os.environ.get(HUGGINGFACE_API_KEY_ENV, "").strip() or None
