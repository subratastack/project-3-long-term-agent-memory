"""Which torch device the local embedding and reranking models run on.

sentence-transformers picks a device itself when given none (and logs "No
device provided, using cuda:0"); resolving it here instead makes the choice
explicit and configurable. Kept free of module-level torch imports so
`cross_encoder.py` can use it without pulling in torch at import time.
"""

from __future__ import annotations

import os

MODEL_DEVICE_ENV = "MODEL_DEVICE"


def model_device() -> str:
    """Return `MODEL_DEVICE` (e.g. "cuda", "cuda:1", "cpu", "mps"), else the best available.

    Without the variable: "cuda" if a GPU is visible, then "mps" on Apple
    silicon, otherwise "cpu" -- the same order sentence-transformers uses.
    """
    configured = os.environ.get(MODEL_DEVICE_ENV, "").strip()
    if configured:
        return configured

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
