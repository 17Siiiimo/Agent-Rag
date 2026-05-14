"""Thread-safe cache for SentenceTransformer models (avoid reload on every query / index)."""

from __future__ import annotations

import threading
from typing import Any

_lock = threading.Lock()
_models: dict[str, Any] = {}


def get_sentence_transformer(model_name: str) -> Any:
    """Return a shared SentenceTransformer instance for ``model_name``."""
    with _lock:
        if model_name not in _models:
            from sentence_transformers import SentenceTransformer

            _models[model_name] = SentenceTransformer(model_name)
        return _models[model_name]
