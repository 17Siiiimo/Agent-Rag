from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


def normalize_text(text: str) -> str:
    text = str(text or "").casefold()
    text = unicodedata.normalize("NFD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def slugify(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "document"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def clamp_bbox(bbox: list[float], width: float, height: float) -> list[float]:
    x1, y1, x2, y2 = bbox
    x1 = max(0.0, min(float(x1), width))
    x2 = max(0.0, min(float(x2), width))
    y1 = max(0.0, min(float(y1), height))
    y2 = max(0.0, min(float(y2), height))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]
