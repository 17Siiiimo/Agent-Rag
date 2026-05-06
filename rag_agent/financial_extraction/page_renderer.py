from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from .utils import ensure_dir


def render_pdf_pages(pdf_path: str, output_dir: str | Path, dpi: int = 300) -> dict[int, str]:
    """Render every PDF page to a PNG image. Returns 1-based page -> image path."""
    out_dir = ensure_dir(output_dir)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    paths: dict[int, str] = {}

    doc = fitz.open(pdf_path)
    try:
        for idx in range(len(doc)):
            page_num = idx + 1
            image_path = out_dir / f"page_{page_num:03d}.png"
            if not image_path.exists() or image_path.stat().st_size == 0:
                pix = doc[idx].get_pixmap(matrix=matrix, alpha=False)
                pix.save(str(image_path))
            paths[page_num] = str(image_path)
    finally:
        doc.close()
    return paths
