from __future__ import annotations

from pathlib import Path

from PIL import Image

from .models import TableCandidate
from .utils import ensure_dir


def crop_table_image(
    candidate: TableCandidate,
    rendered_image_path: str,
    output_dir: str | Path,
    page_size: tuple[float, float],
) -> str:
    out_dir = ensure_dir(output_dir)
    img = Image.open(rendered_image_path).convert("RGB")
    width, height = img.size

    pdf_width, pdf_height = page_size
    sx = width / max(pdf_width, 1.0)
    sy = height / max(pdf_height, 1.0)

    x1, y1, x2, y2 = candidate.bbox
    box = (
        max(0, int(x1 * sx)),
        max(0, int(y1 * sy)),
        min(width, int(x2 * sx)),
        min(height, int(y2 * sy)),
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        box = (0, 0, width, height)
    crop = img.crop(box)
    out_path = out_dir / f"page_{candidate.page_number:03d}_{candidate.table_type.lower()}_crop.png"
    crop.save(out_path)
    return str(out_path)
