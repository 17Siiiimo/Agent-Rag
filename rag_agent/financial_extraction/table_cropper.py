from __future__ import annotations

from pathlib import Path

from PIL import Image

from .models import TableCandidate
from .utils import clamp_bbox, ensure_dir

# PDF points: small margin so row labels / totals are not clipped at crop edges.
_PDF_PADDING_PT = 12.0
# If localization returned an overly tight box, expand toward these fractions of the page.
_MIN_CROP_WIDTH_FRAC = 0.22
_MIN_CROP_HEIGHT_FRAC = 0.08


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
    pad = _PDF_PADDING_PT
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad

    bw = x2 - x1
    bh = y2 - y1
    min_w = pdf_width * _MIN_CROP_WIDTH_FRAC
    min_h = pdf_height * _MIN_CROP_HEIGHT_FRAC
    if bw < min_w:
        cx = (x1 + x2) / 2.0
        half = min_w / 2.0
        x1, x2 = cx - half, cx + half
    if bh < min_h:
        cy = (y1 + y2) / 2.0
        half = min_h / 2.0
        y1, y2 = cy - half, cy + half

    x1, y1, x2, y2 = clamp_bbox([x1, y1, x2, y2], pdf_width, pdf_height)

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
