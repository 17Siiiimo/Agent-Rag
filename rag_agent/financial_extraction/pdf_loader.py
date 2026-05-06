from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from .models import PDFDocument


def load_pdf(pdf_path: str) -> PDFDocument:
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF introuvable: {pdf_path}")
    doc = fitz.open(str(path))
    try:
        return PDFDocument(pdf_id=path.stem, pdf_path=str(path), page_count=len(doc))
    finally:
        doc.close()
