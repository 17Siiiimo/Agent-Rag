from __future__ import annotations

import fitz  # PyMuPDF


def extract_page_texts(pdf_path: str) -> dict[int, str]:
    """Extract page text with PyMuPDF. Returns 1-based page -> text."""
    texts: dict[int, str] = {}
    doc = fitz.open(pdf_path)
    try:
        for idx in range(len(doc)):
            texts[idx + 1] = doc[idx].get_text("text") or ""
    finally:
        doc.close()
    return texts
