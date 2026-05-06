"""
High-quality single-page PDF table extraction pipeline.

Pipeline:
  1. render_page()         — high-DPI rendering (pypdfium2 → pdf2image → PyMuPDF fallback)
  2. preprocess_image()    — grayscale + contrast + sharpening
  3. detect_table_region() — optional crop to table bounding box
  4. extract_with_llm()    — Vision LLM → strict JSON
  5. fallback_extract()    — Camelot lattice → stream → pdfplumber

Target table: "BILAN ACTIF COMPTES SOCIAUX" (or any table name passed by caller).

Usage:
    result = run_pipeline(
        pdf_path="rapport.pdf",
        page_num=12,
        table_name="BILAN ACTIF COMPTES SOCIAUX",
        api_provider="gemini",   # "groq" | "openai" | "gemini"
        api_key="...",
    )
    # result is a dict matching the OUTPUT FORMAT spec below.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy deps — imported lazily so the module loads even without them
# ---------------------------------------------------------------------------

try:
    import pypdfium2 as pdfium  # type: ignore
    _HAS_PDFIUM = True
except ImportError:
    _HAS_PDFIUM = False

try:
    from pdf2image import convert_from_path  # type: ignore
    _HAS_PDF2IMAGE = True
except ImportError:
    _HAS_PDF2IMAGE = False

try:
    import fitz  # PyMuPDF  # type: ignore
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False

try:
    import camelot  # type: ignore
    _HAS_CAMELOT = True
except ImportError:
    _HAS_CAMELOT = False

try:
    import pdfplumber  # type: ignore
    _HAS_PDFPLUMBER = True
except ImportError:
    _HAS_PDFPLUMBER = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_DPI = 350          # preferred rendering DPI
FALLBACK_DPI = 300        # minimum acceptable DPI
CONTRAST_FACTOR = 1.8     # PIL ImageEnhance.Contrast multiplier
SHARPNESS_FACTOR = 2.0    # PIL ImageEnhance.Sharpness multiplier

# ---------------------------------------------------------------------------
# Output type
# ---------------------------------------------------------------------------

ExtractionOutput = Dict[str, Any]
"""
{
  "table_name": str,
  "page_number": int,
  "columns": [str, ...],
  "rows": [
    {
      "label": str,
      "values": {"year_n": str | null, "year_n-1": str | null, ...}
    }
  ],
  "confidence": float,   # 0.0 – 1.0
  "method": str,         # "llm_vision" | "camelot_lattice" | "camelot_stream" | "pdfplumber" | "failed"
}
"""

# ---------------------------------------------------------------------------
# STEP 1 — High-quality page rendering
# ---------------------------------------------------------------------------

def render_page(pdf_path: str, page_num: int, dpi: int = TARGET_DPI) -> Optional[Image.Image]:
    """
    Render *page_num* (1-indexed) of *pdf_path* at *dpi* into a PIL RGB image.

    Resolution ladder:
      1. pypdfium2  (fastest, no system dependency)
      2. pdf2image  (Poppler, best quality)
      3. PyMuPDF    (fitz, last resort — lower quality at equivalent zoom)

    Returns None if all renderers fail.
    """
    # --- pypdfium2 (Option A preferred) ---
    if _HAS_PDFIUM:
        try:
            doc = pdfium.PdfDocument(pdf_path)
            try:
                page = doc[page_num - 1]
                scale = dpi / 72.0  # pypdfium2 native is 72 DPI
                bitmap = page.render(scale=scale, rotation=0)
                img = bitmap.to_pil()
                return img.convert("RGB")
            finally:
                doc.close()
        except Exception as exc:
            logger.warning("render_page: pypdfium2 failed — %s", exc)

    # --- pdf2image / Poppler (Option B) ---
    if _HAS_PDF2IMAGE:
        try:
            pages = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=page_num,
                last_page=page_num,
                fmt="png",
            )
            if pages:
                return pages[0].convert("RGB")
        except Exception as exc:
            logger.warning("render_page: pdf2image failed — %s", exc)

    # --- PyMuPDF fallback (zoom = dpi/72) ---
    if _HAS_FITZ:
        try:
            doc = fitz.open(pdf_path)
            try:
                if page_num < 1 or page_num > len(doc):
                    logger.error("render_page: page %d out of range (%d pages)", page_num, len(doc))
                    return None
                page = doc[page_num - 1]
                zoom = dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img_bytes = pix.tobytes("png")
                return Image.open(io.BytesIO(img_bytes)).convert("RGB")
            finally:
                doc.close()
        except Exception as exc:
            logger.warning("render_page: PyMuPDF failed — %s", exc)

    logger.error("render_page: no PDF renderer available (install pypdfium2 or pdf2image)")
    return None


# ---------------------------------------------------------------------------
# STEP 2 — Image preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Improve readability for Vision LLM:
      1. Convert to grayscale (removes color noise)
      2. Boost contrast  (× CONTRAST_FACTOR)
      3. Sharpen         (× SHARPNESS_FACTOR)
      4. Convert back to RGB (LLMs expect 3-channel)

    Returns a new PIL Image; the original is not modified.
    """
    # 1. Grayscale
    gray = img.convert("L")

    # 2. Contrast
    gray = ImageEnhance.Contrast(gray).enhance(CONTRAST_FACTOR)

    # 3. Sharpening — apply UnsharpMask for finer control than SHARPEN kernel
    gray = gray.filter(ImageFilter.UnsharpMask(radius=1, percent=int(SHARPNESS_FACTOR * 100), threshold=3))

    # 4. Back to RGB
    return gray.convert("RGB")


# ---------------------------------------------------------------------------
# STEP 3 — Table region detection (optional)
# ---------------------------------------------------------------------------

def detect_table_region(img: Image.Image, page_num: int = 1) -> Tuple[Image.Image, bool]:
    """
    Attempt to crop the image to the table bounding box.

    Strategy (in order):
      1. Table Transformer (microsoft/table-transformer-detection) if available
      2. Heuristic: strip top 15 % (header / logo) and bottom 5 % (footer)

    Returns (cropped_image, was_cropped).
    """
    # --- Attempt 1: Table Transformer ---
    try:
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection  # type: ignore
        import torch  # type: ignore

        processor = AutoImageProcessor.from_pretrained(
            "microsoft/table-transformer-detection", use_fast=True
        )
        model = TableTransformerForObjectDetection.from_pretrained(
            "microsoft/table-transformer-detection"
        )
        model.eval()

        inputs = processor(images=img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([img.size[::-1]])
        results = processor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=target_sizes
        )[0]

        if len(results["boxes"]) > 0:
            # Pick the largest detected table box
            boxes = results["boxes"].tolist()
            best_box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            x0, y0, x1, y1 = [int(v) for v in best_box]
            # Add small padding
            pad = 10
            w, h = img.size
            x0 = max(0, x0 - pad)
            y0 = max(0, y0 - pad)
            x1 = min(w, x1 + pad)
            y1 = min(h, y1 + pad)
            cropped = img.crop((x0, y0, x1, y1))
            logger.info("detect_table_region: TATR detected box %s on page %d", best_box, page_num)
            return cropped, True

    except Exception as exc:
        logger.debug("detect_table_region: TATR unavailable or failed — %s", exc)

    # --- Attempt 2: Heuristic crop ---
    w, h = img.size
    top_cut = int(h * 0.12)    # skip ~12 % top (logo, title, page header)
    bottom_cut = int(h * 0.04) # skip ~4 % bottom (page footer)
    if top_cut + bottom_cut < h:
        heuristic = img.crop((0, top_cut, w, h - bottom_cut))
        logger.debug("detect_table_region: heuristic crop applied on page %d", page_num)
        return heuristic, True

    return img, False


# ---------------------------------------------------------------------------
# STEP 4 — Vision LLM extraction
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_BARE_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_STRICT_PROMPT = """\
You are a financial data extraction assistant. Your ONLY task is to extract a table from this image.

Target table: "{table_name}"

Rules:
1. Extract ONLY the table named above. Do NOT extract any other table.
2. Do NOT invent or hallucinate any values. If a cell is empty, use null.
3. Preserve ALL numeric values EXACTLY as printed (spaces, commas, dashes).
4. Preserve row hierarchy / indentation in the label field.
5. The first data column = "year_n", the second = "year_n-1". Add more keys if more columns exist.
6. Return ONLY valid JSON. No markdown fences, no explanation text.

Output format (strict):
{{
  "table_name": "{table_name}",
  "page_number": {page_num},
  "columns": ["label", "year_n", "year_n-1"],
  "rows": [
    {{
      "label": "<row label>",
      "values": {{
        "year_n": "<value or null>",
        "year_n-1": "<value or null>"
      }}
    }}
  ],
  "confidence": 0.95
}}
"""


def _image_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode PIL image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, optimize=False)
    return base64.b64encode(buf.getvalue()).decode()


def _parse_llm_json(raw: str, table_name: str, page_num: int) -> Optional[ExtractionOutput]:
    """
    Parse JSON from LLM response.  Tries JSON block first, then bare JSON search.
    Validates required keys; fills in defaults for missing optional fields.
    """
    if not raw:
        return None

    text = raw.strip()

    # Try fenced block
    m = _JSON_BLOCK_RE.search(text)
    blob = m.group(1) if m else None

    # Try bare JSON
    if blob is None:
        m2 = _BARE_JSON_RE.search(text)
        blob = m2.group(0) if m2 else None

    if blob is None:
        logger.warning("_parse_llm_json: no JSON found in response (len=%d)", len(text))
        return None

    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        logger.warning("_parse_llm_json: JSON decode error — %s", exc)
        return None

    # Normalise required fields
    data.setdefault("table_name", table_name)
    data.setdefault("page_number", page_num)
    data.setdefault("columns", [])
    data.setdefault("rows", [])
    data.setdefault("confidence", 0.7)
    data.setdefault("method", "llm_vision")

    if not isinstance(data.get("rows"), list):
        logger.warning("_parse_llm_json: 'rows' is not a list")
        return None

    return data


def extract_with_llm(
    img: Image.Image,
    table_name: str,
    page_num: int,
    api_provider: str = "gemini",
    api_key: Optional[str] = None,
) -> Optional[ExtractionOutput]:
    """
    Send *img* to a Vision LLM and parse the strict JSON response.

    Supported providers: "groq", "openai", "gemini".
    Returns None if the call fails or the response cannot be parsed.
    """
    api_key = api_key or _resolve_api_key(api_provider)
    if not api_key:
        logger.warning("extract_with_llm: no API key for provider=%s", api_provider)
        return None

    img_b64 = _image_to_b64(img)
    prompt = _STRICT_PROMPT.format(table_name=table_name, page_num=page_num)

    raw: str = ""
    provider = api_provider.lower().strip()

    try:
        if provider == "groq":
            raw = _call_groq(api_key, img_b64, prompt)
        elif provider in ("openai", "gpt-4o", "gpt-4"):
            raw = _call_openai(api_key, img_b64, prompt)
        elif provider == "gemini":
            raw = _call_gemini(api_key, img_b64, prompt)
        else:
            logger.warning("extract_with_llm: unknown provider '%s'", provider)
            return None
    except Exception as exc:
        logger.warning("extract_with_llm: LLM call failed provider=%s — %s", provider, exc)
        return None

    result = _parse_llm_json(raw, table_name, page_num)
    if result:
        result["method"] = f"llm_vision_{provider}"
    return result


# ---------------------------------------------------------------------------
# Provider helpers
# ---------------------------------------------------------------------------

def _resolve_api_key(provider: str) -> Optional[str]:
    """Read API key from environment variables."""
    mapping = {
        "groq":   "GROQ_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gpt-4o": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    env_var = mapping.get(provider.lower())
    return os.environ.get(env_var) if env_var else None


def _call_groq(api_key: str, img_b64: str, prompt: str) -> str:
    import requests  # noqa: PLC0415
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 8192,
    }
    resp = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"] or ""


def _call_openai(api_key: str, img_b64: str, prompt: str) -> str:
    from openai import OpenAI  # noqa: PLC0415
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        temperature=0.0,
        max_tokens=8192,
    )
    return (completion.choices[0].message.content or "").strip()


def _call_gemini(api_key: str, img_b64: str, prompt: str) -> str:
    import requests  # noqa: PLC0415
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    payload = {
        "contents": [
            {
                "parts": [
                    {"inline_data": {"mime_type": "image/png", "data": img_b64}},
                    {"text": prompt},
                ]
            }
        ],
        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192},
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    candidates = resp.json().get("candidates", [])
    if not candidates:
        return ""
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts).strip()


# ---------------------------------------------------------------------------
# STEP 5 — Fallback extraction (no Vision LLM)
# ---------------------------------------------------------------------------

def _camelot_to_output(tables, table_name: str, page_num: int, mode: str) -> Optional[ExtractionOutput]:
    """Convert the first valid Camelot table to the standard output dict."""
    if not tables or len(tables) == 0:
        return None
    df = tables[0].df
    if df.empty or df.shape[0] < 3 or df.shape[1] < 2:
        return None

    headers = df.iloc[0].tolist()
    rows = []
    for _, row in df.iloc[1:].iterrows():
        cells = row.tolist()
        label = str(cells[0]).strip() if cells else ""
        values: Dict[str, Any] = {}
        col_keys = ["year_n", "year_n-1", "brut", "amort_prov"] + [f"col_{i}" for i in range(10)]
        for idx, val in enumerate(cells[1:], start=0):
            key = col_keys[idx] if idx < len(col_keys) else f"col_{idx}"
            values[key] = str(val).strip() or None
        rows.append({"label": label, "values": values})

    return {
        "table_name": table_name,
        "page_number": page_num,
        "columns": [str(h) for h in headers],
        "rows": rows,
        "confidence": 0.55 if mode == "lattice" else 0.40,
        "method": f"camelot_{mode}",
    }


def _pdfplumber_to_output(pdf_path: str, page_num: int, table_name: str) -> Optional[ExtractionOutput]:
    """Extract first table on page with pdfplumber."""
    if not _HAS_PDFPLUMBER:
        return None
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num < 1 or page_num > len(pdf.pages):
                return None
            page = pdf.pages[page_num - 1]
            tables = page.extract_tables()
            if not tables:
                return None
            tbl = tables[0]
            if len(tbl) < 3:
                return None

            headers = tbl[0]
            rows = []
            col_keys = ["year_n", "year_n-1", "brut", "amort_prov"] + [f"col_{i}" for i in range(10)]
            for row in tbl[1:]:
                label = str(row[0]).strip() if row else ""
                values: Dict[str, Any] = {}
                for idx, val in enumerate(row[1:], start=0):
                    key = col_keys[idx] if idx < len(col_keys) else f"col_{idx}"
                    values[key] = str(val).strip() if val else None
                rows.append({"label": label, "values": values})

            return {
                "table_name": table_name,
                "page_number": page_num,
                "columns": [str(h) for h in headers],
                "rows": rows,
                "confidence": 0.30,
                "method": "pdfplumber",
            }
    except Exception as exc:
        logger.warning("_pdfplumber_to_output: failed — %s", exc)
        return None


def fallback_extract(pdf_path: str, page_num: int, table_name: str) -> ExtractionOutput:
    """
    Technical fallback when Vision LLM fails.

    Tries in order:
      1. Camelot lattice  (best for bordered tables)
      2. Camelot stream   (best for whitespace-delimited tables)
      3. pdfplumber       (always available)

    Returns the first successful result, or a "failed" output.
    """
    # 1. Camelot lattice
    if _HAS_CAMELOT:
        try:
            tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor="lattice")
            result = _camelot_to_output(tables, table_name, page_num, "lattice")
            if result:
                logger.info("fallback_extract: camelot lattice OK (page=%d)", page_num)
                return result
        except Exception as exc:
            logger.debug("fallback_extract: camelot lattice failed — %s", exc)

        # 2. Camelot stream
        try:
            tables = camelot.read_pdf(pdf_path, pages=str(page_num), flavor="stream")
            result = _camelot_to_output(tables, table_name, page_num, "stream")
            if result:
                logger.info("fallback_extract: camelot stream OK (page=%d)", page_num)
                return result
        except Exception as exc:
            logger.debug("fallback_extract: camelot stream failed — %s", exc)

    # 3. pdfplumber
    result = _pdfplumber_to_output(pdf_path, page_num, table_name)
    if result:
        logger.info("fallback_extract: pdfplumber OK (page=%d)", page_num)
        return result

    logger.error("fallback_extract: all methods failed (page=%d)", page_num)
    return {
        "table_name": table_name,
        "page_number": page_num,
        "columns": [],
        "rows": [],
        "confidence": 0.0,
        "method": "failed",
    }


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    pdf_path: str,
    page_num: int,
    table_name: str = "BILAN ACTIF COMPTES SOCIAUX",
    api_provider: str = "gemini",
    api_key: Optional[str] = None,
    dpi: int = TARGET_DPI,
    detect_table: bool = True,
) -> ExtractionOutput:
    """
    Full single-page extraction pipeline.

    Parameters
    ----------
    pdf_path     : path to the PDF file
    page_num     : 1-indexed page number
    table_name   : exact table name to extract
    api_provider : "groq" | "openai" | "gemini"
    api_key      : optional override; falls back to env vars
    dpi          : rendering resolution (default 350)
    detect_table : whether to attempt table region cropping (default True)

    Returns
    -------
    dict matching ExtractionOutput spec (table_name, page_number, columns, rows,
    confidence, method).
    """
    logger.info(
        "run_pipeline: START pdf=%s page=%d table='%s' provider=%s dpi=%d",
        pdf_path, page_num, table_name, api_provider, dpi,
    )

    # --- Step 1: Render ---
    img = render_page(pdf_path, page_num, dpi=dpi)
    if img is None:
        logger.error("run_pipeline: render_page failed — using fallback extractor")
        return fallback_extract(pdf_path, page_num, table_name)

    # --- Step 2: Preprocess ---
    img = preprocess_image(img)

    # --- Step 3: Detect table region (optional) ---
    if detect_table:
        img, was_cropped = detect_table_region(img, page_num=page_num)
        if was_cropped:
            logger.debug("run_pipeline: table region cropped")

    # --- Step 4: Vision LLM ---
    result = extract_with_llm(
        img=img,
        table_name=table_name,
        page_num=page_num,
        api_provider=api_provider,
        api_key=api_key,
    )

    if result and result.get("rows"):
        logger.info(
            "run_pipeline: LLM OK — %d rows, confidence=%.2f, method=%s",
            len(result["rows"]),
            result.get("confidence", 0),
            result.get("method", "?"),
        )
        return result

    # --- Step 5: Fallback ---
    logger.warning("run_pipeline: LLM returned no rows — falling back to technical extractors")
    return fallback_extract(pdf_path, page_num, table_name)
