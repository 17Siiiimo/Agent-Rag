"""
Stratégies d'extraction modernes 2025 :

- Groq Structured Outputs (response_format=json_object) → JSON garanti, zéro parsing error
- llama-4-maverick-17b-128e-instruct (texte + vision) → modèle Groq le plus puissant
- pymupdf4llm → conversion PDF→Markdown LLM-ready (bien meilleur que get_text brut)
- Gemini 2.0 Flash → vision fallback haute qualité

Ce module est autonome (pas d'import circulaire avec extraction_agent).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Modèles recommandés 2025 ────────────────────────────────────────────────
# Groq : llama-4-maverick surpasse llama-3.3-70b sur extraction structurée + vision
GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"            # JSON structuré (stable, rapide)
GROQ_MAVERICK_TEXT = "meta-llama/llama-4-maverick-17b-128e-instruct"  # Texte enrichi
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"        # Vision (stable sur ce compte)
GROQ_SCOUT_VISION = "meta-llama/llama-4-scout-17b-16e-instruct"        # Alias compat legacy

# Gemini 2.0 en priorité (meilleur OCR sur PDF scannés)
GEMINI_MODELS_2025 = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]


# ── Helpers PDF ──────────────────────────────────────────────────────────────

def page_to_markdown(pdf_path: str, page_num: int) -> str:
    """
    Extrait le contenu d'une page PDF en Markdown structuré.

    Priorité :
    1. pymupdf4llm  → meilleure qualité pour LLM (préserve structure tableaux)
    2. PyMuPDF brut → fallback rapide
    3. pdfplumber   → dernier recours
    """
    # 1. pymupdf4llm (optimisé LLM)
    try:
        import pymupdf4llm  # type: ignore
        md = pymupdf4llm.to_markdown(pdf_path, pages=[page_num - 1])
        if md and md.strip():
            logger.debug("page_to_markdown: pymupdf4llm ok page=%d", page_num)
            return md
    except Exception as e:
        logger.debug("pymupdf4llm unavailable: %s", e)

    # 2. PyMuPDF brut
    try:
        import pymupdf  # type: ignore
        doc = pymupdf.open(pdf_path)
        try:
            if 1 <= page_num <= len(doc):
                text = doc[page_num - 1].get_text()
                if text and text.strip():
                    return text
        finally:
            doc.close()
    except Exception as e:
        logger.debug("PyMuPDF fallback failed: %s", e)

    # 3. pdfplumber
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(pdf_path) as pdf:
            if 1 <= page_num <= len(pdf.pages):
                return pdf.pages[page_num - 1].extract_text(x_tolerance=2, y_tolerance=2) or ""
    except Exception as e:
        logger.warning("page_to_markdown all methods failed page=%d: %s", page_num, e)

    return ""


def page_to_image_b64(pdf_path: str, page_num: int, zoom: float = 2.0) -> Optional[str]:
    """Convertit une page PDF en image PNG base64 (zoom x2 pour meilleure résolution OCR)."""
    try:
        import pymupdf  # type: ignore
        doc = pymupdf.open(pdf_path)
        try:
            if page_num < 1 or page_num > len(doc):
                return None
            page = doc[page_num - 1]
            mat = pymupdf.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            return base64.b64encode(img_bytes).decode()
        finally:
            doc.close()
    except Exception as e:
        logger.warning("page_to_image_b64 failed page=%d: %s", page_num, e)
        return None


# ── Helpers parsing ──────────────────────────────────────────────────────────

def _json_payload_to_df(payload: dict) -> pd.DataFrame:
    """Convertit un payload {table_name, headers, rows} en DataFrame."""
    if not isinstance(payload, dict):
        return pd.DataFrame()
    headers = payload.get("headers", [])
    rows = payload.get("rows", [])
    if not isinstance(headers, list) or not headers:
        return pd.DataFrame()
    if not isinstance(rows, list):
        return pd.DataFrame()

    norm_headers = [str(h or "").strip() for h in headers]
    out_rows: List[List[str]] = []
    for r in rows:
        if not isinstance(r, list):
            continue
        row = [str(v or "").strip() for v in r]
        if len(row) < len(norm_headers):
            row.extend([""] * (len(norm_headers) - len(row)))
        elif len(row) > len(norm_headers):
            row = row[: len(norm_headers)]
        out_rows.append(row)

    if not out_rows:
        return pd.DataFrame()
    return pd.DataFrame(out_rows, columns=norm_headers)


def _is_valid_df(df: pd.DataFrame, min_rows: int = 3, min_cols: int = 2) -> bool:
    """Vérifie qu'un DataFrame ressemble à un tableau financier valide."""
    if df is None or df.empty:
        return False
    if len(df) < min_rows or len(df.columns) < min_cols:
        return False
    for i in range(len(df.columns)):
        ser = df.iloc[:, i]
        if ser.astype(str).str.contains(r"\d{3,}", regex=True).any():
            return True
    return False


def _extract_markdown_table(content: str) -> str:
    """Extrait un tableau Markdown depuis une réponse LLM (ignore texte parasite)."""
    if not content:
        return ""
    md_match = re.search(r"(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)", content)
    return md_match.group(1) if md_match else content.strip()


def _markdown_table_to_df(table_md: str) -> pd.DataFrame:
    """Convertit un tableau Markdown en DataFrame."""
    lines = [ln.strip() for ln in table_md.splitlines() if ln.strip()]
    if not lines:
        return pd.DataFrame()

    def _split_row(row: str) -> List[str]:
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        return [cell.strip() for cell in row.split("|")]

    header: Optional[List[str]] = None
    rows: List[List[str]] = []
    for line in lines:
        stripped = line.replace("|", "").replace(" ", "").replace("-", "").replace(":", "")
        if not stripped:
            continue  # ligne séparateur
        cells = _split_row(line)
        if header is None:
            header = cells
        else:
            if len(cells) < len(header):
                cells.extend([""] * (len(header) - len(cells)))
            elif len(cells) > len(header):
                cells = cells[: len(header)]
            rows.append(cells)

    if not header:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=header)


# ── Prompt vision ────────────────────────────────────────────────────────────

_VISION_PROMPT = """Cette image est une page de rapport financier marocain.
Extrais UNIQUEMENT le tableau '{table_name}' visible sur cette page.

Instructions strictes :
- Reproduis EXACTEMENT toutes les colonnes visibles (dans leur ordre exact)
- Si colonne "Notes" (ex: "2.12", "2.13") présente entre libellé et dates : inclus-la obligatoirement
- Extrais TOUTES les lignes sans exception (libellés, sous-totaux, totaux, lignes vides)
- Les espaces dans les nombres sont séparateurs de milliers ("13 157 683" = UN seul nombre)
- Cellule vide → laisser vide : |  |
- Retourne UNIQUEMENT un tableau Markdown valide, rien avant ni après

Format attendu :
| Rubrique | Notes | 31/12/2024 | 31/12/2023 |
| --- | --- | --- | --- |
| Libellé | 2.12 | 1 716 269 | 1 670 543 |
| Total | | 2 000 000 | 1 900 000 |"""


# ── Stratégies Groq ──────────────────────────────────────────────────────────

def groq_vision_extract(
    pdf_path: str,
    page_num: int,
    table_name: str,
    api_key: str,
    model: str = GROQ_VISION_MODEL,
) -> pd.DataFrame:
    """
    Niveau 1 — Vision LLM (llama-4-maverick) → Markdown → DataFrame.

    Avantages vs ancien modèle (llama-4-scout) :
    - Contexte 128K tokens (vs 16K)
    - Meilleure précision sur tableaux denses et pages multi-tableaux
    """
    img_b64 = page_to_image_b64(pdf_path, page_num)
    if not img_b64:
        return pd.DataFrame()

    prompt = _VISION_PROMPT.format(table_name=table_name)
    try:
        from groq import Groq  # type: ignore

        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=8192,
        )
        content = (completion.choices[0].message.content or "").strip()
        if not content:
            return pd.DataFrame()
        content = _extract_markdown_table(content)
        df = _markdown_table_to_df(content)
        result = df if _is_valid_df(df) else pd.DataFrame()
        if not result.empty:
            logger.info(
                "groq_vision_extract OK model=%s page=%d shape=%s",
                model, page_num, result.shape,
            )
        return result
    except Exception as e:
        logger.warning("groq_vision_extract model=%s page=%d failed: %s", model, page_num, e)
        return pd.DataFrame()


def groq_structured_json_extract(
    pdf_path: str,
    page_num: int,
    table_name: str,
    api_key: str,
    model: str = GROQ_TEXT_MODEL,
) -> pd.DataFrame:
    """
    Niveau 2 — Groq JSON mode (response_format=json_object) → DataFrame.

    Améliorations vs ancienne implémentation :
    - response_format=json_object : JSON GARANTI, élimine les erreurs de parsing
    - pymupdf4llm : texte de meilleure qualité que PyMuPDF brut
    - max_tokens=8192 : évite les troncatures sur grands tableaux
    """
    text = page_to_markdown(pdf_path, page_num)
    if not text.strip():
        return pd.DataFrame()

    prompt = f"""Extrais le tableau financier '{table_name}' depuis ce document financier marocain.

Réponds UNIQUEMENT avec un objet JSON (structure exacte) :
{{
  "table_name": "{table_name}",
  "headers": ["Rubrique", "Exercice N", "Exercice N-1"],
  "rows": [
    ["Libellé de la ligne", "valeur1", "valeur2"]
  ]
}}

Règles strictes :
- headers : "Rubrique" en premier, puis chaque colonne de données (dates exactes si visibles)
- rows : chaque élément = liste de même longueur que headers
- Inclure TOUTES les lignes : libellés, sous-totaux, totaux (TOTAL DE L'ACTIF, etc.)
- Nombres exacts avec espaces séparateurs ("13 157 683")
- Cellule vide → ""
- Si tableau absent : {{"table_name":"{table_name}","headers":[],"rows":[]}}

Contenu de la page :
{text[:16000]}"""

    try:
        from groq import Groq  # type: ignore

        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Tu es un extracteur de tableaux financiers. Réponds uniquement avec du JSON valide.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=8192,
            response_format={"type": "json_object"},  # ← JSON garanti (Groq Structured Outputs)
        )
        content = (completion.choices[0].message.content or "").strip()
        payload = json.loads(content)  # ne peut pas lever JSONDecodeError grâce au mode json_object
        df = _json_payload_to_df(payload)
        result = df if _is_valid_df(df) else pd.DataFrame()
        if not result.empty:
            logger.info(
                "groq_structured_json_extract OK model=%s page=%d shape=%s",
                model, page_num, result.shape,
            )
        return result
    except Exception as e:
        logger.warning(
            "groq_structured_json_extract model=%s page=%d failed: %s", model, page_num, e
        )
        return pd.DataFrame()


def groq_text_reconstruct(
    pdf_path: str,
    page_num: int,
    table_name: str,
    api_key: str,
    model: str = GROQ_TEXT_MODEL,
) -> pd.DataFrame:
    """
    Niveau 3 — Reconstruction Markdown depuis texte pymupdf4llm.
    Fallback quand JSON structuré échoue (ex: tableau très fragmenté).
    """
    text = page_to_markdown(pdf_path, page_num)
    if not text.strip():
        return pd.DataFrame()

    prompt = f"""Le texte ci-dessous est extrait d'une page de rapport financier marocain.
Reconstruis UNIQUEMENT le tableau "{table_name}" au format Markdown.

Règles :
- Première ligne = en-têtes des colonnes
- Deuxième ligne = | --- | --- | ...
- Conserver les montants exacts (espaces séparateurs de milliers)
- Ne renvoie QUE le tableau Markdown, sans commentaire

Texte de la page :
{text[:14000]}"""

    try:
        from groq import Groq  # type: ignore

        client = Groq(api_key=api_key)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Tu extrais des tableaux financiers en Markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=8192,
        )
        content = (completion.choices[0].message.content or "").strip()
        if not content:
            return pd.DataFrame()
        content = _extract_markdown_table(content)
        df = _markdown_table_to_df(content)
        return df if _is_valid_df(df) else pd.DataFrame()
    except Exception as e:
        logger.warning("groq_text_reconstruct model=%s page=%d failed: %s", model, page_num, e)
        return pd.DataFrame()


# ── Stratégies Gemini ─────────────────────────────────────────────────────────

def gemini_vision_extract(
    pdf_path: str,
    page_num: int,
    table_name: str,
    api_key: str,
) -> pd.DataFrame:
    """
    Vision Gemini 2.0 Flash (fallback si Groq vision échoue).
    Gemini 2.0 Flash a un meilleur OCR sur documents scannés que Gemini 1.5.
    """
    import requests  # type: ignore

    img_b64 = page_to_image_b64(pdf_path, page_num)
    if not img_b64:
        return pd.DataFrame()

    prompt = _VISION_PROMPT.format(table_name=table_name)

    for model in GEMINI_MODELS_2025:
        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent"
            )
            payload: Dict = {
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
            resp = requests.post(f"{url}?key={api_key}", json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            content = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            ).strip()
            if content:
                content = _extract_markdown_table(content)
                df = _markdown_table_to_df(content)
                if _is_valid_df(df):
                    logger.info(
                        "gemini_vision_extract OK model=%s page=%d shape=%s",
                        model, page_num, df.shape,
                    )
                    return df
        except Exception as e:
            logger.warning("gemini_vision_extract model=%s page=%d failed: %s", model, page_num, e)
            continue

    return pd.DataFrame()


def gemini_text_extract(
    pdf_path: str,
    page_num: int,
    table_name: str,
    api_key: str,
) -> pd.DataFrame:
    """Extraction texte Gemini 2.0 Flash avec JSON structuré."""
    import requests  # type: ignore

    text = page_to_markdown(pdf_path, page_num)
    if not text.strip():
        return pd.DataFrame()

    prompt = f"""Extrais le tableau '{table_name}' depuis ce rapport financier marocain.
Renvoie un JSON valide :
{{"table_name": "{table_name}", "headers": [...], "rows": [...]}}
Règles : inclure toutes les lignes, nombres exacts avec espaces, cellules vides = "".
Contenu :
{text[:14000]}"""

    for model in GEMINI_MODELS_2025:
        try:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model}:generateContent"
            )
            payload: Dict = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0, "maxOutputTokens": 8192},
            }
            resp = requests.post(f"{url}?key={api_key}", json=payload, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            content = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            ).strip()
            if content:
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end > start:
                    try:
                        payload_json = json.loads(content[start : end + 1])
                        df = _json_payload_to_df(payload_json)
                        if _is_valid_df(df):
                            return df
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            logger.warning("gemini_text_extract model=%s page=%d failed: %s", model, page_num, e)
            continue

    return pd.DataFrame()
