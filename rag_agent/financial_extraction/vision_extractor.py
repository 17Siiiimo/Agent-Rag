from __future__ import annotations

import base64
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .models import TableCandidate
from .utils import write_json


load_dotenv()

JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_GROQ_KEY_CURSOR = 0


def build_vision_prompt(candidate: TableCandidate, company: str, year: int) -> str:
    return f"""
You are an expert system specialized in extracting structured financial tables from Moroccan listed-company reports.

Context:
- Company: {company}
- Year: {year}
- Target table: {candidate.table_type}
- Scope: {candidate.scope}
- Sector: {candidate.sector}

You are given an IMAGE of a financial table (not raw text).

Your goal:
Extract ONLY the requested table and reconstruct it as a structured table.

Step-by-step reasoning (IMPORTANT):
1. Identify the table title.
2. Identify the column headers (years or metrics).
3. Identify each row label.
4. Align each value to the correct column.
5. Reconstruct the table exactly as displayed.

STRICT RULES:

Structure:
- The table must have:
  -> one list of columns
  -> multiple rows
- Each row must contain:
  -> one label
  -> values aligned to columns

Labels:
- Preserve row labels EXACTLY as shown
- If a label spans multiple lines, merge it into one label

Columns:
- Extract ALL columns (e.g., 2024, 2023)
- Do NOT guess or rename columns

Values:
- Preserve values EXACTLY (including spaces, separators, parentheses, "-")
- Preserve signs, spaces, thousand separators, commas, dots, parentheses and dashes.
- Do NOT normalize numbers
- Keep empty cells as ""
- Preserve every visible row from the table, including section headers and rows without numbers.
- If a visible row has no values, keep the row and set all column values to "".
- Never copy the row label into the values object.
- Do not create duplicate label columns.
- The label must appear only in the "label" field.
- Empty cells must remain "".
- A dash "-" is not empty; preserve it as "-".
- If a value is visibly "-", return "-".
- If the cell is visually blank, return "".

Row types:
- Section/header rows without numeric values must use "row_type": "section_header".
- Normal rows must use "row_type": "data".
- Total rows must use "row_type": "total".
- Result rows such as Resultat net must use "row_type": "result".
- Common section headers include PRODUITS D'EXPLOITATION, CHARGES D'EXPLOITATION, PRODUITS FINANCIERS, CHARGES FINANCIERES, PRODUITS NON COURANTS, CHARGES NON COURANTES.

Table boundaries:
- Extract ONLY {candidate.table_type}
- Ignore:
  -> HORS BILAN
  -> NOTES ANNEXES
  -> ETAT DES DEROGATIONS
  -> TABLEAU DE FINANCEMENT
  -> FLUX DE TRESORERIE
  -> any other table

Target-specific anchors:

If {candidate.table_type} == BILAN_ACTIF:
- Look for: ACTIF, Total actif

If {candidate.table_type} == BILAN_PASSIF:
- Look for: PASSIF, Total passif, Capitaux propres

If {candidate.table_type} == CPC:
- Look for: Compte de Produits et Charges, Resultat net

Validation rules:
- The table MUST contain at least one of its anchor rows:
  -> BILAN_ACTIF -> "Total actif"
  -> BILAN_PASSIF -> "Total passif"
  -> CPC -> "Resultat net"

If the anchor is missing:
-> return target_found=false

Failure cases:
- If the image is unclear
- If the table is not readable
- If the structure is inconsistent

Then return:
{{
  "target_found": false,
  "reason": "explain briefly why"
}}

Return STRICT JSON only:

{{
  "target_found": true,
  "table_type": "{candidate.table_type}",
  "title_detected": "",
  "columns": ["<column1>", "<column2>"],
  "rows": [
    {{
      "label": "<exact row label>",
      "row_type": "section_header | data | total | result",
      "values": {{
        "<column1>": "<value>",
        "<column2>": "<value>"
      }}
    }}
  ],
  "confidence": 0.0,
  "warnings": []
}}

Do not return markdown.
Do not return text outside JSON.
""".strip()


def extract_table_with_vision(
    crop_image_path: str,
    candidate: TableCandidate,
    company: str,
    year: int,
    provider: str = "openai",
    model: str | None = None,
    debug_dir: str | Path | None = None,
) -> dict[str, Any]:
    prompt = build_vision_prompt(candidate, company, year)
    provider = provider.lower().strip()
    raw = ""
    try:
        if provider == "openai":
            raw = _call_openai(crop_image_path, prompt, model or "gpt-4o")
        elif provider == "groq":
            raw = _call_groq(crop_image_path, prompt, model or "meta-llama/llama-4-scout-17b-16e-instruct")
        elif provider == "gemini":
            raw = _call_gemini(crop_image_path, prompt, model or "gemini-2.0-flash")
        else:
            raise ValueError(f"Provider vision inconnu: {provider}")
        if debug_dir:
            write_json(Path(debug_dir) / "raw_llm_response.json", {"provider": provider, "model": model, "text": raw})
        parsed = _parse_json(raw)
    except Exception as exc:
        parsed = {
            "target_found": False,
            "table_type": candidate.table_type,
            "title_detected": "",
            "columns": [],
            "rows": [],
            "confidence": 0.0,
            "warnings": [],
            "error": f"vision_extraction_failed: {exc}",
        }
        if debug_dir:
            write_json(
                Path(debug_dir) / "raw_llm_response.json",
                {"provider": provider, "model": model, "text": raw, "error": parsed["error"]},
            )
    parsed.setdefault("target_found", bool(parsed.get("rows")))
    parsed.setdefault("table_type", candidate.table_type)
    parsed.setdefault("title_detected", "")
    parsed.setdefault("columns", [])
    parsed.setdefault("rows", [])
    parsed.setdefault("confidence", 0.0)
    parsed.setdefault("warnings", [])
    parsed = normalize_extracted_table(parsed)
    if debug_dir:
        write_json(Path(debug_dir) / "extracted_table.json", parsed)
        (Path(debug_dir) / "extracted_table.md").write_text(table_to_markdown(parsed), encoding="utf-8")
    return parsed


def normalize_extracted_table(extracted: dict[str, Any]) -> dict[str, Any]:
    columns = [str(c) for c in (extracted.get("columns") or [])]
    table_type = str(extracted.get("table_type") or "")
    label_columns = {c for c in columns if _is_label_column(c, table_type)}
    columns = [c for c in columns if c not in label_columns]
    rows: list[dict[str, Any]] = []
    for raw_row in extracted.get("rows") or []:
        if not isinstance(raw_row, dict):
            continue
        values = raw_row.get("values") or {}
        label = str(raw_row.get("label") or "")
        if not label:
            for col in label_columns:
                if values.get(col):
                    label = str(values.get(col))
                    break
        cleaned_values = {col: _clean_cell(values.get(col, "")) for col in columns}
        row_type = str(raw_row.get("row_type") or "").strip()
        if row_type not in {"section_header", "data", "total", "result"}:
            row_type = _infer_row_type(label, cleaned_values)
        rows.append(
            {
                "label": label,
                "row_type": row_type,
                "values": cleaned_values,
            }
        )
    extracted["columns"] = columns
    extracted["rows"] = rows
    return extracted


def table_to_markdown(extracted: dict[str, Any]) -> str:
    extracted = normalize_extracted_table(dict(extracted))
    columns = [str(c) for c in (extracted.get("columns") or [])]
    rows = extracted.get("rows") or []
    if not columns:
        return "_No columns extracted._\n"

    headers = ["label", *columns]
    lines = [
        "| " + " | ".join(_md_cell(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = row.get("values") or {}
        line = [_md_cell(row.get("label", ""))]
        line.extend(_md_cell(values.get(col, "")) for col in columns)
        lines.append("| " + " | ".join(line) + " |")
    return "\n".join(lines) + "\n"


def _md_cell(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ").strip()


def _is_label_column(column: str, table_type: str = "") -> bool:
    normalized = str(column or "").strip().casefold()
    table_title_columns = set()
    if table_type == "BILAN_ACTIF":
        table_title_columns.add("actif")
    elif table_type == "BILAN_PASSIF":
        table_title_columns.add("passif")
    elif table_type == "CPC":
        table_title_columns.update({"cpc", "compte de produits et charges"})
    return normalized in {"label", "libelle", "libellé", "nature", "rubrique", "poste", "", *table_title_columns}


def _clean_cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _infer_row_type(label: str, values: dict[str, str]) -> str:
    norm = _ascii_lower(label)
    non_empty_values = [v for v in values.values() if str(v).strip() != ""]
    if "resultat net" in norm:
        return "result"
    if norm.startswith("total") or " total " in f" {norm} ":
        return "total"
    section_markers = [
        "produits d'exploitation",
        "charges d'exploitation",
        "produits financiers",
        "charges financieres",
        "produits non courants",
        "charges non courantes",
        "produits d exploitation",
        "charges d exploitation",
    ]
    if not non_empty_values or any(marker in norm for marker in section_markers):
        return "section_header"
    return "data"


def _ascii_lower(text: str) -> str:
    replacements = str.maketrans({"é": "e", "è": "e", "ê": "e", "à": "a", "â": "a", "î": "i", "ï": "i", "ç": "c", "É": "e", "È": "e"})
    return str(text or "").translate(replacements).casefold()


def _image_b64(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _parse_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = JSON_RE.search(text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        balanced = _first_balanced_json_object(text)
        if balanced:
            return json.loads(balanced)
        raise


def _first_balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _call_openai(image_path: str, prompt: str, model: str) -> str:
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    client = OpenAI(api_key=key)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_image_b64(image_path)}"}},
                ],
            }
        ],
        temperature=0,
        max_tokens=8192,
    )
    return completion.choices[0].message.content or ""


def _call_groq(image_path: str, prompt: str, model: str) -> str:
    import requests

    keys = _groq_api_keys()
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_image_b64(image_path)}"}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 8192,
    }
    url = "https://api.groq.com/openai/v1/chat/completions"
    delays = [8, 20, 45]
    last_error: Exception | None = None
    rotated_keys = _rotate_groq_keys(keys)
    for attempt in range(len(delays) + 1):
        exhausted_this_round = True
        for key_index, key in enumerate(rotated_keys, start=1):
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                last_error = RuntimeError(f"{_provider_http_error('Groq', resp)} key_index={key_index}")
                continue
            if resp.status_code in {401, 403} and len(rotated_keys) > 1:
                last_error = RuntimeError(f"{_provider_http_error('Groq', resp)} key_index={key_index}")
                continue
            try:
                resp.raise_for_status()
            except requests.HTTPError as exc:
                raise RuntimeError(_provider_http_error("Groq", resp)) from exc
            return resp.json()["choices"][0]["message"]["content"] or ""

            exhausted_this_round = False

        if attempt >= len(delays):
            break
        retry_after = delays[attempt]
        if "resp" in locals():
            retry_after = _retry_after_seconds(resp, delays[attempt])
        time.sleep(retry_after)

    raise RuntimeError(
        f"Groq rate limit reached after retries across {len(rotated_keys)} configured key(s). "
        "Please wait a minute, reduce parallel table requests, add more Groq keys, or choose Gemini/OpenAI."
    ) from last_error


def _groq_api_keys() -> list[str]:
    keys: list[str] = []
    for raw_key in (os.environ.get("GROQ_API_KEYS") or "").split(","):
        key = raw_key.strip()
        if key and key not in keys:
            keys.append(key)
    single_key = (os.environ.get("GROQ_API_KEY") or "").strip()
    if single_key and single_key not in keys:
        keys.append(single_key)
    if not keys:
        raise RuntimeError("GROQ_API_KEY or GROQ_API_KEYS is not configured")
    return keys


def _rotate_groq_keys(keys: list[str]) -> list[str]:
    global _GROQ_KEY_CURSOR
    if len(keys) <= 1:
        return keys
    start = _GROQ_KEY_CURSOR % len(keys)
    _GROQ_KEY_CURSOR += 1
    return keys[start:] + keys[:start]


def _call_gemini(image_path: str, prompt: str, model: str) -> str:
    import requests

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not configured")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": "image/png", "data": _image_b64(image_path)}},
                ]
            }
        ],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 8192},
    }
    resp = requests.post(url, json=payload, timeout=120)
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(_provider_http_error("Gemini", resp)) from exc
    parts = resp.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(part.get("text", "") for part in parts)


def _retry_after_seconds(resp, default: int) -> int:
    raw = resp.headers.get("Retry-After")
    if raw:
        try:
            return max(1, min(90, int(float(raw))))
        except ValueError:
            pass
    return default


def _provider_http_error(provider: str, resp) -> str:
    if resp.status_code == 429:
        return (
            f"{provider} rate limit (429 Too Many Requests). "
            "The PDF and crop are OK; wait a little or choose another provider."
        )
    detail = ""
    try:
        detail = resp.json().get("error", {}).get("message", "")
    except Exception:
        detail = (resp.text or "")[:300]
    return f"{provider} API error {resp.status_code}: {detail}".strip()
