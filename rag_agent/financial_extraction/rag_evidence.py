from __future__ import annotations

from pathlib import Path
from typing import Any

from .utils import normalize_text, write_json


def build_table_evidence_chunks(result: dict[str, Any], *, page_text: str = "") -> list[dict[str, Any]]:
    """Build row-level RAG evidence chunks from one extracted financial table result."""
    columns = [str(col) for col in (result.get("columns") or [])]
    rows = result.get("rows") or []
    if not columns or not rows:
        return []

    chunks: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        label = str(row.get("label") or "").strip()
        values = _row_values(row.get("values"), columns)
        chunk_text = _chunk_text(result, label=label, row_type=str(row.get("row_type") or ""), values=values)
        chunks.append(
            {
                "chunk_id": _chunk_id(result, idx, label),
                "chunk_type": "financial_table_row",
                "text": chunk_text,
                "normalized_text": normalize_text(chunk_text),
                "metadata": {
                    "pdf_path": result.get("pdf_path") or "",
                    "company": result.get("company") or "",
                    "year": result.get("year"),
                    "report_type": result.get("report_type") or "",
                    "sector": result.get("sector") or "",
                    "scope": result.get("scope") or "",
                    "target_table": result.get("target_table") or "",
                    "statement_type": result.get("target_table") or "",
                    "selected_page": result.get("selected_page"),
                    "page_number": result.get("selected_page"),
                    "bbox": result.get("bbox") or [],
                    "crop_path": result.get("crop_path") or "",
                    "row_index": idx,
                    "row_label": label,
                    "row_type": row.get("row_type") or "",
                    "columns": columns,
                    "values": values,
                    "confidence": result.get("confidence"),
                    "validation_status": (result.get("validation") or {}).get("status"),
                    "debug_dir": (result.get("debug") or {}).get("dir", ""),
                },
                "evidence": {
                    "page_text_excerpt": _excerpt_around_label(page_text, label),
                    "candidate_evidence": (result.get("debug") or {}).get("candidate_evidence", []),
                    "retrieval": result.get("benchmark_retrieval") or {},
                    "warnings": result.get("warnings") or [],
                },
            }
        )
    return chunks


def attach_and_write_rag_evidence(
    result: dict[str, Any],
    debug_dir: str | Path,
    *,
    page_text: str = "",
) -> list[dict[str, Any]]:
    """Attach RAG chunks to a result and persist them beside extraction artifacts."""
    chunks = build_table_evidence_chunks(result, page_text=page_text)
    result["rag_evidence_chunks"] = chunks
    result["rag_evidence_path"] = str(Path(debug_dir) / "rag_evidence_chunks.json")
    write_json(result["rag_evidence_path"], chunks)
    return chunks


def write_summary_rag_evidence(summary: dict[str, Any], output_dir: str | Path) -> list[dict[str, Any]]:
    """Persist one combined RAG evidence file for all table results in a run."""
    chunks: list[dict[str, Any]] = []
    for result in summary.get("results") or []:
        if isinstance(result, dict):
            chunks.extend(result.get("rag_evidence_chunks") or [])
    path = Path(output_dir) / "rag_evidence_chunks.json"
    summary["rag_evidence_path"] = str(path)
    summary["rag_evidence_chunk_count"] = len(chunks)
    write_json(path, chunks)
    return chunks


def _row_values(raw_values: Any, columns: list[str]) -> dict[str, str]:
    values = raw_values if isinstance(raw_values, dict) else {}
    return {column: str(values.get(column, "") if values.get(column, "") is not None else "") for column in columns}


def _chunk_text(result: dict[str, Any], *, label: str, row_type: str, values: dict[str, str]) -> str:
    parts = [
        f"Company: {result.get('company') or ''}",
        f"Year: {result.get('year') or ''}",
        f"Scope: {result.get('scope') or ''}",
        f"Statement: {result.get('target_table') or ''}",
        f"Row: {label}",
    ]
    if row_type:
        parts.append(f"Row type: {row_type}")
    parts.extend(f"{column}: {value}" for column, value in values.items())
    return " | ".join(parts)


def _chunk_id(result: dict[str, Any], row_index: int, label: str) -> str:
    parts = [
        result.get("company") or "company",
        str(result.get("year") or "year"),
        result.get("scope") or "scope",
        result.get("target_table") or "table",
        str(result.get("selected_page") or "page"),
        str(row_index),
        label[:48] or "row",
    ]
    return normalize_text("_".join(parts)).replace(" ", "_")


def _excerpt_around_label(page_text: str, label: str, *, radius: int = 420) -> str:
    text = str(page_text or "")
    if not text:
        return ""
    label_norm = normalize_text(label)
    text_norm = normalize_text(text)
    idx = text_norm.find(label_norm) if label_norm else -1
    if idx < 0:
        return text[: radius * 2].strip()
    start = max(0, idx - radius)
    end = min(len(text), idx + len(label) + radius)
    return text[start:end].strip()
