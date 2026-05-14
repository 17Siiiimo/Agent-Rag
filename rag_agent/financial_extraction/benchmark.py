from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from .utils import slugify


ValidationResult = Literal["PASS", "FAIL"]
FailureReason = Literal[
    "WRONG_PAGE",
    "WRONG_SCOPE",
    "WRONG_TABLE",
    "BAD_CROP",
    "PARTIAL_TABLE",
    "MISSING_ROWS",
    "OCR_ERROR",
    "OTHER",
]

FAILURE_REASONS: set[str] = {
    "WRONG_PAGE",
    "WRONG_SCOPE",
    "WRONG_TABLE",
    "BAD_CROP",
    "PARTIAL_TABLE",
    "MISSING_ROWS",
    "OCR_ERROR",
    "OTHER",
}


@dataclass(frozen=True)
class FinancialBenchmarkEntry:
    id: str
    pdf_name: str
    pdf_path: str
    company: str
    year: int
    report_type: str
    scope: str
    sector: str
    target_table: str
    expected_page: int | None
    predicted_page: int | None
    top_k_pages: list[int]
    retrieval_scores: dict[str, float | None]
    crop_path: str
    bbox: list[float]
    validation_result: ValidationResult
    validation_type: str
    validated_at: str
    retrieval_latency_ms: int | None
    source_job_id: str
    failure_reason: FailureReason | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "pdf_name": self.pdf_name,
            "pdf_path": self.pdf_path,
            "company": self.company,
            "year": self.year,
            "report_type": self.report_type,
            "scope": self.scope,
            "sector": self.sector,
            "target_table": self.target_table,
            "expected_page": self.expected_page,
            "predicted_page": self.predicted_page,
        }
        if self.failure_reason:
            payload["failure_reason"] = self.failure_reason
        payload.update(
            {
                "top_k_pages": self.top_k_pages,
                "retrieval_scores": self.retrieval_scores,
                "crop_path": self.crop_path,
                "bbox": self.bbox,
                "validation_result": self.validation_result,
                "validation_type": self.validation_type,
                "validated_at": self.validated_at,
                "retrieval_latency_ms": self.retrieval_latency_ms,
                "source_job_id": self.source_job_id,
            }
        )
        return payload


def build_financial_benchmark_entry(
    *,
    source_job_id: str,
    job: dict[str, Any],
    validation_result: ValidationResult,
    failure_reason: str | None = None,
) -> dict[str, Any]:
    validation_result = validation_result.upper()  # type: ignore[assignment]
    if validation_result not in {"PASS", "FAIL"}:
        raise ValueError("validation_result must be PASS or FAIL")

    normalized_failure_reason = _normalize_failure_reason(failure_reason)
    if validation_result == "PASS":
        normalized_failure_reason = None

    predicted_page = _optional_int(job.get("predicted_page") or job.get("page_num") or job.get("selected_page"))
    entry = FinancialBenchmarkEntry(
        id=_entry_id(job),
        pdf_name=Path(str(job.get("pdf_path") or "")).name,
        pdf_path=str(job.get("pdf_path") or ""),
        company=str(job.get("company") or ""),
        year=int(job.get("year") or 0),
        report_type=str(job.get("report_type") or job.get("type_rapport_used") or ""),
        scope=str(job.get("scope") or ""),
        sector=str(job.get("sector") or ""),
        target_table=str(job.get("target_table") or ""),
        expected_page=predicted_page if validation_result == "PASS" else _optional_int(job.get("expected_page")),
        predicted_page=predicted_page,
        top_k_pages=_int_list(job.get("top_k_pages")),
        retrieval_scores=_retrieval_scores(job.get("retrieval_scores")),
        crop_path=str(job.get("crop_path") or ""),
        bbox=_float_list(job.get("bbox")),
        validation_result=validation_result,  # type: ignore[arg-type]
        validation_type="human",
        validated_at=datetime.now().isoformat(timespec="seconds"),
        retrieval_latency_ms=_optional_int(job.get("retrieval_latency_ms")),
        source_job_id=source_job_id,
        failure_reason=normalized_failure_reason,  # type: ignore[arg-type]
    )
    return entry.to_dict()


def save_financial_benchmark_entry(path: Path, collection_key: str, entry: dict[str, Any]) -> Path:
    payload = _read_json(path, default={"version": 1, collection_key: []})
    if not isinstance(payload, dict):
        payload = {"version": 1, collection_key: []}
    entries = payload.setdefault(collection_key, [])
    if not isinstance(entries, list):
        entries = []
        payload[collection_key] = entries

    source_job_id = entry.get("source_job_id")
    entry_id = entry.get("id")
    for idx, existing in enumerate(entries):
        if not isinstance(existing, dict):
            continue
        same_job = source_job_id and existing.get("source_job_id") == source_job_id
        same_case = entry_id and existing.get("id") == entry_id
        if same_job or same_case:
            entries[idx] = entry
            break
    else:
        entries.append(entry)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _entry_id(job: dict[str, Any]) -> str:
    company = str(job.get("emetteur") or job.get("company") or "")
    report_type = str(job.get("report_type") or job.get("type_rapport_used") or "")
    parts = [
        slugify(company),
        str(job.get("year") or ""),
        slugify(report_type),
        slugify(str(job.get("scope") or "")),
        slugify(str(job.get("target_table") or "")),
    ]
    return "_".join(part for part in parts if part)


def _retrieval_scores(raw: Any) -> dict[str, float | None]:
    raw = raw if isinstance(raw, dict) else {}
    return {
        "bm25": _optional_float(raw.get("bm25")),
        "vector": _optional_float(raw.get("vector")),
        "anchor": _optional_float(raw.get("anchor")),
        "scope": _optional_float(raw.get("scope")),
        "title": _optional_float(raw.get("title")),
        "signature": _optional_float(raw.get("signature")),
        "negative_penalty": _optional_float(raw.get("negative_penalty")),
        "final_score": _optional_float(raw.get("final_score")),
    }


def _normalize_failure_reason(value: str | None) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    normalized = str(value).strip().upper()
    if normalized not in FAILURE_REASONS:
        raise ValueError(f"failure_reason must be one of: {', '.join(sorted(FAILURE_REASONS))}")
    return normalized


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        converted = _optional_int(item)
        if converted is not None:
            result.append(converted)
    return result


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    result: list[float] = []
    for item in value:
        converted = _optional_float(item)
        if converted is not None:
            result.append(converted)
    return result


def _read_json(path: Path, default: Any) -> Any:
    if not path.is_file() or path.stat().st_size == 0:
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
