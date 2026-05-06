from __future__ import annotations

from typing import Any

from .keyword_dictionary import NEGATIVE_ANCHORS
from .models import PipelineConfig, TableCandidate
from .utils import normalize_text, write_json


def validate_extracted_table(
    extracted: dict[str, Any],
    candidate: TableCandidate,
    page_text: str,
    cfg: PipelineConfig,
    debug_path: str | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = list(extracted.get("warnings") or [])
    columns = extracted.get("columns") or []
    rows = extracted.get("rows") or []
    confidence = float(extracted.get("confidence") or 0.0)

    if not columns:
        issues.append("missing_columns")
    if not rows:
        issues.append("missing_rows")
    if len(rows) < cfg.min_validation_rows:
        issues.append("too_few_rows")
    if confidence < 0.45:
        warnings.append("low_vision_confidence")

    extracted_text = normalize_text(
        " ".join(
            [
                str(row.get("label", ""))
                for row in rows
                if isinstance(row, dict)
            ]
        )
    )
    for neg in NEGATIVE_ANCHORS.get(candidate.table_type, []):
        if normalize_text(neg) in extracted_text and neg == "hors bilan":
            warnings.append("extracted_rows_mention_hors_bilan_verify_crop")

    status = "approved"
    if issues:
        status = "rejected"
    elif warnings:
        status = "warning"

    report = {"status": status, "issues": issues, "warnings": sorted(set(warnings))}
    if debug_path:
        write_json(debug_path, report)
    return report
