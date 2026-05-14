from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_agent.financial_extraction.document_classifier import detect_sector
from rag_agent.financial_extraction.hybrid_retriever import retrieve_candidate_pages
from rag_agent.financial_extraction.models import PageChunk, PipelineConfig
from rag_agent.financial_extraction.page_indexer import (
    build_page_chunks,
    load_or_build_page_index,
)
from rag_agent.financial_extraction.page_renderer import render_pdf_pages
from rag_agent.financial_extraction.page_text_extractor import extract_page_texts
from rag_agent.financial_extraction.pdf_loader import load_pdf
from rag_agent.financial_extraction.table_cropper import crop_table_image
from rag_agent.financial_extraction.table_localizer import localize_table_candidates
from rag_agent.financial_extraction.utils import ensure_dir, slugify, write_json


DEFAULT_CASES = ROOT / "data" / "financial_benchmark_cases.json"
DEFAULT_OUTPUT = ROOT / "output" / "financial_extraction_debug" / "benchmark"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the financial RAG/crop benchmark without Vision calls.")
    parser.add_argument("--cases", default=str(DEFAULT_CASES), help="JSON file containing golden benchmark cases.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output directory for benchmark artifacts.")
    parser.add_argument("--dpi", type=int, default=120, help="Rendering DPI used for crop validation.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-localize-pages", type=int, default=4)
    parser.add_argument("--pdf-filter", default="", help="Run only cases whose pdf_path contains this text.")
    parser.add_argument("--id-filter", default="", help="Run only cases whose id contains this text.")
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N filtered cases.")
    args = parser.parse_args()

    cases = _load_cases(Path(args.cases))
    cases = _filter_cases(cases, pdf_filter=args.pdf_filter, id_filter=args.id_filter, limit=args.limit)
    if not cases:
        print("No benchmark cases matched the filters.")
        return 1

    out_root = ensure_dir(Path(args.output))
    cfg = PipelineConfig(
        output_dir=out_root,
        dpi=args.dpi,
        top_k_pages=args.top_k,
        candidate_pages_for_localization=args.max_localize_pages,
        use_vision=False,
        cleanup_rendered_pages=False,
    )

    grouped = _group_cases(cases)
    summary: list[dict[str, Any]] = []
    for group_key, group_cases in grouped.items():
        print(f"\nPDF: {group_key}")
        context = _prepare_pdf_context(group_cases[0], cfg, out_root)
        if context.get("error"):
            for case in group_cases:
                row = _failure_row(case, context["error"])
                summary.append(row)
                print(f"  FAIL {case['id']}: {context['error']}")
            continue

        for case in group_cases:
            row = _run_case(context, case, cfg)
            summary.append(row)
            status = "PASS" if row["passed"] else "FAIL"
            print(
                f"  {status} {case['id']}: expected={row['expected_page']} "
                f"selected={row['selected_page']} score={row['retrieval_score']:.3f} "
                f"crop={row['crop_path'] or '-'}"
            )

    report = _build_report(summary)
    write_json(out_root / "summary.json", report)
    _write_csv(out_root / "summary.csv", summary)
    _write_markdown(out_root / "summary.md", report)

    print(f"\nBenchmark: {report['passed_cases']}/{report['total_cases']} passed")
    print(f"JSON: {out_root / 'summary.json'}")
    print(f"CSV:  {out_root / 'summary.csv'}")
    print(f"MD:   {out_root / 'summary.md'}")
    return 0 if report["failed_cases"] == 0 else 2


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases", payload if isinstance(payload, list) else [])
    if not isinstance(cases, list):
        raise ValueError(f"Invalid benchmark cases file: {path}")
    return [case for case in cases if isinstance(case, dict)]


def _filter_cases(cases: list[dict[str, Any]], *, pdf_filter: str, id_filter: str, limit: int) -> list[dict[str, Any]]:
    pdf_filter = pdf_filter.lower().strip()
    id_filter = id_filter.lower().strip()
    filtered = []
    for case in cases:
        if pdf_filter and pdf_filter not in str(case.get("pdf_path", "")).lower():
            continue
        if id_filter and id_filter not in str(case.get("id", "")).lower():
            continue
        filtered.append(case)
    return filtered[:limit] if limit and limit > 0 else filtered


def _group_cases(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[str(case.get("pdf_path", ""))].append(case)
    return dict(grouped)


def _prepare_pdf_context(case: dict[str, Any], cfg: PipelineConfig, out_root: Path) -> dict[str, Any]:
    pdf_path = _resolve_path(str(case["pdf_path"]))
    if not pdf_path.is_file():
        return {"error": f"pdf not found: {pdf_path}"}

    company = str(case["company"])
    year = int(case["year"])
    sector = str(case["sector"])
    report_type = str(case.get("report_type") or "")
    pdf_slug = slugify(pdf_path.stem)
    pdf_out = ensure_dir(out_root / pdf_slug)

    try:
        pdf = load_pdf(str(pdf_path))
        page_chunks_path = pdf_out / "page_chunks.json"
        rendered_dir = ensure_dir(pdf_out / "_rendered_pages")
        page_texts = extract_page_texts(pdf.pdf_path)
        detected_sector = detect_sector(page_texts, sector)
        image_paths = _load_rendered_images(rendered_dir)
        if not image_paths:
            image_paths = render_pdf_pages(pdf.pdf_path, rendered_dir, dpi=cfg.dpi)

        pages = _load_page_chunks(page_chunks_path, pdf.pdf_id, company, year, detected_sector)
        if pages is None:
            pages = build_page_chunks(
                pdf_id=pdf.pdf_id,
                company=company,
                year=year,
                sector=detected_sector,
                page_texts=page_texts,
                image_paths=image_paths,
            )
            write_json(page_chunks_path, [p.to_dict() for p in pages])
        else:
            for page in pages:
                if page.page_number in image_paths:
                    page.image_path = image_paths[page.page_number]
            write_json(page_chunks_path, [p.to_dict() for p in pages])

        index, index_cache_used = load_or_build_page_index(pages, cfg, pdf_out)
        return {
            "pdf": pdf,
            "pdf_path": pdf_path,
            "pdf_out": pdf_out,
            "company": company,
            "year": year,
            "report_type": report_type,
            "sector": detected_sector,
            "page_texts": page_texts,
            "image_paths": image_paths,
            "pages": pages,
            "index": index,
            "index_cache_used": index_cache_used,
        }
    except Exception as exc:
        return {"error": str(exc)}


def _run_case(context: dict[str, Any], case: dict[str, Any], cfg: PipelineConfig) -> dict[str, Any]:
    pdf = context["pdf"]
    target_table = str(case["target_table"])
    scope = str(case["scope"])
    sector = str(case["sector"])
    expected_page = int(case["expected_page"])
    case_dir = ensure_dir(context["pdf_out"] / scope / target_table)

    retrieved = retrieve_candidate_pages(context["index"], target_table, scope, sector, cfg)
    retrieval_payload = [r.to_dict() for r in retrieved]
    write_json(case_dir / "retrieval_candidates.json", retrieval_payload)
    write_json(case_dir / "score_breakdown.json", retrieval_payload)

    candidates = localize_table_candidates(
        pdf.pdf_path,
        retrieved,
        target_table,
        scope,
        sector,
        max_pages=cfg.candidate_pages_for_localization,
        debug_dir=case_dir,
    )
    selected = candidates[0] if candidates else None
    selected_retrieved = (
        next((r for r in retrieved if selected and r.page.page_number == selected.page_number), None)
        if selected
        else None
    )

    selected_page = selected.page_number if selected else (retrieved[0].page.page_number if retrieved else None)
    retrieval_score = float(selected_retrieved.score if selected_retrieved else (retrieved[0].score if retrieved else 0.0))
    crop_path = ""
    bbox: list[float] = []
    confidence = 0.0
    warnings: list[str] = []

    if selected:
        page_size = _page_size(pdf.pdf_path, selected.page_number)
        source_image = context["image_paths"].get(selected.page_number)
        if source_image:
            generated_crop = crop_table_image(selected, source_image, case_dir / "crops", page_size)
            crop_path = str(case_dir / "crop.png")
            shutil.copy2(generated_crop, crop_path)
        else:
            warnings.append("selected page image missing")
        bbox = selected.bbox
        confidence = float(selected.confidence)
    else:
        warnings.append("no table candidate selected")

    selected_payload = {
        "case_id": case["id"],
        "target_table": target_table,
        "scope": scope,
        "sector": sector,
        "expected_page": expected_page,
        "selected_page": selected_page,
        "page_match": selected_page == expected_page,
        "retrieval": selected_retrieved.to_dict() if selected_retrieved else (retrieved[0].to_dict() if retrieved else None),
        "table_candidate": selected.to_dict() if selected else None,
        "crop_path": crop_path,
        "warnings": warnings,
    }
    write_json(case_dir / "selected_page.json", selected_payload)
    write_json(case_dir / "crop_metadata.json", selected_payload)

    return {
        "id": case["id"],
        "pdf_path": str(context["pdf_path"]),
        "company": context["company"],
        "year": context["year"],
        "report_type": context["report_type"],
        "target_table": target_table,
        "scope": scope,
        "sector": sector,
        "expected_page": expected_page,
        "selected_page": selected_page,
        "passed": selected_page == expected_page and bool(crop_path),
        "page_match": selected_page == expected_page,
        "crop_exists": bool(crop_path) and Path(crop_path).is_file(),
        "crop_path": crop_path,
        "bbox": bbox,
        "confidence": confidence,
        "retrieval_score": retrieval_score,
        "warnings": warnings,
        "case_dir": str(case_dir),
    }


def _failure_row(case: dict[str, Any], error: str) -> dict[str, Any]:
    return {
        "id": case.get("id", ""),
        "pdf_path": case.get("pdf_path", ""),
        "company": case.get("company", ""),
        "year": case.get("year", ""),
        "report_type": case.get("report_type", ""),
        "target_table": case.get("target_table", ""),
        "scope": case.get("scope", ""),
        "sector": case.get("sector", ""),
        "expected_page": case.get("expected_page"),
        "selected_page": None,
        "passed": False,
        "page_match": False,
        "crop_exists": False,
        "crop_path": "",
        "bbox": [],
        "confidence": 0.0,
        "retrieval_score": 0.0,
        "warnings": [error],
        "case_dir": "",
    }


def _build_report(summary: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(summary)
    passed = sum(1 for row in summary if row.get("passed"))
    failed_rows = [row for row in summary if not row.get("passed")]
    return {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "accuracy": round(passed / total, 4) if total else 0.0,
        "failed": failed_rows,
        "results": summary,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "id",
        "pdf_path",
        "target_table",
        "scope",
        "sector",
        "expected_page",
        "selected_page",
        "passed",
        "retrieval_score",
        "confidence",
        "crop_path",
        "warnings",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            item = {key: row.get(key, "") for key in columns}
            item["warnings"] = "; ".join(row.get("warnings") or [])
            writer.writerow(item)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Financial Crop Benchmark",
        "",
        f"- Total: {report['total_cases']}",
        f"- Passed: {report['passed_cases']}",
        f"- Failed: {report['failed_cases']}",
        f"- Accuracy: {report['accuracy']:.2%}",
        "",
        "| Status | Case | Expected | Selected | Score | Crop |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in report["results"]:
        status = "PASS" if row.get("passed") else "FAIL"
        lines.append(
            f"| {status} | {row.get('id')} | {row.get('expected_page')} | "
            f"{row.get('selected_page')} | {float(row.get('retrieval_score') or 0):.3f} | "
            f"{row.get('crop_path') or ''} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return ROOT / candidate


def _load_rendered_images(rendered_dir: Path) -> dict[int, str]:
    images: dict[int, str] = {}
    for path in rendered_dir.glob("page_*.png"):
        try:
            page_number = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        images[page_number] = str(path)
    return images


def _load_page_chunks(path: Path, pdf_id: str, company: str, year: int, sector: str) -> list[PageChunk] | None:
    if not path.is_file():
        return None
    try:
        pages = [PageChunk(**item) for item in json.loads(path.read_text(encoding="utf-8"))]
    except Exception:
        return None
    if not pages:
        return None
    if any(page.pdf_id != pdf_id or page.company != company or int(page.year) != int(year) or page.sector != sector for page in pages):
        return None
    return pages


def _page_size(pdf_path: str, page_number: int) -> tuple[float, float]:
    with fitz.open(pdf_path) as doc:
        page = doc[page_number - 1]
        rect = page.rect
        return float(rect.width), float(rect.height)


if __name__ == "__main__":
    raise SystemExit(main())
