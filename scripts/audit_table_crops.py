from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import fitz  # PyMuPDF
from PIL import Image, ImageDraw

from rag_agent.financial_extraction.document_classifier import detect_sector
from rag_agent.financial_extraction.hybrid_retriever import retrieve_candidate_pages
from rag_agent.financial_extraction.models import PipelineConfig, TableCandidate
from rag_agent.financial_extraction.page_indexer import PageAwareIndex, build_page_chunks, build_page_index
from rag_agent.financial_extraction.page_renderer import render_pdf_pages
from rag_agent.financial_extraction.page_text_extractor import extract_page_texts
from rag_agent.financial_extraction.pdf_loader import load_pdf
from rag_agent.financial_extraction.table_cropper import crop_table_image
from rag_agent.financial_extraction.table_localizer import localize_table_candidates
from rag_agent.financial_extraction.utils import ensure_dir, write_json
from rag_agent.financial_extraction.utils import normalize_text


PDF_SOURCES = [
    {
        "name": "awb_annual_2024",
        "company": "Attijariwafa Bank",
        "year": 2024,
        "path": ROOT / "data" / "pdf" / "AWB_RFA_2024_1.pdf",
    },
    {
        "name": "awb_s1_2024",
        "company": "Attijariwafa Bank",
        "year": 2024,
        "path": ROOT / "Émetteur" / "attijariwafa_bank_2024_s1.pdf",
    },
]

CASES = [
    ("BILAN_ACTIF", "comptes_consolides", "bancaire_sdf"),
    ("BILAN_PASSIF", "comptes_consolides", "bancaire_sdf"),
    ("CPC", "comptes_consolides", "bancaire_sdf"),
    ("BILAN_ACTIF", "comptes_sociaux", "bancaire_sdf"),
    ("BILAN_PASSIF", "comptes_sociaux", "bancaire_sdf"),
    ("CPC", "comptes_sociaux", "bancaire_sdf"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit table crops for AWB annual and S1 PDFs without Vision.")
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-localize-pages", type=int, default=4)
    args = parser.parse_args()

    cfg = PipelineConfig(
        dpi=args.dpi,
        top_k_pages=args.top_k,
        candidate_pages_for_localization=args.max_localize_pages,
        use_vision=False,
    )
    root_out = ensure_dir(cfg.output_dir / "crop_audit")
    summary: list[dict[str, Any]] = []

    for source in PDF_SOURCES:
        pdf_path = Path(source["path"])
        pdf_out = ensure_dir(root_out / source["name"])
        print(f"\nPDF: {source['name']} -> {pdf_path}")
        if not pdf_path.is_file():
            warning = f"pdf not found: {pdf_path}"
            print(f"  {warning}")
            for target_table, scope, sector in CASES:
                entry = _summary_entry(source["name"], target_table, scope, sector)
                entry.update({"target_found": False, "reason": warning, "warnings": [warning]})
                summary.append(entry)
            continue

        context = _prepare_pdf_context(pdf_path, source, cfg, pdf_out)
        for target_table, scope, sector in CASES:
            case_name = f"{scope}_{target_table}"
            case_dir = ensure_dir(pdf_out / case_name)
            print(f"  CASE {case_name}")
            entry = _run_case(context, case_dir, source["name"], target_table, scope, sector, cfg)
            summary.append(entry)
            print(
                f"    found={entry['target_found']} page={entry['selected_page']} "
                f"conf={entry['confidence']:.3f} crop={entry['crop_path'] or '-'}"
            )

    write_json(root_out / "summary.json", summary)
    print(f"\nSummary: {root_out / 'summary.json'}")
    return 0


def _prepare_pdf_context(pdf_path: Path, source: dict[str, Any], cfg: PipelineConfig, pdf_out: Path) -> dict[str, Any]:
    pdf = load_pdf(str(pdf_path))
    rendered_dir = ensure_dir(pdf_out / "_rendered_pages")
    page_texts = extract_page_texts(pdf.pdf_path)
    sector = detect_sector(page_texts, "bancaire_sdf")
    image_paths = render_pdf_pages(pdf.pdf_path, rendered_dir, dpi=cfg.dpi)
    pages = build_page_chunks(
        pdf_id=pdf.pdf_id,
        company=source["company"],
        year=source["year"],
        sector=sector,
        page_texts=page_texts,
        image_paths=image_paths,
    )
    index = build_page_index(pages, cfg, debug_dir=pdf_out)
    return {
        "pdf": pdf,
        "source": source,
        "page_texts": page_texts,
        "image_paths": image_paths,
        "index": index,
    }


def _run_case(
    context: dict[str, Any],
    case_dir: Path,
    pdf_name: str,
    target_table: str,
    scope: str,
    sector: str,
    cfg: PipelineConfig,
) -> dict[str, Any]:
    pdf = context["pdf"]
    index: PageAwareIndex = context["index"]
    image_paths: dict[int, str] = context["image_paths"]

    retrieved = retrieve_candidate_pages(index, target_table, scope, sector, cfg)
    write_json(case_dir / "retrieval_candidates.json", [r.to_dict() for r in retrieved])

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

    found, reason, warnings = _target_found(selected, selected_retrieved)
    selected_page = selected.page_number if selected else (retrieved[0].page.page_number if retrieved else None)
    full_page_path = ""
    all_candidates_path = ""
    selected_bbox_path = ""
    crop_path = ""
    bbox: list[float] = []
    confidence = 0.0
    needs_combined = False

    if selected_page and selected_page in image_paths:
        full_page_path = str(case_dir / "full_page.png")
        shutil.copy2(image_paths[selected_page], full_page_path)
        page_size = _page_size(pdf.pdf_path, selected_page)
        same_page_candidates = [c for c in candidates if c.page_number == selected_page]
        all_candidates_path = str(case_dir / "full_page_with_all_candidates.png")
        _draw_bboxes(image_paths[selected_page], same_page_candidates, page_size, all_candidates_path)

        if selected:
            selected_bbox_path = str(case_dir / "selected_bbox.png")
            _draw_bboxes(image_paths[selected_page], [selected], page_size, selected_bbox_path, selected_only=True)
            bbox = selected.bbox
            confidence = float(selected.confidence)
            needs_combined = _needs_target_extraction_from_combined_crop(selected, selected_retrieved)
            if found:
                generated_crop = crop_table_image(selected, image_paths[selected.page_number], case_dir, page_size)
                crop_path = str(case_dir / "crop.png")
                shutil.copy2(generated_crop, crop_path)

    metadata = {
        "pdf_name": pdf_name,
        "target_table": target_table,
        "scope": scope,
        "sector": sector,
        "selected_page": selected_page,
        "target_found": found,
        "crop_path": crop_path,
        "bbox": bbox,
        "confidence": confidence,
        "needs_target_extraction_from_combined_crop": needs_combined,
        "reason": reason,
        "warnings": warnings,
        "full_page_path": full_page_path,
        "full_page_with_all_candidates_path": all_candidates_path,
        "selected_bbox_path": selected_bbox_path,
        "selected_candidate": selected.to_dict() if selected else None,
        "selected_retrieval": selected_retrieved.to_dict() if selected_retrieved else None,
    }
    write_json(case_dir / "crop_metadata.json", metadata)
    return metadata


def _target_found(selected: TableCandidate | None, retrieved) -> tuple[bool, str, list[str]]:
    warnings: list[str] = []
    if selected is None or retrieved is None:
        return False, "scope/table not detected", warnings
    if "fallback:page_region" in selected.evidence:
        return False, "scope/table not detected", ["only fallback page region was localized"]
    if retrieved.scope_score <= 0:
        return False, "scope/table not detected", ["requested scope was not detected on selected page"]
    if retrieved.target_anchor_score <= 0 and retrieved.title_score <= 0:
        return False, "scope/table not detected", ["target anchors/title were not detected on selected page"]
    page_text = normalize_text(retrieved.page.page_text)
    if selected.table_type == "CPC" and "flux de tresorerie" in page_text:
        return False, "scope/table not detected", ["cash-flow page, not main CPC"]
    if selected.confidence < 0.45:
        return False, "scope/table not detected", ["low localization confidence"]
    return True, "", warnings


def _needs_target_extraction_from_combined_crop(selected: TableCandidate, retrieved) -> bool:
    if retrieved is None:
        return False
    target_count = len(retrieved.page.target_candidates)
    return target_count >= 2 and selected.table_type in retrieved.page.target_candidates


def _draw_bboxes(
    image_path: str,
    candidates: list[TableCandidate],
    page_size: tuple[float, float],
    output_path: str,
    selected_only: bool = False,
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    pdf_width, pdf_height = page_size
    sx = img.width / max(pdf_width, 1.0)
    sy = img.height / max(pdf_height, 1.0)

    colors = {
        "BILAN_ACTIF": "lime",
        "BILAN_PASSIF": "dodgerblue",
        "CPC": "red",
    }
    for idx, candidate in enumerate(candidates, start=1):
        x1, y1, x2, y2 = candidate.bbox
        box = [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]
        color = "yellow" if selected_only else colors.get(candidate.table_type, "orange")
        width = 6 if selected_only else 4
        draw.rectangle(box, outline=color, width=width)
        label = f"{idx}:{candidate.table_type} {candidate.confidence:.2f}"
        draw.rectangle([box[0], max(0, box[1] - 18), box[0] + 260, box[1]], fill="black")
        draw.text((box[0] + 4, max(0, box[1] - 16)), label, fill=color)

    img.save(output_path)


def _page_size(pdf_path: str, page_number: int) -> tuple[float, float]:
    doc = fitz.open(pdf_path)
    try:
        rect = doc[page_number - 1].rect
        return float(rect.width), float(rect.height)
    finally:
        doc.close()


def _summary_entry(pdf_name: str, target_table: str, scope: str, sector: str) -> dict[str, Any]:
    return {
        "pdf_name": pdf_name,
        "target_table": target_table,
        "scope": scope,
        "sector": sector,
        "selected_page": None,
        "target_found": False,
        "crop_path": "",
        "bbox": [],
        "confidence": 0.0,
        "needs_target_extraction_from_combined_crop": False,
        "reason": "",
        "warnings": [],
    }


if __name__ == "__main__":
    raise SystemExit(main())
