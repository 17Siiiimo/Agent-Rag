from __future__ import annotations

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
from rag_agent.financial_extraction.page_indexer import build_page_chunks, build_page_index
from rag_agent.financial_extraction.page_renderer import render_pdf_pages
from rag_agent.financial_extraction.page_text_extractor import extract_page_texts
from rag_agent.financial_extraction.pdf_loader import load_pdf
from rag_agent.financial_extraction.table_cropper import crop_table_image
from rag_agent.financial_extraction.table_localizer import localize_table_candidates
from rag_agent.financial_extraction.utils import ensure_dir, write_json


PDF_PATH = next(ROOT.glob("*metteur/atlantasanad_2024_s1.pdf"))
COMPANY = "AtlantaSanad Assurance"
YEAR = 2024
SECTOR = "assurance"
SCOPES = ["comptes_consolides", "comptes_sociaux"]
TARGETS = ["BILAN_ACTIF", "BILAN_PASSIF", "CPC"]
OUT_ROOT = ROOT / "output" / "financial_extraction_debug" / "atlantasanad_2024_s1_assurance"


def main() -> int:
    cfg = PipelineConfig(
        output_dir=OUT_ROOT,
        dpi=120,
        top_k_pages=8,
        candidate_pages_for_localization=4,
        use_vision=False,
    )

    pdf = load_pdf(str(PDF_PATH))
    rendered_dir = ensure_dir(OUT_ROOT / "_rendered_pages")
    page_texts = extract_page_texts(pdf.pdf_path)
    sector = detect_sector(page_texts, SECTOR)
    image_paths = render_pdf_pages(pdf.pdf_path, rendered_dir, dpi=cfg.dpi)
    pages = build_page_chunks(
        pdf_id=pdf.pdf_id,
        company=COMPANY,
        year=YEAR,
        sector=sector,
        page_texts=page_texts,
        image_paths=image_paths,
    )

    summary: list[dict[str, Any]] = []
    for scope in SCOPES:
        for target_table in TARGETS:
            target_dir = ensure_dir(OUT_ROOT / scope / target_table)
            index = build_page_index(pages, cfg, debug_dir=target_dir)
            retrieved = retrieve_candidate_pages(index, target_table, scope, sector, cfg)
            retrieval_payload = [r.to_dict() for r in retrieved]
            write_json(target_dir / "retrieval_candidates.json", retrieval_payload)
            write_json(target_dir / "score_breakdown.json", retrieval_payload)

            candidates = localize_table_candidates(
                pdf.pdf_path,
                retrieved,
                target_table,
                scope,
                sector,
                max_pages=cfg.candidate_pages_for_localization,
                debug_dir=target_dir,
            )
            selected = candidates[0] if candidates else None
            selected_retrieved = (
                next((r for r in retrieved if selected and r.page.page_number == selected.page_number), None)
                if selected
                else None
            )

            target_found = bool(selected and selected_retrieved and selected_retrieved.scope_score > 0)
            reason = "" if target_found else "scope/table not detected"
            crop_path = ""
            bbox_path = ""
            warnings: list[str] = []
            needs_combined = False

            if selected:
                page_size = _page_size(pdf.pdf_path, selected.page_number)
                bbox_path = str(target_dir / "selected_bbox.png")
                _draw_selected_bbox(image_paths[selected.page_number], selected, page_size, bbox_path)
                if target_found:
                    generated_crop = crop_table_image(selected, image_paths[selected.page_number], target_dir, page_size)
                    crop_path = str(target_dir / "crop.png")
                    shutil.copy2(generated_crop, crop_path)
                    needs_combined = _needs_target_extraction_from_combined_crop(pdf.pdf_path, selected)
                    if needs_combined:
                        warnings.append("selected crop may include another table/section; verify manually")
                if selected_retrieved and selected_retrieved.scope_score <= 0:
                    warnings.append("selected page scope score is 0; possible scope confusion")
            else:
                warnings.append("no table candidate selected")

            metadata = {
                "pdf_path": str(PDF_PATH),
                "company": COMPANY,
                "year": YEAR,
                "target_table": target_table,
                "scope": scope,
                "sector": sector,
                "selected_page": selected.page_number if selected else None,
                "target_found": target_found,
                "reason": reason,
                "crop_path": crop_path,
                "bbox": selected.bbox if selected else [],
                "confidence": float(selected.confidence) if selected else 0.0,
                "needs_target_extraction_from_combined_crop": needs_combined,
                "warnings": warnings,
                "selected_bbox_path": bbox_path,
                "selected_candidate": selected.to_dict() if selected else None,
                "selected_retrieval": selected_retrieved.to_dict() if selected_retrieved else None,
            }
            write_json(target_dir / "selected_page.json", {
                "target_table": target_table,
                "scope": scope,
                "sector": sector,
                "selected_page": selected.page_number if selected else None,
                "retrieval": selected_retrieved.to_dict() if selected_retrieved else None,
                "table_candidate": selected.to_dict() if selected else None,
            })
            write_json(target_dir / "crop_metadata.json", metadata)
            summary.append(metadata)
            _print_case(scope, target_table, retrieved[:5], metadata)

    write_json(OUT_ROOT / "summary.json", summary)
    print(f"\nSummary JSON: {OUT_ROOT / 'summary.json'}")
    return 0


def _print_case(scope: str, target_table: str, top5, metadata: dict[str, Any]) -> None:
    print("\n" + "=" * 104)
    print(f"Scope: {scope} | Target table: {target_table}")
    print(f"Selected page: {metadata['selected_page']}")
    print(f"Target found: {metadata['target_found']} {metadata['reason']}")
    print("Top 5 retrieved pages:")
    for idx, retrieved in enumerate(top5, start=1):
        d = retrieved.to_dict()
        print(
            f"  #{idx} page={d['page_number']} final={d['final_score']:.3f} "
            f"bm25={d['bm25_score']:.3f} vector={d['vector_score']:.3f} "
            f"anchor={d['target_anchor_score']:.3f} scope={d['scope_score']:.3f} "
            f"sector={d['sector_score']:.3f} title={d['title_score']:.3f}"
        )
        print(f"     matched anchors: {', '.join(d['matched_anchors'][:10]) or '-'}")
        print(f"     detected scopes: {', '.join(d['detected_scope']) or '-'}")
        print(f"     detected targets: {', '.join(d['detected_target_candidates']) or '-'}")
    print(f"Detected bbox: {metadata['bbox']}")
    print(f"Crop path: {metadata['crop_path'] or '-'}")
    print(
        "Crop type: "
        + ("combined" if metadata["needs_target_extraction_from_combined_crop"] else "single-target")
    )
    print(f"Warnings: {', '.join(metadata['warnings']) or '-'}")


def _needs_target_extraction_from_combined_crop(pdf_path: str, selected: TableCandidate) -> bool:
    text = _clip_text(pdf_path, selected.page_number, selected.bbox)
    norm = " ".join(text.lower().split())
    if selected.table_type == "BILAN_ACTIF":
        return "passif" in norm
    if selected.table_type == "BILAN_PASSIF":
        return "bilan actif" in norm or "actif exercice" in norm
    if selected.table_type == "CPC":
        blocked = ["bilan actif", "bilan passif", "etat des soldes", "notes annexes"]
        return any(marker in norm for marker in blocked)
    return False


def _clip_text(pdf_path: str, page_number: int, bbox: list[float]) -> str:
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        return page.get_text("text", clip=fitz.Rect(*bbox)) or ""
    finally:
        doc.close()


def _draw_selected_bbox(
    image_path: str,
    candidate: TableCandidate,
    page_size: tuple[float, float],
    output_path: str,
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    pdf_width, pdf_height = page_size
    sx = img.width / max(pdf_width, 1.0)
    sy = img.height / max(pdf_height, 1.0)
    x1, y1, x2, y2 = candidate.bbox
    box = [int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)]
    draw.rectangle(box, outline="yellow", width=6)
    label = f"{candidate.table_type} {candidate.confidence:.2f}"
    draw.rectangle([box[0], max(0, box[1] - 20), box[0] + 280, box[1]], fill="black")
    draw.text((box[0] + 4, max(0, box[1] - 17)), label, fill="yellow")
    img.save(output_path)


def _page_size(pdf_path: str, page_number: int) -> tuple[float, float]:
    doc = fitz.open(pdf_path)
    try:
        rect = doc[page_number - 1].rect
        return float(rect.width), float(rect.height)
    finally:
        doc.close()


if __name__ == "__main__":
    raise SystemExit(main())
