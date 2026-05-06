from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_agent.financial_extraction.document_classifier import detect_sector
from rag_agent.financial_extraction.hybrid_retriever import retrieve_candidate_pages
from rag_agent.financial_extraction.models import PipelineConfig
from rag_agent.financial_extraction.page_indexer import build_page_chunks, build_page_index
from rag_agent.financial_extraction.page_renderer import render_pdf_pages
from rag_agent.financial_extraction.page_text_extractor import extract_page_texts
from rag_agent.financial_extraction.pdf_loader import load_pdf
from rag_agent.financial_extraction.table_cropper import crop_table_image
from rag_agent.financial_extraction.table_localizer import localize_table_candidates
from rag_agent.financial_extraction.utils import ensure_dir, slugify, write_json

import fitz  # PyMuPDF


CASES = [
    ("BILAN_ACTIF", "comptes_consolides", "bancaire_sdf"),
    ("BILAN_PASSIF", "comptes_consolides", "bancaire_sdf"),
    ("CPC", "comptes_consolides", "bancaire_sdf"),
    ("BILAN_ACTIF", "comptes_sociaux", "bancaire_sdf"),
    ("BILAN_PASSIF", "comptes_sociaux", "bancaire_sdf"),
    ("CPC", "comptes_sociaux", "bancaire_sdf"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit page-aware Moroccan financial RAG retrieval without Vision.")
    parser.add_argument("--pdf", default=_default_pdf(), help="Path to annual report PDF.")
    parser.add_argument("--company", default="Attijariwafa Bank")
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    pdf = load_pdf(args.pdf)
    cfg = PipelineConfig(dpi=args.dpi, top_k_pages=args.top_k, candidate_pages_for_localization=4, use_vision=False)
    audit_dir = ensure_dir(cfg.output_dir / "audit" / slugify(args.company) / str(args.year) / pdf.pdf_id)
    rendered_dir = ensure_dir(audit_dir / "rendered_pages")
    crop_dir = ensure_dir(audit_dir / "crops")

    print(f"PDF: {pdf.pdf_path}")
    print(f"Pages: {pdf.page_count}")
    print(f"Audit dir: {audit_dir}")

    page_texts = extract_page_texts(pdf.pdf_path)
    sector = detect_sector(page_texts, "bancaire_sdf")
    image_paths = render_pdf_pages(pdf.pdf_path, rendered_dir, dpi=cfg.dpi)
    pages = build_page_chunks(
        pdf_id=pdf.pdf_id,
        company=args.company,
        year=args.year,
        sector=sector,
        page_texts=page_texts,
        image_paths=image_paths,
    )
    index = build_page_index(pages, cfg, debug_dir=audit_dir)

    for target_table, scope, case_sector in CASES:
        case_name = f"{scope}_{target_table}"
        case_dir = ensure_dir(audit_dir / case_name)
        print("\n" + "=" * 96)
        print(f"CASE: {target_table} | {scope} | {case_sector}")

        retrieved = retrieve_candidate_pages(index, target_table, scope, case_sector, cfg)
        top5 = retrieved[:5]
        write_json(case_dir / "retrieval_candidates.json", [r.to_dict() for r in retrieved])
        write_json(case_dir / "score_breakdown.json", [r.to_dict() for r in retrieved])

        for rank, r in enumerate(top5, start=1):
            d = r.to_dict()
            print(
                f"#{rank} page={d['page_number']} final={d['final_score']:.3f} "
                f"bm25={d['bm25_score']:.3f} vector={d['vector_score']:.3f} "
                f"anchor={d['target_anchor_score']:.3f} scope={d['scope_score']:.3f} "
                f"sector={d['sector_score']:.3f} title={d['title_score']:.3f}"
            )
            print(f"   anchors: {', '.join(d['matched_anchors'][:8]) or '-'}")
            print(f"   detected scope: {', '.join(d['detected_scope']) or '-'}")
            print(f"   detected targets: {', '.join(d['detected_target_candidates']) or '-'}")

        candidates = localize_table_candidates(
            pdf.pdf_path,
            retrieved,
            target_table,
            scope,
            case_sector,
            max_pages=cfg.candidate_pages_for_localization,
            debug_dir=case_dir,
        )
        selected = candidates[0] if candidates else None
        crop_path = ""
        if selected:
            page_size = _page_size(pdf.pdf_path, selected.page_number)
            crop_path = crop_table_image(selected, image_paths[selected.page_number], crop_dir / case_name, page_size)
            selected_retrieved = next((r for r in retrieved if r.page.page_number == selected.page_number), None)
            write_json(
                case_dir / "selected_page.json",
                {
                    "page_number": selected.page_number,
                    "retrieval": selected_retrieved.to_dict() if selected_retrieved else None,
                    "table_candidate": selected.to_dict(),
                },
            )
            write_json(
                case_dir / "selected_crop_validation.json",
                {
                    "crop_path": crop_path,
                    "page_number": selected.page_number,
                    "bbox": selected.bbox,
                    "candidate_confidence": selected.confidence,
                    "candidate_evidence": selected.evidence,
                    "vision_called": False,
                },
            )

        print(f"Selected page: {selected.page_number if selected else '-'}")
        print(f"Crop path: {crop_path or '-'}")

    return 0


def _default_pdf() -> str:
    candidates = [
        ROOT / "data" / "pdf" / "AWB_RFA_2024_1.pdf",
        ROOT / "Émetteur" / "attijariwafa_bank_2024_annuel.pdf",
        ROOT / "data" / "pdfs" / "attijariwafa_bank_2024_consolides.pdf",
    ]
    for path in candidates:
        if path.is_file():
            return str(path)
    return str(candidates[0])


def _page_size(pdf_path: str, page_number: int) -> tuple[float, float]:
    doc = fitz.open(pdf_path)
    try:
        rect = doc[page_number - 1].rect
        return float(rect.width), float(rect.height)
    finally:
        doc.close()


if __name__ == "__main__":
    raise SystemExit(main())
