"""
Tests E2E Phase 1 : scraper seul, extraction seule, pipeline complet.

Depuis la racine du projet (RAG/) :
  python -m rag_agent.test_e2e --scraper-only --emetteur agma --year 2024
  python -m rag_agent.test_e2e --extract-only data/pdfs/agma_2024_sociaux.pdf:11 --tableau "BILAN ACTIF"
  python -m rag_agent.test_e2e --emetteur agma --year 2024 --tableau "BILAN ACTIF"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permet d'exécuter depuis la racine du projet
_root = Path(__file__).resolve().parent.parent
if _root not in sys.path:
    sys.path.insert(0, str(_root))


def _test_scraper_only(emetteur: str, year: int, type_comptes: str = "sociaux") -> int:
    """Test 1 : téléchargement AMMC uniquement."""
    from rag_agent.scraper import AMMCScraper, DEFAULT_PDF_DIR

    scraper = AMMCScraper()
    result = scraper.fetch(emetteur=emetteur, year=year, type_comptes=type_comptes)
    if not result.success:
        print(f"[FAIL] [DOWNLOAD] {emetteur} {year} -> {result.error}")
        return 1
    print(f"[OK] [DOWNLOAD] {emetteur} {year} -> {result.pdf_path}")
    print(f"Taille : {result.size_mb} MB" + (" (cache)" if result.from_cache else ""))
    return 0


def _test_extract_only(pdf_page_spec: str, tableau: str) -> int:
    """
    Test 2 : extraction seule sur une page connue.
    pdf_page_spec : "chemin/vers/fichier.pdf:11" (numéro de page 1-based).
    """
    from rag_agent.extraction_agent import ExtractionAgent

    if ":" not in pdf_page_spec:
        print("[FAIL] --extract-only attend format: chemin.pdf:numero_page")
        return 1
    path_str, page_str = pdf_page_spec.rsplit(":", 1)
    try:
        page_num = int(page_str)
    except ValueError:
        print("[FAIL] Numero de page invalide")
        return 1
    pdf_path = Path(path_str)
    if not pdf_path.is_file():
        # Fallback data/pdf
        fallback = Path("data/pdf") / pdf_path.name
        if fallback.is_file():
            pdf_path = fallback
        else:
            print(f"[FAIL] PDF introuvable : {path_str}")
            return 1

    agent = ExtractionAgent(use_llm=True)
    result = agent.extract(str(pdf_path), page_num, tableau)
    rows, cols = result.df.shape if result.df is not None and not result.df.empty else (0, 0)
    conf_pct = int(result.confidence * 100)
    if result.df is not None and not result.df.empty:
        print(f"[OK] [pdfplumber|positional|llm_groq] '{tableau}' p.{page_num}")
        print(f"  -> {rows}L x {cols}C  conf={conf_pct}%")
        return 0
    print(f"[FAIL] Extraction vide (methode={result.method})")
    return 1


def _test_pipeline_full(
    emetteur: str,
    year: int,
    tableau: str,
    type_comptes: str = "sociaux",
) -> int:
    """Test 3 : pipeline complet (scraper → index → find page → extract → Excel)."""
    from rag_agent.pipeline_phase1 import Phase1Pipeline

    pipeline = Phase1Pipeline()
    result = pipeline.run(
        emetteur=emetteur,
        year=year,
        tableau=tableau,
        type_comptes=type_comptes,
    )
    if not result.success:
        print(f"[FAIL] Pipeline : {result.error}")
        return 1
    rows = len(result.df)
    cols = len(result.df.columns)
    conf_pct = int((result.confidence or 0) * 100)
    print(f"[OK] {emetteur} {year} | '{tableau}' | page {result.page_num} | {rows}L x {cols}C | methode={result.method}")
    print(f"Excel : {result.excel_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Tests E2E Phase 1 RAG")
    parser.add_argument("--scraper-only", action="store_true", help="Test scraper AMMC uniquement")
    parser.add_argument("--extract-only", metavar="PDF:PAGE", help="Test extraction sur PDF:PAGE")
    parser.add_argument("--emetteur", default="agma", help="Émetteur (ex: agma)")
    parser.add_argument("--year", type=int, default=2024, help="Année")
    parser.add_argument("--tableau", default="BILAN ACTIF", help="Nom du tableau")
    parser.add_argument("--type-comptes", default="sociaux", choices=["sociaux", "consolides"])
    args = parser.parse_args()

    if args.scraper_only:
        return _test_scraper_only(args.emetteur, args.year, args.type_comptes)
    if args.extract_only:
        return _test_extract_only(args.extract_only, args.tableau)
    return _test_pipeline_full(args.emetteur, args.year, args.tableau, args.type_comptes)


if __name__ == "__main__":
    sys.exit(main())
