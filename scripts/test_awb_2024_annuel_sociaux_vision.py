from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rag_agent.financial_extraction import extract_financial_tables


def main() -> int:
    pdf_path = next(ROOT.glob("*metteur/attijariwafa_bank_2024_annuel.pdf"))
    output_dir = ROOT / "output" / "financial_extraction_debug" / "awb_2024_annuel_sociaux_vision"
    payload = {
        "pdf_path": str(pdf_path),
        "company": "Attijariwafa Bank",
        "year": 2024,
        "report_type": "rapport_annuel",
        "scope": "comptes_sociaux",
        "sector": "bancaire_sdf",
        "target_tables": ["BILAN_ACTIF", "BILAN_PASSIF", "CPC"],
        "provider": "groq",
    }

    summary = extract_financial_tables(payload, output_dir=output_dir)
    print(f"\nSummary JSON: {output_dir / 'summary.json'}")
    for result in summary["results"]:
        validation = result.get("validation") or {}
        print("\n" + "=" * 96)
        print(f"Target table: {result.get('target_table')}")
        print(f"Selected page: {result.get('selected_page')}")
        print(f"Crop path: {result.get('crop_path') or '-'}")
        print(f"target_found: {result.get('target_found')}")
        print(f"columns: {len(result.get('columns') or [])}")
        print(f"rows: {len(result.get('rows') or [])}")
        print(f"validation status: {validation.get('status')}")
        print(f"warnings/errors: {', '.join(result.get('warnings') or []) or result.get('error') or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
