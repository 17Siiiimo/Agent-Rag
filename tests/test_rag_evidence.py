from __future__ import annotations

import unittest

from rag_agent.financial_extraction.rag_evidence import build_table_evidence_chunks


class RagEvidenceTests(unittest.TestCase):
    def test_builds_row_level_chunks_with_source_metadata(self) -> None:
        result = {
            "pdf_path": "data/pdfs/sample.pdf",
            "company": "Sample Co",
            "year": 2024,
            "report_type": "rapport_annuel",
            "sector": "bancaire_sdf",
            "scope": "comptes_sociaux",
            "target_table": "BILAN_ACTIF",
            "selected_page": 12,
            "bbox": [10, 20, 300, 500],
            "crop_path": "output/crop.png",
            "columns": ["2024", "2023"],
            "rows": [
                {
                    "label": "Total actif",
                    "row_type": "total",
                    "values": {"2024": "1 000", "2023": "900"},
                }
            ],
            "confidence": 0.87,
            "validation": {"status": "approved"},
            "debug": {"dir": "output/debug", "candidate_evidence": ["anchor:Total actif"]},
        }

        chunks = build_table_evidence_chunks(result, page_text="Le Total actif est visible sur la page.")

        self.assertEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk["chunk_type"], "financial_table_row")
        self.assertIn("Total actif", chunk["text"])
        self.assertEqual(chunk["metadata"]["company"], "Sample Co")
        self.assertEqual(chunk["metadata"]["page_number"], 12)
        self.assertEqual(chunk["metadata"]["values"]["2024"], "1 000")
        self.assertEqual(chunk["evidence"]["candidate_evidence"], ["anchor:Total actif"])


if __name__ == "__main__":
    unittest.main()
