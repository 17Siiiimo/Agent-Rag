from __future__ import annotations

import unittest

from rag_agent.financial_extraction.models import PageChunk, RetrievedPage, TableCandidate
from rag_agent.financial_extraction.table_localizer import _candidate_rank_score, _pages_for_localization


class TableLocalizerRankingTests(unittest.TestCase):
    def test_cpc_label_signature_beats_weak_fallback_page(self) -> None:
        fallback = TableCandidate(
            pdf_id="doc",
            page_number=182,
            scope="comptes_consolides",
            sector="autres_cgnc",
            table_type="CPC",
            bbox=[0, 0, 100, 100],
            confidence=0.494,
            evidence=["title_hit:compte de resultat", "scope_signature_score:0.000", "fallback:page_region"],
        )
        label_match = TableCandidate(
            pdf_id="doc",
            page_number=161,
            scope="comptes_consolides",
            sector="autres_cgnc",
            table_type="CPC",
            bbox=[0, 0, 100, 100],
            confidence=0.35,
            evidence=["localized:CPC", "title_confirmed:False"],
        )

        fallback_score = _candidate_rank_score(
            fallback,
            retrieval_score=0.494,
            scope_score=1.0,
            signature_score=0.0,
            anchor_score=0.005,
        )
        label_match_score = _candidate_rank_score(
            label_match,
            retrieval_score=0.480,
            scope_score=1.0,
            signature_score=0.419,
            anchor_score=0.382,
        )

        self.assertGreater(label_match_score, fallback_score)

    def test_signature_rich_cpc_page_is_localized_even_beyond_top_four(self) -> None:
        pages = [
            _retrieved(page_number=166, score=0.567, signature=0.0, anchor=0.027),
            _retrieved(page_number=170, score=0.515, signature=0.0, anchor=0.0),
            _retrieved(page_number=182, score=0.494, signature=0.0, anchor=0.005),
            _retrieved(page_number=171, score=0.486, signature=0.0, anchor=0.0),
            _retrieved(page_number=161, score=0.480, signature=0.419, anchor=0.382),
        ]

        selected = _pages_for_localization(pages, max_pages=4, target_table="CPC")

        self.assertEqual([item.page.page_number for item in selected], [166, 170, 182, 171, 161])

def _retrieved(*, page_number: int, score: float, signature: float, anchor: float) -> RetrievedPage:
    return RetrievedPage(
        page=PageChunk(
            pdf_id="doc",
            company="Company",
            year=2023,
            page_number=page_number,
            scope_candidates=["comptes_consolides"],
            sector="autres_cgnc",
            page_text="",
            detected_titles=[],
            target_candidates=["CPC"],
            anchors=[],
        ),
        score=score,
        target_anchor_score=anchor,
        scope_signature_score=signature,
        scope_score=1.0,
    )


if __name__ == "__main__":
    unittest.main()
