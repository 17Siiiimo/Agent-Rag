from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


Scope = Literal["comptes_consolides", "comptes_sociaux"]
Sector = Literal["bancaire_sdf", "assurance", "autres_cgnc"]
TargetTable = Literal["BILAN_ACTIF", "BILAN_PASSIF", "CPC"]


@dataclass
class PipelineConfig:
    output_dir: Path = Path("output") / "financial_extraction_debug"
    dpi: int = 300
    top_k_pages: int = 8
    candidate_pages_for_localization: int = 4
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    bm25_weight: float = 0.35
    vector_weight: float = 0.25
    target_anchor_weight: float = 0.15
    scope_weight: float = 0.10
    sector_weight: float = 0.10
    title_weight: float = 0.05
    vision_provider: str = "openai"
    vision_model: str | None = None
    min_validation_rows: int = 3
    use_vision: bool = True
    save_debug: bool = True
    cleanup_rendered_pages: bool = True


@dataclass
class PDFDocument:
    pdf_id: str
    pdf_path: str
    page_count: int


@dataclass
class PageChunk:
    pdf_id: str
    company: str
    year: int
    page_number: int
    scope_candidates: list[str]
    sector: str
    page_text: str
    detected_titles: list[str]
    target_candidates: list[str]
    anchors: list[str]
    image_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def embedding_text(self) -> str:
        return (
            f"company: {self.company} "
            f"year: {self.year} "
            f"sector: {self.sector} "
            f"scope: {' '.join(self.scope_candidates)} "
            f"targets: {' '.join(self.target_candidates)} "
            f"titles: {' '.join(self.detected_titles)} "
            f"anchors: {' '.join(self.anchors)} "
            f"page text: {self.page_text}"
        )


@dataclass
class TableCandidate:
    pdf_id: str
    page_number: int
    scope: str
    sector: str
    table_type: str
    bbox: list[float]
    confidence: float
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievedPage:
    page: PageChunk
    score: float
    bm25_score: float = 0.0
    vector_score: float = 0.0
    target_anchor_score: float = 0.0
    scope_signature_score: float = 0.0
    opposite_scope_signature_score: float = 0.0
    scope_score: float = 0.0
    sector_score: float = 0.0
    title_score: float = 0.0
    matched_anchors: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page.page_number,
            "final_score": self.score,
            "bm25_score": self.bm25_score,
            "vector_score": self.vector_score,
            "target_anchor_score": self.target_anchor_score,
            "scope_signature_score": self.scope_signature_score,
            "opposite_scope_signature_score": self.opposite_scope_signature_score,
            "scope_score": self.scope_score,
            "sector_score": self.sector_score,
            "title_score": self.title_score,
            "matched_anchors": self.matched_anchors,
            "detected_scope": self.page.scope_candidates,
            "detected_target_candidates": self.page.target_candidates,
            "evidence": self.evidence,
            "page": self.page.to_dict(),
        }
