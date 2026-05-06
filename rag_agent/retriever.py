import json
import os
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import RAGConfig, ensure_dirs


@dataclass
class RetrievedChunk:
    id: str
    doc_id: str
    page: int
    kind: Literal["text", "table"]
    content: str
    section_hint: Optional[str]
    score: float


class SimpleVectorRetriever:
    def __init__(
        self,
        cfg: Optional[RAGConfig] = None,
        index_dir_override: Optional[str] = None,
    ):
        self.cfg = cfg or RAGConfig()
        ensure_dirs(self.cfg)

        index_dir = index_dir_override or self.cfg.index_dir
        os.makedirs(index_dir, exist_ok=True)

        index_path = os.path.join(index_dir, "embeddings.npy")
        meta_path = os.path.join(index_dir, "metadata.json")

        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            raise RuntimeError(
                "Index not found. Run ingestion first (python -m rag_agent.ingest)."
            )

        self.embeddings = np.load(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.model = SentenceTransformer(self.cfg.model_name)

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        kind_filter: Optional[Sequence[str]] = None,
    ) -> List[RetrievedChunk]:
        k = k or self.cfg.top_k
        q_vec = self.model.encode([query], convert_to_numpy=True)[0]

        # cosine similarity
        emb = self.embeddings
        norms = np.linalg.norm(emb, axis=1) * np.linalg.norm(q_vec)
        scores = (emb @ q_vec) / (norms + 1e-10)

        idx_sorted = np.argsort(-scores)
        results: List[RetrievedChunk] = []

        for idx in idx_sorted:
            meta = self.meta[idx]
            if kind_filter and meta["kind"] not in kind_filter:
                continue
            results.append(
                RetrievedChunk(
                    id=meta["id"],
                    doc_id=meta["doc_id"],
                    page=meta["page"],
                    kind=meta["kind"],
                    content=meta["content"],
                    section_hint=meta.get("section_hint"),
                    score=float(scores[idx]),
                )
            )
            if len(results) >= k:
                break

        return results


def classify_intent(question: str) -> Tuple[str, Optional[str]]:
    """
    Heuristic classifier for the type of financial table requested.
    Returns (intent, table_type).
    intent in {"generic_qa", "table_reconstruction"}
    table_type in {"bilan", "compte_resultat", "flux_tresorerie", None}
    """
    q_lower = question.lower()
    table_type: Optional[str] = None

    if "bilan" in q_lower or "balance sheet" in q_lower:
        table_type = "bilan"
    elif (
        "compte de résultat" in q_lower
        or "compte de resultat" in q_lower
        or "income statement" in q_lower
        or "comptes de produits et charges" in q_lower
        or "compte de produits et charges" in q_lower
        or "compte de produits et de charges" in q_lower
        or "cpc" in q_lower  # abréviation fréquente
    ):
        table_type = "compte_resultat"
    elif "flux de trésorerie" in q_lower or "flux de tresorerie" in q_lower or "cash flow" in q_lower:
        table_type = "flux_tresorerie"

    if any(
        kw in q_lower
        for kw in [
            "tableau",
            "table",
            "bilan",
            "compte de resultat",
            "compte de résultat",
            "flux de trésorerie",
            "flux de tresorerie",
            "cash flow",
        ]
    ):
        intent = "table_reconstruction"
    else:
        intent = "generic_qa"

    return intent, table_type

