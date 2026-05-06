from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .document_classifier import detect_anchors, detect_scope_ranges, detect_target_candidates, detect_titles
from .models import PageChunk, PipelineConfig
from .utils import write_json


class PageAwareIndex:
    def __init__(self, pages: list[PageChunk], embeddings: np.ndarray | None = None, faiss_index=None):
        self.pages = pages
        self.embeddings = embeddings
        self.faiss_index = faiss_index


def build_page_chunks(
    *,
    pdf_id: str,
    company: str,
    year: int,
    sector: str,
    page_texts: dict[int, str],
    image_paths: dict[int, str],
) -> list[PageChunk]:
    pages: list[PageChunk] = []
    scope_by_page = detect_scope_ranges(page_texts)
    for page_number in sorted(page_texts):
        text = page_texts[page_number]
        pages.append(
            PageChunk(
                pdf_id=pdf_id,
                company=company,
                year=year,
                page_number=page_number,
                scope_candidates=scope_by_page.get(page_number, []),
                sector=sector,
                page_text=text,
                detected_titles=detect_titles(text),
                target_candidates=detect_target_candidates(text, sector),
                anchors=detect_anchors(text, sector),
                image_path=image_paths.get(page_number, ""),
            )
        )
    return pages


def load_page_chunks_cache(
    cache_path: str | Path,
    *,
    pdf_id: str,
    company: str,
    year: int,
    sector: str,
) -> list[PageChunk] | None:
    path = Path(cache_path)
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        pages = [PageChunk(**item) for item in payload]
    except Exception:
        return None
    if not pages:
        return None
    if any(p.pdf_id != pdf_id or p.company != company or int(p.year) != int(year) or p.sector != sector for p in pages):
        return None
    if any(p.image_path and not Path(p.image_path).is_file() for p in pages):
        return None
    return pages


def build_page_index(
    pages: Sequence[PageChunk],
    cfg: PipelineConfig,
    debug_dir: str | Path | None = None,
) -> PageAwareIndex:
    texts = [p.embedding_text() for p in pages]
    embeddings = _encode_texts(texts, cfg.embedding_model)
    index = PageAwareIndex(list(pages), embeddings, _build_faiss_index(embeddings))
    if debug_dir:
        write_json(Path(debug_dir) / "page_chunks.json", [p.to_dict() for p in pages])
    return index


def load_or_build_page_index(
    pages: Sequence[PageChunk],
    cfg: PipelineConfig,
    cache_dir: str | Path,
) -> tuple[PageAwareIndex, bool]:
    cache = Path(cache_dir)
    embeddings_path = cache / "page_embeddings.npy"
    faiss_path = cache / "faiss.index"
    metadata_path = cache / "index_metadata.json"
    expected = _index_metadata(pages, cfg)

    cached = _load_cached_index(list(pages), embeddings_path, faiss_path, metadata_path, expected)
    if cached is not None:
        return cached, True

    index = build_page_index(pages, cfg, debug_dir=None)
    _save_index_cache(index, embeddings_path, faiss_path, metadata_path, expected)
    return index, False


def _encode_texts(texts: list[str], model_name: str) -> np.ndarray | None:
    if not texts:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(emb, dtype="float32")
    except Exception:
        # The retriever degrades to BM25 + anchors if the embedding model is absent.
        return None


def _build_faiss_index(embeddings: np.ndarray | None):
    if embeddings is None or len(embeddings) == 0:
        return None
    try:
        import faiss

        emb = np.asarray(embeddings, dtype="float32")
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        emb = emb / norms
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        return index
    except Exception:
        return None


def _index_metadata(pages: Sequence[PageChunk], cfg: PipelineConfig) -> dict:
    return {
        "embedding_model": cfg.embedding_model,
        "page_count": len(pages),
        "pdf_id": pages[0].pdf_id if pages else "",
        "company": pages[0].company if pages else "",
        "year": pages[0].year if pages else None,
        "sector": pages[0].sector if pages else "",
        "page_numbers": [p.page_number for p in pages],
    }


def _load_cached_index(
    pages: list[PageChunk],
    embeddings_path: Path,
    faiss_path: Path,
    metadata_path: Path,
    expected_metadata: dict,
) -> PageAwareIndex | None:
    if not embeddings_path.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if metadata != expected_metadata:
        return None
    try:
        embeddings = np.load(embeddings_path).astype("float32")
    except Exception:
        return None
    if len(embeddings) != len(pages):
        return None

    faiss_index = None
    if faiss_path.is_file():
        try:
            import faiss

            faiss_index = faiss.read_index(str(faiss_path))
        except Exception:
            faiss_index = _build_faiss_index(embeddings)
    else:
        faiss_index = _build_faiss_index(embeddings)
    return PageAwareIndex(pages, embeddings, faiss_index)


def _save_index_cache(
    index: PageAwareIndex,
    embeddings_path: Path,
    faiss_path: Path,
    metadata_path: Path,
    metadata: dict,
) -> None:
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    if index.embeddings is not None:
        np.save(embeddings_path, np.asarray(index.embeddings, dtype="float32"))
    if index.faiss_index is not None:
        try:
            import faiss

            faiss.write_index(index.faiss_index, str(faiss_path))
        except Exception:
            pass
    write_json(metadata_path, metadata)
