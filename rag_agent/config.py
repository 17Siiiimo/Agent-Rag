import os
from pathlib import Path
from dataclasses import dataclass

try:
    # Permet d'aligner le dossier de cache PDF avec la convention du scraper.
    from .scraper import DEFAULT_PDF_DIR as _DEFAULT_PDF_DIR
except Exception:
    _DEFAULT_PDF_DIR = Path(os.path.join("data", "pdfs"))


@dataclass
class RAGConfig:
    data_dir: str = "data"
    raw_pdf_dir: str = str(_DEFAULT_PDF_DIR)
    index_dir: str = os.path.join("data", "index")
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    top_k: int = 8


def ensure_dirs(cfg: RAGConfig) -> None:
    os.makedirs(cfg.data_dir, exist_ok=True)
    os.makedirs(cfg.raw_pdf_dir, exist_ok=True)
    os.makedirs(cfg.index_dir, exist_ok=True)

