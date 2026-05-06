"""
Chemins d'index par émetteur / année / type de rapport.

Structure cible :
  data/index/{emetteur_safe}/{year}/{annuel|s1}/index/
    ├── embeddings.npy
    └── metadata.json

Compatibilité : les anciens dossiers sans segment type_rapport
  data/index/{emetteur_safe}/{year}/index/
sont encore reconnus en lecture pour ne pas réindexer les PDFs déjà traités.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional


def normalize_report_code(type_rapport: Optional[str]) -> str:
    tr = (type_rapport or "annuel").strip().lower()
    tr = tr.replace("é", "e").replace("è", "e")
    if "1er semestre" in tr or "premier semestre" in tr or "semestr" in tr or tr in ("s1", "rfs"):
        return "s1"
    return "annuel"


def _stem_from_pdf_name(pdf_name: str) -> str:
    base = os.path.basename(pdf_name)
    return base[:-4] if base.lower().endswith(".pdf") else base


def infer_index_dir_for_pdf(
    pdf_name: str,
    base_index_dir: str,
    type_rapport_override: Optional[str] = None,
) -> str:
    """
    Dossier d'index où ÉCRIRE (toujours la structure avec type_rapport).

    - {emetteur}_{year}_annuel.pdf ou _s1.pdf -> .../{year}/{annuel|s1}/index
    - {emetteur}_{year}.pdf -> .../{year}/{override ou annuel}/index
    - {emetteur}_{year}_{autre}.pdf (suffixe non annuel/s1) -> .../{year}/{override ou annuel}/index
    """
    stem = _stem_from_pdf_name(pdf_name)
    parts = stem.split("_")
    default_rep = normalize_report_code(type_rapport_override)
    rep_codes = {"annuel", "s1"}

    if len(parts) >= 3 and parts[-1].lower() in rep_codes and parts[-2].isdigit():
        emetteur_safe = "_".join(parts[:-2])
        year = parts[-2]
        rep = parts[-1].lower()
        return os.path.join(base_index_dir, emetteur_safe, year, rep, "index")

    if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 4:
        emetteur_safe = "_".join(parts[:-1])
        year = parts[-1]
        return os.path.join(base_index_dir, emetteur_safe, year, default_rep, "index")

    if len(parts) >= 3:
        year_prev = parts[-2]
        if year_prev.isdigit():
            emetteur_safe = "_".join(parts[:-2])
            return os.path.join(base_index_dir, emetteur_safe, year_prev, default_rep, "index")

    return base_index_dir


def infer_legacy_index_dir_for_pdf(pdf_name: str, base_index_dir: str) -> Optional[str]:
    """Ancien chemin : {base}/{emetteur}/{year}/index/ (sans annuel|s1)."""
    stem = _stem_from_pdf_name(pdf_name)
    parts = stem.split("_")
    if len(parts) < 2:
        return None
    if parts[-1].isdigit() and len(parts[-1]) == 4:
        emetteur_safe = "_".join(parts[:-1])
        return os.path.join(base_index_dir, emetteur_safe, parts[-1], "index")
    if len(parts) >= 3 and parts[-2].isdigit():
        emetteur_safe = "_".join(parts[:-2])
        return os.path.join(base_index_dir, emetteur_safe, parts[-2], "index")
    return None


def _index_has_doc_id(index_dir: str, doc_id: str) -> bool:
    meta_path = os.path.join(index_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return False
    doc_l = (doc_id or "").lower()
    if not doc_l.endswith(".pdf"):
        doc_l = doc_l + ".pdf"
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f) or []
    except (json.JSONDecodeError, OSError):
        return False
    for m in meta:
        if (m.get("doc_id") or "").lower() == doc_l:
            return True
    return False


def index_dir_candidates_for_read(
    pdf_name: str,
    base_index_dir: str,
    type_rapport_override: Optional[str] = None,
) -> List[str]:
    """Ordre : index cible (avec type rapport), puis ancien dossier sans segment si pertinent."""
    out: List[str] = []
    primary = infer_index_dir_for_pdf(pdf_name, base_index_dir, type_rapport_override)
    stem = _stem_from_pdf_name(pdf_name)
    parts = stem.split("_")
    rep_codes = {"annuel", "s1"}
    explicit_rep: Optional[str] = None
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].lower() in rep_codes:
        explicit_rep = parts[-1].lower()

    if primary != base_index_dir and primary not in out:
        out.append(primary)
    legacy = infer_legacy_index_dir_for_pdf(pdf_name, base_index_dir)
    # PDF nommé ..._2024_s1.pdf ou ..._2024_annuel.pdf : ne pas retomber sur
    # data/index/.../2024/index/ (ancien plat) pour éviter de lire l'index annuel par erreur.
    if legacy and legacy not in out and explicit_rep is None:
        out.append(legacy)
    return out


def resolve_index_dir_for_read(
    pdf_name: str,
    base_index_dir: str,
    type_rapport_override: Optional[str] = None,
) -> str:
    """
    Dossier d'index à utiliser pour la lecture (métadonnées / RAG).
    Choisit le premier candidat qui contient déjà ce PDF dans metadata.json.
    Sinon retourne le chemin d'écriture cible (pour cohérence après premier build).
    """
    doc_id = os.path.basename(pdf_name)
    if not doc_id.lower().endswith(".pdf"):
        doc_id = doc_id + ".pdf"
    primary = infer_index_dir_for_pdf(pdf_name, base_index_dir, type_rapport_override)
    for d in index_dir_candidates_for_read(pdf_name, base_index_dir, type_rapport_override):
        emb = os.path.join(d, "embeddings.npy")
        if os.path.isfile(emb) and _index_has_doc_id(d, doc_id):
            return d
    return primary if primary != base_index_dir else base_index_dir


def is_index_complete_for_pdf(
    pdf_name: str,
    base_index_dir: str,
    type_rapport_override: Optional[str] = None,
) -> bool:
    """True si embeddings + metadata existent et contiennent ce doc_id (nouveau ou ancien chemin)."""
    doc_id = os.path.basename(pdf_name)
    if not doc_id.lower().endswith(".pdf"):
        doc_id = doc_id + ".pdf"
    for d in index_dir_candidates_for_read(pdf_name, base_index_dir, type_rapport_override):
        emb = os.path.join(d, "embeddings.npy")
        meta = os.path.join(d, "metadata.json")
        if os.path.isfile(emb) and os.path.isfile(meta) and _index_has_doc_id(d, doc_id):
            return True
    return False
