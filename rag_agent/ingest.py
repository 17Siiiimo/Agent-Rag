import json
import os
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
import pdfplumber
from sentence_transformers import SentenceTransformer

from .config import RAGConfig, ensure_dirs
from .index_paths import infer_index_dir_for_pdf

try:
    import pymupdf
except ImportError:
    pymupdf = None  # type: ignore


@dataclass
class Chunk:
    id: str
    doc_id: str
    page: int
    kind: str  # "text" or "table"
    content: str
    section_hint: Optional[str] = None


# ── Règles de détection de section_hint ──────────────────────────────────────
# Format : (hint_name, seuil_minimal, [(keyword, poids), ...])
# Le hint est assigné si la somme des poids des mots-clés trouvés >= seuil.
# Les mots-clés sont des lignes/cellules qui n'apparaissent QUE sur la vraie page
# du tableau (pas dans le sommaire ni dans le texte narratif).
_SECTION_HINT_RULES: List[Tuple[str, int, List[Tuple[str, int]]]] = [
    # ── CGNC Comptes Sociaux ──────────────────────────────────────────────────
    ("bilan_actif_cgnc", 10, [
        ("bilan actif", 16),
        ("actif (en mad)", 14),
        ("immobilisations en non-valeur", 15),
        ("actif circulant (hors trésorerie)", 12),
        ("trésorerie - actif", 10),
        ("tresorerie - actif", 10),
        ("total de l'actif", 7),
        ("actif immobilisé", 6),
        ("créances de l'actif circulant", 10),
        ("créances actif circulant", 12),
        # Lignes CGNC standard du bilan actif (A) à (I)
        ("immobilisations incorporelles", 8),
        ("immobilisations corporelles", 8),
        ("immobilisations financières", 8),
        ("immobilisations financieres", 8),
        ("écarts de conversion - actif", 12),
        ("ecarts de conversion - actif", 12),
        ("titres et valeurs de placement", 8),
    ]),
    ("bilan_passif_cgnc", 10, [
        ("bilan passif", 16),
        ("passif (en mad)", 14),
        ("dettes de financement", 12),
        ("passif circulant (hors trésorerie)", 12),
        ("report à nouveau", 8),
        ("total du passif", 7),
        ("capitaux propres assimilés", 10),
        ("primes d'émission, de fusion, d'apport", 10),
        ("trésorerie - passif", 10),
        # Lignes CGNC standard du bilan passif (A) à (F) + trésorerie
        ("total des capitaux propres", 12),
        ("capitaux propres", 6),
        ("provisions durables pour risques et charges", 14),
        ("écarts de conversion - passif", 12),
        ("ecarts de conversion - passif", 12),
        ("dettes du passif circulant", 10),
        ("tresorerie - passif", 10),
    ]),
    ("cpc_cgnc", 12, [
        ("comptes de produits et charges", 16),
        ("compte de produits et charges", 14),
        ("produits d'exploitation", 10),
        ("charges d'exploitation", 10),
        ("résultat d'exploitation", 8),
        ("résultat non courant", 10),
        ("impôts sur les résultats", 8),
        ("chiffre d'affaires", 6),
        # Lignes CGNC standard du CPC
        ("produits financiers", 8),
        ("charges financières", 8),
        ("charges financieres", 8),
        ("résultat financier", 10),
        ("resultat financier", 10),
        ("résultat courant", 8),
        ("resultat courant", 8),
        ("produits non courants", 10),
        ("charges non courantes", 10),
        ("résultat avant impôts", 10),
        ("resultat avant impots", 10),
        ("total des produits", 8),
        ("total des charges", 8),
    ]),
    # ── PCEC Bancaire ─────────────────────────────────────────────────────────
    ("bilan_actif_pcec", 12, [
        ("valeurs en caisse, banques centrales", 18),
        ("créances sur les établissements de crédit et assimilés", 15),
        ("créances sur la clientèle", 10),
        ("titres de transaction", 8),
        ("titres de placement", 8),
        ("total de l'actif", 6),
        # Lignes PCEC standard du bilan actif bancaire
        ("valeurs en caisse, banques centrales, trésor public, service des chèques postaux", 20),
        ("créances acquises par affacturage", 14),
        ("titres de transaction et de placement", 12),
        ("autres actifs", 4),
        ("titres de participation et emplois assimilés", 14),
        ("créances subordonnées", 12),
        ("dépôts d'investissement et wakala bil istithmar placés", 18),
        ("immobilisations données en crédit-bail et en location", 14),
        ("immobilisations données en ijara", 16),
        ("immobilisations incorporelles", 6),
        ("immobilisations corporelles", 6),
        ("total actif", 6),
    ]),
    ("bilan_passif_pcec", 10, [
        ("dettes envers les établissements de crédit et assimilés", 18),
        ("dépôts de la clientèle", 14),
        ("titres de créance émis", 10),
        ("dettes subordonnées", 8),
        ("total du passif", 6),
    ]),
    ("cpc_pcec", 10, [
        # Uniquement les lignes STRUCTURELLES du CPC principal (totaux et en-têtes de section)
        # — PAS les sous-détails qui apparaissent sur les pages d'annexes.
        ("produits d'exploitation bancaire", 18),
        ("charges d'exploitation bancaire", 18),
        ("produit net bancaire", 20),
        ("résultat brut d'exploitation", 16),
        ("charges générales d'exploitation", 14),
        ("dotations aux provisions et pertes sur créances irrécouvrables", 18),
        ("dotations aux provisions pour créances et engagements par signature en souffrance", 20),
        ("reprises de provisions et récupérations sur créances amorties", 18),
        ("résultat courant", 12),
        ("résultat avant impôts", 12),
        ("impôts sur les résultats", 12),
        ("résultat net de l'exercice", 14),
        ("intérêts et produits assimilés", 14),
        ("intérêts et charges assimilées", 14),
        ("coefficient d'exploitation", 12),
        # Finance islamique — lignes structurelles uniquement
        ("produits sur titres de moudaraba et moucharaka", 18),
        ("charges sur titres de moudaraba et moucharaka", 18),
        ("produits sur immobilisations données en ijara", 18),
        ("charges sur immobilisations données en ijara", 18),
        ("transfert de charges sur dépôts d'investissement reçus", 18),
        ("transfert de produits sur dépôts d'investissement reçus", 18),
    ]),
    # ── IFRS Consolidé ────────────────────────────────────────────────────────
    ("bilan_actif_ifrs", 12, [
        ("goodwill", 10),
        ("total actif non courant", 14),
        ("total actif courant", 14),
        ("actifs d'impôts différés", 10),
        ("participations dans les sociétés mises en équivalence", 12),
        ("immeubles de placement", 8),
        # Lignes IFRS standard du bilan actif consolidé
        ("immobilisations incorporelles", 6),
        ("immobilisations corporelles", 6),
        ("autres actifs financiers", 8),
        ("total actifs non courants", 14),
        ("créances clients", 10),
        ("stocks et encours", 10),
        ("total actifs courants", 14),
        ("total actif", 6),
    ]),
    ("bilan_passif_ifrs", 10, [
        ("capitaux propres - part du groupe", 18),
        ("intérêts minoritaires", 12),
        ("résultat net - part du groupe", 10),
        ("total passif non courant", 12),
        ("total passif courant", 12),
        ("dettes financières non courantes", 10),
        # Lignes IFRS standard du bilan passif consolidé
        ("capital", 4),
        ("primes d'émission et de fusion", 10),
        ("réserves consolidées", 12),
        ("reserves consolidees", 12),
        ("capitaux propres part du groupe", 16),
        ("capitaux propres part des minoritaires", 14),
        ("capitaux propres d'ensemble", 16),
        ("total des passifs non courants", 12),
        ("total dettes courantes", 12),
        ("total passif", 6),
    ]),
    ("cpc_ifrs", 10, [
        ("résultat net - part du groupe", 18),
        ("résultat net - part des minoritaires", 16),
        ("résultat opérationnel", 12),
        ("coût des ventes", 10),
        ("marge brute", 10),
        ("charge d'impôt sur le résultat", 8),
        # Lignes IFRS standard du CPC consolidé
        ("chiffre d'affaires", 6),
        ("produits des activités ordinaires", 14),
        ("charges d'exploitation courantes", 12),
        ("résultat d'exploitation courant", 12),
        ("autres produits et charges d'exploitation", 12),
        ("résultat des activités opérationnelles", 14),
        ("résultat financier", 8),
        ("résultat avant impôts des entreprises intégrées", 16),
        ("résultat net des entreprises intégrées", 16),
        ("résultat net des activités poursuivies", 14),
        ("résultat de l'ensemble consolidé", 16),
    ]),
    # ── Assurance ─────────────────────────────────────────────────────────────
    ("bilan_actif_assurance", 12, [
        ("placements représentatifs des provisions techniques", 18),
        ("créances nées d'opérations d'assurance", 12),
        ("placements des entreprises d'assurance", 14),
        ("actifs incorporels", 7),
    ]),
    ("bilan_passif_assurance", 10, [
        ("provisions techniques", 12),
        ("primes non acquises", 12),
        ("provisions pour sinistres", 12),
        ("dettes nées d'opérations d'assurance", 10),
    ]),
    ("cpc_assurance", 10, [
        ("primes émises", 16),
        ("charges de sinistres", 14),
        ("résultat technique", 14),
        ("primes acquises", 10),
        ("résultat non technique", 10),
    ]),
]


_SECTION_HINT_EXCLUSIONS = [
    "tableau des flux de trésorerie",
    "tableau des flux de tresorerie",
    "flux de trésorerie",
    "flux de tresorerie",
    "tableau de variation des capitaux propres",
    "tableau de financement",
    # "hors bilan" et "état des soldes de gestion" retirés des exclusions globales :
    # les PDF bancaires (ex CIH) regroupent Bilan + CPC + Hors Bilan + ESG sur une
    # seule page — ces exclusions bloquaient la détection correcte des tableaux.
]


def _normalize(text: str) -> str:
    """Supprime les accents et met en minuscules — gère les PDF OCR sans accents."""
    return unicodedata.normalize("NFD", text.lower()).encode("ascii", "ignore").decode("ascii")


# Pré-normaliser tous les mots-clés une seule fois au chargement
_SECTION_HINT_RULES_NORM = [
    (hint, threshold, [(_normalize(kw), w) for kw, w in keywords])
    for hint, threshold, keywords in _SECTION_HINT_RULES
]

_SECTION_HINT_EXCLUSIONS_NORM = [_normalize(e) for e in _SECTION_HINT_EXCLUSIONS]


def _detect_section_hint(text: str) -> Optional[str]:
    """
    Analyse le texte d'une page PDF et retourne le type de tableau financier
    qu'elle contient, ou None si non identifié.

    Utilise des mots-clés d'ancrage (lignes de tableau) qui ne figurent que
    sur la vraie page du tableau, pas dans le sommaire ni dans le texte narratif.
    En cas de conflit, retourne le hint avec le score le plus élevé.
    """
    if not text or len(text.strip()) < 50:
        return None
    t = _normalize(text)

    # Exclure les pages qui sont clairement un autre tableau
    for excl in _SECTION_HINT_EXCLUSIONS_NORM:
        if excl in t:
            return None

    best_hint: Optional[str] = None
    best_score = 0

    for hint_name, threshold, keywords in _SECTION_HINT_RULES_NORM:
        matched = [(kw, w) for kw, w in keywords if kw in t]
        score = sum(w for _, w in matched)
        # Exiger un nombre minimum de mots-clés trouvés pour éviter les faux positifs
        # sur un seul mot-clé à poids élevé. La formule s'adapte à la taille de la liste :
        #   ≤10 mots  → 3 min  |  11-20 → 4 min  |  21-35 → 5 min  |  36+ → 7 min
        n = len(keywords)
        min_matches = 3 if n <= 10 else 4 if n <= 20 else 5 if n <= 35 else 7
        if score >= threshold and len(matched) >= min_matches and score > best_score:
            best_score = score
            best_hint = hint_name

    return best_hint


def _pdf_to_chunks(pdf_path: str, show_progress: bool = True) -> List[Chunk]:
    doc_id = os.path.basename(pdf_path)
    chunks: List[Chunk] = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for page_idx, page in enumerate(pdf.pages, start=1):
            if show_progress and (page_idx == 1 or page_idx % 10 == 0 or page_idx == total_pages):
                print(f"  {doc_id}: page {page_idx}/{total_pages}...", flush=True)
            try:
                text = page.extract_text() or ""
            except Exception as e:
                print(f"  Warning: skip page {page_idx} ({e})", flush=True)
                text = ""
            if text.strip():
                chunks.append(
                    Chunk(
                        id=f"{doc_id}_p{page_idx}_text",
                        doc_id=doc_id,
                        page=page_idx,
                        kind="text",
                        content=text,
                        section_hint=_detect_section_hint(text),
                    )
                )

            try:
                tables = page.extract_tables() or []
            except Exception as e:
                print(f"  Warning: tables skip page {page_idx} ({e})", flush=True)
                tables = []
            for t_idx, table in enumerate(tables, start=1):
                # Simple markdown rendering of the table
                if not table:
                    continue
                header = table[0]
                rows = table[1:] if len(table) > 1 else []
                md_lines = []
                if header:
                    md_lines.append("| " + " | ".join(h or "" for h in header) + " |")
                    md_lines.append(
                        "| "
                        + " | ".join("---" for _ in header)
                        + " |"
                    )
                for row in rows:
                    md_lines.append(
                        "| " + " | ".join((c or "").strip() for c in row) + " |"
                    )
                table_md = "\n".join(md_lines)

                chunks.append(
                    Chunk(
                        id=f"{doc_id}_p{page_idx}_t{t_idx}",
                        doc_id=doc_id,
                        page=page_idx,
                        kind="table",
                        content=table_md,
                        section_hint=None,
                    )
                )

    return chunks


def _pdf_to_chunks_fast(pdf_path: str, show_progress: bool = True) -> List[Chunk]:
    """Extraction texte seule avec PyMuPDF (5–10x plus rapide, pas de tableaux)."""
    if pymupdf is None:
        raise RuntimeError("Mode rapide nécessite: pip install pymupdf")
    doc_id = os.path.basename(pdf_path)
    chunks: List[Chunk] = []
    doc = pymupdf.open(pdf_path)
    try:
        total_pages = len(doc)
        for page_idx in range(1, total_pages + 1):
            if show_progress and (
                page_idx == 1 or page_idx % 20 == 0 or page_idx == total_pages
            ):
                print(f"  {doc_id}: page {page_idx}/{total_pages}...", flush=True)
            try:
                page = doc[page_idx - 1]
                text = page.get_text() or ""
            except Exception as e:
                print(f"  Warning: skip page {page_idx} ({e})", flush=True)
                text = ""
            if text.strip():
                chunks.append(
                    Chunk(
                        id=f"{doc_id}_p{page_idx}_text",
                        doc_id=doc_id,
                        page=page_idx,
                        kind="text",
                        content=text,
                        section_hint=_detect_section_hint(text),
                    )
                )
    finally:
        doc.close()
    return chunks


def _process_one_pdf(args: tuple) -> List[Chunk]:
    """Worker pour le pool de processus (fonction top-level pour pickle)."""
    pdf_path, show_progress, use_fast = args
    if use_fast and pymupdf is not None:
        return _pdf_to_chunks_fast(pdf_path, show_progress=show_progress)
    return _pdf_to_chunks(pdf_path, show_progress=show_progress)


def build_index(
    cfg: Optional[RAGConfig] = None,
    pdf_paths: Optional[List[str]] = None,
    use_fast: bool = False,
    workers: int = 1,
    index_dir_override: Optional[str] = None,
) -> None:
    """
    Ingest PDFs and build a vector index. If pdf_paths is given, use only
    those files; otherwise use all PDFs in cfg.raw_pdf_dir.
    use_fast: PyMuPDF text-only (faster, no table extraction).
    workers: number of PDFs processed in parallel (default 1).
    """
    cfg = cfg or RAGConfig()
    ensure_dirs(cfg)
    if use_fast and pymupdf is None:
        raise RuntimeError("Mode rapide nécessite: pip install pymupdf")

    if pdf_paths is not None:
        pdf_files = [
            os.path.abspath(p) if not os.path.isabs(p) else p
            for p in pdf_paths
            if p and p.lower().endswith(".pdf")
        ]
        for p in pdf_files:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"PDF introuvable: {p}")
    else:
        pdf_files = [
            os.path.join(cfg.raw_pdf_dir, f)
            for f in os.listdir(cfg.raw_pdf_dir)
            if f.lower().endswith(".pdf")
        ]
    if not pdf_files:
        raise RuntimeError(
            "Aucun PDF valide indiqué (vérifiez le chemin avec --pdf)."
            if pdf_paths is not None
            else f"Aucun PDF dans '{cfg.raw_pdf_dir}'. Mettez vos rapports financiers dedans."
        )

    index_dir = index_dir_override or cfg.index_dir
    index_path = os.path.join(index_dir, "embeddings.npy")
    meta_path = os.path.join(index_dir, "metadata.json")
    os.makedirs(index_dir, exist_ok=True)

    # Si l'index existe déjà, on évite de réencoder les PDFs déjà indexés,
    # et on concatène uniquement les nouveaux chunks.
    existing_meta: List[dict] = []
    existing_embeddings = None
    existing_doc_ids: set[str] = set()
    index_exists = os.path.isfile(index_path) and os.path.isfile(meta_path)
    if index_exists:
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                existing_meta = json.load(f) or []
            for m in existing_meta:
                doc = (m.get("doc_id") or "")
                if doc:
                    existing_doc_ids.add(doc.lower())
            existing_embeddings = np.load(index_path)
        except Exception:
            # En cas de corruption/incompatibilité, on retombe sur un rebuild complet.
            existing_meta = []
            existing_doc_ids = set()
            existing_embeddings = None

    def _pdf_doc_id(path: str) -> str:
        return os.path.basename(path)

    # Garder uniquement les PDFs non encore présents dans existing_meta
    new_pdf_files: List[str] = []
    if existing_doc_ids:
        for p in pdf_files:
            if _pdf_doc_id(p).lower() not in existing_doc_ids:
                new_pdf_files.append(p)
    else:
        new_pdf_files = list(pdf_files)

    if not new_pdf_files:
        print("Index already contains requested PDF(s); no rebuild needed.")
        return

    # Extraction uniquement sur les nouveaux PDFs
    all_chunks: List[Chunk] = []
    # Désactiver le détail page par page si plusieurs workers
    show_progress = workers <= 1

    if workers <= 1:
        for i, path in enumerate(new_pdf_files, start=1):
            name = os.path.basename(path)
            print(f"PDF {i}/{len(new_pdf_files)}: {name}", flush=True)
            if use_fast and pymupdf is not None:
                all_chunks.extend(_pdf_to_chunks_fast(path, show_progress=True))
            else:
                all_chunks.extend(_pdf_to_chunks(path, show_progress=True))
    else:
        n = min(workers, len(new_pdf_files))
        print(f"Extraction en parallèle ({n} workers)...", flush=True)
        task_args = [(path, False, use_fast) for path in new_pdf_files]
        with ProcessPoolExecutor(max_workers=n) as pool:
            for i, future in enumerate(
                as_completed(pool.submit(_process_one_pdf, a) for a in task_args), 1
            ):
                chunks = future.result()
                all_chunks.extend(chunks)
                print(
                    f"  PDF {i}/{len(new_pdf_files)} terminé ({len(chunks)} chunks)",
                    flush=True,
                )

    texts = [c.content for c in all_chunks]

    model = SentenceTransformer(cfg.model_name)
    new_embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    new_meta = [asdict(c) for c in all_chunks]

    if existing_embeddings is not None and len(existing_meta) > 0:
        # Concaténation : nouvel index = ancien + nouveaux chunks.
        # (On suppose que le modèle/embedding dimension n'a pas changé.)
        if existing_embeddings.shape[1] != new_embeddings.shape[1]:
            # Incompatible => rebuild complet.
            print("Embedding dimension mismatch; rebuilding full index.")
            np.save(index_path, new_embeddings)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(new_meta, f, ensure_ascii=False, indent=2)
        else:
            merged_embeddings = np.concatenate([existing_embeddings, new_embeddings], axis=0)
            merged_meta = existing_meta + new_meta
            np.save(index_path, merged_embeddings)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(merged_meta, f, ensure_ascii=False, indent=2)
    else:
        # Premier index : écriture directe
        np.save(index_path, new_embeddings)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(new_meta, f, ensure_ascii=False, indent=2)

    print(f"Indexed {len(all_chunks)} chunks from {len(new_pdf_files)} PDFs.")


def auto_index_all_pdfs(
    cfg: Optional["RAGConfig"] = None,
    use_fast: bool = False,
    force: bool = False,
) -> None:
    """
    Indexe automatiquement tous les PDFs présents dans cfg.raw_pdf_dir
    qui ne sont pas encore indexés dans leur dossier per-emetteur.

    Chaque PDF est indexé dans :
        data/index/{emetteur}/{year}/{annuel|s1}/index/

    Args:
        use_fast: extraction PyMuPDF texte seul (plus rapide, sans tableaux)
        force: forcer le rebuild même si l'index existe déjà
    """
    cfg = cfg or RAGConfig()
    ensure_dirs(cfg)

    pdf_dir = cfg.raw_pdf_dir
    if not os.path.isdir(pdf_dir):
        print(f"Dossier PDF introuvable: {pdf_dir}")
        return

    pdf_files = sorted(
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    )
    if not pdf_files:
        print(f"Aucun PDF trouvé dans '{pdf_dir}'.")
        return

    print(f"Auto-indexation : {len(pdf_files)} PDF(s) trouvé(s) dans '{pdf_dir}'.")
    indexed = 0
    skipped = 0

    for pdf_path in pdf_files:
        pdf_name = os.path.basename(pdf_path)
        index_dir = infer_index_dir_for_pdf(pdf_name, cfg.index_dir)

        if not force:
            meta_path = os.path.join(index_dir, "metadata.json")
            emb_path = os.path.join(index_dir, "embeddings.npy")
            if os.path.isfile(meta_path) and os.path.isfile(emb_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f) or []
                    doc_ids = {(m.get("doc_id") or "").lower() for m in meta}
                    if pdf_name.lower() in doc_ids:
                        print(f"  [OK] {pdf_name} → déjà indexé")
                        skipped += 1
                        continue
                except Exception:
                    pass

        print(f"  [→] {pdf_name} → {index_dir}")
        build_index(cfg, pdf_paths=[pdf_path], use_fast=use_fast, index_dir_override=index_dir)
        indexed += 1

    print(f"\nTerminé : {indexed} PDF(s) indexé(s), {skipped} déjà à jour.")


if __name__ == "__main__":
    build_index()

