"""
Détection et segmentation de blocs de tableaux sur une page PDF.

Résout le problème central du projet : pages avec 2-3 tableaux distincts
(ex: Bilan Actif + Bilan Passif + CPC sur la même page), où l'extraction
de la page entière produit un 3e tableau partiel.

Approche :
- PyMuPDF (déjà installé) → blocs texte avec coordonnées (x0, y0, x1, y1)
- Détection de "frontières" de tableaux par mots-clés (pas de dépendance ML)
- Extraction ciblée par zone (clip rect)
- Compatible avec pymupdf4llm pour meilleure qualité de texte par zone

Usage :
    from rag_agent.multi_table_detector import detect_multi_table_zones, find_target_block

    blocks = detect_multi_table_zones(pdf_path, page_num)
    if len(blocks) > 1:
        target = find_target_block(blocks, "BILAN ACTIF")
        zone_text = extract_zone_text(pdf_path, page_num, target)
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Mots-clés déclencheurs de nouveau tableau ──────────────────────────────
# Quand un de ces mots-clés apparaît au début d'un bloc texte sur la page,
# on considère qu'un nouveau tableau commence.
_TABLE_START_KEYWORDS: List[str] = [
    # Titres de tableaux
    "bilan actif",
    "bilan passif",
    "compte de produits et charges",
    "comptes de produits et charges",
    "tableau des flux",
    "etat des soldes de gestion",
    "hors bilan",
    # Premières lignes spécifiques (CGNC)
    "actif (en mad)",
    "passif (en mad)",
    "immobilisations en non-valeur",
    # Premières lignes spécifiques (PCEC bancaire)
    "valeurs en caisse",
    "interets et produits assimiles",
    "produits d'exploitation bancaire",
    # Premières lignes PCEC passif (côté passif du bilan bancaire)
    "dettes envers les etablissements de credit",
    "dettes envers la clientele",
    "banques centrales, tresor public, service des cheques postaux",
    "capitaux propres et assimiles",
    # Premières lignes spécifiques (IFRS)
    "actifs financiers",
    "prets et creances sur la clientele",
    "chiffre d'affaires",
    "produits des activites ordinaires",
]

# ── Correspondance keyword → type de tableau ────────────────────────────────
_KEYWORD_TO_TYPE: List[tuple] = [
    ("bilan actif", "bilan_actif"),
    ("actif (en mad)", "bilan_actif"),
    ("immobilisations en non-valeur", "bilan_actif"),
    ("valeurs en caisse", "bilan_actif"),
    ("actifs financiers", "bilan_actif"),
    ("prets et creances", "bilan_actif"),
    ("bilan passif", "bilan_passif"),
    ("passif (en mad)", "bilan_passif"),
    ("capitaux propres", "bilan_passif"),
    ("dettes envers les etablissements de credit", "bilan_passif"),
    ("dettes envers la clientele", "bilan_passif"),
    ("capitaux propres et assimiles", "bilan_passif"),
    ("compte de produits", "cpc"),
    ("comptes de produits", "cpc"),
    ("produits d'exploitation bancaire", "cpc"),
    ("interets et produits assimiles", "cpc"),
    ("chiffre d'affaires", "cpc"),
    ("produits des activites ordinaires", "cpc"),
    ("tableau des flux", "flux_tresorerie"),
    ("hors bilan", "hors_bilan"),
]


@dataclass
class TableBlock:
    """Un bloc de tableau détecté sur une page PDF."""

    page_num: int
    y_start: float          # Coordonnée verticale de début (points PDF)
    y_end: float            # Coordonnée verticale de fin
    x_start: float          # Coordonnée horizontale gauche
    x_end: float            # Coordonnée horizontale droite
    text_sample: str        # Premiers mots du bloc (pour debug et identification)
    table_type_hint: Optional[str] = None  # "bilan_actif", "bilan_passif", "cpc", ...
    block_index: int = 0    # Ordre d'apparition sur la page (0 = premier)
    confidence: float = 0.0  # Confiance de la détection (0.0-1.0)
    trigger_keyword: str = ""  # Mot-clé qui a déclenché la détection


def _normalize(text: str) -> str:
    """Supprime accents et met en minuscules."""
    return (
        unicodedata.normalize("NFD", str(text).lower())
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def _extend_blocks_x_bounds(
    blocks: List[TableBlock],
    text_blocks: list,
    page_width: float,
) -> None:
    """
    Étend x_start/x_end de chaque TableBlock en scannant les blocs texte
    tombant dans sa plage y. Pour les layouts multi-colonnes (ex: ACTIF gauche,
    PASSIF droite sur la même ligne), la frontière x entre colonnes adjacentes
    est calculée comme le midpoint entre leurs titres respectifs.

    Corrige le cas où x_end du bloc ne couvre que le titre et non les données.
    """
    if not blocks:
        return

    # Regrouper les blocs par rangée de titres (y_start ± 20 pts = même rangée)
    visited = [False] * len(blocks)
    row_groups: List[List[int]] = []
    for i in range(len(blocks)):
        if visited[i]:
            continue
        group = [i]
        visited[i] = True
        for j in range(len(blocks)):
            if not visited[j] and abs(blocks[j].y_start - blocks[i].y_start) <= 20:
                group.append(j)
                visited[j] = True
        row_groups.append(group)

    for group_indices in row_groups:
        row_blks = sorted([blocks[i] for i in group_indices], key=lambda b: b.x_start)

        # Frontières x entre colonnes adjacentes (midpoints)
        x_lefts: List[float] = []
        x_rights: List[float] = []
        for i, blk in enumerate(row_blks):
            xl = 0.0 if i == 0 else (row_blks[i - 1].x_end + blk.x_start) / 2
            xr = page_width if i == len(row_blks) - 1 else (blk.x_end + row_blks[i + 1].x_start) / 2
            x_lefts.append(xl)
            x_rights.append(xr)

        # Étendre chaque bloc avec les données dans sa zone x/y
        for blk, xl, xr in zip(row_blks, x_lefts, x_rights):
            x_min = blk.x_start
            x_max = blk.x_end
            for raw_b in text_blocks:
                bx0, by0, bx1, by1 = raw_b[0], raw_b[1], raw_b[2], raw_b[3]
                bx_center = (bx0 + bx1) / 2
                # Le bloc texte doit être dans la zone y ET dans la zone x de la colonne
                if (by0 >= blk.y_start - 5 and by1 <= blk.y_end + 10
                        and xl <= bx_center <= xr):
                    x_min = min(x_min, bx0)
                    x_max = max(x_max, bx1)
            blk.x_start = max(0.0, x_min - 3)
            blk.x_end = min(page_width, x_max + 3)


def _merge_adjacent_same_type_blocks(
    blocks: List[TableBlock],
) -> List[TableBlock]:
    """
    Fusionne ou supprime les blocs du même table_type_hint dans deux cas :

    1. Blocs adjacents (gap < 15 pts) :
       Ex: titre CPC (y=[60,136]) + données CPC (y=[136,842]) fragmentés par
       PyMuPDF → fusionnés en y=[60,842].

    2. Blocs imbriqués (l'un est contenu dans l'autre) :
       Ex: faux positif "Dont actifs financiers AFS" (y=[317,345]) déclenché
       par le keyword "actifs financiers", mais sa plage y est entièrement
       à l'intérieur du vrai bloc ACTIF (y=[168,575]) → supprimé.

    Les blocs de types différents (ex: bilan_actif + bilan_passif) ne sont
    jamais fusionnés ni supprimés.
    """
    if len(blocks) <= 1:
        return blocks

    merged: List[TableBlock] = []
    skip: set = set()

    for i, blk in enumerate(blocks):
        if i in skip:
            continue
        current = blk
        for j in range(i + 1, len(blocks)):
            if j in skip:
                continue
            other = blocks[j]
            same_type = (
                current.table_type_hint is not None
                and current.table_type_hint == other.table_type_hint
            )
            if not same_type:
                continue

            adjacent = abs(other.y_start - current.y_end) < 15
            contained = (
                other.y_start >= current.y_start - 5
                and other.y_end <= current.y_end + 5
            )

            if adjacent:
                # Fusionner les deux blocs en un seul
                current = TableBlock(
                    page_num=current.page_num,
                    y_start=min(current.y_start, other.y_start),
                    y_end=max(current.y_end, other.y_end),
                    x_start=min(current.x_start, other.x_start),
                    x_end=max(current.x_end, other.x_end),
                    text_sample=current.text_sample,
                    table_type_hint=current.table_type_hint,
                    block_index=current.block_index,
                    confidence=max(current.confidence, other.confidence),
                    trigger_keyword=current.trigger_keyword,
                )
                skip.add(j)
                logger.debug(
                    "_merge_adjacent_same_type_blocks: fusion adjacente %d+%d "
                    "type=%s y=[%.0f,%.0f]",
                    i, j, current.table_type_hint, current.y_start, current.y_end,
                )
            elif contained:
                # Supprimer le sous-bloc (faux positif keyword sur une sous-ligne)
                skip.add(j)
                logger.debug(
                    "_merge_adjacent_same_type_blocks: suppression sous-bloc %d "
                    "y=[%.0f,%.0f] contenu dans bloc %d y=[%.0f,%.0f]",
                    j, other.y_start, other.y_end,
                    i, current.y_start, current.y_end,
                )
        merged.append(current)

    return merged


def _detect_table_type_from_text(text: str) -> Optional[str]:
    """Infère le type de tableau depuis un extrait de texte."""
    n = _normalize(text)
    for keyword, ttype in _KEYWORD_TO_TYPE:
        if keyword in n:
            return ttype
    return None


def _is_table_boundary(text: str) -> tuple[bool, str]:
    """
    Détecte si un texte marque le début d'un nouveau tableau.
    Retourne (is_boundary, trigger_keyword).
    """
    n = _normalize(text.strip()[:200])
    for kw in _TABLE_START_KEYWORDS:
        if kw in n:
            return True, kw
    return False, ""


def detect_multi_table_zones(pdf_path: str, page_num: int) -> List[TableBlock]:
    """
    Détecte les zones de tableaux sur une page PDF.

    Algorithme (deux passes) :
    ──────────────────────────
    Passe 1 — Keyword-direct (prioritaire) :
        Chaque bloc texte PyMuPDF qui contient un mot-clé de tableau
        (ex: "BILAN ACTIF", "HORS BILAN", "COMPTE DE PRODUITS ET CHARGES")
        devient directement un TableBlock avec ses vraies coordonnées x/y.

        Avantage : gère nativement les layouts complexes —
        · Layout 2 colonnes (BILAN ACTIF + BILAN PASSIF côte à côte au même y)
        · Layouts mixtes (tableaux résumés sur une seule page)
        · Pages compactes avec plusieurs tableaux sans gap vertical

    Passe 2 — Sequential-gap (fallback) :
        Si la passe 1 ne trouve aucun bloc avec mot-clé (page atypique),
        l'ancien algorithme par gap vertical prend le relais pour ne pas
        régresser sur les pages précédemment bien gérées.

    Returns:
        Liste de TableBlock ordonnés (y_start croissant puis x_start croissant).
        - 0 ou 1 bloc → page mono-tableau (traitement standard)
        - 2+ blocs   → page multi-tableaux (extraction ciblée recommandée)
    """
    try:
        import pymupdf  # type: ignore
    except ImportError:
        logger.warning("PyMuPDF requis pour detect_multi_table_zones")
        return []

    try:
        doc = pymupdf.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            doc.close()
            return []

        page = doc[page_num - 1]
        page_height = page.rect.height
        page_width = page.rect.width

        # Blocs texte : (x0, y0, x1, y1, text, block_no, block_type)
        raw_blocks = page.get_text("blocks")
        doc.close()

        # Conserver uniquement les blocs texte non vides (block_type == 0)
        text_blocks = [
            b for b in raw_blocks
            if len(b) >= 7 and b[6] == 0 and str(b[4]).strip()
        ]
        # Trier : y croissant, puis x croissant (gère les colonnes côte à côte)
        text_blocks.sort(key=lambda b: (b[1], b[0]))

        if not text_blocks:
            return []

    except Exception as e:
        logger.warning("detect_multi_table_zones PyMuPDF failed page=%d: %s", page_num, e)
        return []

    # ══════════════════════════════════════════════════════════════════════════
    # Passe 1 : Keyword-direct
    # Chaque bloc PyMuPDF contenant un mot-clé de tableau → 1 TableBlock.
    # Cette approche gère correctement les layouts 2 colonnes car PyMuPDF
    # groupe déjà le contenu par zone cohérente ; on exploite ces zones
    # directement sans imposer de contrainte de gap vertical.
    # ══════════════════════════════════════════════════════════════════════════
    keyword_blocks: List[TableBlock] = []

    for raw_b in text_blocks:
        x0, y0, x1, y1, text, _, _ = raw_b
        text_stripped = str(text).strip()

        is_boundary, trigger = _is_table_boundary(text_stripped)
        if not is_boundary:
            continue

        ttype = _detect_table_type_from_text(text_stripped[:300])
        tb = TableBlock(
            page_num=page_num,
            y_start=float(y0),
            y_end=float(y1),
            x_start=max(0.0, float(x0)),
            x_end=min(page_width, float(x1)),
            text_sample=text_stripped[:200],
            table_type_hint=ttype,
            block_index=len(keyword_blocks),
            confidence=0.90 if ttype else 0.65,
            trigger_keyword=trigger,
        )
        keyword_blocks.append(tb)

    if keyword_blocks:
        # Étendre y_end de chaque bloc jusqu'au y_start du prochain bloc
        # sur une ligne différente (ou jusqu'au bas de page).
        # Cela permet de capturer tout le contenu du tableau, pas seulement
        # le titre. Les blocs en layout 2 colonnes (même y_start) sont traités
        # indépendamment : ils s'étendent tous jusqu'au même prochain y_start.
        for i, blk in enumerate(keyword_blocks):
            # Cherche le premier bloc dont y_start est nettement plus bas
            # (> 20 pts pour ne pas confondre deux titres sur la même ligne)
            next_y = page_height
            for other in keyword_blocks:
                if other.y_start > blk.y_start + 20:
                    next_y = min(next_y, other.y_start)
            blk.y_end = next_y

        # Étendre les x_start/x_end en scannant toutes les données du tableau.
        # Corrige le cas où x_end ne couvre que le titre (trop étroit).
        _extend_blocks_x_bounds(keyword_blocks, text_blocks, page_width)

        # Fusionner les blocs adjacents du même type (ex: titre CPC + données CPC
        # fragmentés par PyMuPDF en deux blocs séparés sur la même page).
        keyword_blocks = _merge_adjacent_same_type_blocks(keyword_blocks)

        logger.debug(
            "detect_multi_table_zones page=%d → %d bloc(s) [keyword-direct]: %s",
            page_num,
            len(keyword_blocks),
            [b.table_type_hint for b in keyword_blocks],
        )
        return keyword_blocks

    # ══════════════════════════════════════════════════════════════════════════
    # Passe 2 : Sequential-gap (fallback pour pages sans mot-clé explicite)
    # Algorithme original : segmentation par gap vertical + mots-clés.
    # ══════════════════════════════════════════════════════════════════════════
    logger.debug(
        "detect_multi_table_zones page=%d: passe keyword-direct → 0 bloc, "
        "fallback sequential-gap",
        page_num,
    )

    gap_blocks: List[TableBlock] = []
    current_texts: List[str] = []
    current_y_start: float = text_blocks[0][1]
    current_x_min: float = float("inf")
    current_x_max: float = float("-inf")
    current_trigger: str = ""

    def _save_gap_block(y_end: float) -> None:
        nonlocal current_texts, current_y_start, current_x_min, current_x_max, current_trigger
        if not current_texts:
            return
        sample = " ".join(current_texts[:4])
        ttype = _detect_table_type_from_text(sample)
        tb = TableBlock(
            page_num=page_num,
            y_start=current_y_start,
            y_end=y_end,
            x_start=max(0.0, current_x_min),
            x_end=min(page_width, current_x_max),
            text_sample=sample[:200],
            table_type_hint=ttype,
            block_index=len(gap_blocks),
            confidence=0.80 if ttype else 0.50,
            trigger_keyword=current_trigger,
        )
        gap_blocks.append(tb)

    for i, b in enumerate(text_blocks):
        x0, y0, x1, y1, text, _, _ = b
        text_stripped = str(text).strip()

        if i == 0:
            current_y_start = y0
            current_x_min = x0
            current_x_max = x1
            current_texts = [text_stripped]
            continue

        is_boundary, trigger = _is_table_boundary(text_stripped)
        gap = y0 - text_blocks[i - 1][3]
        is_significant_gap = gap > page_height * 0.03

        if is_boundary and is_significant_gap and current_texts:
            _save_gap_block(y0)
            current_y_start = y0
            current_x_min = x0
            current_x_max = x1
            current_texts = [text_stripped]
            current_trigger = trigger
        else:
            current_texts.append(text_stripped)
            current_x_min = min(current_x_min, x0)
            current_x_max = max(current_x_max, x1)

    last_y = text_blocks[-1][3] if text_blocks else page_height
    _save_gap_block(last_y)

    logger.debug(
        "detect_multi_table_zones page=%d → %d bloc(s) [sequential-gap]: %s",
        page_num,
        len(gap_blocks),
        [b.table_type_hint for b in gap_blocks],
    )
    return gap_blocks


def find_target_block(blocks: List[TableBlock], table_name: str) -> Optional[TableBlock]:
    """
    Parmi les blocs détectés, trouve celui qui correspond au tableau cible.

    Stratégie :
    1. Match exact par table_type_hint
    2. Match par mots-clés dans text_sample
    3. Fallback : premier bloc
    """
    if not blocks:
        return None

    target_type = _detect_table_type_from_text(table_name)

    # 1. Match par type
    if target_type:
        for block in blocks:
            if block.table_type_hint == target_type:
                return block

    # 2. Match par mots-clés du nom de tableau dans le texte du bloc
    n_target = _normalize(table_name)
    significant_words = [w for w in n_target.split() if len(w) > 4]
    if significant_words:
        best_block = None
        best_match_count = 0
        for block in blocks:
            n_sample = _normalize(block.text_sample)
            match_count = sum(1 for w in significant_words if w in n_sample)
            if match_count > best_match_count:
                best_match_count = match_count
                best_block = block
        if best_block and best_match_count > 0:
            return best_block

    # 3. Aucun match valide → retourne None pour forcer le fallback page entière
    # (retourner le premier bloc par défaut enverrait une mauvaise zone au LLM)
    return None


def extract_zone_text(
    pdf_path: str,
    page_num: int,
    block: TableBlock,
    margin: float = 5.0,
) -> str:
    """
    Extrait le texte d'une zone spécifique d'une page (entre block.y_start et block.y_end).
    Ajoute une marge pour ne pas couper les bordures de cellules.

    Args:
        margin: Marge en points PDF (défaut 5pt)
    """
    try:
        import pymupdf  # type: ignore

        doc = pymupdf.open(pdf_path)
        page = doc[page_num - 1]
        page_rect = page.rect

        clip = pymupdf.Rect(
            0,
            max(0, block.y_start - margin),
            page_rect.width,
            min(page_rect.height, block.y_end + margin),
        )
        text = page.get_text(clip=clip)
        doc.close()
        return text or ""
    except Exception as e:
        logger.warning("extract_zone_text failed page=%d block=%d: %s", page_num, block.block_index, e)
        return ""


def extract_zone_image_b64(
    pdf_path: str,
    page_num: int,
    block: TableBlock,
    zoom: float = 2.0,
    margin: float = 10.0,
) -> Optional[str]:
    """
    Extrait une zone spécifique d'une page en image PNG base64.
    Idéal pour envoyer uniquement le tableau ciblé au LLM Vision
    (évite la confusion avec les autres tableaux sur la même page).

    Args:
        zoom: Facteur de zoom (2.0 = résolution double pour meilleur OCR)
        margin: Marge en points PDF
    """
    try:
        import base64
        import pymupdf  # type: ignore

        doc = pymupdf.open(pdf_path)
        page = doc[page_num - 1]
        page_rect = page.rect

        clip = pymupdf.Rect(
            0,
            max(0, block.y_start - margin),
            page_rect.width,
            min(page_rect.height, block.y_end + margin),
        )
        mat = pymupdf.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=clip)
        img_bytes = pix.tobytes("png")
        doc.close()
        return base64.b64encode(img_bytes).decode()
    except Exception as e:
        logger.warning(
            "extract_zone_image_b64 failed page=%d block=%d: %s",
            page_num, block.block_index, e,
        )
        return None


def is_multi_table_page(blocks: List[TableBlock]) -> bool:
    """Retourne True si la page contient plusieurs tableaux distincts."""
    return len(blocks) >= 2
