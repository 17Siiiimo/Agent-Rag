"""
Extraction de tableaux financiers via LLM (Groq, Gemini, OpenAI).

1) Vision (Groq ou Gemini) — image de la page → tableau Markdown → DataFrame
2) Texte structuré (Groq, Gemini ou OpenAI) — texte de la page → JSON strict → DataFrame
3) Texte markdown (fallback LLM)
4) Fallback technique (pdfplumber / Camelot / Tabula) si LLM échoue

Usage:
    agent = ExtractionAgent(use_llm=True)
    result = agent.extract(pdf_path, page_num, table_name)
    result.method  # "llm_groq_vision" | "llm_groq" | "failed"
"""

from __future__ import annotations

import logging
import os
import re
import unicodedata
import json
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import pandas as pd
import pdfplumber
import requests

logger = logging.getLogger(__name__)

MIN_ROWS = 3
MIN_COLS = 2

try:
    import pymupdf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pymupdf = None  # type: ignore

try:
    import camelot  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    camelot = None

try:
    import tabula  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tabula = None

# ── Configuration Vision LLM ──────────────────────────────────────────────────
# Résolution de rendu PDF → image (DPI). zoom = dpi / 72.
VISION_RENDER_DPI: int = 300        # DPI défaut (pages texte normales)
VISION_DPI_SCANNED: int = 350       # DPI élevé pour pages scannées / image-heavy
# Seuil heuristique : si le texte extrait par PyMuPDF < N caractères,
# la page est considérée comme scannée (peu ou pas de texte natif).
VISION_SCANNED_TEXT_THRESHOLD: int = 200
# Post-processing PIL (autocontrast + unsharp mask).
# False = désactivé sur pages normales ; True = activé automatiquement sur pages scannées.
VISION_ENABLE_ENHANCE: bool = False
# Sauvegarder les images envoyées au LLM dans output/debug/vision_inputs/ (debug).
VISION_DEBUG_SAVE_INPUTS: bool = True
# Activer le crop intelligent de la zone du tableau avant envoi au LLM.
VISION_ENABLE_TABLE_CROP: bool = False
# Padding (pixels) autour de la bounding box détectée.
VISION_CROP_PADDING: int = 20
# Ratio minimal largeur crop / largeur page (sécurité anti-crop trop étroit).
VISION_MIN_CROP_WIDTH_RATIO: float = 0.30
# Ratio minimal hauteur crop / hauteur page (sécurité anti-crop trop court).
VISION_MIN_CROP_HEIGHT_RATIO: float = 0.15
# Dossier de sortie pour les images debug.
VISION_DEBUG_OUTPUT_DIR: str = "trash"
# Sauvegarder la réponse brute du Vision LLM (texte JSON/Markdown exact) pour debug.
VISION_DEBUG_SAVE_RAW_OUTPUT: bool = False
# Dossier de sortie pour les réponses brutes LLM.
VISION_DEBUG_RAW_OUTPUT_DIR: str = "output/debug/vision_raw"
# ── Multi-table isolation (nouvelle logique) ──────────────────────────────────
# Activer l'isolation automatique du tableau cible quand plusieurs tableaux sont
# détectés sur la même page (ex: BILAN ACTIF + BILAN PASSIF sur la même page).
# True = toujours activé (recommandé) ; False = désactivé (comportement legacy).
VISION_MULTI_TABLE_ISOLATION: bool = False
# Sauvegarder les fichiers debug de l'isolation multi-tableaux :
#   page_{N}_full.png               — page complète rendue
#   page_{N}_target_title_detect.json — résultat de la détection
#   page_{N}_{table}_crop.png       — crop du tableau cible
VISION_DEBUG_MULTI_TABLE_ISOLATION: bool = False
# ── Pipeline coarse-to-fine ───────────────────────────────────────────────────
# Activer le pipeline coarse-to-fine (localisation LLM passe 1 → crop → extraction passe 2).
# False par défaut (comportement legacy préservé) ; mettre True pour activer.
VISION_COARSE_TO_FINE: bool = False
# Seuil de surface (pixels) au-delà duquel un crop est découpé en sous-crops verticaux.
VISION_SPLIT_PIXEL_THRESHOLD: int = 2_400_000  # ≈ 1600×1500 px
# Seuil hauteur_crop / hauteur_page pour déclencher le découpage vertical.
VISION_SPLIT_HEIGHT_RATIO: float = 0.75
# Nombre de sous-crops verticaux en cas de découpage (2 ou 3).
VISION_SPLIT_N_SUBCROP: int = 2
# Chevauchement en pixels entre sous-crops adjacents (évite les coupures de lignes).
VISION_SPLIT_OVERLAP_PX: int = 80
# ── Mode localisation seule ───────────────────────────────────────────────────
# Quand True : _simple_vision_extract s'arrête après la localisation visuelle du tableau.
# Aucune extraction, aucun crop, aucun split. Résultat JSON loggué, DataFrame vide retourné.
# Utiliser pour déboguer / valider les bboxes avant d'activer l'extraction complète.
VISION_LOCALIZATION_ONLY: bool = False
# ─────────────────────────────────────────────────────────────────────────────
# ── Pipeline multi-tableaux (PHASES 1 → 4) ───────────────────────────────────
# Activer le pipeline multi-tableaux (détection LLM → crop local → split local
# → extraction sur crop). Tenté en priorité dans _simple_vision_extract si True.
# False = désactivé, comportement legacy préservé.
MULTI_TABLE_PAGE_MODE: bool = False
MULTI_TABLE_EAGER_EXTRACT: bool = False
# Activer le découpage vertical automatique sur les crops larges.
MULTI_TABLE_SPLIT_ENABLED: bool = True
# Ratio crop_height / page_height au-delà duquel le split est déclenché.
MULTI_TABLE_SPLIT_HEIGHT_RATIO: float = 0.55
# Surface pixel (crop_w × crop_h) au-delà de laquelle le split est déclenché.
MULTI_TABLE_SPLIT_PIXEL_THRESHOLD: int = 1_600_000
# Nombre de sous-crops verticaux par défaut (3 si crop très haut).
MULTI_TABLE_SPLIT_DEFAULT_N: int = 2
# Chevauchement en pixels entre sous-crops adjacents (évite les coupures de lignes).
MULTI_TABLE_SPLIT_OVERLAP_PX: int = 80
# Padding pixels ajouté autour du crop du tableau sélectionné.
MULTI_TABLE_CROP_PADDING_PX: int = 12
# Sauvegarder les images intermédiaires du pipeline multi-tableaux (debug).
MULTI_TABLE_DEBUG_SAVE_IMAGES: bool = False
# Sauvegarder les crops du pipeline simplifié (Phase 2 : _crop_detected_tables).
MULTI_TABLE_DEBUG_SAVE_CROPS: bool = False
MULTI_TABLE_VALIDATE_SELECTED_CROP: bool = True
MULTI_TABLE_VALIDATE_ONLY_AMBIGUOUS: bool = True
# ─────────────────────────────────────────────────────────────────────────────


def _is_valid_table(df: pd.DataFrame) -> bool:
    """Vérifie qu'un DataFrame ressemble à un tableau financier."""
    if df is None or df.empty:
        return False
    if len(df) < MIN_ROWS or len(df.columns) < MIN_COLS:
        return False
    # Itérer par position pour éviter df[col] = DataFrame quand colonnes dupliquées
    for i in range(len(df.columns)):
        ser = df.iloc[:, i]
        if hasattr(ser, "str") and ser.astype(str).str.contains(r"\d{3,}", regex=True).any():
            return True
    return False


def _is_usable_partial(df: pd.DataFrame) -> bool:
    """Au moins 3 lignes et 2 colonnes pour ne pas renvoyer du faux tableau (ex. 2x1)."""
    if df is None or df.empty:
        return False
    return len(df) >= MIN_ROWS and len(df.columns) >= MIN_COLS


def _is_usable_llm_table(df: pd.DataFrame) -> bool:
    """
    Critère minimal pour accepter un résultat Vision LLM sans fallback technique.

    On autorise les tableaux partiels (ex: 1 ligne extraite) tant que:
    - au moins 2 colonnes sont présentes
    - au moins une ligne contient un libellé non vide
    """
    if df is None or df.empty:
        return False
    if len(df.columns) < MIN_COLS:
        return False
    try:
        first_col = df.iloc[:, 0].astype(str).str.strip()
        return (first_col != "").any()
    except Exception:
        return False


def _table_family(table_name: str) -> str:
    """Famille normalisee du tableau demande."""
    name = _norm(table_name or "").replace("\n", " ")
    if "actif" in name and "passif" not in name:
        return "bilan_actif"
    if "passif" in name:
        return "bilan_passif"
    if (
        "cpc" in name
        or "compte de produits" in name
        or "produits et charges" in name
        or "compte de resultat" in name
    ):
        return "cpc"
    return "unknown"


def _target_section_type(table_name: str) -> Optional[str]:
    """Nom de section attendu par le prompt de localisation pleine page."""
    family = _table_family(table_name)
    if family == "bilan_actif":
        return "BILAN_ACTIF"
    if family == "bilan_passif":
        return "BILAN_PASSIF"
    if family == "cpc":
        return "CPC"
    return None


_TABLE_ANCHORS: dict[str, tuple[tuple[str, float], ...]] = {
    "bilan_actif": (
        ("bilan actif", 28.0),
        ("total de l'actif", 42.0),
        ("total actif", 36.0),
        ("immobilisations", 14.0),
        ("actif circulant", 18.0),
        ("tresorerie actif", 18.0),
        ("creances de l'actif circulant", 20.0),
        ("valeurs en caisse", 18.0),
    ),
    "bilan_passif": (
        ("bilan passif", 28.0),
        ("total du passif", 42.0),
        ("total passif", 36.0),
        ("capitaux propres", 18.0),
        ("dettes de financement", 16.0),
        ("passif circulant", 18.0),
        ("tresorerie passif", 18.0),
        ("depots de la clientele", 18.0),
    ),
    "cpc": (
        ("compte de produits et charges", 30.0),
        ("comptes de produits et charges", 30.0),
        ("compte de resultat consolide", 34.0),
        ("compte de resultats consolide", 30.0),
        ("produits d'exploitation", 18.0),
        ("charges d'exploitation", 18.0),
        ("resultat d'exploitation", 18.0),
        ("resultat courant", 14.0),
        ("resultat net", 36.0),
        ("produit net bancaire", 22.0),
        ("interets et produits assimiles", 22.0),
        ("interets et charges assimiles", 22.0),
        ("marge d'interet", 28.0),
        ("marge sur commissions", 28.0),
        ("resultat net part du groupe", 30.0),
    ),
}


_TABLE_NEGATIVE_ANCHORS: dict[str, tuple[str, ...]] = {
    "bilan_actif": ("bilan passif", "total du passif", "compte de produits et charges", "resultat net"),
    "bilan_passif": ("bilan actif", "total de l'actif", "compte de produits et charges", "resultat net"),
    "cpc": ("bilan actif", "bilan passif", "total de l'actif", "total du passif"),
}


def _score_text_for_table(text: str, table_name: str) -> tuple[float, dict[str, Any]]:
    """Score local d'un texte/crop pour verifier qu'il correspond au tableau cible."""
    family = _table_family(table_name)
    text_norm = _norm(text or "")
    text_ns = re.sub(r"\s+", " ", text_norm)
    score = 0.0
    positive_hits: list[str] = []
    negative_hits: list[str] = []

    for anchor, weight in _TABLE_ANCHORS.get(family, ()):
        anchor_norm = _norm(anchor)
        if anchor_norm in text_ns:
            score += weight
            positive_hits.append(anchor)

    for anchor in _TABLE_NEGATIVE_ANCHORS.get(family, ()):
        anchor_norm = _norm(anchor)
        if anchor_norm in text_ns:
            score -= 35.0
            negative_hits.append(anchor)

    if re.search(r"\d[\d\s.,]{2,}", text_norm):
        score += 8.0
    if len(text_norm.strip()) < 80:
        score -= 25.0
    if len(positive_hits) >= 3:
        score += 15.0

    return score, {
        "family": family,
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "text_len": len(text_norm),
    }


def _extract_text_from_bbox_px(
    pdf_path: str,
    page_num: int,
    bbox_px: list[int],
    image_size: tuple[int, int],
    padding_px: int = 0,
) -> str:
    """Extrait le texte PDF qui correspond approximativement a une bbox image."""
    if pymupdf is None:
        return ""
    try:
        img_w, img_h = image_size
        if img_w <= 0 or img_h <= 0:
            return ""
        x0, y0, x1, y1 = [int(v) for v in bbox_px]
        x0 = max(0, x0 - padding_px)
        y0 = max(0, y0 - padding_px)
        x1 = min(img_w, x1 + padding_px)
        y1 = min(img_h, y1 + padding_px)
        doc = pymupdf.open(pdf_path)
        try:
            page = doc[page_num - 1]
            rect = page.rect
            clip = pymupdf.Rect(
                x0 / img_w * rect.width,
                y0 / img_h * rect.height,
                x1 / img_w * rect.width,
                y1 / img_h * rect.height,
            )
            return page.get_text(clip=clip) or ""
        finally:
            doc.close()
    except Exception as exc:
        logger.debug("_extract_text_from_bbox_px failed: %s", exc)
        return ""


def _table_completeness(df: pd.DataFrame, table_name: str) -> dict[str, Any]:
    """Wrapper tolerant autour du controle de completude."""
    try:
        from .completeness_checker import check_completeness
        return check_completeness(df, table_name)
    except Exception as exc:
        logger.debug("check_completeness unavailable: %s", exc)
        return {
            "completeness_score": 1.0 if _is_valid_table(df) else 0.0,
            "is_complete": _is_valid_table(df),
            "missing_anchors": [],
        }


def _is_acceptable_extraction(df: pd.DataFrame, table_name: str) -> tuple[bool, dict[str, Any]]:
    """Accepte un tableau seulement s'il est valide et raisonnablement complet."""
    completeness = _table_completeness(df, table_name)
    score = float(completeness.get("completeness_score") or 0.0)
    is_complete = bool(completeness.get("is_complete"))
    if is_complete and _is_usable_llm_table(df):
        return True, completeness
    # Certains tableaux partiels restent utiles, mais on exige un score minimal
    # pour eviter de retourner le mauvais crop avec quelques chiffres.
    if _is_valid_table(df) and score >= 0.50:
        return True, completeness
    return False, completeness


def _detect_table_schema(table_name: str) -> Tuple[str, Optional[int]]:
    """
    Détecte le type de tableau et le nombre attendu de colonnes de données (hors colonne Rubrique).

    Retourne (table_type, expected_data_cols) où expected_data_cols peut être None si inconnu/variable.
    """
    name = (table_name or "").lower().replace("\n", " ")
    table_type = "unknown"
    expected: Optional[int] = None

    is_consolide = "consolid" in name
    is_sociaux = "sociaux" in name or "social" in name

    if "flux de trésorerie" in name or "tableau des flux" in name:
        table_type = "flux_tresorerie"
        expected = 2  # Exercice N | Exercice N-1
    elif (
        "cpc" in name
        or "compte de produits et charges" in name
        or "comptes de produits et charges" in name
        or "produits et charges" in name
        or "compte de résultat" in name
        or "compte de resultat" in name
    ):
        table_type = "cpc"
        expected = 3  # Exercice N | Exercice N-1 | Variation
    elif "bilan" in name and "actif" in name and "passif" not in name:
        table_type = "bilan_actif_consolide" if is_consolide else "bilan_actif_sociaux"
        # Hors colonne Rubrique
        expected = 4  # Brut | Amort/Prov | Net N | Net N-1 (sociaux) ou Notes | N | N-1 | Variation (consolide)
    elif "bilan" in name and "passif" in name:
        table_type = "bilan_passif_consolide" if is_consolide else "bilan_passif_sociaux"
        # Hors colonne Rubrique
        expected = 4 if is_consolide else 2
    elif "esg" in name or "tableau de bord" in name:
        table_type = "esg"
        expected = None

    return table_type, expected


_TWO_NUMBERS_RE = re.compile(r"(\d[\d\s.,\-]*\d)\s{2,}(\d[\d\s.,\-]*\d)")


_NUMBER_FRAGMENT_RE = re.compile(r"^\s*[\d\s]{1,15}\s*$")


def _is_number_fragment(val: str) -> bool:
    """True si la valeur ressemble à un fragment de nombre (ex: '683', '157')."""
    s = str(val).strip()
    if not s or s == "-":
        return False
    return bool(_NUMBER_FRAGMENT_RE.match(s)) and len(s.replace(" ", "")) <= 6


def _merge_split_number_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    """
    Fusionne les colonnes adjacentes dont les valeurs sont des fragments de nombres.
    Ex: col "13 157" + col "683" → col "13 157 683"
    Cela arrive quand les espaces (séparateurs de milliers) trompent l'extracteur.
    """
    if df is None or df.empty or df.shape[1] < 3:
        return df, False

    fixed = False
    cols = list(df.columns)
    i = 1  # On commence après la colonne Rubrique
    while i < len(cols) - 1:
        col_curr = cols[i]
        col_next = cols[i + 1]
        curr_vals = df.iloc[:, i].astype(str)
        next_vals = df.iloc[:, i + 1].astype(str)

        # Nombre de lignes où next ressemble à un fragment de nombre court
        n_rows = max(1, len(df))
        frag_count = sum(
            1 for v in next_vals
            if _is_number_fragment(v)
        )
        # Si plus de 50% des lignes de la colonne suivante sont des fragments → fusionner
        if frag_count / n_rows > 0.5:
            merged_vals = curr_vals.str.strip() + " " + next_vals.str.strip()
            merged_vals = merged_vals.str.strip()
            df = df.copy()
            df.iloc[:, i] = merged_vals
            df = df.drop(columns=[col_next])
            cols = list(df.columns)
            fixed = True
            # Ne pas incrémenter i : re-vérifier la colonne courante avec la nouvelle suivante
        else:
            i += 1

    return df, fixed


def _validate_and_fix_columns(df: pd.DataFrame, table_name: str) -> Tuple[pd.DataFrame, bool]:
    """
    Valide et corrige les colonnes d'un tableau selon un schéma connu.

    - Détecte le type de tableau via table_name.
    - Connaît le nombre attendu de colonnes de données (hors colonne Rubrique).
    - Si une colonne de trop : supprime les colonnes entièrement vides ou quasi-vides (<10% cellules remplies).
    - Si une colonne de moins et qu'une colonne contient visiblement deux valeurs collées,
      tente de la scinder en deux.

    Retourne (df_corrigé, was_fixed).
    """
    if df is None or df.empty or len(df.columns) <= 1:
        return df, False

    table_type, expected_data_cols = _detect_table_schema(table_name)
    if expected_data_cols is None:
        return df, False

    nrows, ncols = df.shape
    if ncols <= 1:
        return df, False

    # On considère la première colonne comme Rubrique, le reste = colonnes de données
    data_col_indices = list(range(1, ncols))
    actual_data_cols = len(data_col_indices)
    was_fixed = False

    def non_empty_ratio(col_idx: int) -> float:
        col = df.iloc[:, col_idx]
        non_empty = col.astype(str).apply(lambda v: bool(str(v).strip())).sum()
        return non_empty / max(1, nrows)

    # Trop de colonnes :
    # Historique : on supprimait jusqu'à retomber sur le schéma attendu.
    # Problème : certains PDFs (ex: CPC sur plusieurs pages) peuvent avoir une
    # colonne peu remplie mais utile (elle peut contenir des chiffres).
    # Ici, on ne supprime que les colonnes vraiment inutiles (quasi vides ET
    # sans aucun chiffre), pour éviter de perdre des colonnes valides.
    if actual_data_cols > expected_data_cols:
        # Trier les colonnes candidates par ratio croissant
        candidates = sorted(
            data_col_indices,
            key=lambda j: non_empty_ratio(j),
        )
        for j in candidates:
            if actual_data_cols <= expected_data_cols:
                break
            # Ne supprimer que si la colonne est quasi vide ET ne contient aucun chiffre
            col = df.iloc[:, j].astype(str)
            has_digits = col.str.contains(r"\d", regex=True).any()
            if non_empty_ratio(j) < 0.02 and not has_digits:
                col_name = df.columns[j]
                df = df.drop(columns=[col_name])
                was_fixed = True
                ncols = df.shape[1]
                data_col_indices = list(range(1, ncols))
                actual_data_cols = len(data_col_indices)

    # Pas assez de colonnes : tenter de scinder une colonne contenant deux valeurs collées
    if actual_data_cols < expected_data_cols:

        def count_two_numbers(col_idx: int) -> int:
            col = df.iloc[:, col_idx].astype(str)
            return sum(1 for v in col if _TWO_NUMBERS_RE.search(v.strip()))

        best_idx = None
        best_count = 0
        for j in data_col_indices:
            c = count_two_numbers(j)
            if c > best_count:
                best_count = c
                best_idx = j

        if best_idx is not None and best_count >= max(1, int(0.2 * nrows)):
            col_name = df.columns[best_idx]
            ser = df.iloc[:, best_idx].astype(str)
            left_vals = ser.apply(lambda v: _split_cell_into_n_values(v, 2)[0])
            right_vals = ser.apply(lambda v: _split_cell_into_n_values(v, 2)[1])
            df[col_name] = left_vals
            insert_pos = best_idx + 1
            new_col_name = f"{col_name}_2"
            df.insert(insert_pos, new_col_name, right_vals)
            was_fixed = True

    return df, was_fixed


# Regex pour en-têtes de type date (ex. 31/12/2024, 1/1/2023, 31-12-2024)
_DATE_HEADER_RE = re.compile(r"^\s*\d{1,2}[/-]\d{1,2}[/-]\d{4}\s*$")
# Substring pour détecter une date dans une cellule
_DATE_SUBSTR_RE = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{4}")


def _extract_all_dates(text: str) -> List[str]:
    """Extrait toutes les dates (DD/MM/YYYY ou DD-MM-YYYY) d'une chaîne. Une colonne peut contenir « 31/12/2025 31/12/2024 » -> 2 colonnes."""
    if not text or not isinstance(text, str):
        return []
    return _DATE_SUBSTR_RE.findall(text.strip())


def _is_date_header(value: str) -> bool:
    """True si la chaîne ressemble à une date type DD/MM/YYYY (en-tête de colonne Comptes Sociaux)."""
    if not value or not isinstance(value, str):
        return False
    return bool(_DATE_HEADER_RE.match(value.strip()))


def _cell_contains_date(value: str) -> bool:
    """True si la chaîne contient un motif date DD/MM/YYYY (pour détecter en-têtes sur plusieurs lignes)."""
    if not value or not isinstance(value, str):
        return False
    return bool(_DATE_SUBSTR_RE.search(str(value).strip()))


def _is_generic_column_name(s: str) -> bool:
    """True si le nom de colonne est générique (col, col_1, 0, 1...) → on préfère lire la cellule."""
    if not s:
        return True
    s = str(s).strip()
    if re.match(r"^col(_\d+)?$", s, re.I):
        return True
    if s.isdigit():
        return True
    return False


def _get_header_for_col(df: pd.DataFrame, j: int) -> str:
    """
    En-tête de la colonne j : nom de colonne ou première cellule.
    Si le nom est générique (col, col_1), on lit les 5 premières lignes pour trouver une date (ex. « 31/12/2025 31/12/2024 »).
    """
    try:
        c = df.columns[j]
        s = str(c or "").strip()
        if s and (_is_date_header(s) or _cell_contains_date(s)):
            return s
        if s and not _is_generic_column_name(s):
            return s
    except Exception:
        pass
    # Chercher une date dans les 5 premières lignes (dates parfois sous BILAN ACTIF / PASSIF)
    for row_idx in range(min(5, len(df))):
        try:
            cell = str(df.iloc[row_idx, j] or "").strip()
            if _is_date_header(cell):
                return cell
            if _cell_contains_date(cell):
                # Garder la cellule complète pour permettre _extract_all_dates (ex. « 31/12/2025 31/12/2024 » → 2 colonnes)
                return cell
        except Exception:
            pass
    if len(df) > 0:
        try:
            return str(df.iloc[0, j] or "").strip()
        except Exception:
            pass
    return ""


def _col_has_date_in_header(df: pd.DataFrame, j: int) -> bool:
    """True si la colonne j a une date dans son en-tête (nom de colonne ou une des 5 premières lignes)."""
    h = _get_header_for_col(df, j)
    return _is_date_header(h) or _cell_contains_date(h)


def _cell_looks_numeric(value: str) -> bool:
    """True si la cellule contient un nombre (ex. 1 362 860, 1.5, 1,234)."""
    if not value or not isinstance(value, str):
        return False
    s = value.strip().replace(" ", "").replace("\xa0", "")
    return bool(re.search(r"^\d[\d\s,.\-]*\d$|^\d+$", s))


def _find_value_column_indices(df: pd.DataFrame, start_col: int, max_cols: int = 4) -> List[int]:
    """
    Fallback quand aucune date n'est détectée : colonnes qui contiennent des montants (chiffres)
    dans les lignes 2 à min(12, nrows). Retourne [start_col, j1, j2, ...] (max start_col + max_cols).
    """
    ncols = len(df.columns)
    indices = [start_col]
    for j in range(start_col + 1, min(start_col + max_cols + 1, ncols)):
        # Au moins une cellule numérique dans les lignes 2..12
        for row_idx in range(2, min(13, len(df))):
            try:
                cell = str(df.iloc[row_idx, j] or "").strip()
                if _cell_looks_numeric(cell):
                    indices.append(j)
                    break
            except Exception:
                pass
    return indices


def _date_column_indices(df: pd.DataFrame, start_col: int) -> List[int]:
    """
    Comptes Sociaux : 1 colonne nom (rubrique) + colonnes dates.
    On garde start_col (rubrique), puis on ajoute les colonnes qui ont une date en en-tête.
    On peut sauter des colonnes « trou » (ex. libellé sur plusieurs colonnes) : on s'arrête
    seulement après avoir trouvé au moins une colonne date puis une colonne sans date.
    """
    ncols = len(df.columns)
    indices = [start_col]
    found_any_date_col = False
    for j in range(start_col + 1, ncols):
        if _col_has_date_in_header(df, j):
            indices.append(j)
            found_any_date_col = True
        elif found_any_date_col:
            break
    return indices


def _split_cell_into_n_values(cell: str, n: int) -> List[str]:
    """
    Découpe une cellule qui contient n valeurs (ex. « 1 793 793 1 362 860 » pour 2 colonnes).
    D’abord par 2+ espaces ; sinon partage des tokens en n groupes.
    """
    if n <= 1:
        return [cell.strip()] if cell else [""]
    s = (cell or "").strip()
    if not s:
        return [""] * n
    parts = re.split(r"\s{2,}", s)
    if len(parts) >= n:
        return (parts[:n] + [""] * n)[:n]
    tokens = s.split()
    if not tokens:
        return [""] * n
    size = (len(tokens) + n - 1) // n
    result = []
    for i in range(n):
        start = i * size
        end = min((i + 1) * size, len(tokens))
        result.append(" ".join(tokens[start:end]) if start < end else "")
    return (result + [""] * n)[:n]


def _column_names_for_section(
    df: pd.DataFrame, col_indices: List[int], expanded_dates: Optional[List[List[str]]] = None
) -> List[str]:
    """Noms de colonnes : Rubrique + en-têtes (une par date ; si une col. contient 2 dates, expanded_dates a 2 noms)."""
    names = ["Rubrique"]
    if expanded_dates is not None:
        for date_list in expanded_dates:
            names.extend(date_list)
        return names
    for i, j in enumerate(col_indices[1:], start=1):
        h = _get_header_for_col(df, j)
        names.append(h if (_is_date_header(h) or _cell_contains_date(h)) else f"Exercice {i}")
    return names


def _build_section_df(df: pd.DataFrame, col_indices: List[int]) -> pd.DataFrame:
    """
    Construit le DataFrame de la section : Rubrique + colonnes dates.
    Si une colonne a un en-tête avec plusieurs dates (ex. « 31/12/2025 31/12/2024 »), on la découpe en une colonne par date.
    """
    if len(col_indices) <= 1:
        return df.iloc[:, col_indices].copy()
    start_col = col_indices[0]
    first_date_col = col_indices[1]
    # Rubrique : fusion des colonnes [start_col .. first_date_col)
    if first_date_col > start_col + 1:
        rubrique_parts = df.iloc[:, start_col:first_date_col]
        rubrique = rubrique_parts.apply(
            lambda row: " ".join(str(x).strip() for x in row if str(x).strip()), axis=1
        )
    else:
        rubrique = df.iloc[:, start_col].copy()

    # Colonnes dates : une colonne par date ; si une col. contient « 31/12/2025 31/12/2024 », on crée 2 colonnes
    date_blocks: List[pd.Series] = []
    expanded_dates: List[List[str]] = []
    for j in col_indices[1:]:
        h = _get_header_for_col(df, j)
        dates = _extract_all_dates(h)
        if len(dates) >= 2:
            ser = df.iloc[:, j]
            for i in range(len(dates)):
                part = ser.apply(
                    lambda c, idx=i, n=len(dates): _split_cell_into_n_values(str(c or ""), n)[idx]
                )
                date_blocks.append(part)
            expanded_dates.append(dates)
        else:
            date_blocks.append(df.iloc[:, j])
            expanded_dates.append([h] if (_is_date_header(h) or _cell_contains_date(h)) else [f"Exercice {len(date_blocks) + 1}"])

    out = pd.concat([rubrique] + date_blocks, axis=1)
    names = ["Rubrique"] + [name for sub in expanded_dates for name in sub]
    out.columns = names[: len(out.columns)]
    return out


def _row_text(row, df: pd.DataFrame) -> str:
    """Texte normalisé d'une ligne pour détecter les sections (actif, passif, hors bilan)."""
    try:
        return " ".join(str(v) for v in row.astype(str).values).strip().lower()
    except Exception:
        return " ".join(str(row.get(c, "")) for c in df.columns).strip().lower()


def _filter_bilan_section(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """
    Sur une page qui contient Actif et Passif côte à côte (Comptes Sociaux : 1 colonne nom + 2 colonnes dates par bloc),
    ne garde que la section demandée et les colonnes de montants associées (rubrique + colonnes de valeurs).
    """
    if df is None or df.empty or len(df) < 2:
        return df
    name = table_name.lower().replace("\n", " ")
    ncols = len(df.columns)
    # Première colonne pour détecter les lignes actif / passif
    first_col_series = df.iloc[:, 0]
    col_text = first_col_series.astype(str).str.strip().str.lower()

    want_actif = "bilan actif" in name or ("actif" in name and "passif" not in name and "hors" not in name)
    want_passif = "bilan passif" in name or ("passif" in name and "hors" not in name)
    want_hors_bilan = "hors bilan" in name

    if want_actif:
        end_idx = len(df)
        for i in range(len(df)):
            cell = col_text.iloc[i]
            if "passif" in cell and "actif" not in cell:
                end_idx = i
                break
        for i in range(end_idx - 1, -1, -1):
            cell = col_text.iloc[i]
            if "total" in cell and "actif" in cell:
                end_idx = i + 1
                break
        out = df.iloc[:end_idx]
        # Comptes sociaux – ACTIF : une rubrique + jusqu'à 4 colonnes de montants
        # (Brut, amortissements/provisions, net exercice, net exercice précédent).
        col_indices = _date_column_indices(out, 0)
        # Si les dates ne sont pas clairement détectées, on tombe sur les colonnes "numériques"
        if len(col_indices) < 2:
            col_indices = _find_value_column_indices(out, 0, max_cols=4)
        # Fallback minimal pour éviter de perdre les données
        if len(col_indices) < 2 and len(out.columns) >= 3:
            col_indices = [0, 1, 2]
        elif len(col_indices) < 2 and len(out.columns) >= 2:
            col_indices = [0, 1]
        # Si seulement 2 colonnes détectées mais plus sont disponibles, inclure les suivantes
        ncols_out = len(out.columns)
        if len(col_indices) == 2 and ncols_out >= 4:
            col_indices = [0, 1, 2, 3]
        elif len(col_indices) == 2 and ncols_out == 3:
            col_indices = [0, 1, 2]
        out = _build_section_df(out, col_indices)
        return out

    if want_passif:
        # Ancres structurelles pour mieux isoler le vrai bloc passif quand plusieurs tableaux
        # coexistent sur la même page (cas fréquent en consolidé assurance).
        passif_needles = (
            "passif",
            "capitaux propres",
            "interets minoritaires",
            "intérêts minoritaires",
            "dettes de financement",
            "provisions techniques",
            "passif circulant",
            "tresorerie - passif",
            "trésorerie - passif",
            "tresorerie-passif",
            "trésorerie-passif",
            "total passif",
            "total du passif",
        )
        actif_needles = (
            "actif immobilise",
            "actif immobilisé",
            "actif circulant",
            "immobilisations",
            "creances de l'actif",
            "créances de l'actif",
            "valeurs en caisse",
            "total actif",
            "total de l'actif",
            "tresorerie - actif",
            "trésorerie - actif",
            "tresorerie-actif",
            "trésorerie-actif",
        )

        def _passif_signal(row_txt: str) -> int:
            return sum(1 for n in passif_needles if n in row_txt)

        def _actif_signal(row_txt: str) -> int:
            return sum(1 for n in actif_needles if n in row_txt)

        passif_start = None
        for i in range(len(df)):
            cell = col_text.iloc[i]
            row_full = _row_text(df.iloc[i], df)
            if "passif" in cell or ("passif" in row_full and "hors bilan" not in row_full):
                passif_start = i
                break
        # Fallback intelligent : première ligne avec signal passif dominant
        if passif_start is None:
            for i in range(len(df)):
                row_full = _row_text(df.iloc[i], df)
                if _passif_signal(row_full) >= 1 and _passif_signal(row_full) >= _actif_signal(row_full):
                    passif_start = i
                    break
        # Si "passif" n'est trouvé qu'à mi-tableau (ex: "Autres passifs"),
        # c'est que les premières lignes (Banques centrales, Dépôts...) ne contiennent
        # pas le mot "passif" — on commence depuis le début du DataFrame.
        if passif_start is None or passif_start > len(df) // 3:
            passif_start = 0
        end_passif = len(df)
        stop_markers = (
            "perimetre de consolidation",
            "périmètre de consolidation",
            "methode de consolidation",
            "méthode de consolidation",
            "integration globale",
            "intégration globale",
            "mise en equivalence",
            "mise en équivalence",
            "pourcentage",
        )
        for i in range(passif_start, len(df)):
            cell = col_text.iloc[i]
            row_full = _row_text(df.iloc[i], df)
            if "hors bilan" in cell or "hors bilan" in row_full:
                end_passif = i
                break
            if "total" in cell and "passif" in cell:
                end_passif = i + 1
                break
            # Éviter de mélanger le tableau "BILAN PASSIF" avec des tableaux narratifs
            # présents en bas de page (ex: périmètre de consolidation).
            if any(m in row_full for m in stop_markers):
                end_passif = i
                break
        out = df.iloc[passif_start:end_passif]
        # Comptes sociaux : trouver où commence le bloc passif (rubrique + dates)
        col_indices = None
        for start_col in [0, 3, 6]:
            if start_col >= ncols:
                continue
            indices = _date_column_indices(out, start_col)
            if len(indices) >= 2:
                col_indices = indices
                break
        if col_indices is None:
            col_indices = _date_column_indices(out, 0)
        # Fallback : si aucune date détectée, garder les colonnes qui contiennent des montants
        if len(col_indices) < 2:
            for start in [0, 3, 6]:
                if start >= ncols:
                    continue
                value_indices = _find_value_column_indices(out, start, max_cols=3)
                if len(value_indices) >= 2:
                    col_indices = value_indices
                    break
            if len(col_indices) < 2 and ncols >= 3:
                col_indices = [0, 1, 2]
            elif len(col_indices) < 2 and ncols >= 2:
                col_indices = [0, 1]
        # Si seulement 2 colonnes détectées mais le tableau en a 3+, inclure la 3e
        if len(col_indices) == 2 and ncols >= 3 and col_indices[1] + 1 < ncols:
            col_indices = list(col_indices) + [col_indices[1] + 1]
        out = _build_section_df(out, col_indices)

        # Nettoyage final : enlever les lignes clairement "actif"/tableau voisin.
        if not out.empty and "Rubrique" in out.columns:
            rub = out["Rubrique"].astype(str).str.lower().str.strip()
            keep_mask = []
            for txt in rub:
                if any(
                    bad in txt
                    for bad in (
                        "perimetre de consolidation",
                        "périmètre de consolidation",
                        "methode de consolidation",
                        "méthode de consolidation",
                        "integration globale",
                        "intégration globale",
                        "mise en equivalence",
                        "mise en équivalence",
                        "societe",
                        "société",
                    )
                ):
                    keep_mask.append(False)
                    continue
                # Toujours garder les lignes totalement vides (séparateurs) et totaux passif.
                if txt == "" or ("total" in txt and "passif" in txt):
                    keep_mask.append(True)
                    continue
                p_sig = _passif_signal(txt)
                a_sig = _actif_signal(txt)
                # Rejeter uniquement les lignes où le signal actif domine clairement.
                keep_mask.append(not (a_sig >= 2 and a_sig > p_sig))
            cleaned = out.loc[keep_mask].reset_index(drop=True)
            # Sécurité : ne pas retourner un tableau trop vidé.
            if len(cleaned) >= max(3, len(out) // 3):
                out = cleaned
        return out

    if want_hors_bilan:
        for i in range(len(df)):
            cell = col_text.iloc[i]
            row_full = _row_text(df.iloc[i], df)
            if "hors bilan" in cell or "hors bilan" in row_full:
                return df.iloc[i:]
        return df

    # CPC : Compte de produits et charges / Compte de résultat (et variante consolidé)
    want_cpc = (
        "cpc" in name
        or "compte de produits" in name
        or "produits et charges" in name
        or "compte de résultat" in name
        or "compte de resultat" in name
    )
    if want_cpc:
        end_idx = len(df)
        for i in range(len(df)):
            row_full = _row_text(df.iloc[i], df)
            if "état des dérogations" in row_full or "derogations" in row_full:
                end_idx = i
                break
            if "créances sur les établissements" in row_full or "créances sur la clientèle" in row_full:
                end_idx = i
                break
            if "tableau des flux" in row_full or "flux de trésorerie" in row_full:
                end_idx = i
                break
        start_idx = 0
        result = df.iloc[start_idx:end_idx]
        if len(result) >= 3:
            return result
    return df


def _pdfplumber_extract(pdf_path: str, page_num: int) -> pd.DataFrame:
    """Niveau 1 : pdfplumber avec paramètres optimisés."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num < 1 or page_num > len(pdf.pages):
                return pd.DataFrame()
            page = pdf.pages[page_num - 1]
            settings = {
                "vertical_strategy": "lines",
                "horizontal_strategy": "lines",
                "snap_tolerance": 4,
                "snap_x_tolerance": 4,
                "snap_y_tolerance": 4,
                "join_tolerance": 4,
                "edge_min_length": 5,
                "intersection_tolerance": 5,
                "text_x_tolerance": 4,
                "text_y_tolerance": 4,
            }
            tables = page.extract_tables(settings)
            if not tables:
                settings["vertical_strategy"] = "text"
                settings["horizontal_strategy"] = "text"
                tables = page.extract_tables(settings)
            if not tables:
                return pd.DataFrame()
            best = max(
                tables,
                key=lambda t: len(t) * len(t[0]) if t and t[0] else 0,
            )
            if not best or len(best) < 2:
                return pd.DataFrame()
            max_c = max(len(r) for r in best)
            normed = [(list(r) + [""] * max_c)[:max_c] for r in best]
            headers = [str(h or "").strip() for h in normed[0]]
            rows = [[str(c or "").strip() for c in row] for row in normed[1:]]
            df = pd.DataFrame(rows, columns=headers)
            df = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]
            df = df.loc[:, df.apply(lambda c: any(str(v).strip() for v in c))]
            return df.reset_index(drop=True)
    except Exception as e:
        logger.debug("pdfplumber extract failed: %s", e)
        return pd.DataFrame()


def _camelot_extract(pdf_path: str, page_num: int, table_name: str) -> pd.DataFrame:
    """
    Niveau 1 : Camelot (lattice puis stream).

    - Essaie d'abord flavor="lattice" pour les tableaux avec bordures.
    - Si aucun tableau ou accuracy < 80, essaie flavor="stream".
    - Retourne le DataFrame du meilleur tableau (plus grand nombre de cellules).
    - Si Camelot n'est pas installé, log un warning et retourne un DataFrame vide.
    """
    if camelot is None:
        logger.warning("Camelot non installé, niveau Camelot ignoré pour %s p.%s", pdf_path, page_num)
        return pd.DataFrame()

    try:
        pages = str(page_num)
        best_table = None
        best_cells = 0

        def _update_best(tables):
            nonlocal best_table, best_cells
            for t in tables:
                df_t = t.df if hasattr(t, "df") else None
                if df_t is None or df_t.empty:
                    continue
                cells = df_t.shape[0] * df_t.shape[1]
                if cells > best_cells:
                    best_cells = cells
                    best_table = df_t

        # 1) Lattice
        tables_lattice = camelot.read_pdf(pdf_path, pages=pages, flavor="lattice")
        max_acc_lattice = max((t.parsing_report.get("accuracy", 0) for t in tables_lattice), default=0) if tables_lattice else 0
        if tables_lattice:
            _update_best(tables_lattice)

        # 2) Stream si aucun tableau ou accuracy faible
        if not tables_lattice or max_acc_lattice < 80:
            tables_stream = camelot.read_pdf(
                pdf_path,
                pages=pages,
                flavor="stream",
                edge_tol=50,
                row_tol=10,
            )
            if tables_stream:
                _update_best(tables_stream)

        if best_table is None or best_table.empty:
            return pd.DataFrame()

        df = best_table.copy()
        df = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]
        df = df.loc[:, df.apply(lambda c: any(str(v).strip() for v in c))]
        return df.reset_index(drop=True)
    except Exception as e:
        logger.debug("Camelot extract failed: %s", e)
        return pd.DataFrame()


def _tabula_extract(pdf_path: str, page_num: int) -> pd.DataFrame:
    """
    Extraction via Tabula-py (lattice puis stream).

    - Nécessite Java + tabula-py installés.
    - Essaie d'abord lattice=True (tableaux avec bordures).
    - Si aucun tableau utilisable, essaie stream=True, guess=False.
    """
    if tabula is None:
        logger.debug("Tabula non installé, niveau Tabula ignoré pour %s p.%s", pdf_path, page_num)
        return pd.DataFrame()

    try:
        page = str(page_num)

        def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.copy()
            df = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]
            df = df.loc[:, df.apply(lambda c: any(str(v).strip() for v in c))]
            return df.reset_index(drop=True)

        # 1) Lattice
        dfs = tabula.read_pdf(
            pdf_path,
            pages=page,
            multiple_tables=False,
            lattice=True,
            pandas_options={"header": None},
        )
        if dfs:
            df_lat = _clean_df(dfs[0])
            if _is_valid_table(df_lat):
                return df_lat

        # 2) Stream
        dfs = tabula.read_pdf(
            pdf_path,
            pages=page,
            multiple_tables=False,
            stream=True,
            guess=False,
            pandas_options={"header": None},
        )
        if dfs:
            df_stream = _clean_df(dfs[0])
            if _is_valid_table(df_stream) or _is_usable_partial(df_stream):
                return df_stream

        return pd.DataFrame()
    except Exception as e:
        logger.debug("Tabula extract failed: %s", e)
        return pd.DataFrame()


def _markdown_table_to_dataframe(table_md: str) -> pd.DataFrame:
    """Convertit un tableau Markdown en DataFrame."""
    lines = [l.strip() for l in table_md.splitlines() if l.strip()]
    if not lines:
        return pd.DataFrame()

    def _split_row(row: str) -> List[str]:
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        return [cell.strip() for cell in row.split("|")]

    header = None
    rows: List[List[str]] = []
    for line in lines:
        if set(line.replace("|", "").replace(" ", "")) == {"-"}:
            continue
        cells = _split_row(line)
        if header is None:
            header = cells
        else:
            if len(cells) < len(header):
                cells.extend([""] * (len(header) - len(cells)))
            elif len(cells) > len(header):
                cells = cells[: len(header)]
            rows.append(cells)

    if header is None:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=header)


def _extract_markdown_table(content: str) -> str:
    if not content:
        return ""
    md_match = re.search(r"(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)", content)
    return md_match.group(1) if md_match else content.strip()


def _normalize_api_provider(api_provider: Optional[str]) -> str:
    p = (api_provider or "groq").strip().lower()
    if p in {"gpt", "openai", "gpt-5", "gpt-5.4"}:
        return "gpt-5.4"
    if p in {"gemini", "google"}:
        return "gemini"
    return "groq"


def _provider_api_key(api_provider: str, api_key: Optional[str] = None) -> str:
    if api_key:
        return api_key
    provider = _normalize_api_provider(api_provider)
    if provider == "gemini":
        return os.environ.get("GEMINI_API_KEY", "")
    if provider == "gpt-5.4":
        return os.environ.get("OPENAI_API_KEY", "")
    return os.environ.get("GROQ_API_KEY", "")


GROQ_TEXT_MODEL_CANDIDATES = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

GROQ_VISION_MODEL_CANDIDATES = [
    "meta-llama/llama-4-scout-17b-16e-instruct",      # modèle vision actif
    "meta-llama/llama-4-maverick-17b-128e-instruct",  # fallback llama-4
    "llama-3.2-90b-vision-preview",                   # legacy (decommissioned)
    "llama-3.2-11b-vision-preview",                   # legacy (decommissioned)
]


def _is_model_not_available_error(exc: Exception) -> bool:
    """Détecte les erreurs de modèle indisponible/non autorisé côté provider."""
    msg = str(exc).lower()
    return (
        "model_not_found" in msg
        or "does not exist" in msg
        or "do not have access" in msg
        or "404" in msg
        or "decommissioned" in msg   # modèles Groq dépréciés (llama-3.2-90b, etc.)
        or "model_decommissioned" in msg
    )


def _is_rate_limit_error(exc: Exception) -> bool:
    """Détecte les erreurs de quota/rate limit côté provider."""
    msg = str(exc).lower()
    return (
        "rate_limit_exceeded" in msg
        or "rate limit" in msg
        or "too many requests" in msg
        or "429" in msg
        or "tokens per day" in msg
    )


def _groq_chat_with_model_fallback(
    api_key: str,
    model_candidates: List[str],
    messages: List[dict],
    temperature: float,
    max_tokens: Optional[int] = None,
    response_format: Optional[dict] = None,
) -> str:
    """Appelle Groq en essayant plusieurs modèles jusqu'à succès."""
    from groq import Groq

    client = Groq(api_key=api_key)
    last_error: Optional[Exception] = None

    for model_name in model_candidates:
        try:
            kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                kwargs["max_tokens"] = max_tokens
            if response_format is not None:
                kwargs["response_format"] = response_format
            completion = client.chat.completions.create(**kwargs)
            content = (completion.choices[0].message.content or "").strip()
            finish_reason = getattr(completion.choices[0], "finish_reason", None)
            if finish_reason == "length":
                logger.warning(
                    "Groq model=%s réponse tronquée (finish_reason=length, longueur=%d) — augmenter max_tokens",
                    model_name, len(content),
                )
            if content:
                logger.info("Groq call success with model=%s", model_name)
                return content
        except Exception as e:
            last_error = e
            if _is_model_not_available_error(e):
                logger.warning("Groq model unavailable (%s): %s", model_name, e)
                continue
            if _is_rate_limit_error(e):
                logger.warning("Groq model rate-limited (%s): %s", model_name, e)
                continue
            raise

    if last_error:
        raise last_error
    return ""


def _gemini_generate_content(
    api_key: str,
    text_prompt: str,
    inline_image_b64: Optional[str] = None,
    temperature: float = 0.1,
    max_output_tokens: int = 4096,
) -> str:
    """
    Appelle Gemini avec fallback de modèles pour éviter les 404 selon les comptes.
    """
    model_candidates = [
        "gemini-2.0-flash",   # Priorité 2025 : meilleur OCR sur PDF scannés
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ]
    last_error: Optional[Exception] = None

    for model_name in model_candidates:
        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta/models/"
                f"{model_name}:generateContent"
            )
            parts: list[dict] = [{"text": text_prompt}]
            if inline_image_b64:
                parts.append(
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": inline_image_b64,
                        }
                    }
                )
            payload = {
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "temperature": temperature,
                    "maxOutputTokens": max_output_tokens,
                },
            }
            resp = requests.post(
                f"{url}?key={api_key}",
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()
            content = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            ).strip()
            if content:
                logger.info("Gemini call success with model=%s", model_name)
                return content
        except Exception as e:
            last_error = e
            logger.warning("Gemini model failed (%s): %s", model_name, e)
            continue

    if last_error:
        raise last_error
    return ""


def _page_text_for_groq(pdf_path: str, page_num: int) -> str:
    """
    Texte d'une page pour extraction LLM.

    Priorité :
    1. pymupdf4llm  → Markdown structuré LLM-ready (meilleure qualité)
    2. PyMuPDF brut → rapide
    3. pdfplumber   → dernier recours
    """
    # 1. pymupdf4llm (meilleure qualité pour les tableaux)
    try:
        import pymupdf4llm  # type: ignore
        md = pymupdf4llm.to_markdown(pdf_path, pages=[page_num - 1])
        if md and md.strip():
            return md
    except Exception:
        pass

    # 2. PyMuPDF brut
    if pymupdf is not None:
        try:
            doc = pymupdf.open(pdf_path)
            try:
                if page_num < 1 or page_num > len(doc):
                    return ""
                return doc[page_num - 1].get_text() or ""
            finally:
                doc.close()
        except Exception as e:
            logger.debug("PyMuPDF texte page pour Groq: %s", e)

    # 3. pdfplumber
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num < 1 or page_num > len(pdf.pages):
                return ""
            return pdf.pages[page_num - 1].extract_text(x_tolerance=2, y_tolerance=2) or ""
    except Exception as e:
        logger.warning("Impossible de lire la page pour Groq texte: %s", e)
        return ""


def _extract_json_object(content: str) -> str:
    """Isole un objet JSON dans une réponse LLM (avec ou sans ```json)."""
    if not content:
        return ""
    text = content.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S | re.I)
    if fence_match:
        text = fence_match.group(1).strip()
    start = text.find("{")
    if start == -1:
        return ""
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()
    end = text.rfind("}")
    if end > start:
        return text[start : end + 1].strip()
    return ""


def _safe_load_llm_json(raw_json: str) -> Optional[dict]:
    """
    Parse un JSON LLM avec petites réparations défensives.

    Cas gérés:
    - backslashes invalides dans des chaînes (ex: D\'EXPLOITATION)
    - virgules terminales avant } ou ]
    """
    if not raw_json:
        return None

    candidate = raw_json.strip()
    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass

    # Supprime les échappements invalides (\' etc.) tout en conservant les
    # échappements JSON valides (\n, \uXXXX, \", \\ ...).
    candidate = re.sub(r"\\(?![\"\\/bfnrtu])", "", candidate)
    # Supprime les trailing commas fréquentes en sortie LLM.
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)

    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _looks_like_no_image_response(content: str) -> bool:
    """
    Détecte les réponses non exploitables où le modèle indique ne pas voir l'image.
    """
    if not content:
        return False
    txt = content.lower()
    no_image_markers = [
        "je ne vois pas l'image",
        "je ne vois pas l’image",
        "i can't see the image",
        "i cannot see the image",
        "cannot view the image",
        "unable to view the image",
        "please provide the image",
        "pouvez-vous me fournir l'image",
        "pouvez-vous me fournir l’image",
    ]
    return any(m in txt for m in no_image_markers)


def _json_table_to_dataframe(payload: dict) -> pd.DataFrame:
    """Convertit un payload JSON {headers, rows} en DataFrame."""
    if not isinstance(payload, dict):
        return pd.DataFrame()

    headers = payload.get("headers")
    if headers is None:
        headers = payload.get("columns")
    rows = payload.get("rows")

    # New cropped-section schema:
    # {"section_type": "...", "columns": [...], "rows": [{"label": "...", ...}]}
    if isinstance(headers, list) and isinstance(rows, list) and rows and all(isinstance(r, dict) for r in rows):
        norm_headers = [str(h if h is not None else "").strip() for h in headers]
        if not norm_headers:
            return pd.DataFrame()

        def _dict_cell(row: dict, header: str, idx: int) -> Any:
            candidates = [header]
            h_norm = _norm(header)
            h_lower = str(header or "").strip().lower()
            if h_lower:
                candidates.append(h_lower)
            if h_norm in {"label", "libelle", "rubrique"} or idx == 0:
                candidates.extend(["label", "libelle", "rubrique"])
            if "note" in h_norm or h_norm in {"ref", "reference"}:
                candidates.extend(["notes", "note", "ref", "reference", "NOTES"])
            if "2024" in h_norm:
                candidates.extend(["value_2024", "2024", "30/06/2024", "31/12/2024"])
            if "2023" in h_norm:
                candidates.extend(["value_2023", "2023", "31/12/2023", "30/06/2023"])
            for key in candidates:
                if key in row:
                    return row.get(key)
            row_norm_map = {_norm(str(k)): v for k, v in row.items()}
            for key in candidates:
                key_norm = _norm(str(key))
                if key_norm in row_norm_map:
                    return row_norm_map[key_norm]
            return None

        out_rows_dict: List[List[str]] = []
        for row in rows:
            vals: List[str] = []
            for idx, h in enumerate(norm_headers):
                v = _dict_cell(row, h, idx)
                vals.append("" if v is None else str(v).strip())
            out_rows_dict.append(vals)
        if not out_rows_dict:
            return pd.DataFrame()
        df = pd.DataFrame(out_rows_dict, columns=norm_headers)
        logger.info(
            "_json_table_to_dataframe: object rows OK shape=%s headers=%s",
            df.shape, norm_headers[:5],
        )
        return df
    if not isinstance(headers, list) or not isinstance(rows, list):
        return pd.DataFrame()
    if not headers:
        return pd.DataFrame()

    norm_headers = [str(h or "").strip() for h in headers]
    if not any(norm_headers):
        return pd.DataFrame()

    out_rows: List[List[str]] = []
    for r in rows:
        if not isinstance(r, list):
            continue
        vals = [str(v or "").strip() for v in r]
        if len(vals) < len(norm_headers):
            vals.extend([""] * (len(norm_headers) - len(vals)))
        elif len(vals) > len(norm_headers):
            vals = vals[: len(norm_headers)]
        out_rows.append(vals)

    if not out_rows:
        return pd.DataFrame()

    return pd.DataFrame(out_rows, columns=norm_headers)


def _llm_structured_extract(
    pdf_path: str,
    page_num: int,
    table_name: str,
    api_provider: str = "groq",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Extraction LLM en JSON strict puis conversion DataFrame."""
    provider = _normalize_api_provider(api_provider)
    api_key = _provider_api_key(provider, api_key=api_key)
    if not api_key:
        return pd.DataFrame()

    text = _page_text_for_groq(pdf_path, page_num)
    if not text.strip():
        return pd.DataFrame()

    prompt = f"""Tu reçois le texte OCR/extrait d'une page de rapport financier.
Ta mission: extraire UNIQUEMENT le tableau '{table_name}' et renvoyer UN JSON valide.

Format JSON obligatoire (et rien d'autre):
{{
  "table_name": "{table_name}",
  "headers": ["colonne_1", "colonne_2", "..."],
  "rows": [
    ["val11", "val12", "..."],
    ["val21", "val22", "..."]
  ]
}}

Règles strictes:
- Réponds uniquement par du JSON pur, sans markdown.
- Garde les montants tels qu'ils apparaissent.
- Ne coupe pas un nombre avec séparateurs de milliers.
- Si le tableau n'est pas trouvé, renvoie:
  {{"table_name":"{table_name}","headers":[],"rows":[]}}

Texte page:
{text[:14000]}
"""
    try:
        content = ""
        if provider == "groq":
            content = _groq_chat_with_model_fallback(
                api_key=api_key,
                model_candidates=GROQ_TEXT_MODEL_CANDIDATES,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un extracteur de tableaux financiers. Tu renvoies seulement du JSON valide.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=8192,  # 8192 pour éviter troncature sur grands tableaux
                response_format={"type": "json_object"},  # JSON garanti (Groq Structured Outputs)
            )
        elif provider == "gemini":
            content = _gemini_generate_content(
                api_key=api_key,
                text_prompt=prompt,
                temperature=0.0,
                max_output_tokens=8192,
            )
        else:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un extracteur de tableaux financiers. Tu renvoies seulement du JSON valide.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )
            content = (completion.choices[0].message.content or "").strip()

        obj = _extract_json_object(content)
        if not obj:
            return pd.DataFrame()
        payload = json.loads(obj)
        df = _json_table_to_dataframe(payload)
        return df if _is_valid_table(df) else pd.DataFrame()
    except Exception as e:
        logger.warning("LLM structured JSON failed provider=%s: %s", provider, e)
        return pd.DataFrame()


def _llm_reconstruct(
    pdf_path: str,
    page_num: int,
    table_name: str,
    api_provider: str = "groq",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Reconstruction texte via Groq, Gemini ou OpenAI."""
    provider = _normalize_api_provider(api_provider)
    api_key = _provider_api_key(provider, api_key=api_key)
    if not api_key:
        logger.warning("API key manquante pour provider=%s", provider)
        return pd.DataFrame()

    text = _page_text_for_groq(pdf_path, page_num)

    if not text.strip():
        return pd.DataFrame()

    prompt = f"""Le texte ci-dessous provient d'une page de rapport financier. Reconstruis UNIQUEMENT le tableau correspondant à "{table_name}" au format Markdown.

Règles :
- Une ligne par ligne du tableau (libellés + chiffres).
- Séparateur de colonnes : | (pipe).
- Première ligne = en-têtes des colonnes.
- Deuxième ligne = séparateur | --- | --- | ...
- Conserve les montants exacts (espaces, virgules, points).
- Ne renvoie QUE le tableau Markdown, sans commentaire avant ou après.

Texte de la page :

{text[:12000]}
"""

    try:
        content = ""
        if provider == "groq":
            content = _groq_chat_with_model_fallback(
                api_key=api_key,
                model_candidates=GROQ_TEXT_MODEL_CANDIDATES,
                messages=[
                    {"role": "system", "content": "Tu extrais des tableaux financiers. Tu réponds uniquement par un tableau Markdown valide."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=8192,  # 8192 pour éviter troncature sur grands tableaux
            )
        elif provider == "gemini":
            content = _gemini_generate_content(
                api_key=api_key,
                text_prompt=prompt,
                temperature=0.1,
                max_output_tokens=8192,
            )
        else:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Tu extrais des tableaux financiers. Tu réponds uniquement par un tableau Markdown valide."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            content = (completion.choices[0].message.content or "").strip()

        if not content:
            return pd.DataFrame()
        content = _extract_markdown_table(content)
        df = _markdown_table_to_dataframe(content)
        return df if _is_valid_table(df) else pd.DataFrame()
    except Exception as e:
        logger.warning("LLM texte failed provider=%s: %s", provider, e)
        return pd.DataFrame()


def _llm_vision_extract(
    pdf_path: str,
    page_num: int,
    table_name: str,
    api_provider: str = "groq",
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Niveau Vision : convertit la page PDF en image et envoie à Groq Vision ou Gemini.
    Contourne tous les problèmes de bordures, espaces blancs et sections séparées.
    Retourne un DataFrame complet avec toutes les lignes et colonnes du tableau.
    """
    from dotenv import load_dotenv
    load_dotenv()
    provider = _normalize_api_provider(api_provider)
    api_key = _provider_api_key(provider, api_key=api_key)
    if not api_key:
        logger.warning("API key manquante pour provider=%s (vision)", provider)
        return pd.DataFrame()

    # Rendu HQ via pipeline centralisé (DPI adaptatif, enhance automatique).
    # Fallback vers zoom x2 legacy uniquement si PIL ou fitz indisponibles.
    try:
        import base64, io
        dpi, enhance = _choose_dpi(pdf_path, page_num)
        pil_img = _render_pdf_page_high_quality(pdf_path, page_num, dpi=dpi, enhance=enhance)
        if pil_img is not None:
            buf = io.BytesIO()
            pil_img.save(buf, format="PNG", optimize=False)
            img_b64 = base64.b64encode(buf.getvalue()).decode()
        else:
            # Fallback legacy uniquement si HQ échoue
            import fitz
            doc = fitz.open(pdf_path)
            if page_num < 1 or page_num > len(doc):
                return pd.DataFrame()
            pix = doc[page_num - 1].get_pixmap(matrix=fitz.Matrix(2, 2))
            img_b64 = base64.b64encode(pix.tobytes("png")).decode()
            doc.close()
            logger.warning(
                "_llm_vision_extract: HQ échoué, fallback 144 DPI page=%d", page_num
            )
    except Exception as e:
        logger.warning("Vision: impossible de préparer l'image page=%d: %s", page_num, e)
        return pd.DataFrame()

    # Prompt unique intelligent (famille + contexte détectés automatiquement)
    prompt = _build_vision_prompt(table_name=table_name)

    try:
        content = ""
        if provider == "groq":
            content = _groq_chat_with_model_fallback(
                api_key=api_key,
                model_candidates=GROQ_VISION_MODEL_CANDIDATES,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
                temperature=0.0,
                max_tokens=8192,
            )
        elif provider == "gemini":
            content = _gemini_generate_content(
                api_key=api_key,
                text_prompt=prompt,
                inline_image_b64=img_b64,
                temperature=0.0,
                max_output_tokens=8192,
            )
        else:
            logger.warning("Provider=%s non supporte en vision, fallback texte.", provider)
            return pd.DataFrame()

        if not content:
            return pd.DataFrame()

        # Extraire le tableau Markdown depuis la réponse (ignorer texte parasite)
        content = _extract_markdown_table(content)

        df = _markdown_table_to_dataframe(content)
        if not _is_valid_table(df):
            logger.warning("Vision: tableau Markdown invalide:\n%s", content[:300])
            return pd.DataFrame()
        return df

    except Exception as e:
        logger.warning("Vision failed provider=%s: %s", provider, e)
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
#  Pipeline Vision — rendu haute qualité + crop intelligent + debug
# ═══════════════════════════════════════════════════════════════════════════════


def _render_pdf_page_high_quality(
    pdf_path: str,
    page_num: int,
    dpi: int = VISION_RENDER_DPI,
    enhance: bool = VISION_ENABLE_ENHANCE,
) -> "Optional[PIL.Image.Image]":  # noqa: F821 — PIL imported inside
    """
    Rend une page PDF en image PIL haute qualité via PyMuPDF.

    Utilise un zoom matrix correct (zoom = dpi / 72) et exporte en PNG sans
    compression destructive.  Post-processing PIL léger optionnel.

    Args:
        pdf_path:  Chemin vers le fichier PDF.
        page_num:  Numéro de page 1-based.
        dpi:       Résolution cible (défaut VISION_RENDER_DPI = 300 DPI).
        enhance:   Si True : autocontrast + léger unsharp mask.

    Returns:
        PIL.Image RGB, ou None si échec (PyMuPDF ou Pillow absent).
    """
    try:
        import fitz  # PyMuPDF
        from PIL import Image, ImageOps, ImageFilter  # type: ignore
    except ImportError as e:
        logger.warning("_render_pdf_page_high_quality: dépendance manquante (%s)", e)
        return None

    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)

        doc = fitz.open(pdf_path)
        try:
            if page_num < 1 or page_num > len(doc):
                logger.warning(
                    "_render_pdf_page_high_quality: page %d hors bornes (doc=%d pages)",
                    page_num, len(doc),
                )
                return None
            page = doc[page_num - 1]
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        finally:
            doc.close()

        logger.debug(
            "_render_pdf_page_high_quality: page=%d dpi=%d zoom=%.2f size=%dx%d",
            page_num, dpi, zoom, img.width, img.height,
        )

        if enhance:
            img = ImageOps.autocontrast(img, cutoff=0.5)
            img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3))
            logger.debug("_render_pdf_page_high_quality: enhance appliqué (autocontrast + unsharp)")

        return img

    except Exception as e:
        logger.warning("_render_pdf_page_high_quality: erreur page=%d: %s", page_num, e)
        return None


def _choose_dpi(pdf_path: str, page_num: int) -> tuple:
    """
    Choisit le DPI de rendu et l'activation de l'enhancement selon le type de page.

    Heuristique simple et robuste :
    - Page avec peu de texte natif (< VISION_SCANNED_TEXT_THRESHOLD chars)
      → page scannée ou image-heavy → 350 DPI + enhance=True
    - Page avec texte natif suffisant
      → page normale → VISION_RENDER_DPI (300) + enhance=False

    Returns:
        (dpi: int, enhance: bool)
    """
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        try:
            if page_num < 1 or page_num > len(doc):
                return VISION_RENDER_DPI, VISION_ENABLE_ENHANCE
            text = doc[page_num - 1].get_text() or ""
        finally:
            doc.close()
        is_scanned = len(text.strip()) < VISION_SCANNED_TEXT_THRESHOLD
    except Exception:
        return VISION_RENDER_DPI, VISION_ENABLE_ENHANCE

    if is_scanned:
        logger.info(
            "_choose_dpi: page=%d → scannée/image-heavy (texte=%d chars) → %d DPI + enhance=True",
            page_num, len(text.strip()), VISION_DPI_SCANNED,
        )
        return VISION_DPI_SCANNED, True
    else:
        logger.info(
            "_choose_dpi: page=%d → normale (texte=%d chars) → %d DPI + enhance=False",
            page_num, len(text.strip()), VISION_RENDER_DPI,
        )
        return VISION_RENDER_DPI, False


def _detect_table_bbox(
    pdf_path: str,
    page_num: int,
    table_name: str,
) -> "Optional[Tuple[float, float, float, float]]":
    """
    Détecte la zone probable du tableau dans la page via les blocs de texte PyMuPDF.

    Stratégie pragmatique (niveau 1, sans IA lourde) :
    1. Normalise le nom du tableau et tokenise.
    2. Parcourt les blocs de texte de la page (page.get_text("blocks")).
    3. Identifie le bloc qui contient le titre du tableau (score tokens ≥ 50%).
    4. Collecte les blocs suivants jusqu'à un marqueur d'arrêt ou fin de page.
    5. Construit la bounding box englobante en coordonnées PDF (points).

    Args:
        pdf_path:   Chemin vers le fichier PDF.
        page_num:   Numéro de page 1-based.
        table_name: Nom du tableau cible (ex. "BILAN ACTIF").

    Returns:
        (x0, y0, x1, y1) en coordonnées PDF (points), ou None si non détecté.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.debug("_detect_table_bbox: PyMuPDF non disponible")
        return None

    try:
        doc = fitz.open(pdf_path)
        try:
            if page_num < 1 or page_num > len(doc):
                return None
            page = doc[page_num - 1]
            # blocks : liste de tuples (x0, y0, x1, y1, text, block_no, block_type)
            blocks = page.get_text("blocks")
        finally:
            doc.close()
    except Exception as e:
        logger.warning("_detect_table_bbox: impossible de lire les blocs: %s", e)
        return None

    if not blocks:
        return None

    # Normalisation du nom du tableau pour comparaison robuste
    name_norm = unicodedata.normalize("NFC", (table_name or "").lower().strip())
    name_tokens = [t for t in name_norm.split() if len(t) >= 3]
    if not name_tokens:
        return None

    # Patterns qui signalent la fin du tableau (blocs narratifs sans chiffres)
    _STOP_PATTERNS = (
        "état des dérogations",
        "périmètre de consolidation",
        "perimetre de consolidation",
        "méthode de consolidation",
        "methode de consolidation",
        "tableau des flux",
        "flux de trésorerie",
    )
    _NUMBER_RE = re.compile(r"\d{2,}")

    # ── Étape 1 : localiser le bloc titre ──────────────────────────────────
    title_block_idx: Optional[int] = None
    title_y1: Optional[float] = None

    for i, block in enumerate(blocks):
        bx0, by0, bx1, by1, btext, *_ = block
        btext_norm = unicodedata.normalize("NFC", (btext or "").lower().strip())
        match_score = sum(1 for tok in name_tokens if tok in btext_norm)
        if match_score >= max(1, len(name_tokens) // 2):
            title_block_idx = i
            title_y1 = float(by1)
            logger.debug(
                "_detect_table_bbox: titre trouvé bloc=%d y=[%.1f,%.1f] score=%d/%d texte=%r",
                i, by0, by1, match_score, len(name_tokens), btext[:80],
            )
            break

    if title_block_idx is None or title_y1 is None:
        logger.debug("_detect_table_bbox: titre '%s' non trouvé sur page %d", table_name, page_num)
        return None

    # ── Étape 2 : collecter les blocs du tableau ────────────────────────────
    table_blocks = []
    for block in blocks[title_block_idx:]:
        bx0, by0, bx1, by1, btext, *_ = block
        if float(by0) < title_y1 - 5:
            continue  # bloc au-dessus du titre (ne devrait pas arriver mais sécurité)
        btext_clean = (btext or "").strip()
        btext_lower = btext_clean.lower()
        # Stop si bloc narratif sans chiffres correspondant à un marqueur connu
        if any(p in btext_lower for p in _STOP_PATTERNS) and not _NUMBER_RE.search(btext_clean):
            logger.debug("_detect_table_bbox: stop pattern '%s'", btext_clean[:60])
            break
        table_blocks.append(block)

    if not table_blocks:
        return None

    # ── Étape 3 : bounding box englobante ──────────────────────────────────
    xs0 = [b[0] for b in table_blocks]
    ys0 = [b[1] for b in table_blocks]
    xs1 = [b[2] for b in table_blocks]
    ys1 = [b[3] for b in table_blocks]

    bbox = (min(xs0), min(ys0), max(xs1), max(ys1))
    logger.debug(
        "_detect_table_bbox: bbox=(%.1f,%.1f,%.1f,%.1f) blocs=%d",
        *bbox, len(table_blocks),
    )
    return bbox


def _crop_table_region(
    image: "PIL.Image.Image",  # noqa: F821
    bbox_pdf: "Tuple[float, float, float, float]",
    page_rect_pdf: "Tuple[float, float, float, float]",
    padding: int = VISION_CROP_PADDING,
    min_width_ratio: float = VISION_MIN_CROP_WIDTH_RATIO,
    min_height_ratio: float = VISION_MIN_CROP_HEIGHT_RATIO,
) -> "Tuple[PIL.Image.Image, bool]":  # noqa: F821
    """
    Crop la zone du tableau dans une image PIL.

    Convertit les coordonnées PDF (points) en coordonnées pixels via les
    facteurs d'échelle page_rect → image.size, applique le padding et
    valide la taille minimale du crop.

    Args:
        image:            Image PIL pleine page (source).
        bbox_pdf:         (x0, y0, x1, y1) en points PDF.
        page_rect_pdf:    (x0, y0, x1, y1) du rect de page PDF.
        padding:          Pixels ajoutés de chaque côté du crop.
        min_width_ratio:  Ratio minimal largeur_crop / largeur_image.
        min_height_ratio: Ratio minimal hauteur_crop / hauteur_image.

    Returns:
        (image_finale, crop_appliqué) — crop_appliqué=False si fallback page entière.
    """
    try:
        img_w, img_h = image.size
        pdf_x0, pdf_y0, pdf_x1, pdf_y1 = page_rect_pdf
        pdf_w = pdf_x1 - pdf_x0
        pdf_h = pdf_y1 - pdf_y0

        if pdf_w <= 0 or pdf_h <= 0:
            logger.warning("_crop_table_region: page_rect invalide, fallback page entière")
            return image, False

        scale_x = img_w / pdf_w
        scale_y = img_h / pdf_h

        bx0, by0, bx1, by1 = bbox_pdf
        px0 = int((bx0 - pdf_x0) * scale_x) - padding
        py0 = int((by0 - pdf_y0) * scale_y) - padding
        px1 = int((bx1 - pdf_x0) * scale_x) + padding
        py1 = int((by1 - pdf_y0) * scale_y) + padding

        # Clamp aux limites de l'image
        px0 = max(0, px0)
        py0 = max(0, py0)
        px1 = min(img_w, px1)
        py1 = min(img_h, py1)

        crop_w = px1 - px0
        crop_h = py1 - py0

        if crop_w <= 0 or crop_h <= 0:
            logger.warning("_crop_table_region: crop nul après clamping, fallback page entière")
            return image, False

        if crop_w / img_w < min_width_ratio:
            logger.warning(
                "_crop_table_region: crop trop étroit (%.1f%% < %.0f%%), fallback page entière",
                100 * crop_w / img_w, 100 * min_width_ratio,
            )
            return image, False

        if crop_h / img_h < min_height_ratio:
            logger.warning(
                "_crop_table_region: crop trop court (%.1f%% < %.0f%%), fallback page entière",
                100 * crop_h / img_h, 100 * min_height_ratio,
            )
            return image, False

        cropped = image.crop((px0, py0, px1, py1))
        logger.info(
            "_crop_table_region: crop OK pixels=(%d,%d,%d,%d) size=%dx%d (page=%dx%d)",
            px0, py0, px1, py1, crop_w, crop_h, img_w, img_h,
        )
        return cropped, True

    except Exception as e:
        logger.warning("_crop_table_region: erreur: %s — fallback page entière", e)
        return image, False


def _debug_save_image(
    image: "PIL.Image.Image",  # noqa: F821
    pdf_path: str,
    page_num: int,
    table_name: str,
    suffix: str = "full",
    output_dir: str = VISION_DEBUG_OUTPUT_DIR,
) -> "Optional[str]":
    """
    Sauvegarde une image PIL dans le dossier de debug vision.

    Nom de fichier : {pdf_stem}_{TABLE_NAME_SAFE}_page{N}_{suffix}.png

    Args:
        image:      Image PIL à sauvegarder.
        pdf_path:   Chemin source du PDF (pour nommer le fichier).
        page_num:   Numéro de page.
        table_name: Nom du tableau (intégré dans le nom de fichier).
        suffix:     "full" ou "crop" pour distinguer les deux variantes.
        output_dir: Dossier de destination.

    Returns:
        Chemin du fichier sauvegardé, ou None si échec.
    """
    import pathlib

    def _pretty_pdf_folder_name(path: str) -> str:
        stem = pathlib.Path(path).stem
        parts = [p for p in re.split(r"[_\-\s]+", stem) if p]
        year_idx = next((i for i, p in enumerate(parts) if re.fullmatch(r"20\d{2}", p)), None)
        if year_idx is None:
            return stem
        issuer_parts = parts[:year_idx]
        year = parts[year_idx]
        report_parts = parts[year_idx + 1:]

        def _issuer_token(token: str) -> str:
            if token.isalpha() and len(token) <= 4:
                return token.upper()
            return token[:1].upper() + token[1:].lower()

        issuer = " ".join(_issuer_token(p) for p in issuer_parts) or stem
        report = " ".join(p.lower() for p in report_parts)
        return " ".join(p for p in [issuer, year, report] if p).strip()

    try:
        out_dir = pathlib.Path(output_dir) / _pretty_pdf_folder_name(pdf_path)
        out_dir.mkdir(parents=True, exist_ok=True)

        pdf_stem = pathlib.Path(pdf_path).stem
        table_safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", table_name.upper())[:40]
        filename = f"{pdf_stem}_{table_safe}_page{page_num}_{suffix}.png"
        filepath = out_dir / filename

        image.save(str(filepath), format="PNG", optimize=False)
        logger.info("_debug_save_image: sauvegardé → %s", filepath)
        return str(filepath)

    except Exception as e:
        logger.warning("_debug_save_image: sauvegarde impossible: %s", e)
        return None


def _prepare_vision_image(
    pdf_path: str,
    page_num: int,
    table_name: str,
    dpi: Optional[int] = None,
    enhance: Optional[bool] = None,
    enable_crop: bool = VISION_ENABLE_TABLE_CROP,
    debug_save: bool = VISION_DEBUG_SAVE_INPUTS,
    crop_padding: int = VISION_CROP_PADDING,
) -> "Optional[str]":
    """
    Pipeline complet de préparation d'image avant appel au Vision LLM.

    Étapes :
    1. Choix adaptatif du DPI et de l'enhancement via _choose_dpi()
       (300 DPI normal / 350 DPI + enhance sur pages scannées).
    2. Rendu haute qualité (PyMuPDF → PIL RGB).
    3. Détection de la zone du tableau via blocs de texte (optionnel).
    4. Crop de la région du tableau avec sécurité fallback (optionnel).
    5. Sauvegarde debug des images pleine page et/ou croppée (optionnel).
    6. Conversion finale en base64 PNG.

    En cas d'échec de n'importe quelle étape, fallback propre vers la méthode
    précédente (zoom x2 simple via _page_to_image_b64).

    Args:
        pdf_path:     Chemin vers le fichier PDF.
        page_num:     Numéro de page 1-based.
        table_name:   Nom du tableau (pour crop + nommage debug).
        dpi:          Forcer un DPI spécifique. Si None, choix adaptatif automatique.
        enhance:      Forcer l'état de l'enhancement. Si None, choix adaptatif automatique.
        enable_crop:  Activer le crop intelligent.
        debug_save:   Sauvegarder les images dans VISION_DEBUG_OUTPUT_DIR.
        crop_padding: Padding pixels autour du crop.

    Returns:
        Image base64 PNG prête pour le Vision LLM, ou None si échec total.
    """
    import base64
    import io

    # ── DPI et enhance adaptatifs (si non forcés par l'appelant) ─────────────
    if dpi is None or enhance is None:
        auto_dpi, auto_enhance = _choose_dpi(pdf_path, page_num)
        if dpi is None:
            dpi = auto_dpi
        if enhance is None:
            enhance = auto_enhance

    logger.info(
        "_prepare_vision_image: début — pdf=%s page=%d table=%r dpi=%d "
        "enhance=%s crop=%s debug=%s",
        pdf_path, page_num, table_name, dpi, enhance, enable_crop, debug_save,
    )

    # ── Étape 1 : rendu haute qualité ────────────────────────────────────────
    full_image = _render_pdf_page_high_quality(
        pdf_path=pdf_path,
        page_num=page_num,
        dpi=dpi,
        enhance=enhance,
    )
    if full_image is None:
        logger.warning(
            "_prepare_vision_image: rendu haute qualité échoué — fallback zoom x2 legacy"
        )
        return _page_to_image_b64(pdf_path, page_num)

    logger.info(
        "_prepare_vision_image: rendu OK dpi=%d size=%dx%d",
        dpi, full_image.width, full_image.height,
    )

    # ── Étape 2 (debug) : sauvegarde image pleine page ───────────────────────
    if debug_save:
        _debug_save_image(
            full_image,
            pdf_path,
            page_num,
            table_name,
            suffix="llm_input_full_page" if not enable_crop else "full",
        )

    # ── Étape 3 : crop intelligent de la zone du tableau ─────────────────────
    # full_page_image  = image pleine page (toujours disponible)
    # cropped_image    = crop si détecté et valide
    # final_image      = image envoyée au LLM (crop si valide, sinon full_page)
    final_image = full_image
    method_used = "full_page"

    if enable_crop:
        bbox_pdf = _detect_table_bbox(pdf_path, page_num, table_name)

        if bbox_pdf is not None:
            # Récupérer le rect de la page pour la conversion coordonnées PDF → pixels
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(pdf_path)
                try:
                    pr = doc[page_num - 1].rect
                    page_rect_tuple: Tuple[float, float, float, float] = (
                        pr.x0, pr.y0, pr.x1, pr.y1
                    )
                finally:
                    doc.close()
            except Exception as e:
                logger.warning(
                    "_prepare_vision_image: impossible de lire page_rect: %s — A4 fallback", e
                )
                page_rect_tuple = (0.0, 0.0, 595.0, 842.0)

            cropped, crop_ok = _crop_table_region(
                image=full_image,
                bbox_pdf=bbox_pdf,
                page_rect_pdf=page_rect_tuple,
                padding=crop_padding,
            )

            if crop_ok:
                final_image = cropped
                method_used = "cropped_table"
                if debug_save:
                    _debug_save_image(cropped, pdf_path, page_num, table_name, suffix="crop")
                logger.info(
                    "_prepare_vision_image: crop appliqué size=%dx%d",
                    cropped.width, cropped.height,
                )
            else:
                logger.info(
                    "_prepare_vision_image: crop invalide → page entière utilisée"
                )
        else:
            logger.info(
                "_prepare_vision_image: bbox non détectée → page entière utilisée"
            )
    else:
        logger.debug("_prepare_vision_image: crop désactivé (VISION_ENABLE_TABLE_CROP=False)")

    logger.info(
        "_prepare_vision_image: méthode finale=%s image_finale=%dx%d",
        method_used, final_image.width, final_image.height,
    )

    # ── Étape 4 : conversion base64 ──────────────────────────────────────────
    try:
        buf = io.BytesIO()
        final_image.save(buf, format="PNG", optimize=False)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as e:
        logger.warning("_prepare_vision_image: conversion base64 échouée: %s", e)
        return None


def _page_to_image_b64(pdf_path: str, page_num: int, zoom: float = 2.0) -> Optional[str]:
    """Convertit une page PDF en image PNG base64 (zoom x2 pour meilleure résolution OCR)."""
    try:
        import fitz  # PyMuPDF
        import base64
        doc = fitz.open(pdf_path)
        try:
            if page_num < 1 or page_num > len(doc):
                return None
            page = doc[page_num - 1]
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            return base64.b64encode(img_bytes).decode()
        finally:
            doc.close()
    except Exception as e:
        logger.warning("_page_to_image_b64: page %d échouée: %s", page_num, e)
        return None


def _vision_json_to_dataframe(payload: dict, table_name: str = "") -> pd.DataFrame:
    """
    Convertit un payload JSON Vision {headers, rows} en DataFrame robuste.

    Différences avec _json_table_to_dataframe (utilisé pour les extractions texte) :
    - Pad/trim chaque ligne au nombre exact d'en-têtes avec log détaillé.
    - Ne convertit jamais les valeurs numériques (tout reste str).
    - Rejette les lignes entièrement vides avant de construire le DataFrame.
    - Valide la cohérence minimale (≥ 1 header, ≥ 1 ligne non vide).

    Args:
        payload:    Dict avec clés "headers" et "rows".
        table_name: Nom du tableau pour les logs.

    Returns:
        DataFrame ou DataFrame vide si structure invalide.
    """
    if not isinstance(payload, dict):
        logger.warning("_vision_json_to_dataframe [%s]: payload n'est pas un dict", table_name)
        return pd.DataFrame()

    sections = payload.get("sections")
    if isinstance(sections, list) and sections:
        target_section = _target_section_type(table_name)
        selected_section = None
        for section in sections:
            if not isinstance(section, dict):
                continue
            section_type = str(section.get("section_type") or "").strip().upper()
            if target_section and section_type == target_section:
                selected_section = section
                break
        if selected_section is None:
            dict_sections = [s for s in sections if isinstance(s, dict)]
            if len(dict_sections) == 1:
                selected_section = dict_sections[0]
        if selected_section is not None:
            logger.info(
                "_vision_json_to_dataframe [%s]: multi-section payload -> selected=%s",
                table_name,
                selected_section.get("section_type"),
            )
            return _vision_json_to_dataframe(selected_section, table_name=table_name)
        logger.warning(
            "_vision_json_to_dataframe [%s]: multi-section payload sans section cible=%s",
            table_name,
            target_section,
        )
        return pd.DataFrame()

    headers = payload.get("headers")
    if headers is None:
        headers = payload.get("columns")
    rows = payload.get("rows")

    if isinstance(headers, list) and isinstance(rows, list) and rows and all(isinstance(r, dict) for r in rows):
        norm_headers = [str(h if h is not None else "").strip() for h in headers]
        if not norm_headers:
            return pd.DataFrame()

        def _dict_cell(row: dict, header: str, idx: int) -> Any:
            candidates = [header]
            h_norm = _norm(header)
            h_lower = str(header or "").strip().lower()
            if h_lower:
                candidates.append(h_lower)
            if h_norm in {"label", "libelle", "rubrique"} or idx == 0:
                candidates.extend(["label", "libelle", "rubrique"])
            if "note" in h_norm or h_norm in {"ref", "reference"}:
                candidates.extend(["notes", "note", "ref", "reference", "NOTES"])
            if "2024" in h_norm:
                candidates.extend(["value_current", "current", "value_2024", "2024", "30/06/2024", "31/12/2024"])
            if "2023" in h_norm:
                candidates.extend(["value_previous", "previous", "value_2023", "2023", "31/12/2023", "30/06/2023"])
            if idx == 1 and "note" not in h_norm and h_norm not in {"ref", "reference"}:
                candidates.extend(["value_current", "current", "value_2024"])
            if idx >= 2 and "note" not in h_norm and h_norm not in {"ref", "reference"}:
                candidates.extend(["value_previous", "previous", "value_2023"])
            for key in candidates:
                if key in row:
                    return row.get(key)
            row_norm_map = {_norm(str(k)): v for k, v in row.items()}
            for key in candidates:
                key_norm = _norm(str(key))
                if key_norm in row_norm_map:
                    return row_norm_map[key_norm]
            return None

        out_rows_dict: List[List[str]] = []
        for row in rows:
            vals: List[str] = []
            for idx, h in enumerate(norm_headers):
                v = _dict_cell(row, h, idx)
                vals.append("" if v is None else str(v).strip())
            out_rows_dict.append(vals)
        if not out_rows_dict:
            return pd.DataFrame()
        df = pd.DataFrame(out_rows_dict, columns=norm_headers)
        logger.info(
            "_vision_json_to_dataframe [%s]: object rows OK shape=%s headers=%s",
            table_name, df.shape, norm_headers[:5],
        )
        return df

    if not isinstance(headers, list) or not headers:
        logger.warning("_vision_json_to_dataframe [%s]: headers manquants ou vides", table_name)
        return pd.DataFrame()

    if not isinstance(rows, list):
        logger.warning("_vision_json_to_dataframe [%s]: clé 'rows' manquante", table_name)
        return pd.DataFrame()

    n_cols = len(headers)
    norm_headers = [str(h if h is not None else "").strip() for h in headers]

    padded_short = 0
    trimmed_long = 0
    skipped_non_list = 0
    out_rows: List[List[str]] = []

    for i, row in enumerate(rows):
        if not isinstance(row, list):
            skipped_non_list += 1
            logger.debug(
                "_vision_json_to_dataframe [%s]: ligne %d ignorée (type=%s): %r",
                table_name, i, type(row).__name__, row,
            )
            continue

        # Normaliser : None → "" et cast str sans conversion numérique
        vals = [str(v if v is not None else "").strip() for v in row]

        if len(vals) < n_cols:
            vals.extend([""] * (n_cols - len(vals)))
            padded_short += 1
        elif len(vals) > n_cols:
            vals = vals[:n_cols]
            trimmed_long += 1

        out_rows.append(vals)

    if skipped_non_list:
        logger.warning(
            "_vision_json_to_dataframe [%s]: %d lignes ignorées (pas des listes)",
            table_name, skipped_non_list,
        )
    if padded_short:
        logger.info(
            "_vision_json_to_dataframe [%s]: %d lignes courtes paddées → %d colonnes",
            table_name, padded_short, n_cols,
        )
    if trimmed_long:
        logger.info(
            "_vision_json_to_dataframe [%s]: %d lignes longues tronquées → %d colonnes",
            table_name, trimmed_long, n_cols,
        )

    if not out_rows:
        logger.warning("_vision_json_to_dataframe [%s]: aucune ligne valide extraite", table_name)
        return pd.DataFrame()

    df = pd.DataFrame(out_rows, columns=norm_headers)
    logger.info(
        "_vision_json_to_dataframe [%s]: shape=%s headers=%s",
        table_name, df.shape, norm_headers[:5],
    )
    return df


def _blank_dash_cells_from_page_text(
    df: pd.DataFrame,
    pdf_path: str,
    page_num: int,
) -> pd.DataFrame:
    """
    Corrige les hallucinations Vision sur les lignes dont le PDF natif indique "- -".

    Exemple: si la page contient "Passifs financiers ... - -", les colonnes
    numeriques correspondantes doivent rester vides meme si le LLM a invente
    des montants.
    """
    if df is None or df.empty or pymupdf is None:
        return df
    try:
        doc = pymupdf.open(pdf_path)
        try:
            if page_num < 1 or page_num > len(doc):
                return df
            text = doc[page_num - 1].get_text("text") or ""
        finally:
            doc.close()
    except Exception as exc:
        logger.debug("_blank_dash_cells_from_page_text: texte PDF indisponible: %s", exc)
        return df

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return df

    def _is_value_col(col: Any, idx: int) -> bool:
        c = _norm(str(col))
        if idx == 0:
            return False
        if c in {"notes", "note", "ref", "reference"}:
            return False
        if c in {"is_subrow", "indent_level", "is_total", "confidence"}:
            return False
        return True

    value_col_indices = [i for i, col in enumerate(df.columns) if _is_value_col(col, i)]
    if not value_col_indices:
        return df

    def _line_has_dash_values(line: str) -> bool:
        normalized = (
            str(line)
            .replace("–", "-")
            .replace("—", "-")
            .replace("−", "-")
        )
        # Deux cellules vides imprimees en tirets, sans montant reel sur la meme ligne.
        if not re.search(r"(?<!\w)-\s+-", normalized):
            return False
        return not bool(re.search(r"\d[\d\s.,]{2,}", normalized))

    dash_lines = [(_norm(ln), ln) for ln in lines if _line_has_dash_values(ln)]
    if not dash_lines:
        return df

    fixed = df.copy()
    blanked = 0
    for row_idx in range(len(fixed)):
        label = str(fixed.iat[row_idx, 0] if len(fixed.columns) else "").strip()
        label_norm = _norm(label)
        if len(label_norm) < 12:
            continue
        for line_norm, _line in dash_lines:
            if label_norm in line_norm or line_norm in label_norm:
                for col_idx in value_col_indices:
                    if str(fixed.iat[row_idx, col_idx]).strip():
                        blanked += 1
                    fixed.iat[row_idx, col_idx] = ""
                break
    if blanked:
        logger.info(
            "_blank_dash_cells_from_page_text: %d cellule(s) numeriques videes selon tirets PDF",
            blanked,
        )
    return fixed


def _pdf_value_row_records(pdf_path: str, page_num: int) -> list[dict[str, Any]]:
    """Extrait des lignes candidates (libelle + 2 valeurs) depuis les mots natifs PDF."""
    if pymupdf is None:
        return []
    try:
        doc = pymupdf.open(pdf_path)
        try:
            if page_num < 1 or page_num > len(doc):
                return []
            words = doc[page_num - 1].get_text("words") or []
        finally:
            doc.close()
    except Exception as exc:
        logger.debug("_pdf_value_row_records: mots PDF indisponibles: %s", exc)
        return []

    grouped: dict[tuple[int, int], list[tuple[float, float, str]]] = {}
    for w in words:
        if len(w) < 8:
            continue
        x0, y0, x1, _y1, txt, block_no, line_no, _word_no = w[:8]
        grouped.setdefault((int(block_no), int(line_no)), []).append((float(x0), float(x1), str(txt)))

    ordered_lines: list[dict[str, Any]] = []
    for (block_no, line_no), items in sorted(grouped.items()):
        items_sorted = sorted(items, key=lambda item: item[0])
        line_text = " ".join(t for _x0, _x1, t in items_sorted).strip()
        ordered_lines.append({
            "block_no": block_no,
            "line_no": line_no,
            "items": items_sorted,
            "text": line_text,
            "norm": _norm(line_text),
        })

    def _is_value_token(token: str) -> bool:
        t = str(token).strip().replace("–", "-").replace("—", "-").replace("−", "-")
        if t == "-":
            return True
        return bool(re.fullmatch(r"\(?\d[\d.,]*\)?", t))

    def _split_values(items: list[tuple[float, float, str]]) -> Optional[tuple[int, str, str]]:
        trailing: list[tuple[float, float, str]] = []
        for item in reversed(items):
            if _is_value_token(item[2]):
                trailing.append(item)
                continue
            break
        trailing = list(reversed(trailing))
        if len(trailing) < 2:
            return None
        centers = [((x0 + x1) / 2.0) for x0, x1, _t in trailing]
        if len(trailing) == 2:
            split_at = 1
        else:
            gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
            split_at = gaps.index(max(gaps)) + 1
        left = trailing[:split_at]
        right = trailing[split_at:]
        if not left or not right:
            return None

        def _value(parts: list[tuple[float, float, str]]) -> str:
            toks = [p[2].strip().replace("–", "-").replace("—", "-").replace("−", "-") for p in parts]
            if all(t == "-" for t in toks):
                return ""
            return " ".join(toks)

        first_value_idx = len(items) - len(trailing)
        return first_value_idx, _value(left), _value(right)

    records: list[dict[str, Any]] = []
    previous_without_values: Optional[dict[str, Any]] = None
    for line in ordered_lines:
        split = _split_values(line["items"])
        if split is None:
            previous_without_values = line
            continue
        first_value_idx, value_current, value_previous = split
        label_items = line["items"][:first_value_idx]
        label = " ".join(t for _x0, _x1, t in label_items).strip()
        if not label:
            previous_without_values = None
            continue
        labels = [label]
        if previous_without_values and previous_without_values["block_no"] == line["block_no"]:
            prev_norm = previous_without_values["norm"]
            if prev_norm and not any(h in prev_norm for h in ("bilan", "passif ifrs", "actif ifrs", "compte de resultat")):
                labels.insert(0, f"{previous_without_values['text']} {label}".strip())
        for candidate_label in labels:
            records.append({
                "label": candidate_label,
                "label_norm": _norm(candidate_label),
                "value_current": value_current,
                "value_previous": value_previous,
                "block_no": line["block_no"],
                "line_no": line["line_no"],
            })
        previous_without_values = None
    if len(records) >= 5:
        return records

    # Beaucoup de PDFs AMMC exposent chaque cellule sur une ligne separee:
    # libelle, valeur N, valeur N-1. On reconstruit alors les lignes sequentiellement.
    try:
        doc = pymupdf.open(pdf_path)
        try:
            text = doc[page_num - 1].get_text("text") or ""
        finally:
            doc.close()
    except Exception:
        return records

    def _is_value_line(line: str) -> bool:
        s = str(line).strip().replace("–", "-").replace("—", "-").replace("−", "-")
        if s == "-":
            return True
        return bool(re.fullmatch(r"-?\d[\d\s.,]*", s))

    def _clean_value(line: str) -> str:
        s = str(line).strip().replace("–", "-").replace("—", "-").replace("−", "-")
        return "" if s == "-" else s

    def _is_header_or_deco(line: str) -> bool:
        n = _norm(line)
        if not n:
            return True
        if re.fullmatch(r"20\d{2}.*", n) and any(m in n for m in ("juin", "dec", "12", "06")):
            return True
        return n in {
            "bilan consolide ifrs",
            "actif ifrs",
            "passif ifrs",
            "compte de produits et charges consolide ifrs",
            "compte de produits et charges ifrs consolides",
        }

    text_records: list[dict[str, Any]] = []
    label_parts: list[str] = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if _is_header_or_deco(line):
            i += 1
            continue
        if _is_value_line(line):
            if label_parts and i + 1 < len(lines) and _is_value_line(lines[i + 1]):
                label = " ".join(label_parts).strip()
                text_records.append({
                    "label": label,
                    "label_norm": _norm(label),
                    "value_current": _clean_value(line),
                    "value_previous": _clean_value(lines[i + 1]),
                    "block_no": 0,
                    "line_no": i,
                })
                label_parts = []
                i += 2
                continue
            label_parts = []
            i += 1
            continue
        label_parts.append(line)
        i += 1
    return text_records or records


def _value_columns_for_df(df: pd.DataFrame) -> list[int]:
    def _is_value_col(col: Any, idx: int) -> bool:
        c = _norm(str(col))
        if idx == 0:
            return False
        if c in {"notes", "note", "ref", "reference"}:
            return False
        if c in {"is_subrow", "indent_level", "is_total", "confidence"}:
            return False
        return True

    return [i for i, col in enumerate(df.columns) if _is_value_col(col, i)]


def _source_reconstruct_bilan_if_available(
    df: pd.DataFrame,
    pdf_path: str,
    page_num: int,
    table_name: str,
) -> pd.DataFrame:
    """Reconstruit ACTIF/PASSIF depuis le texte PDF natif si un segment fiable existe."""
    family = _table_family(table_name)
    if family not in {"bilan_actif", "bilan_passif"}:
        return df
    records = _pdf_value_row_records(pdf_path, page_num)
    if not records:
        return df
    total_key = "total passif" if family == "bilan_passif" else "total actif"
    total_indices = [i for i, r in enumerate(records) if total_key in r["label_norm"]]
    if not total_indices:
        return df
    total_idx = total_indices[-1]
    total_block = records[total_idx]["block_no"]
    if family == "bilan_passif":
        start_markers = ("banques centrales", "passifs financiers")
    else:
        start_markers = ("valeurs en caisse", "actifs financiers")
    marker_indices = [
        i for i in range(0, total_idx + 1)
        if records[i]["block_no"] == total_block
        and any(marker in records[i]["label_norm"] for marker in start_markers)
    ]
    if marker_indices:
        window_start = max(0, total_idx - 45)
        in_window = [i for i in marker_indices if i >= window_start]
        start_idx = (in_window[0] if in_window else marker_indices[-1])
        # Pour PASSIF, la ligne "Banques centrales..." precede souvent
        # "Passifs financiers..."; on garde cette premiere ligne si elle est juste avant.
        if family == "bilan_passif" and start_idx > 0:
            prev = records[start_idx - 1]
            if prev["block_no"] == total_block and "banques centrales" in prev["label_norm"]:
                start_idx -= 1
    else:
        start_idx = total_idx
        stop_totals = {"total actif", "total de l'actif"} if family == "bilan_passif" else {"total passif", "total du passif"}
        while start_idx > 0 and records[start_idx - 1]["block_no"] == total_block:
            prev_norm = records[start_idx - 1]["label_norm"]
            if any(anchor in prev_norm for anchor in stop_totals):
                break
            start_idx -= 1
    segment = records[start_idx: total_idx + 1]
    if len(segment) < 8:
        return df

    columns = list(df.columns) if df is not None and not df.empty else ["label", "value_current", "value_previous"]
    value_cols = _value_columns_for_df(pd.DataFrame(columns=columns))
    if len(value_cols) < 2:
        columns = ["label", "value_current", "value_previous"]
        value_cols = [1, 2]

    out_rows: list[list[str]] = []
    for rec in segment:
        vals = [""] * len(columns)
        vals[0] = rec["label"]
        vals[value_cols[0]] = rec["value_current"]
        vals[value_cols[1]] = rec["value_previous"]
        out_rows.append(vals)
    rebuilt = pd.DataFrame(out_rows, columns=columns)
    logger.info(
        "_source_reconstruct_bilan_if_available: %s reconstruit depuis PDF natif shape=%s",
        table_name,
        rebuilt.shape,
    )
    return rebuilt


def _correct_values_from_pdf_source(
    df: pd.DataFrame,
    pdf_path: str,
    page_num: int,
    table_name: str,
) -> pd.DataFrame:
    """Remplace les valeurs LLM par les valeurs exactes du texte PDF quand possible."""
    if df is None or df.empty:
        return df
    rebuilt = _source_reconstruct_bilan_if_available(df, pdf_path, page_num, table_name)
    if rebuilt is not df:
        return rebuilt

    records = _pdf_value_row_records(pdf_path, page_num)
    if not records:
        return _blank_dash_cells_from_page_text(df, pdf_path=pdf_path, page_num=page_num)
    import difflib

    fixed = df.copy()
    value_cols = _value_columns_for_df(fixed)
    if len(value_cols) < 2:
        return _blank_dash_cells_from_page_text(fixed, pdf_path=pdf_path, page_num=page_num)
    corrections = 0
    for row_idx in range(len(fixed)):
        label_norm = _norm(str(fixed.iat[row_idx, 0]))
        if len(label_norm) < 10:
            continue
        best_score = 0.0
        best: Optional[dict[str, Any]] = None
        for rec in records:
            rec_norm = rec["label_norm"]
            score = difflib.SequenceMatcher(None, label_norm, rec_norm).ratio()
            if label_norm in rec_norm or rec_norm in label_norm:
                score = max(score, 0.92)
            if score > best_score:
                best_score = score
                best = rec
        if best is not None and best_score >= 0.84:
            new_vals = [best["value_current"], best["value_previous"]]
            for col_idx, new_val in zip(value_cols[:2], new_vals):
                if str(fixed.iat[row_idx, col_idx]).strip() != str(new_val).strip():
                    corrections += 1
                fixed.iat[row_idx, col_idx] = new_val
    if corrections:
        logger.info("_correct_values_from_pdf_source: %d valeur(s) corrigees depuis PDF natif", corrections)
    return _blank_dash_cells_from_page_text(fixed, pdf_path=pdf_path, page_num=page_num)


def _vision_save_raw_output(
    content: str,
    pdf_path: str,
    page_num: int,
    table_name: str,
    output_dir: str = VISION_DEBUG_RAW_OUTPUT_DIR,
) -> None:
    """
    Sauvegarde la réponse brute du Vision LLM dans un fichier texte de debug.

    Nom de fichier : {pdf_stem}_{TABLE_SAFE}_page{N}_raw_llm.txt
    """
    import pathlib

    try:
        out_dir = pathlib.Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_stem = pathlib.Path(pdf_path).stem
        table_safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", table_name.upper())[:40]
        filename = f"{pdf_stem}_{table_safe}_page{page_num}_raw_llm.txt"
        filepath = out_dir / filename
        filepath.write_text(content, encoding="utf-8")
        logger.info("_vision_save_raw_output: sauvegardé → %s", filepath)
    except Exception as e:
        logger.warning("_vision_save_raw_output: impossible de sauvegarder: %s", e)


def _vision_parse_response(
    content: str,
    table_name: str,
    pdf_path: str,
    page_num: int,
    debug_save: bool = VISION_DEBUG_SAVE_RAW_OUTPUT,
) -> pd.DataFrame:
    """
    Parse la réponse brute d'un Vision LLM vers un DataFrame.

    Stratégie en deux passes :
    1. Tente le parsing JSON strict ({headers, rows}) via _vision_json_to_dataframe.
    2. Si JSON absent ou invalide, fallback vers le parsing Markdown existant.

    La sauvegarde debug du raw output se fait ici, avant tout parsing.

    Args:
        content:    Réponse brute du LLM (JSON ou Markdown).
        table_name: Nom du tableau (pour logs + nommage fichier debug).
        pdf_path:   Chemin PDF (pour nommage fichier debug).
        page_num:   Numéro de page (pour nommage fichier debug).
        debug_save: Si True, sauvegarde le raw output dans VISION_DEBUG_RAW_OUTPUT_DIR.

    Returns:
        DataFrame ou DataFrame vide si aucun parsing n'a fonctionné.
    """
    if not content or not content.strip():
        logger.warning("_vision_parse_response [%s]: contenu vide", table_name)
        return pd.DataFrame()

    # ── Debug : sauvegarde de la réponse brute ────────────────────────────────
    if debug_save:
        _vision_save_raw_output(content, pdf_path, page_num, table_name)

    # ── Passe 1 : parsing JSON {headers, rows} ────────────────────────────────
    json_str = _extract_json_object(content)
    if json_str:
        try:
            payload = _safe_load_llm_json(json_str)
            if payload is None:
                raise ValueError("JSON invalide après normalisation")
            df = _vision_json_to_dataframe(payload, table_name=table_name)
            if _is_valid_table(df):
                logger.info(
                    "_vision_parse_response [%s]: JSON OK shape=%s",
                    table_name, df.shape,
                )
                return df
            if _is_usable_llm_table(df):
                logger.warning(
                    "_vision_parse_response [%s]: JSON partiel accepté pour continuité LLM shape=%s",
                    table_name, df.shape,
                )
                return df
            logger.debug(
                "_vision_parse_response [%s]: JSON parsé mais tableau invalide (%s) → fallback Markdown",
                table_name, df.shape if df is not None else "None",
            )
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(
                "_vision_parse_response [%s]: JSON invalide (%s) → fallback Markdown",
                table_name, e,
            )

    # ── Passe 2 : fallback parsing Markdown (comportement antérieur) ──────────
    md = _extract_markdown_table(content)
    if md:
        df = _markdown_table_to_dataframe(md)
        if _is_valid_table(df):
            logger.info(
                "_vision_parse_response [%s]: Markdown OK shape=%s",
                table_name, df.shape,
            )
            return df
        if _is_usable_llm_table(df):
            logger.warning(
                "_vision_parse_response [%s]: Markdown partiel accepté pour continuité LLM shape=%s",
                table_name, df.shape,
            )
            return df

    logger.warning(
        "_vision_parse_response [%s]: ni JSON ni Markdown valide dans la réponse LLM "
        "(longueur=%d, extrait=%r)",
        table_name, len(content), content[:150],
    )
    return pd.DataFrame()


def _build_vision_prompt(
    table_name: str,
    type_comptes: Optional[str] = None,
    secteur: Optional[str] = None,
) -> str:
    """
    Construit le prompt Vision LLM unique et intelligent pour l'extraction de tableaux
    financiers marocains.

    Un seul prompt couvre les 3 familles (ACTIF / PASSIF / CPC) et les variantes
    (sociaux / consolidés, banque / assurance / autre).

    Args:
        table_name:   Nom exact du tableau à extraire (ex: "BILAN PASSIF · COMPTES SOCIAUX").
        type_comptes: "sociaux", "consolides", ou None si inconnu.
        secteur:      "banque", "assurance", "autre", ou None si inconnu.

    Returns:
        Prompt string prêt à envoyer au Vision LLM.
    """
    name_l = (table_name or "").lower()
    name_norm = _norm(table_name or "")
    type_norm = _norm(type_comptes or "")
    is_consolidated = "consolid" in name_norm or "consolid" in type_norm

    # ── Détection automatique de la famille de tableau ────────────────────────
    is_actif   = "actif"  in name_l and "passif" not in name_l
    is_passif  = "passif" in name_l
    is_cpc     = (
        "cpc" in name_l
        or "produits et charges" in name_l
        or "compte de r" in name_l  # résultat / resultat
    )

    # ── Hint contextuel (ajouté uniquement si pertinent) ─────────────────────
    context_hints: list = []
    if is_actif:
        context_hints.append(
            "Ce tableau liste les emplois (actif) : immobilisations, créances, trésorerie."
        )
        if secteur == "banque":
            context_hints.append(
                "Format bancaire PCEC : colonnes typiques = Exercice N | Exercice N-1. "
                "Lignes : créances sur établissements de crédit, créances sur la clientèle, "
                "titres de placement, immobilisations, autres actifs, total de l'actif."
            )
        elif secteur == "assurance":
            context_hints.append(
                "Format assurance : colonnes = Brut | Provisions | Net N | Net N-1 possible."
            )
        else:
            context_hints.append(
                "Format CGNC sociaux : colonnes = Brut | Amort/Prov | Net N | Net N-1."
            )
    elif is_passif:
        context_hints.append(
            "Ce tableau liste les ressources (passif) : capitaux propres, dettes, provisions."
        )
        if secteur == "banque":
            context_hints.append(
                "Format bancaire PCEC : colonnes typiques = Exercice N | Exercice N-1. "
                "Lignes : dépôts de la clientèle, dettes envers établissements de crédit, "
                "titres de créance émis, provisions, capital, réserves, résultat net."
            )
        elif secteur == "assurance":
            context_hints.append(
                "Format assurance : provisions techniques présentes côté passif."
            )
        else:
            context_hints.append(
                "Format CGNC : capitaux propres assimilés, dettes de financement, "
                "passif circulant, trésorerie passif, total du passif."
            )
    elif is_cpc:
        context_hints.append(
            "Ce tableau est un compte de résultat : produits en haut, charges en dessous, "
            "résultat net en bas."
        )
        if secteur == "banque":
            context_hints.append(
                "Format PCEC bancaire : produit net bancaire, charges générales d'exploitation, "
                "dotations aux provisions, résultat courant, résultat net."
            )

    if is_consolidated:
        context_hints.append(
            "Comptes consolidés : une colonne 'Notes' ou 'Ref' peut être présente entre "
            "le libellé et les montants — inclure cette colonne si visible."
        )

    if is_consolidated:
        context_hints.append(
            "Regle obligatoire consolidee : si l'en-tete contient NOTES 30/06/2024 "
            "31/12/2023, la sortie doit garder une colonne notes entre label et les "
            "colonnes de dates. Ne jamais fusionner les notes avec le libelle ou les montants."
        )

    if is_consolidated and is_cpc:
        context_hints.append(
            "Banques consolidees : le CPC est souvent intitule 'COMPTE DE RESULTAT "
            "CONSOLIDE'. Il faut le traiter comme le tableau CPC consolide demande."
        )
        context_hints.append(
            "Lignes attendues possibles : Interets et produits assimiles, Interets et "
            "charges assimiles, MARGE D'INTERET, Commissions (produits), Commissions "
            "(charges), MARGE SUR COMMISSIONS, gains/pertes nets sur instruments "
            "financiers, PRODUIT NET BANCAIRE, Charges generales d'exploitation, "
            "RESULTAT BRUT D'EXPLOITATION, Cout du risque de credit, RESULTAT "
            "D'EXPLOITATION, RESULTAT AVANT IMPOTS, RESULTAT NET, Interets "
            "minoritaires, RESULTAT NET PART DU GROUPE, resultat de base/dilue par action."
        )

    context_block = ""
    if context_hints:
        context_block = "\nContexte du tableau :\n" + "\n".join(f"- {h}" for h in context_hints)

    section_type = _target_section_type(table_name) or "CPC"
    section_rules = {
        "BILAN_ACTIF": (
            "On the page, extract the ACTIF section only. It may be titled \"BILAN CONSOLIDE IFRS\" "
            "with subheader \"ACTIF IFRS\". It starts at the ACTIF header row "
            "and, if visible, ends at \"Total de l'Actif\", \"TOTAL ACTIF\", or equivalent. Do not include PASSIF, HORS BILAN, "
            "CPC, notes, footer, or page header."
        ),
        "BILAN_PASSIF": (
            "On the page, extract the PASSIF section only. It may be titled \"BILAN CONSOLIDE IFRS\" "
            "with subheader \"PASSIF IFRS\". It starts at the PASSIF header row "
            "and, if visible, ends at \"Total du Passif\", \"TOTAL PASSIF\", or equivalent. Do not include HORS BILAN, CPC, notes, "
            "footer, or any following section."
        ),
        "CPC": (
            "On the page, extract the Compte de produits et charges / Compte de resultat "
            "consolide section only. In Moroccan bank consolidated statements, CPC may be titled "
            "\"COMPTE DE RESULTAT CONSOLIDE\" or \"COMPTE DE PRODUITS ET CHARGES CONSOLIDE IFRS\". It starts at the CPC/result title/table header and "
            "contains CPC rows until the section ends. Do not include "
            "BILAN, HORS BILAN, notes, footer, or unrelated sections."
        ),
    }
    section_description = section_rules.get(section_type, "")
    columns_example = (
        '["label", "NOTES", "30/06/2024", "31/12/2023"]'
        if is_consolidated
        else '["label", "value_2024", "value_2023"]'
    )
    row_notes_example = '"NOTES": "1.1",\n      ' if is_consolidated else ""
    consolidated_rules = ""
    if is_consolidated:
        consolidated_rules = """
CONSOLIDATED STATEMENT COLUMN RULES:
- Actively look for a visible narrow NOTES/Note/Ref column between the label column and the date/value columns.
- If the header row is "NOTES 30/06/2024 31/12/2023", the output columns MUST be ["label", "NOTES", "30/06/2024", "31/12/2023"].
- NEVER drop the notes column when it is visible.
- NEVER merge note numbers into the label or amount columns.
- If a row has no visible note, set "NOTES": null.
- Notes may be numeric or textual, for example "1", "2.3", "11", "A", or empty.
"""
    multi_block_rules = """
FULL-PAGE MULTI-BLOCK RULES:
- The page may contain up to 4 visual blocks arranged in a grid.
- Common layout: BILAN_ACTIF upper-left, BILAN_PASSIF upper-right, CPC lower-left, FLUX DE TRESORERIE lower-right.
- Extract ONLY the expected target section named above.
- Completely ignore FLUX DE TRESORERIE IFRS, TABLEAU DES FLUX DE TRESORERIE, HORS BILAN, ETAT DE VARIATION DES CAPITAUX PROPRES, notes, legal footer text, website URL, page number, and decorative content.
- Never extract any row from ignored sections, especially the lower-right FLUX block.
- If another target section is visible but it is not the expected section, ignore it.
"""
    return f"""You are extracting data from a Moroccan financial statement page image.

The image may be a FULL PAGE containing several sections, or a crop.
Your job is to locate and extract ONLY ONE target section:
- BILAN_ACTIF
OR
- BILAN_PASSIF
OR
- CPC

EXPECTED SECTION TYPE:
- {section_type}
{context_block}

DESCRIPTIVE EXTRACTION CONTEXT:
- The image is from a Moroccan AMMC-style financial report and may show multiple financial sections.
- Target table requested by the user: "{table_name}".
- {section_description}
- Read the table visually from top to bottom and left to right.
- Extract only rows that belong to the expected section, even if other tables are visible on the same page.
- If there are colored bars, headers, or totals, use them only to understand the table boundaries.
- Ignore page decorations, logos, page numbers, notes, and any section outside the expected section.
- Never merge ACTIF with PASSIF.
- Never merge PASSIF with HORS BILAN.
- Never merge CPC / COMPTE DE RESULTAT CONSOLIDE with another table.
{multi_block_rules}

TASK:
Extract all visible rows exactly as they appear.

RULES:
- Preserve exact French labels.
- Preserve row order exactly.
- Preserve empty values as null.
- If a numeric cell is printed as "-" or is visually blank, return null/empty. NEVER replace "-" or blank cells with guessed numbers.
- If a row has two printed dashes like "- -" under the numeric columns, both numeric values MUST be null/empty.
- Preserve indentation hierarchy using indent_level.
- Do not invent missing rows.
- Do not invent, estimate, recalculate, or shift values from neighboring rows.
- Do not normalize numbers.
- Keep values exactly as printed, including spaces.
- If a row is partially visible, include it with low confidence.
- Use visible date/value headers as column names when possible.
- The first column MUST be "label".
- Preserve every visible business column from the table header.
- Include metadata fields on every row: is_subrow, indent_level, is_total, confidence.
- Mark is_total=true for total/subtotal rows such as "Total de l'Actif", "Total du Passif", "Résultat net", "TOTAL".
- Use indent_level=0 for main rows, 1 or more for visually indented subrows.
- Do not include table titles as data rows unless they are part of the table header itself.
- Do not return Markdown. Do not wrap the JSON in ```json fences.
{consolidated_rules}

OUTPUT STRICT JSON ONLY:
{{
  "section_type": "{section_type}",
  "columns": {columns_example},
  "rows": [
    {{
      "label": "Créances sur la clientèle",
      {row_notes_example}"value_2024": "271 414 638",
      "value_2023": "246 950 715",
      "is_subrow": false,
      "indent_level": 0,
      "is_total": false,
      "confidence": 0.99
    }}
  ]
}}

If the section is not found: {{"section_type": "{section_type}", "columns": [], "rows": []}}
STRICT JSON ONLY."""

    prompt = f"""Tu es un extracteur de données financières.
Tu reçois l'image d'une page de rapport financier marocain.
Ta mission : transcrire fidèlement le tableau intitulé « {table_name } ».
{context_block}

RÈGLES ABSOLUES — transcription exacte, pas de reconstruction :
1. Extrais UNIQUEMENT le tableau demandé. Si la page contient plusieurs tableaux (ex: ACTIF et PASSIF côte à côte), ne prends que celui dont le titre correspond.
2. Conserve TOUTES les lignes visibles sans exception : libellés principaux, sous-items, sous-totaux, totaux, lignes sans valeur numérique.
   → UNE LIGNE VISIBLE DANS LE TABLEAU = UNE LIGNE DANS TA RÉPONSE, même si toutes ses cellules numériques sont vides.
   → Ne supprime JAMAIS une ligne sous prétexte qu'elle n'a pas de montant.
   → Exemple : si le tableau contient "Certificats de Sukuks" sans montant, tu dois inclure ["Certificats de Sukuks", "", ""] dans les rows.
3. Conserve TOUTES les colonnes dans leur ordre exact d'apparition.
4. Recopie les libellés mot pour mot, tels qu'ils apparaissent à l'écran (avec ponctuation, points, tirets, majuscules).
5. Recopie les montants exactement comme affichés : garde les espaces séparateurs de milliers ("13 157 683" reste "13 157 683"), les signes négatifs, les parenthèses, les tirets, les zéros.
6. Cellule visuellement vide → retourne "" (chaîne vide). Ne rien inventer.
7. Valeur illisible → écrire [ILLISIBLE].
8. Ne corrige pas, ne reformate pas, ne normalise pas les nombres.
9. Ne supprime pas les lignes de codes ou repères (ex : (A), (B), I., II., notes "2.12").

FORMAT DE SORTIE — JSON strict, sans prose autour :
{{
  "headers": ["Rubrique", "Colonne1", "Colonne2", ...],
  "rows": [
    ["Libellé ligne 1", "valeur1", "valeur2"],
    ["Sous-item sans montant", "", ""],
    ["Libellé ligne 3", "valeur1", "valeur2"]
  ]
}}

Si le tableau n'est pas trouvé sur l'image : {{"headers": [], "rows": []}}
Ne mets aucun texte avant ni après le JSON."""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
#  2e passe Vision — détection et correction ciblée des lignes suspectes
# ─────────────────────────────────────────────────────────────────────────────

# Valeurs considérées comme "vides" dans un DataFrame extrait par LLM.
_EMPTY_CELL_VALUES: frozenset = frozenset({"", "nan", "none", "n/a", "nd"})

# Longueur minimale du libellé pour qu'une ligne soit candidate à la relecture
# (évite de relancer sur des séparateurs ou codes d'une lettre).
_SUSPICIOUS_MIN_LABEL_LEN: int = 4


def _detect_suspicious_rows(df: pd.DataFrame, table_name: str) -> List[str]:
    """
    Détecte les lignes dont le libellé est non vide mais toutes les valeurs
    numériques sont absentes — cas typiques d'oubli LLM ou de conversion `-` → vide.

    Heuristiques :
    - libellé non vide, non trivial (len >= _SUSPICIOUS_MIN_LABEL_LEN)
    - TOUTES les colonnes numériques (col 1+) sont vides / nan
    - libellé ne ressemble pas à un en-tête de colonne ou à un sous-item

    Exclusions explicites (lignes intentionnellement sans montant) :
    - libellés commençant par un marqueur de sous-item (`. `, `- `, `• `, `* `)
    - libellés très courts (codes, numéros de note)
    - libellés ressemblant à une date seule

    Returns:
        Liste ordonnée des libellés suspects (dans l'ordre du DataFrame).
    """
    if df is None or df.empty or len(df.columns) < 2:
        return []

    # Marqueurs de début indiquant un sous-item (intentionnellement sans montant)
    _SUBITEM_PREFIXES = (".", "-", "•", "*", "–", "→", ">")

    value_cols = list(df.columns[1:])
    suspicious: List[str] = []

    for _, row in df.iterrows():
        label = str(row.iloc[0]).strip()
        # Ignorer libellés vides ou triviaux
        if not label or label.lower() in _EMPTY_CELL_VALUES:
            continue
        if len(label) < _SUSPICIOUS_MIN_LABEL_LEN:
            continue
        # Ignorer lignes ressemblant à une date seule (en-têtes de colonnes)
        if re.match(r"^\d{2}[/\-]\d{2}[/\-]\d{2,4}$", label):
            continue
        # Ignorer les sous-items (ex: ". Certificats de Sukuks", "- dont dépôts vue")
        # Ces lignes sont intentionnellement sans montant dans le PDF.
        if any(label.startswith(p) for p in _SUBITEM_PREFIXES):
            continue

        values = [str(row[c]).strip().lower() for c in value_cols]
        all_empty = all(v in _EMPTY_CELL_VALUES for v in values)
        if all_empty:
            suspicious.append(label)

    if suspicious:
        logger.info(
            "_detect_suspicious_rows [%s]: %d ligne(s) suspecte(s) → %s",
            table_name, len(suspicious),
            [s[:40] for s in suspicious[:6]],
        )
    else:
        logger.debug("_detect_suspicious_rows [%s]: aucune ligne suspecte", table_name)

    return suspicious


def _build_targeted_recheck_prompt(
    table_name: str,
    suspicious_rows: List[str],
    headers: List[str],
) -> str:
    """
    Prompt de 2e passe : relecture ciblée de lignes précises seulement.
    Ne réextrait pas tout le tableau — uniquement les lignes listées.
    """
    rows_block = "\n".join(f'  - "{r}"' for r in suspicious_rows)
    # En-têtes numériques (tout sauf la première colonne = libellé)
    num_headers = " | ".join(headers[1:]) if len(headers) > 1 else "Exercice N | Exercice N-1"

    return f"""Tu es un extracteur de données financières de précision.
Tu reçois l'image d'une page de rapport financier marocain.
Le tableau principal s'intitule « {table_name} ».

MISSION : relire uniquement les lignes listées ci-dessous et retourner leurs valeurs exactes.

Lignes à relire ({len(suspicious_rows)}) :
{rows_block}

RÈGLES ABSOLUES :
1. Cherche chaque libellé UNIQUEMENT dans le tableau « {table_name} ».
   Si d'autres tableaux sont visibles sur la page, ignore-les totalement.
2. TRANSCRIPTION STRICTE — ne lis que ce qui est visuellement présent dans la cellule :
   - cellule visuellement vide dans l'image → retourne `""` (chaîne vide) — NE PAS inventer une valeur
   - cellule contenant `-` → retourne `"-"` (NE PAS la remplacer par vide)
   - cellule avec un nombre → retourne le nombre tel qu'affiché, espaces séparateurs inclus
   - conserve les parenthèses, signes négatifs, virgules, zéros
   - NE PAS copier une valeur depuis une autre ligne ou une autre colonne
   - NE PAS interpoler, NE PAS estimer, NE PAS déduire
3. Si le libellé est introuvable sur la page → retourne `[ILLISIBLE]` pour ses valeurs.
4. Retourne EXACTEMENT autant de lignes que demandées, dans le même ordre.

RAPPEL CRITIQUE : une cellule vide dans le tableau PDF = `""` dans ta réponse. Jamais une valeur inventée.

FORMAT DE SORTIE — JSON strict, aucune prose autour :
{{
  "rows": [
    ["libellé exact tel qu'il apparaît dans le tableau", "valeur_col1", "valeur_col2"],
    ["libellé avec cellule vide légitime", "12 345", ""],
    ["libellé introuvable", "[ILLISIBLE]", "[ILLISIBLE]"]
  ]
}}"""


def _call_vision_llm(
    img_b64: str,
    prompt: str,
    provider: str,
    api_key: str,
) -> str:
    """
    Appelle le Vision LLM avec une image base64 et un prompt.
    Centralise la logique multi-provider pour éviter la duplication entre
    la 1ère extraction et la 2e passe de relecture.

    Returns:
        Contenu textuel de la réponse, ou "" si échec.
    """
    providers_to_try = [provider]
    for fallback_provider in ("gemini", "gpt-5.4"):
        if fallback_provider != provider and _provider_api_key(fallback_provider):
            providers_to_try.append(fallback_provider)

    for current_provider in providers_to_try:
        current_api_key = api_key if current_provider == provider else _provider_api_key(current_provider)
        if not current_api_key:
            continue

        try:
            if current_provider == "groq":
                content = _groq_chat_with_model_fallback(
                    api_key=current_api_key,
                    model_candidates=GROQ_VISION_MODEL_CANDIDATES,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                    temperature=0.0,
                    max_tokens=8192,
                )
            elif current_provider == "gpt-5.4":
                from openai import OpenAI  # type: ignore
                client = OpenAI(api_key=current_api_key)
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                            {"type": "text", "text": prompt},
                        ],
                    }],
                    temperature=0.0,
                    max_tokens=8192,
                )
                content = (completion.choices[0].message.content or "").strip()
            elif current_provider == "gemini":
                content = _gemini_generate_content(
                    api_key=current_api_key,
                    text_prompt=prompt,
                    inline_image_b64=img_b64,
                    temperature=0.0,
                    max_output_tokens=8192,
                )
            else:
                continue

            if content:
                if _looks_like_no_image_response(content):
                    logger.warning(
                        "_call_vision_llm: réponse non-vision détectée provider=%s (le modèle dit ne pas voir l'image) -> retry modèle/provider suivant",
                        current_provider,
                    )
                    continue
                if current_provider != provider:
                    logger.warning(
                        "_call_vision_llm fallback provider used: %s -> %s",
                        provider,
                        current_provider,
                    )
                return content

        except Exception as e:
            logger.warning("_call_vision_llm provider=%s failed: %s", current_provider, e)

    return ""


def _vision_targeted_recheck(
    img_b64: str,
    table_name: str,
    suspicious_rows: List[str],
    headers: List[str],
    provider: str,
    api_key: str,
) -> dict:
    """
    2e passe Vision : envoie la même image au LLM avec un prompt ciblé
    sur les lignes suspectes uniquement.

    Returns:
        Dict {libellé_normalisé: [val_col1, val_col2, ...]} des corrections trouvées.
        Retourne {} si la passe échoue ou ne produit rien d'exploitable.
    """
    if not suspicious_rows or not img_b64:
        return {}

    prompt = _build_targeted_recheck_prompt(table_name, suspicious_rows, headers)
    logger.info(
        "_vision_targeted_recheck: lancement 2e passe provider=%s — %d ligne(s): %s",
        provider, len(suspicious_rows), [s[:30] for s in suspicious_rows],
    )

    content = _call_vision_llm(img_b64, prompt, provider, api_key)
    if not content:
        logger.warning("_vision_targeted_recheck: réponse vide du LLM")
        return {}

    # Parsing JSON de la réponse ciblée
    json_str = _extract_json_object(content)
    if not json_str:
        logger.warning(
            "_vision_targeted_recheck: pas de JSON dans la réponse (extrait=%r)",
            content[:120],
        )
        return {}

    try:
        payload = _safe_load_llm_json(json_str)
        if payload is None:
            raise ValueError("JSON invalide après normalisation")
        rows_raw = payload.get("rows", [])
        if not isinstance(rows_raw, list):
            return {}

        corrections: dict = {}
        for row in rows_raw:
            if not isinstance(row, list) or len(row) < 2:
                continue
            label = str(row[0]).strip()
            values = [str(v).strip() for v in row[1:]]
            # Ignorer les lignes entièrement [ILLISIBLE]
            if all(v == "[ILLISIBLE]" for v in values):
                continue
            corrections[label.lower()] = {"label": label, "values": values}

        logger.info(
            "_vision_targeted_recheck: %d correction(s) reçue(s): %s",
            len(corrections), list(corrections.keys())[:6],
        )
        return corrections

    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("_vision_targeted_recheck: JSON invalide (%s)", e)
        return {}


def _merge_targeted_corrections(
    df: pd.DataFrame,
    corrections: dict,
    table_name: str,
) -> pd.DataFrame:
    """
    Fusionne les corrections de la 2e passe dans le DataFrame de la 1ère extraction.

    Règles de fusion :
    - Matching libellé par normalisation (minuscules, espaces homogènes)
    - Applique la correction UNIQUEMENT si la cellule cible est actuellement vide
    - N'écrase jamais une valeur déjà renseignée
    - Conserve la structure et l'ordre du DataFrame intact

    Returns:
        DataFrame avec les corrections appliquées.
    """
    if not corrections or df is None or df.empty:
        return df

    df = df.copy()
    value_cols = list(df.columns[1:])
    applied = 0

    def _norm_label(s: str) -> str:
        return re.sub(r"\s+", " ", s.lower().strip())

    for idx, row in df.iterrows():
        label_raw = str(row.iloc[0]).strip()
        label_norm = _norm_label(label_raw)
        if not label_norm:
            continue

        # Recherche par correspondance exacte normalisée d'abord,
        # puis par inclusion (label de la correction contenu dans celui du df ou vice-versa)
        correction = corrections.get(label_norm)
        if correction is None:
            for key, corr in corrections.items():
                if key in label_norm or label_norm in key:
                    correction = corr
                    break

        if correction is None:
            continue

        corr_values = correction["values"]
        changed = False
        for col_idx, col in enumerate(value_cols):
            if col_idx >= len(corr_values):
                break
            current = str(df.at[idx, col]).strip()
            is_empty = current.lower() in _EMPTY_CELL_VALUES
            corr_val = corr_values[col_idx]
            # Appliquer uniquement si la cellule est vide et la correction non triviale
            if is_empty and corr_val and corr_val not in ("", "[ILLISIBLE]"):
                df.at[idx, col] = corr_val
                changed = True

        if changed:
            applied += 1
            logger.info(
                "_merge_targeted_corrections [%s]: ligne corrigée → %r",
                table_name, label_raw[:50],
            )

    logger.info(
        "_merge_targeted_corrections [%s]: %d/%d ligne(s) modifiée(s)",
        table_name, applied, len(corrections),
    )
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  Isolation multi-tableaux — localiser, cropper, extraire uniquement le tableau
#  cible avant d'envoyer l'image au Vision LLM.
#
#  Résout le problème : quand une page contient plusieurs tableaux (ex: BILAN
#  ACTIF + BILAN PASSIF), le LLM confond les zones et produit un résultat mixte.
#  Ces helpers détectent visuellement la zone du tableau demandé et cropent
#  uniquement cette zone avant l'envoi.
#
#  Helpers publics :
#    locate_target_table_by_title(pdf_path, page_num, title, ...)
#    crop_table_under_title(pdf_path, page_num, block, ...)
#    extract_target_table_only(pdf_path, page_num, table_name, ...)
# ═══════════════════════════════════════════════════════════════════════════════


def _save_detection_json(
    detection_info: dict,
    page_num: int,
    debug_dir: str,
) -> None:
    """
    Sauvegarde le JSON de détection multi-tableaux dans :
    ``{debug_dir}/page_{N}_target_title_detect.json``
    """
    import pathlib

    try:
        out_dir = pathlib.Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        filename = f"page_{page_num}_target_title_detect.json"
        filepath = out_dir / filename
        filepath.write_text(
            json.dumps(detection_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("_save_detection_json: sauvegardé → %s", filepath)
    except Exception as e:
        logger.warning("_save_detection_json: sauvegarde impossible: %s", e)


def locate_target_table_by_title(
    pdf_path: str,
    page_num: int,
    title: str,
    debug_dir: str = VISION_DEBUG_OUTPUT_DIR,
    save_debug: bool = True,
) -> Tuple[Optional[object], dict]:
    """
    Détecte le bloc de tableau correspondant au titre cible sur la page.

    Utilise ``detect_multi_table_zones()`` pour identifier tous les tableaux
    présents, puis ``find_target_block()`` pour isoler celui qui correspond
    au titre demandé.

    Fichier debug produit (si save_debug=True) :
        ``{debug_dir}/page_{N}_target_title_detect.json``

    Args:
        pdf_path:   Chemin vers le fichier PDF.
        page_num:   Numéro de page 1-based.
        title:      Nom du tableau cible (ex: "BILAN ACTIF").
        debug_dir:  Dossier de sortie pour les fichiers debug.
        save_debug: Si True, sauvegarde le JSON de détection.

    Returns:
        (target_block, detection_info)
        - target_block   : TableBlock ou None si non trouvé / page mono-tableau.
        - detection_info : dict JSON sérialisable avec toutes les infos de debug.
    """
    try:
        from rag_agent.multi_table_detector import (
            detect_multi_table_zones,
            find_target_block,
            is_multi_table_page,
        )
    except ImportError as e:
        logger.warning("locate_target_table_by_title: multi_table_detector indisponible: %s", e)
        return None, {"error": str(e), "is_multi_table": False, "target_found": False}

    detection_info: dict = {
        "pdf_path": pdf_path,
        "page_num": page_num,
        "title_searched": title,
        "blocks_detected": 0,
        "is_multi_table": False,
        "target_found": False,
        "target_block": None,
        "blocks": [],
    }

    try:
        blocks = detect_multi_table_zones(pdf_path, page_num)
    except Exception as e:
        logger.warning("locate_target_table_by_title: detect_multi_table_zones failed: %s", e)
        blocks = []

    detection_info["blocks_detected"] = len(blocks)
    detection_info["is_multi_table"] = is_multi_table_page(blocks)
    detection_info["blocks"] = [
        {
            "index": b.block_index,
            "y_start": round(b.y_start, 2),
            "y_end": round(b.y_end, 2),
            "x_start": round(b.x_start, 2),
            "x_end": round(b.x_end, 2),
            "table_type_hint": b.table_type_hint,
            "trigger_keyword": b.trigger_keyword,
            "text_sample": b.text_sample[:100],
            "confidence": round(b.confidence, 3),
        }
        for b in blocks
    ]
    # Conserver les objets TableBlock pour usage interne (génération de candidates)
    detection_info["_raw_blocks"] = blocks

    if not blocks:
        logger.info(
            "locate_target_table_by_title: aucun bloc détecté page=%d — fallback page entière",
            page_num,
        )
        if save_debug:
            _save_detection_json(detection_info, page_num, debug_dir)
        return None, detection_info

    target = find_target_block(blocks, title)
    if target is not None:
        detection_info["target_found"] = True
        detection_info["target_block"] = {
            "index": target.block_index,
            "y_start": round(target.y_start, 2),
            "y_end": round(target.y_end, 2),
            "x_start": round(target.x_start, 2),
            "x_end": round(target.x_end, 2),
            "table_type_hint": target.table_type_hint,
            "trigger_keyword": target.trigger_keyword,
            "text_sample": target.text_sample[:150],
            "confidence": round(target.confidence, 3),
        }
        logger.info(
            "locate_target_table_by_title: cible trouvée page=%d bloc_idx=%d "
            "y=[%.1f, %.1f] type=%s nb_blocs_page=%d",
            page_num,
            target.block_index,
            target.y_start,
            target.y_end,
            target.table_type_hint,
            len(blocks),
        )
    else:
        logger.warning(
            "locate_target_table_by_title: titre '%s' non trouvé page=%d "
            "(blocs détectés=%d) — fallback page entière",
            title, page_num, len(blocks),
        )

    if save_debug:
        _save_detection_json(detection_info, page_num, debug_dir)

    return target, detection_info


def crop_table_under_title(
    pdf_path: str,
    page_num: int,
    block: object,
    dpi: int = VISION_RENDER_DPI,
    margin: float = 10.0,
    debug_dir: str = VISION_DEBUG_OUTPUT_DIR,
    save_debug: bool = True,
    title_safe: str = "",
) -> Tuple[Optional[object], Optional[str]]:
    """
    Crop la zone d'un TableBlock détecté et retourne ``(PIL image, base64 string)``.

    Rend la zone délimitée par ``block.y_start`` / ``block.y_end`` (avec marge)
    au DPI demandé via PyMuPDF, puis convertit en PNG base64 prêt pour un Vision LLM.

    Fichier debug produit (si save_debug=True et title_safe non vide) :
        ``{debug_dir}/page_{N}_{title_safe}_crop.png``

    Args:
        pdf_path:   Chemin vers le fichier PDF.
        page_num:   Numéro de page 1-based.
        block:      TableBlock retourné par locate_target_table_by_title().
        dpi:        Résolution de rendu (défaut VISION_RENDER_DPI = 300 DPI).
        margin:     Marge en points PDF ajoutée en haut et en bas du clip.
        debug_dir:  Dossier de destination pour le fichier debug.
        save_debug: Si True, sauvegarde l'image crop dans debug_dir.
        title_safe: Suffixe court pour nommer le fichier debug
                    (ex: "BILAN_ACTIF" → ``page_19_bilan_actif_crop.png``).

    Returns:
        (pil_image, b64_string) ou (None, None) si échec.
    """
    import base64
    import io
    import pathlib

    try:
        import fitz  # PyMuPDF
        from PIL import Image  # type: ignore
    except ImportError as e:
        logger.warning("crop_table_under_title: dépendance manquante (%s)", e)
        return None, None

    try:
        doc = fitz.open(pdf_path)
        try:
            if page_num < 1 or page_num > len(doc):
                logger.warning("crop_table_under_title: page %d hors bornes", page_num)
                return None, None

            page = doc[page_num - 1]
            pr = page.rect

            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)

            # Clip rect avec marge, clampé aux limites de la page.
            # Utilise les vraies coordonnées x du bloc pour isoler une colonne
            # quand la page est en layout 2+ colonnes (ex: BILAN ACTIF à gauche,
            # BILAN PASSIF à droite sur la même page).
            clip = fitz.Rect(
                max(pr.x0, block.x_start - margin),
                max(pr.y0, block.y_start - margin),
                min(pr.x1, block.x_end + margin),
                min(pr.y1, block.y_end + margin),
            )

            pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        finally:
            doc.close()

        logger.info(
            "crop_table_under_title: crop OK page=%d y=[%.1f, %.1f] "
            "dpi=%d size=%dx%d",
            page_num, block.y_start, block.y_end, dpi, img.width, img.height,
        )

        # Sauvegarde debug
        if save_debug and title_safe:
            title_slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", title_safe.lower())[:30]
            out_dir = pathlib.Path(debug_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            filename = f"page_{page_num}_{title_slug}_crop.png"
            filepath = out_dir / filename
            img.save(str(filepath), format="PNG", optimize=False)
            logger.info("crop_table_under_title: sauvegardé → %s", filepath)

        # Conversion base64
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        return img, b64

    except Exception as e:
        logger.warning("crop_table_under_title: erreur page=%d: %s", page_num, e)
        return None, None


# ── Helpers pour isolation multi-tableaux robuste ─────────────────────────────

def _infer_table_type_from_name(table_name: str) -> Optional[str]:
    """Infère bilan_actif / bilan_passif / cpc depuis le nom du tableau."""
    import unicodedata
    n = (
        unicodedata.normalize("NFD", str(table_name).lower())
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    if "passif" in n:
        return "bilan_passif"
    if "actif" in n:
        return "bilan_actif"
    if "cpc" in n or "compte de produits" in n or "resultat" in n:
        return "cpc"
    return None


def _expand_table_bbox(
    x0: float, y0: float, x1: float, y1: float,
    page_rect,
    expand_right: float = 0.0,
    expand_bottom: float = 0.0,
    expand_left: float = 0.0,
    expand_top: float = 0.0,
) -> Tuple[float, float, float, float]:
    """Étend une bbox en points PDF, clampée aux limites de page_rect."""
    return (
        max(page_rect.x0, x0 - expand_left),
        max(page_rect.y0, y0 - expand_top),
        min(page_rect.x1, x1 + expand_right),
        min(page_rect.y1, y1 + expand_bottom),
    )


def _get_bbox_text_fitz(pdf_path: str, page_num: int, bbox: Tuple) -> str:
    """Extrait le texte brut d'une zone bbox (en points PDF) via PyMuPDF."""
    try:
        import fitz  # type: ignore
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_num - 1]
            rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
            text = page.get_text(clip=rect) or ""
        finally:
            doc.close()
        return text
    except Exception:
        return ""


def _score_crop_candidate(text: str, table_name: str) -> Tuple[float, dict]:
    """
    Score un crop candidat en analysant le texte extrait de la zone.

    Returns:
        (score, reasons_dict)
    """
    import re
    import unicodedata

    def _norm(t: str) -> str:
        return (
            unicodedata.normalize("NFD", str(t).lower())
            .encode("ascii", "ignore")
            .decode("ascii")
        )

    n = _norm(text)
    table_type = _infer_table_type_from_name(table_name)
    score = 0.0
    reasons: dict = {
        "table_type": table_type,
        "title_match": False,
        "total_match": False,
        "col_headers": False,
        "row_count": 0,
        "number_density": 0.0,
        "wrong_table_penalty": False,
        "truncated_penalty": False,
    }

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    reasons["row_count"] = len(lines)

    numbers = re.findall(r"\b\d[\d\s,.]*\d\b", text)
    reasons["number_density"] = round(len(numbers) / max(len(text), 1) * 100, 2)

    # Présence de colonnes dates ou NOTES
    if re.search(r"\b(20\d{2}|19\d{2})\b", text):
        score += 0.5
        reasons["col_headers"] = True
    if "notes" in n or "note" in n:
        score += 0.3

    if table_type == "bilan_actif":
        if "actif" in n:
            score += 2.0
            reasons["title_match"] = True
        if "total actif" in n or "total de l actif" in n:
            score += 3.0
            reasons["total_match"] = True
        # Pénalité si le crop contient principalement du PASSIF
        if "total passif" in n or ("passif" in n and "actif" not in n):
            score -= 2.0
            reasons["wrong_table_penalty"] = True

    elif table_type == "bilan_passif":
        if "passif" in n:
            score += 2.0
            reasons["title_match"] = True
        if "total passif" in n or "total du passif" in n:
            score += 3.0
            reasons["total_match"] = True
        # Pénalité si le crop contient principalement de l'ACTIF
        if "total actif" in n or ("actif" in n and "passif" not in n):
            score -= 2.0
            reasons["wrong_table_penalty"] = True

    elif table_type == "cpc":
        for kw in ["compte de produits", "cpc", "resultat net",
                   "produits d exploitation", "charges d exploitation",
                   "chiffre d affaires"]:
            if kw in n:
                score += 1.5
                reasons["title_match"] = True
                break
        for kw in ["resultat net", "resultat d exploitation",
                   "resultat avant impots", "benefice net"]:
            if kw in n:
                score += 2.0
                reasons["total_match"] = True
                break
    else:
        if any(w in n for w in ["total", "actif", "passif", "resultat"]):
            score += 1.0
            reasons["title_match"] = True

    # Bonus si le nombre de lignes est raisonnable (tableau non tronqué)
    if reasons["row_count"] >= 10:
        score += 1.0
    elif reasons["row_count"] < 5:
        score -= 1.0
        reasons["truncated_penalty"] = True

    # Bonus densité de nombres
    if reasons["number_density"] > 3.0:
        score += 0.5

    return round(score, 3), reasons


def _generate_crop_candidates(
    block,            # TableBlock
    all_blocks: list, # List[TableBlock]
    page_rect,
    table_name: str,
) -> list:
    """
    Génère plusieurs régions candidates (bbox en points PDF) pour le tableau cible.

    Candidates produites :
      base             – bbox détectée telle quelle
      expand_right     – expansion droite modérée
      expand_bottom    – expansion bas modérée
      expand_right_bottom – expansion droite + bas modérée
      expand_large     – expansion large droite + bas
      actif_left_column / actif_left_half – heuristique ACTIF (gauche)
      passif_right_column / passif_right_half – heuristique PASSIF (droite)
      cpc_below_bilan / cpc_full_width – heuristique CPC (dessous)
      full_width_y_zone – pleine largeur, zone y du bloc (dernier recours)
    """
    pr = page_rect
    pw = pr.x1 - pr.x0
    ph = pr.y1 - pr.y0

    x0, y0, x1, y1 = block.x_start, block.y_start, block.x_end, block.y_end
    exp_r = pw * 0.12
    exp_b = ph * 0.08
    exp_r_lg = pw * 0.25
    exp_b_lg = ph * 0.18

    table_type = _infer_table_type_from_name(table_name)
    co_level = [b for b in all_blocks if abs(b.y_start - block.y_start) <= 20 and b is not block]

    candidates = []
    candidates.append({"bbox": _expand_table_bbox(x0, y0, x1, y1, pr),                                             "label": "base"})
    candidates.append({"bbox": _expand_table_bbox(x0, y0, x1, y1, pr, expand_right=exp_r),                         "label": "expand_right"})
    candidates.append({"bbox": _expand_table_bbox(x0, y0, x1, y1, pr, expand_bottom=exp_b),                        "label": "expand_bottom"})
    candidates.append({"bbox": _expand_table_bbox(x0, y0, x1, y1, pr, expand_right=exp_r, expand_bottom=exp_b),    "label": "expand_right_bottom"})
    candidates.append({"bbox": _expand_table_bbox(x0, y0, x1, y1, pr, expand_right=exp_r_lg, expand_bottom=exp_b_lg), "label": "expand_large"})

    if table_type == "bilan_actif" and co_level:
        right_sib = [b for b in co_level if b.x_start > x0]
        if right_sib:
            xb = min(b.x_start for b in right_sib)
            candidates.append({"bbox": _expand_table_bbox(x0, y0, xb, y1, pr, expand_bottom=exp_b),     "label": "actif_left_column"})
        else:
            candidates.append({"bbox": _expand_table_bbox(pr.x0, y0, pw * 0.55, y1, pr, expand_bottom=exp_b), "label": "actif_left_half"})

    elif table_type == "bilan_passif" and co_level:
        left_sib = [b for b in co_level if b.x_end < x1]
        if left_sib:
            xb = max(b.x_end for b in left_sib)
            candidates.append({"bbox": _expand_table_bbox(xb, y0, pr.x1, y1, pr, expand_bottom=exp_b),  "label": "passif_right_column"})
        else:
            candidates.append({"bbox": _expand_table_bbox(pw * 0.45, y0, pr.x1, y1, pr, expand_bottom=exp_b), "label": "passif_right_half"})

    elif table_type == "cpc":
        above = [b for b in all_blocks if b.y_end < block.y_start - 20]
        if above:
            cpc_y = max(b.y_end for b in above)
            candidates.append({"bbox": _expand_table_bbox(pr.x0, cpc_y, pr.x1, pr.y1, pr),             "label": "cpc_below_bilan"})
        candidates.append({"bbox": _expand_table_bbox(pr.x0, y0, pr.x1, y1, pr, expand_bottom=exp_b),  "label": "cpc_full_width"})

    # Fallback pleine largeur
    candidates.append({"bbox": _expand_table_bbox(pr.x0, y0, pr.x1, y1, pr), "label": "full_width_y_zone"})

    return candidates


def _pick_best_crop_candidate(
    pdf_path: str,
    page_num: int,
    candidates: list,
    table_name: str,
) -> Tuple[Optional[dict], list]:
    """
    Score chaque candidat bbox et retourne le meilleur + la liste complète scorée.

    Returns:
        (best_candidate_dict, all_scored_list)
        best_candidate a les clés : bbox, label, score, reasons
    """
    scored = []
    for cand in candidates:
        bbox = cand["bbox"]
        text = _get_bbox_text_fitz(pdf_path, page_num, bbox)
        score, reasons = _score_crop_candidate(text, table_name)
        entry = {**cand, "score": score, "reasons": reasons, "text_len": len(text)}
        scored.append(entry)
        logger.info(
            "_pick_best_crop_candidate: [%-22s] bbox=(%.0f,%.0f,%.0f,%.0f) "
            "score=%5.2f title=%-5s total=%-5s rows=%3d penalty=%s",
            cand["label"],
            bbox[0], bbox[1], bbox[2], bbox[3],
            score,
            reasons.get("title_match"),
            reasons.get("total_match"),
            reasons.get("row_count"),
            reasons.get("wrong_table_penalty"),
        )

    scored.sort(key=lambda c: c["score"], reverse=True)
    best = scored[0]
    logger.info(
        "_pick_best_crop_candidate: MEILLEUR=[%s] score=%.3f bbox=(%.0f,%.0f,%.0f,%.0f)",
        best["label"], best["score"],
        best["bbox"][0], best["bbox"][1], best["bbox"][2], best["bbox"][3],
    )
    return best, scored


def extract_target_table_only(
    pdf_path: str,
    page_num: int,
    table_name: str,
    dpi: Optional[int] = None,
    enhance: Optional[bool] = None,
    debug_save: bool = VISION_DEBUG_MULTI_TABLE_ISOLATION,
    debug_dir: str = VISION_DEBUG_OUTPUT_DIR,
) -> Tuple[Optional[str], dict]:
    """
    Pipeline d'isolation du tableau cible avant envoi au Vision LLM.

    Étapes :
    1. Rend la page complète en haute qualité.
       → Sauvegarde ``{debug_dir}/page_{N}_full.png`` (si debug_save=True).
    2. Détecte tous les tableaux présents sur la page via PyMuPDF.
       → Sauvegarde ``{debug_dir}/page_{N}_target_title_detect.json``.
    3. Si plusieurs tableaux sont détectés ET le tableau cible est trouvé :
       a. Crop la zone du tableau cible.
       b. Sauvegarde ``{debug_dir}/page_{N}_{table}_crop.png``.
       c. Retourne l'image crop en base64 → Vision LLM ne voit qu'un seul tableau.
    4. Sinon (page mono-tableau ou titre introuvable) :
       Retourne la page entière en base64 (comportement legacy).

    Args:
        pdf_path:   Chemin vers le fichier PDF.
        page_num:   Numéro de page 1-based.
        table_name: Nom du tableau cible (ex: "BILAN ACTIF").
        dpi:        DPI de rendu. Si None, choix adaptatif automatique.
        enhance:    Enhancement PIL. Si None, choix adaptatif automatique.
        debug_save: Si True, sauvegarde les fichiers debug.
        debug_dir:  Dossier de destination pour les fichiers debug.

    Returns:
        (img_b64, info_dict)
        - img_b64    : image PNG base64 prête pour le Vision LLM, ou None si échec.
        - info_dict  : dict avec les métadonnées de la détection.
    """
    import base64
    import io
    import pathlib

    # DPI et enhance adaptatifs si non forcés
    if dpi is None or enhance is None:
        auto_dpi, auto_enhance = _choose_dpi(pdf_path, page_num)
        if dpi is None:
            dpi = auto_dpi
        if enhance is None:
            enhance = auto_enhance

    info: dict = {
        "page_num": page_num,
        "table_name": table_name,
        "dpi": dpi,
        "method": "full_page",
        "is_multi_table": False,
        "blocks_detected": 0,
        "target_found": False,
    }

    logger.info(
        "extract_target_table_only: début — page=%d table=%r dpi=%d enhance=%s debug=%s",
        page_num, table_name, dpi, enhance, debug_save,
    )

    # ── Étape 1 : rendu page complète HQ ─────────────────────────────────────
    full_image = _render_pdf_page_high_quality(
        pdf_path=pdf_path,
        page_num=page_num,
        dpi=dpi,
        enhance=enhance,
    )
    if full_image is None:
        logger.warning(
            "extract_target_table_only: rendu HQ échoué page=%d → fallback zoom×2",
            page_num,
        )
        info["method"] = "full_page_fallback_zoom2"
        return _page_to_image_b64(pdf_path, page_num), info

    logger.info(
        "extract_target_table_only: page rendue %dx%d",
        full_image.width, full_image.height,
    )

    # Debug : sauvegarder page complète avec nom simple page_{N}_full.png
    if debug_save:
        try:
            out_dir = pathlib.Path(debug_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            full_path = out_dir / f"page_{page_num}_full.png"
            full_image.save(str(full_path), format="PNG", optimize=False)
            logger.info("extract_target_table_only: full page sauvegardée → %s", full_path)
        except Exception as exc:
            logger.warning(
                "extract_target_table_only: sauvegarde full page échouée: %s", exc
            )

    # ── Étape 2 : détection multi-tableaux + localisation du tableau cible ────
    target_block, detection_info = locate_target_table_by_title(
        pdf_path=pdf_path,
        page_num=page_num,
        title=table_name,
        debug_dir=debug_dir,
        save_debug=debug_save,
    )

    info["is_multi_table"] = detection_info.get("is_multi_table", False)
    info["blocks_detected"] = detection_info.get("blocks_detected", 0)
    info["target_found"] = detection_info.get("target_found", False)

    # ── Étape 3 : crop si page multi-tableaux ET tableau cible localisé ───────
    if info["is_multi_table"] and target_block is not None:
        logger.info(
            "extract_target_table_only: page multi-tableaux (%d blocs) → "
            "génération candidates pour '%s'",
            info["blocks_detected"], table_name,
        )

        # Récupérer le page_rect pour les expansions de bbox
        _page_rect = None
        try:
            import fitz as _fitz_pr  # type: ignore
            _doc_pr = _fitz_pr.open(pdf_path)
            _page_rect = _doc_pr[page_num - 1].rect
            _doc_pr.close()
        except Exception as _e:
            logger.warning("extract_target_table_only: page_rect indisponible: %s", _e)

        # Récupérer tous les blocs (pour heuristiques layout)
        _all_blocks = detection_info.get("_raw_blocks", [])

        best_candidate = None
        _SCORE_THRESHOLD = 1.5  # score minimum pour accepter un candidat

        if _page_rect is not None:
            candidates = _generate_crop_candidates(
                target_block, _all_blocks, _page_rect, table_name
            )
            logger.info(
                "extract_target_table_only: %d candidates générées pour '%s'",
                len(candidates), table_name,
            )
            best_candidate, _all_scores = _pick_best_crop_candidate(
                pdf_path, page_num, candidates, table_name
            )
            info["crop_candidates_count"] = len(candidates)
            info["crop_best_label"] = best_candidate["label"]
            info["crop_best_score"] = best_candidate["score"]

        # ── Utiliser le meilleur candidat si score suffisant ──────────────────
        if best_candidate and best_candidate["score"] >= _SCORE_THRESHOLD:
            bbox = best_candidate["bbox"]
            logger.info(
                "extract_target_table_only: candidat retenu=[%s] score=%.3f "
                "bbox=(%.0f,%.0f,%.0f,%.0f)",
                best_candidate["label"], best_candidate["score"],
                bbox[0], bbox[1], bbox[2], bbox[3],
            )
            try:
                from rag_agent.multi_table_detector import TableBlock as _TB  # type: ignore
                optimal_block = _TB(
                    page_num=page_num,
                    y_start=bbox[1],
                    y_end=bbox[3],
                    x_start=bbox[0],
                    x_end=bbox[2],
                    text_sample=target_block.text_sample,
                    table_type_hint=target_block.table_type_hint,
                )
            except Exception:
                optimal_block = target_block  # fallback sécurisé

            _, crop_b64 = crop_table_under_title(
                pdf_path=pdf_path,
                page_num=page_num,
                block=optimal_block,
                dpi=dpi,
                debug_dir=debug_dir,
                save_debug=debug_save,
                title_safe=table_name,
            )
            if crop_b64:
                info["method"] = f"multi_table_crop_{best_candidate['label']}"
                logger.info(
                    "extract_target_table_only: crop candidat réussi — "
                    "table='%s' méthode='%s'",
                    table_name, info["method"],
                )
                return crop_b64, info
            logger.warning(
                "extract_target_table_only: crop candidat [%s] échoué → "
                "fallback crop bloc original",
                best_candidate["label"],
            )
        else:
            if best_candidate:
                logger.warning(
                    "extract_target_table_only: score faible (meilleur=[%s] score=%.3f) → "
                    "fallback crop bloc original",
                    best_candidate["label"], best_candidate["score"],
                )
            else:
                logger.warning(
                    "extract_target_table_only: aucun candidat disponible → "
                    "fallback crop bloc original",
                )

        # ── Fallback 1 : crop du bloc original (logique précédente) ──────────
        _, crop_b64 = crop_table_under_title(
            pdf_path=pdf_path,
            page_num=page_num,
            block=target_block,
            dpi=dpi,
            debug_dir=debug_dir,
            save_debug=debug_save,
            title_safe=table_name,
        )
        if crop_b64:
            info["method"] = "multi_table_crop_base_fallback"
            logger.info(
                "extract_target_table_only: crop bloc original réussi — "
                "table='%s' (type=%s)",
                table_name, target_block.table_type_hint,
            )
            return crop_b64, info

        # ── Fallback 2 : page entière (loggé explicitement) ──────────────────
        logger.warning(
            "extract_target_table_only: tous les crops ont échoué page=%d → "
            "FALLBACK page entière",
            page_num,
        )

    elif info["is_multi_table"]:
        logger.warning(
            "extract_target_table_only: page multi-tableaux MAIS titre '%s' non trouvé "
            "page=%d → page entière (risque de confusion LLM)",
            table_name, page_num,
        )
    else:
        logger.info(
            "extract_target_table_only: page mono-tableau page=%d → page entière",
            page_num,
        )

    # ── Fallback : retourner la page entière en base64 ────────────────────────
    try:
        buf = io.BytesIO()
        full_image.save(buf, format="PNG", optimize=False)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode()
        return img_b64, info
    except Exception as exc:
        logger.warning(
            "extract_target_table_only: conversion base64 page entière échouée: %s — "
            "fallback zoom×2",
            exc,
        )
        return _page_to_image_b64(pdf_path, page_num), info


# ═══════════════════════════════════════════════════════════════════════════════
#  Crop 2 modes — PyMuPDF+Pillow (PDF texte) / OpenCV (page scannée)
#
#  Stratégie prioritaire :
#    1. Mode texte  : PyMuPDF cherche le titre dans les blocs texte → Pillow crop.
#    2. Mode scanné : OpenCV détecte les lignes H+V du tableau → Pillow crop.
#
#  Helpers publics :
#    locate_table_title_with_pymupdf(pdf_path, page_num, table_name, ...)
#    crop_region_with_pillow(image, bbox_px, padding)
#    detect_table_region_with_opencv(image, page_num, debug_dir, save_debug)
#    crop_target_table(pdf_path, page_num, table_name, ...)
# ═══════════════════════════════════════════════════════════════════════════════


def _save_title_debug_json(
    title_info: dict,
    page_num: int,
    debug_dir: str,
) -> None:
    """Sauvegarde le JSON de localisation de titre dans page_{N}_title_found.json."""
    import pathlib

    try:
        out_dir = pathlib.Path(debug_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        filepath = out_dir / f"page_{page_num}_title_found.json"
        filepath.write_text(
            json.dumps(title_info, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info("_save_title_debug_json: sauvegardé → %s", filepath)
    except Exception as e:
        logger.warning("_save_title_debug_json: sauvegarde impossible: %s", e)


def locate_table_title_with_pymupdf(
    pdf_path: str,
    page_num: int,
    table_name: str,
    margin_below: float = 20.0,
    debug_dir: str = VISION_DEBUG_OUTPUT_DIR,
    save_debug: bool = True,
) -> Tuple[Optional[dict], dict]:
    """
    Cherche le titre du tableau cible dans les blocs texte de la page via PyMuPDF.

    Stratégie :
    1. Extrait tous les blocs texte (get_text("dict")).
    2. Normalise chaque bloc et le compare aux mots-clés du table_name.
    3. Le bloc avec le meilleur score (≥ 50 % des mots-clés) est retenu comme titre.
    4. Estime la bbox du tableau : toute la largeur de la page, depuis le bas du
       titre jusqu'au prochain marqueur de section (ou bas de page).

    Debug produit (si save_debug=True) :
        {debug_dir}/page_{N}_title_found.json

    Returns:
        (table_bbox_pdf, info)
        - table_bbox_pdf : dict {x0, y0, x1, y1} en points PDF, ou None si non trouvé.
        - info           : dict JSON-sérialisable avec toutes les métadonnées.
    """
    import unicodedata as _ud

    def _norm(s: str) -> str:
        s = _ud.normalize("NFD", s.lower())
        return "".join(c for c in s if _ud.category(c) != "Mn")

    info: dict = {
        "pdf_path": pdf_path,
        "page_num": page_num,
        "table_name": table_name,
        "mode": "pymupdf_title_search",
        "title_found": False,
        "title_text": None,
        "title_bbox": None,
        "table_bbox": None,
        "score": 0.0,
        "blocks_scanned": 0,
    }

    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.warning("locate_table_title_with_pymupdf: PyMuPDF (fitz) non disponible")
        return None, info

    keywords = [_norm(w) for w in table_name.split() if len(w) >= 2]
    if not keywords:
        logger.warning(
            "locate_table_title_with_pymupdf: table_name trop court ou vide: %r", table_name
        )
        return None, info

    try:
        doc = fitz.open(pdf_path)
        try:
            if page_num < 1 or page_num > len(doc):
                logger.warning("locate_table_title_with_pymupdf: page %d hors bornes", page_num)
                return None, info
            page = doc[page_num - 1]
            page_rect = page.rect
            page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        finally:
            doc.close()
    except Exception as e:
        logger.warning("locate_table_title_with_pymupdf: erreur PyMuPDF: %s", e)
        return None, info

    # Collecter tous les blocs texte avec leur bbox
    text_blocks: List[dict] = []
    for blk in page_dict.get("blocks", []):
        if blk.get("type") != 0:  # type 0 = texte
            continue
        spans_text = [
            span.get("text", "")
            for line in blk.get("lines", [])
            for span in line.get("spans", [])
        ]
        full_text = " ".join(spans_text).strip()
        if not full_text:
            continue
        text_blocks.append({"text": full_text, "bbox": blk.get("bbox")})

    info["blocks_scanned"] = len(text_blocks)

    if not text_blocks:
        logger.info("locate_table_title_with_pymupdf: aucun bloc texte page=%d", page_num)
        if save_debug:
            _save_title_debug_json(info, page_num, debug_dir)
        return None, info

    # Scorer chaque bloc : proportion de mots-clés présents
    best_score = 0.0
    best_idx = -1
    for i, blk in enumerate(text_blocks):
        norm_text = _norm(blk["text"])
        hits = sum(1 for kw in keywords if kw in norm_text)
        score = hits / len(keywords)
        if score > best_score:
            best_score = score
            best_idx = i

    SCORE_THRESHOLD = 0.5  # au moins 50 % des mots-clés présents
    if best_score < SCORE_THRESHOLD or best_idx == -1:
        logger.info(
            "locate_table_title_with_pymupdf: titre %r non trouvé page=%d "
            "(best_score=%.2f < %.2f)",
            table_name, page_num, best_score, SCORE_THRESHOLD,
        )
        info["score"] = round(best_score, 3)
        if save_debug:
            _save_title_debug_json(info, page_num, debug_dir)
        return None, info

    title_blk = text_blocks[best_idx]
    tx0, ty0, tx1, ty1 = title_blk["bbox"]

    # Estimer y_end du tableau : chercher le prochain marqueur de section
    END_MARKERS = [
        "état des", "derogations", "note ", "informations complémentaires",
        "attestation", "rapport", "tableau des flux", "flux de trésorerie",
    ]
    y_end = float(page_rect.y1)
    for blk in sorted(text_blocks, key=lambda b: b["bbox"][1]):
        if blk["bbox"][1] <= ty1 + margin_below:
            continue
        candidate_norm = _norm(blk["text"])
        if any(m in candidate_norm for m in END_MARKERS):
            y_end = float(blk["bbox"][1])
            break

    table_bbox = {
        "x0": float(page_rect.x0),
        "y0": float(ty1),
        "x1": float(page_rect.x1),
        "y1": y_end,
    }
    title_bbox_dict = {
        "x0": float(tx0), "y0": float(ty0),
        "x1": float(tx1), "y1": float(ty1),
    }

    info.update({
        "title_found": True,
        "title_text": title_blk["text"][:120],
        "title_bbox": title_bbox_dict,
        "table_bbox": table_bbox,
        "score": round(best_score, 3),
    })

    logger.info(
        "locate_table_title_with_pymupdf: titre trouvé page=%d score=%.2f "
        "title_y=[%.1f, %.1f] table_y=[%.1f, %.1f]",
        page_num, best_score, ty0, ty1, table_bbox["y0"], table_bbox["y1"],
    )

    if save_debug:
        _save_title_debug_json(info, page_num, debug_dir)

    return table_bbox, info


def crop_region_with_pillow(
    image: "object",  # PIL.Image.Image
    bbox_px: Tuple[int, int, int, int],
    padding: int = 10,
) -> "Optional[object]":  # PIL.Image.Image
    """
    Crop une zone d'une image PIL avec padding et clamping aux limites de l'image.

    Args:
        image:   Image PIL source (pleine page).
        bbox_px: (x0, y0, x1, y1) en pixels dans l'image source.
        padding: Marge en pixels ajoutée de chaque côté.

    Returns:
        Image PIL croppée, ou None si la bbox est invalide.
    """
    try:
        img_w, img_h = image.size
    except Exception:
        logger.warning("crop_region_with_pillow: objet image invalide")
        return None

    x0, y0, x1, y1 = bbox_px
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(img_w, x1 + padding)
    y1 = min(img_h, y1 + padding)

    if x1 <= x0 or y1 <= y0:
        logger.warning(
            "crop_region_with_pillow: bbox invalide après clamping (%d,%d,%d,%d)",
            x0, y0, x1, y1,
        )
        return None

    cropped = image.crop((x0, y0, x1, y1))
    logger.info(
        "crop_region_with_pillow: crop OK (%d,%d,%d,%d) → %dx%d px",
        x0, y0, x1, y1, cropped.width, cropped.height,
    )
    return cropped


def detect_table_region_with_opencv(
    image: "object",  # PIL.Image.Image
    page_num: int = 0,
    debug_dir: str = VISION_DEBUG_OUTPUT_DIR,
    save_debug: bool = True,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Détecte la région principale du tableau dans une image via OpenCV.

    Utilisé en fallback pour les pages scannées (sans texte natif).

    Stratégie :
    1. Conversion PIL → numpy BGR.
    2. Niveaux de gris + seuillage adaptatif.
    3. Morphologie pour isoler les lignes horizontales et verticales.
    4. Fusion H+V → masque des cellules de tableau.
    5. Contours → plus grande bbox dépassant 5 % de la surface page.

    Debug produit (si save_debug=True) :
        {debug_dir}/page_{N}_opencv_detect.png — image avec rectangle rouge détecté.

    Returns:
        (x0, y0, x1, y1) en pixels, ou None si aucune région trouvée.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning(
            "detect_table_region_with_opencv: OpenCV (cv2) ou numpy non disponible"
        )
        return None

    try:
        img_np = np.array(image.convert("RGB"))
        img_bgr = img_np[:, :, ::-1].copy()
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15, C=4,
        )

        img_h, img_w = gray.shape

        # Noyaux morphologiques calibrés sur la taille de l'image
        h_len = max(30, img_w // 20)
        v_len = max(30, img_h // 20)
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))

        h_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=2)
        v_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel, iterations=2)

        table_mask = cv2.add(h_lines, v_lines)
        dilate_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        table_mask = cv2.dilate(table_mask, dilate_k, iterations=3)

        contours, _ = cv2.findContours(
            table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            logger.info(
                "detect_table_region_with_opencv: aucun contour trouvé page=%d", page_num
            )
            return None

        # Plus grande bbox dépassant 5 % de la surface totale
        best_bbox: Optional[Tuple[int, int, int, int]] = None
        best_area = 0
        min_area = 0.05 * img_w * img_h
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h
            if area < min_area:
                continue
            if area > best_area:
                best_area = area
                best_bbox = (x, y, x + w, y + h)

        if best_bbox is None:
            logger.info(
                "detect_table_region_with_opencv: aucune bbox significative page=%d", page_num
            )
            return None

        x0, y0, x1, y1 = best_bbox
        logger.info(
            "detect_table_region_with_opencv: bbox détectée page=%d (%d,%d,%d,%d) %dx%d px",
            page_num, x0, y0, x1, y1, x1 - x0, y1 - y0,
        )

        # Debug : image avec rectangle rouge
        if save_debug:
            import pathlib
            try:
                debug_img = img_bgr.copy()
                cv2.rectangle(debug_img, (x0, y0), (x1, y1), (0, 0, 255), 3)
                out_dir = pathlib.Path(debug_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"page_{page_num}_opencv_detect.png"
                cv2.imwrite(str(out_path), debug_img)
                logger.info(
                    "detect_table_region_with_opencv: debug sauvegardé → %s", out_path
                )
            except Exception as exc:
                logger.warning(
                    "detect_table_region_with_opencv: sauvegarde debug échouée: %s", exc
                )

        return (x0, y0, x1, y1)

    except Exception as e:
        logger.warning("detect_table_region_with_opencv: erreur: %s", e)
        return None


def crop_target_table(
    pdf_path: str,
    page_num: int,
    table_name: str,
    dpi: Optional[int] = None,
    padding: int = 15,
    debug_dir: str = VISION_DEBUG_OUTPUT_DIR,
    save_debug: bool = True,
) -> Tuple[Optional[object], Optional[str], dict]:
    """
    Crop le tableau cible selon une stratégie en 2 modes prioritaires.

    Mode 1 — PDF texte (PyMuPDF + Pillow) :
        1. Cherche le titre du tableau dans les blocs texte via PyMuPDF.
        2. Convertit la bbox PDF (points) en pixels selon le DPI de rendu.
        3. Crop la zone avec Pillow.

    Mode 2 — Page scannée / fallback (OpenCV + Pillow) :
        Déclenché si la page est scannée (texte < VISION_SCANNED_TEXT_THRESHOLD)
        OU si le titre n'est pas trouvé en Mode 1.
        1. Détecte la région du tableau via les lignes H+V (OpenCV).
        2. Crop la bbox avec Pillow.

    Fichiers debug produits (si save_debug=True) :
        page_{N}_full.png              — page complète rendue
        page_{N}_title_found.json      — résultat localisation titre (Mode 1)
        page_{N}_opencv_detect.png     — détection OpenCV avec rectangle (Mode 2)
        page_{N}_{table_safe}_crop.png — crop final envoyé au Vision LLM

    Args:
        pdf_path:   Chemin vers le fichier PDF.
        page_num:   Numéro de page 1-based.
        table_name: Nom du tableau cible (ex: "BILAN ACTIF").
        dpi:        DPI de rendu. Si None, choix adaptatif automatique.
        padding:    Pixels de marge autour du crop final.
        debug_dir:  Dossier de sortie pour les fichiers debug.
        save_debug: Si True, sauvegarde tous les fichiers debug.

    Returns:
        (pil_image, b64_string, info_dict)
        - pil_image  : image PIL croppée (ou page entière si tout échoue).
        - b64_string : image PNG en base64 prête pour un Vision LLM.
        - info_dict  : dict avec mode, méthode, crop_applied, et métadonnées debug.
    """
    import base64
    import io
    import pathlib

    info: dict = {
        "page_num": page_num,
        "table_name": table_name,
        "mode": None,
        "is_scanned": False,
        "title_found": False,
        "opencv_used": False,
        "crop_applied": False,
        "dpi": dpi,
        "method": "full_page",
    }

    # ── Étape 0 : DPI adaptatif + détection page scannée ──────────────────────
    auto_dpi, auto_enhance = _choose_dpi(pdf_path, page_num)
    if dpi is None:
        dpi = auto_dpi
    info["dpi"] = dpi
    info["is_scanned"] = (dpi >= VISION_DPI_SCANNED)

    # ── Étape 1 : rendu page complète HQ ──────────────────────────────────────
    full_image = _render_pdf_page_high_quality(
        pdf_path, page_num, dpi=dpi, enhance=auto_enhance
    )
    if full_image is None:
        logger.warning("crop_target_table: rendu HQ échoué page=%d", page_num)
        return None, _page_to_image_b64(pdf_path, page_num), info

    if save_debug:
        try:
            out_dir = pathlib.Path(debug_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            full_path = out_dir / f"page_{page_num}_full.png"
            full_image.save(str(full_path), format="PNG", optimize=False)
            logger.info("crop_target_table: full page sauvegardée → %s", full_path)
        except Exception as exc:
            logger.warning("crop_target_table: sauvegarde full page échouée: %s", exc)

    cropped_image = None

    # ── Mode 1 : PDF texte — PyMuPDF title search + Pillow crop ───────────────
    if not info["is_scanned"]:
        info["mode"] = "text_pdf"
        logger.info(
            "crop_target_table: Mode 1 (texte PDF) — PyMuPDF title search page=%d", page_num
        )

        table_bbox_pdf, title_info = locate_table_title_with_pymupdf(
            pdf_path=pdf_path,
            page_num=page_num,
            table_name=table_name,
            debug_dir=debug_dir,
            save_debug=save_debug,
        )
        info["title_found"] = title_info.get("title_found", False)

        if table_bbox_pdf is not None:
            # Conversion PDF points → pixels : scale = dpi / 72
            scale = dpi / 72.0
            bbox_px = (
                int(table_bbox_pdf["x0"] * scale),
                int(table_bbox_pdf["y0"] * scale),
                int(table_bbox_pdf["x1"] * scale),
                int(table_bbox_pdf["y1"] * scale),
            )
            cropped_image = crop_region_with_pillow(full_image, bbox_px, padding=padding)
            if cropped_image is not None:
                info["method"] = "pymupdf_title_crop"
                info["crop_applied"] = True
                logger.info(
                    "crop_target_table: Mode 1 OK — titre trouvé et croppé page=%d", page_num
                )

    # ── Mode 2 : scanned ou titre non trouvé — OpenCV detect + Pillow crop ────
    if cropped_image is None:
        if info["is_scanned"]:
            info["mode"] = "scanned_opencv"
            logger.info(
                "crop_target_table: Mode 2 (scanné) — OpenCV detect page=%d", page_num
            )
        else:
            info["mode"] = "text_pdf_opencv_fallback"
            logger.info(
                "crop_target_table: Mode 2 fallback (titre non trouvé) — "
                "OpenCV detect page=%d",
                page_num,
            )

        info["opencv_used"] = True
        bbox_px = detect_table_region_with_opencv(
            full_image,
            page_num=page_num,
            debug_dir=debug_dir,
            save_debug=save_debug,
        )
        if bbox_px is not None:
            cropped_image = crop_region_with_pillow(full_image, bbox_px, padding=padding)
            if cropped_image is not None:
                info["method"] = "opencv_crop"
                info["crop_applied"] = True
                logger.info(
                    "crop_target_table: Mode 2 OK — bbox OpenCV croppée page=%d bbox=%s",
                    page_num, bbox_px,
                )
        else:
            logger.warning(
                "crop_target_table: Mode 2 échoué page=%d — fallback page entière", page_num
            )

    # ── Fallback : page entière ────────────────────────────────────────────────
    final_image = cropped_image if cropped_image is not None else full_image
    if not info["crop_applied"]:
        info["method"] = "full_page"
        logger.info("crop_target_table: fallback page entière page=%d", page_num)

    # Debug : crop final
    if save_debug and info["crop_applied"]:
        try:
            table_safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", table_name.lower())[:30]
            out_dir = pathlib.Path(debug_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            crop_path = out_dir / f"page_{page_num}_{table_safe}_crop.png"
            final_image.save(str(crop_path), format="PNG", optimize=False)
            logger.info("crop_target_table: crop final sauvegardé → %s", crop_path)
        except Exception as exc:
            logger.warning("crop_target_table: sauvegarde crop final échouée: %s", exc)

    # Conversion base64
    try:
        buf = io.BytesIO()
        final_image.save(buf, format="PNG", optimize=False)
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
    except Exception as exc:
        logger.warning("crop_target_table: conversion base64 échouée: %s", exc)
        b64 = _page_to_image_b64(pdf_path, page_num)

    return final_image, b64, info


# ─────────────────────────────────────────────────────────────────────────────
#  Pipeline coarse-to-fine : localisation visuelle → crop → extraction
# ─────────────────────────────────────────────────────────────────────────────


def _build_localization_prompt(table_name: str) -> str:
    """Prompt passe 1 : localiser le tableau sans extraire les données."""
    return f"""You are analyzing a financial report page image.
Your task: locate the table titled "{table_name}" in this image.

Return ONLY a JSON object with these exact fields:
- "table_found": true or false
- "bbox_norm": [x0, y0, x1, y1] as normalized coordinates between 0.0 and 1.0
  (origin = top-left corner, x = horizontal axis, y = vertical axis).
  The bbox must cover the ENTIRE table including its title row and last data row.
- "location_hint": one of "upper_left", "upper_right", "lower_left", "lower_right",
  "upper_half", "lower_half", "left_column", "right_column", "full_page"

If the table is not visible: {{"table_found": false, "bbox_norm": null, "location_hint": null}}

Constraints:
- All bbox_norm values must be between 0.0 and 1.0
- x1 > x0 and y1 > y0
- Add a small margin (≈1-2%) around the table
- Return ONLY the JSON object, no prose before or after

Example: {{"table_found": true, "bbox_norm": [0.02, 0.08, 0.98, 0.55], "location_hint": "upper_half"}}"""


def _vision_locate_table_on_full_page(
    pdf_path: str,
    page_num: int,
    table_name: str,
    api_provider: str,
    api_key: Optional[str] = None,
    dpi: Optional[int] = None,
    enhance: Optional[bool] = None,
) -> dict:
    """
    Passe 1 coarse-to-fine : localisation visuelle du tableau cible sur la page complète.

    Envoie la page entière au Vision LLM avec un prompt de localisation (pas d'extraction).
    Retourne une bounding box normalisée (0–1) couvrant le tableau demandé.

    Args:
        pdf_path:     Chemin vers le fichier PDF.
        page_num:     Numéro de page 1-based.
        table_name:   Nom du tableau cible.
        api_provider: Provider Vision LLM ("groq" | "gemini" | "gpt-5.4").
        api_key:      Clé API (auto-détectée si None).
        dpi:          DPI de rendu (auto si None).
        enhance:      Enhancement PIL (auto si None).

    Returns:
        {
            "table_found": bool,
            "bbox_norm": [x0, y0, x1, y1] | None,  # coordonnées normalisées 0–1
            "location_hint": str | None,
        }
    """
    _FAILED: dict = {"table_found": False, "bbox_norm": None, "location_hint": None}

    provider = _normalize_api_provider(api_provider)
    resolved_key = _provider_api_key(provider, api_key=api_key)
    if not resolved_key:
        logger.warning(
            "_vision_locate_table_on_full_page: API key manquante provider=%s", provider
        )
        return _FAILED

    if dpi is None or enhance is None:
        auto_dpi, auto_enhance = _choose_dpi(pdf_path, page_num)
        if dpi is None:
            dpi = auto_dpi
        if enhance is None:
            enhance = auto_enhance

    full_image = _render_pdf_page_high_quality(
        pdf_path=pdf_path,
        page_num=page_num,
        dpi=dpi,
        enhance=enhance,
    )
    if full_image is None:
        logger.warning(
            "_vision_locate_table_on_full_page: rendu page échoué page=%d", page_num
        )
        return _FAILED

    img_b64 = _img_to_b64(full_image)
    if not img_b64:
        return _FAILED

    prompt = _build_localization_prompt(table_name)
    logger.info(
        "_vision_locate_table_on_full_page: appel LLM provider=%s page=%d table=%r",
        provider, page_num, table_name,
    )

    try:
        raw = _call_vision_llm(
            img_b64=img_b64,
            prompt=prompt,
            provider=provider,
            api_key=resolved_key,
        )
    except Exception as exc:
        logger.warning("_vision_locate_table_on_full_page: LLM échoué: %s", exc)
        return _FAILED

    if not raw:
        logger.warning(
            "_vision_locate_table_on_full_page: réponse vide provider=%s page=%d", provider, page_num
        )
        return _FAILED

    try:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            logger.warning(
                "_vision_locate_table_on_full_page: aucun JSON dans la réponse: %r", raw[:300]
            )
            return _FAILED

        payload = json.loads(json_match.group())
        table_found = bool(payload.get("table_found", False))
        bbox_norm = payload.get("bbox_norm")
        location_hint = payload.get("location_hint")

        if not table_found or bbox_norm is None:
            logger.info(
                "_vision_locate_table_on_full_page: tableau '%s' non localisé par le LLM",
                table_name,
            )
            return {"table_found": False, "bbox_norm": None, "location_hint": location_hint}

        if (
            not isinstance(bbox_norm, list)
            or len(bbox_norm) != 4
            or not all(isinstance(v, (int, float)) for v in bbox_norm)
        ):
            logger.warning(
                "_vision_locate_table_on_full_page: bbox_norm invalide: %r", bbox_norm
            )
            return _FAILED

        x0, y0, x1, y1 = [float(v) for v in bbox_norm]
        x0 = max(0.0, min(1.0, x0))
        y0 = max(0.0, min(1.0, y0))
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))

        if x1 <= x0 or y1 <= y0:
            logger.warning(
                "_vision_locate_table_on_full_page: bbox incohérente [%.3f,%.3f,%.3f,%.3f]",
                x0, y0, x1, y1,
            )
            return _FAILED

        if (x1 - x0) < 0.05 or (y1 - y0) < 0.05:
            logger.warning(
                "_vision_locate_table_on_full_page: bbox trop petite [%.3f,%.3f,%.3f,%.3f]",
                x0, y0, x1, y1,
            )
            return _FAILED

        logger.info(
            "_vision_locate_table_on_full_page: bbox=[%.3f,%.3f,%.3f,%.3f] hint=%r",
            x0, y0, x1, y1, location_hint,
        )
        return {"table_found": True, "bbox_norm": [x0, y0, x1, y1], "location_hint": location_hint}

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "_vision_locate_table_on_full_page: parse JSON échoué: %s — réponse: %r",
            exc, raw[:400],
        )
        return _FAILED


def _vision_locate_table_only(
    pdf_path: str,
    page_num: int,
    table_name: str,
    api_provider: str,
    api_key: Optional[str] = None,
    dpi: Optional[int] = None,
    enhance: Optional[bool] = None,
) -> dict:
    """
    Localisation visuelle du tableau sur la page complète — sans extraction, sans crop.

    Envoie la page entière au Vision LLM avec le prompt de localisation uniquement.
    Retourne les coordonnées du tableau en normalisé (0–1) ET en pixels absolus.

    Args:
        pdf_path:     Chemin vers le fichier PDF.
        page_num:     Numéro de page 1-based.
        table_name:   Nom du tableau à localiser.
        api_provider: Provider Vision LLM ("groq" | "gemini" | "gpt-5.4").
        api_key:      Clé API (auto-détectée si None).
        dpi:          DPI de rendu (auto si None).
        enhance:      Enhancement PIL (auto si None).

    Returns:
        {
            "table_found": bool,
            "table_name":  str,
            "bbox_norm":   [x0, y0, x1, y1] normalisé 0–1, ou None,
            "bbox_px":     [x0_px, y0_px, x1_px, y1_px] en pixels, ou None,
        }
    """
    _FAILED = {
        "table_found": False,
        "table_name": table_name,
        "bbox_norm": None,
        "bbox_px": None,
    }

    provider = _normalize_api_provider(api_provider)
    resolved_key = _provider_api_key(provider, api_key=api_key)
    if not resolved_key:
        logger.warning(
            "_vision_locate_table_only: API key manquante provider=%s — table=%r",
            provider, table_name,
        )
        return _FAILED

    if dpi is None or enhance is None:
        auto_dpi, auto_enhance = _choose_dpi(pdf_path, page_num)
        if dpi is None:
            dpi = auto_dpi
        if enhance is None:
            enhance = auto_enhance

    # Rendu page complète — une seule fois, pas de crop
    full_image = _render_pdf_page_high_quality(
        pdf_path=pdf_path,
        page_num=page_num,
        dpi=dpi,
        enhance=enhance,
    )
    if full_image is None:
        logger.warning(
            "_vision_locate_table_only: rendu page échoué — page=%d table=%r",
            page_num, table_name,
        )
        return _FAILED

    img_w, img_h = full_image.size
    logger.info(
        "_vision_locate_table_only: page=%d table=%r dpi=%d size=%dx%d — envoi au LLM",
        page_num, table_name, dpi, img_w, img_h,
    )

    img_b64 = _img_to_b64(full_image)
    if not img_b64:
        logger.warning(
            "_vision_locate_table_only: conversion base64 échouée — page=%d table=%r",
            page_num, table_name,
        )
        return _FAILED

    prompt = _build_localization_prompt(table_name)

    try:
        raw = _call_vision_llm(
            img_b64=img_b64,
            prompt=prompt,
            provider=provider,
            api_key=resolved_key,
        )
    except Exception as exc:
        logger.warning(
            "_vision_locate_table_only: appel LLM échoué — page=%d table=%r : %s",
            page_num, table_name, exc,
        )
        return _FAILED

    if not raw:
        logger.warning(
            "_vision_locate_table_only: réponse vide — page=%d provider=%s table=%r",
            page_num, provider, table_name,
        )
        return _FAILED

    # Parse JSON de localisation
    try:
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            logger.warning(
                "_vision_locate_table_only: aucun JSON dans la réponse — table=%r réponse=%r",
                table_name, raw[:300],
            )
            return _FAILED

        payload = json.loads(json_match.group())
        table_found = bool(payload.get("table_found", False))
        bbox_norm = payload.get("bbox_norm")

        if not table_found or bbox_norm is None:
            logger.info(
                "_vision_locate_table_only: tableau non localisé — page=%d table=%r",
                page_num, table_name,
            )
            return _FAILED

        if (
            not isinstance(bbox_norm, list)
            or len(bbox_norm) != 4
            or not all(isinstance(v, (int, float)) for v in bbox_norm)
        ):
            logger.warning(
                "_vision_locate_table_only: bbox_norm invalide=%r — page=%d table=%r",
                bbox_norm, page_num, table_name,
            )
            return _FAILED

        x0, y0, x1, y1 = [float(v) for v in bbox_norm]
        x0 = max(0.0, min(1.0, x0))
        y0 = max(0.0, min(1.0, y0))
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))

        if x1 <= x0 or y1 <= y0:
            logger.warning(
                "_vision_locate_table_only: bbox incohérente [%.3f,%.3f,%.3f,%.3f] — table=%r",
                x0, y0, x1, y1, table_name,
            )
            return _FAILED

        if (x1 - x0) < 0.05 or (y1 - y0) < 0.05:
            logger.warning(
                "_vision_locate_table_only: bbox trop petite [%.3f,%.3f,%.3f,%.3f] — table=%r",
                x0, y0, x1, y1, table_name,
            )
            return _FAILED

        # Conversion normalisé → pixels
        x0_px = int(x0 * img_w)
        y0_px = int(y0 * img_h)
        x1_px = int(x1 * img_w)
        y1_px = int(y1 * img_h)

        result = {
            "table_found": True,
            "table_name": table_name,
            "bbox_norm": [x0, y0, x1, y1],
            "bbox_px": [x0_px, y0_px, x1_px, y1_px],
        }
        logger.info(
            "_vision_locate_table_only: OK — table=%r bbox_norm=[%.3f,%.3f,%.3f,%.3f] "
            "bbox_px=[%d,%d,%d,%d]",
            table_name, x0, y0, x1, y1, x0_px, y0_px, x1_px, y1_px,
        )

        # Sauvegarde debug : page complète avec bbox rouge dessinée dessus
        try:
            import pathlib
            from PIL import ImageDraw  # type: ignore

            viz = full_image.copy()
            draw = ImageDraw.Draw(viz)
            draw.rectangle([x0_px, y0_px, x1_px, y1_px], outline="red", width=4)

            out_dir = pathlib.Path(VISION_DEBUG_OUTPUT_DIR)
            out_dir.mkdir(parents=True, exist_ok=True)
            pdf_stem = pathlib.Path(pdf_path).stem
            table_safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", table_name.upper())[:40]
            viz_path = out_dir / f"{pdf_stem}_{table_safe}_page{page_num}_localization.png"
            viz.save(str(viz_path), format="PNG")
            logger.info("_vision_locate_table_only: debug image → %s", viz_path)
        except Exception as exc_viz:
            logger.warning("_vision_locate_table_only: sauvegarde debug échouée: %s", exc_viz)

        return result

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "_vision_locate_table_only: parse JSON échoué — table=%r : %s — réponse=%r",
            table_name, exc, raw[:400],
        )
        return _FAILED


def _img_to_b64(image: "PIL.Image.Image") -> Optional[str]:  # noqa: F821
    """Convertit une image PIL en chaîne base64 PNG."""
    import base64
    import io

    try:
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=False)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()
    except Exception as exc:
        logger.warning("_img_to_b64: conversion échouée: %s", exc)
        return None


def _crop_from_bbox_norm(
    image: "PIL.Image.Image",  # noqa: F821
    bbox_norm: List[float],
    padding_ratio: float = 0.005,
) -> "Optional[PIL.Image.Image]":  # noqa: F821
    """
    Crop une image PIL à partir d'une bounding box normalisée [x0, y0, x1, y1] (0–1).

    Ajoute un léger padding relatif à la taille de l'image pour ne pas couper les bords.

    Args:
        image:         Image PIL source.
        bbox_norm:     [x0, y0, x1, y1] en coordonnées normalisées (0–1).
        padding_ratio: Fraction de w/h ajoutée comme padding de chaque côté.

    Returns:
        Image PIL croppée, ou None si le crop est invalide.
    """
    try:
        img_w, img_h = image.size
        x0_n, y0_n, x1_n, y1_n = bbox_norm

        pad_x = int(img_w * padding_ratio)
        pad_y = int(img_h * padding_ratio)

        px0 = max(0, int(x0_n * img_w) - pad_x)
        py0 = max(0, int(y0_n * img_h) - pad_y)
        px1 = min(img_w, int(x1_n * img_w) + pad_x)
        py1 = min(img_h, int(y1_n * img_h) + pad_y)

        crop_w = px1 - px0
        crop_h = py1 - py0

        if crop_w <= 0 or crop_h <= 0:
            logger.warning("_crop_from_bbox_norm: crop nul px=(%d,%d,%d,%d)", px0, py0, px1, py1)
            return None

        cropped = image.crop((px0, py0, px1, py1))
        logger.info(
            "_crop_from_bbox_norm: pixels=(%d,%d,%d,%d) size=%dx%d (page=%dx%d)",
            px0, py0, px1, py1, crop_w, crop_h, img_w, img_h,
        )
        return cropped

    except Exception as exc:
        logger.warning("_crop_from_bbox_norm: erreur: %s", exc)
        return None


def _should_split_vertically(
    crop_image: "PIL.Image.Image",  # noqa: F821
    full_image: "PIL.Image.Image",  # noqa: F821
    pixel_threshold: int = VISION_SPLIT_PIXEL_THRESHOLD,
    height_ratio_threshold: float = VISION_SPLIT_HEIGHT_RATIO,
) -> bool:
    """
    Détermine si un crop est trop grand/dense pour être envoyé en une seule fois.

    Heuristiques (OU logique) :
    - Surface totale (pixels) du crop > pixel_threshold
    - Rapport hauteur_crop / hauteur_page > height_ratio_threshold

    Returns:
        True si le crop doit être découpé verticalement.
    """
    crop_w, crop_h = crop_image.size
    _, full_h = full_image.size

    surface = crop_w * crop_h
    height_ratio = crop_h / full_h if full_h > 0 else 1.0

    decision = surface > pixel_threshold or height_ratio > height_ratio_threshold
    logger.info(
        "_should_split_vertically: surface=%d (seuil=%d) height_ratio=%.2f (seuil=%.2f) → split=%s",
        surface, pixel_threshold, height_ratio, height_ratio_threshold, decision,
    )
    return decision


def _split_image_vertically(
    image: "PIL.Image.Image",  # noqa: F821
    n_parts: int = VISION_SPLIT_N_SUBCROP,
    overlap_px: int = VISION_SPLIT_OVERLAP_PX,
) -> "List[PIL.Image.Image]":  # noqa: F821
    """
    Découpe une image en n_parts sous-crops verticaux avec chevauchement.

    Le chevauchement (overlap_px) assure qu'une ligne coupée en deux sur une frontière
    apparaît dans les deux sous-crops adjacents → le LLM peut la lire dans l'un ou l'autre.

    Args:
        image:      Image PIL à découper.
        n_parts:    Nombre de tranches (1 = aucun découpage).
        overlap_px: Chevauchement en pixels entre tranches adjacentes.

    Returns:
        Liste ordonnée haut → bas de n_parts images PIL.
    """
    if n_parts <= 1:
        return [image]

    img_w, img_h = image.size
    slice_h = img_h // n_parts

    if slice_h <= 0:
        logger.warning("_split_image_vertically: slice_h=%d invalide → retour image entière", slice_h)
        return [image]

    parts = []
    for i in range(n_parts):
        y0 = max(0, i * slice_h - (overlap_px if i > 0 else 0))
        y1 = min(img_h, (i + 1) * slice_h + (overlap_px if i < n_parts - 1 else 0))
        part = image.crop((0, y0, img_w, y1))
        logger.info(
            "_split_image_vertically: sous-crop %d/%d y=(%d,%d) size=%dx%d",
            i + 1, n_parts, y0, y1, img_w, y1 - y0,
        )
        parts.append(part)

    return parts


def _merge_vision_dataframes(
    dfs: "List[pd.DataFrame]",
    table_name: str = "",
) -> pd.DataFrame:
    """
    Fusionne plusieurs DataFrames extraits de sous-crops verticaux.

    Stratégie :
    - Headers pris du premier DataFrame non vide (référence de schéma).
    - Lignes concaténées dans l'ordre (haut → bas, ordre naturel du tableau).
    - DataFrames suivants alignés sur les colonnes de référence (manquantes → vides).
    - Doublons exacts supprimés (peuvent apparaître dans la zone overlap_px).

    Args:
        dfs:        Liste de DataFrames dans l'ordre haut → bas.
        table_name: Nom du tableau (pour les logs).

    Returns:
        DataFrame fusionné, ou DataFrame vide si tous les inputs sont vides.
    """
    valid_dfs = [df for df in dfs if df is not None and not df.empty]
    if not valid_dfs:
        logger.warning("_merge_vision_dataframes [%s]: tous les DataFrames sont vides", table_name)
        return pd.DataFrame()

    if len(valid_dfs) == 1:
        return valid_dfs[0].reset_index(drop=True)

    ref_cols = list(valid_dfs[0].columns)
    aligned = []
    for i, df in enumerate(valid_dfs):
        aligned.append(df.reindex(columns=ref_cols, fill_value=""))
        logger.info(
            "_merge_vision_dataframes [%s]: sous-crop %d/%d shape=%s",
            table_name, i + 1, len(valid_dfs), df.shape,
        )

    merged = pd.concat(aligned, ignore_index=True)
    before = len(merged)
    merged = merged.drop_duplicates().reset_index(drop=True)
    after = len(merged)
    if before != after:
        logger.info(
            "_merge_vision_dataframes [%s]: %d doublons supprimés (%d → %d lignes)",
            table_name, before - after, before, after,
        )

    logger.info(
        "_merge_vision_dataframes [%s]: fusion finale shape=%s",
        table_name, merged.shape,
    )
    return merged


def _extract_from_b64(
    img_b64: str,
    table_name: str,
    provider: str,
    api_key: str,
    type_comptes: Optional[str] = None,
    secteur: Optional[str] = None,
    pdf_path: str = "",
    page_num: int = 0,
) -> pd.DataFrame:
    """Extraction directe depuis un img_b64 préparé — helper interne pour _coarse_to_fine_vision_extract."""
    prompt = _build_vision_prompt(
        table_name=table_name,
        type_comptes=type_comptes,
        secteur=secteur,
    )
    try:
        raw = _call_vision_llm(img_b64=img_b64, prompt=prompt, provider=provider, api_key=api_key)
        if not raw:
            return pd.DataFrame()
        return _vision_parse_response(
            content=raw,
            table_name=table_name,
            pdf_path=pdf_path,
            page_num=page_num,
        )
    except Exception as exc:
        logger.warning("_extract_from_b64: failed: %s", exc)
        return pd.DataFrame()


def _coarse_to_fine_vision_extract(
    pdf_path: str,
    page_num: int,
    table_name: str,
    api_provider: str = "groq",
    api_key: Optional[str] = None,
    type_comptes: Optional[str] = None,
    secteur: Optional[str] = None,
    debug_save: bool = VISION_DEBUG_SAVE_INPUTS,
) -> pd.DataFrame:
    """
    Pipeline coarse-to-fine d'extraction de tableau financier.

    Étapes :
    1. Rendu page complète HQ.
    2. Passe 1 — localisation : page entière → Vision LLM → bbox_norm du tableau
       (sans extraction de données).
    3. Crop du tableau à partir de la bbox_norm (coordonnées normalisées → pixels).
    4. Si le crop est trop grand/dense (surface ou hauteur relative) :
       découpage en VISION_SPLIT_N_SUBCROP sous-crops verticaux avec chevauchement.
    5. Passe 2 — extraction : chaque (sous-)crop → Vision LLM → DataFrame.
    6. Fusion ordonnée des DataFrames (haut → bas) + déduplication doublons overlap.
    7. Fallback vers page entière si la localisation échoue.

    Args:
        pdf_path:     Chemin vers le fichier PDF.
        page_num:     Numéro de page 1-based.
        table_name:   Nom du tableau à extraire.
        api_provider: Provider LLM ("groq" | "gemini" | "gpt-5.4").
        api_key:      Clé API (auto-détectée si None).
        type_comptes: "sociaux" | "consolides" | None.
        secteur:      "banque" | "assurance" | "autre" | None.
        debug_save:   Sauvegarder les images crop dans le dossier debug.

    Returns:
        DataFrame extrait, ou DataFrame vide si échec total.
    """
    provider = _normalize_api_provider(api_provider)
    resolved_key = _provider_api_key(provider, api_key=api_key)
    if not resolved_key:
        logger.warning(
            "_coarse_to_fine_vision_extract: API key manquante provider=%s", provider
        )
        return pd.DataFrame()

    dpi, enhance = _choose_dpi(pdf_path, page_num)

    # ── Étape 1 : rendu page complète HQ ─────────────────────────────────────
    full_image = _render_pdf_page_high_quality(
        pdf_path=pdf_path,
        page_num=page_num,
        dpi=dpi,
        enhance=enhance,
    )
    if full_image is None:
        logger.warning(
            "_coarse_to_fine_vision_extract: rendu page échoué → fallback zoom×2",
            page_num,
        )
        img_b64_full = _page_to_image_b64(pdf_path, page_num)
        if not img_b64_full:
            return pd.DataFrame()
        return _extract_from_b64(
            img_b64=img_b64_full,
            table_name=table_name,
            provider=provider,
            api_key=resolved_key,
            type_comptes=type_comptes,
            secteur=secteur,
            pdf_path=pdf_path,
            page_num=page_num,
        )

    logger.info(
        "_coarse_to_fine_vision_extract: page rendue %dx%d page=%d table=%r",
        full_image.width, full_image.height, page_num, table_name,
    )
    if debug_save:
        _debug_save_image(full_image, pdf_path, page_num, table_name, suffix="c2f_full")

    # ── Étape 2 : passe 1 — localisation visuelle ────────────────────────────
    loc = _vision_locate_table_on_full_page(
        pdf_path=pdf_path,
        page_num=page_num,
        table_name=table_name,
        api_provider=provider,
        api_key=resolved_key,
        dpi=dpi,
        enhance=enhance,
    )

    # ── Étape 3 : crop ou fallback page entière ───────────────────────────────
    if loc["table_found"] and loc["bbox_norm"] is not None:
        crop_image = _crop_from_bbox_norm(full_image, loc["bbox_norm"])
        if crop_image is None:
            logger.warning(
                "_coarse_to_fine_vision_extract: crop_from_bbox_norm échoué → page entière"
            )
            crop_image = full_image
            use_full_fallback = True
        else:
            use_full_fallback = False
            logger.info(
                "_coarse_to_fine_vision_extract: crop OK size=%dx%d hint=%r",
                crop_image.width, crop_image.height, loc["location_hint"],
            )
            if debug_save:
                _debug_save_image(crop_image, pdf_path, page_num, table_name, suffix="c2f_crop")
    else:
        logger.info(
            "_coarse_to_fine_vision_extract: localisation échouée → page entière utilisée"
        )
        crop_image = full_image
        use_full_fallback = True

    # ── Étape 4 : découpage vertical si crop trop dense ───────────────────────
    if not use_full_fallback and _should_split_vertically(crop_image, full_image):
        logger.info(
            "_coarse_to_fine_vision_extract: crop dense → découpage en %d sous-crops",
            VISION_SPLIT_N_SUBCROP,
        )
        sub_images = _split_image_vertically(crop_image, n_parts=VISION_SPLIT_N_SUBCROP)
    else:
        sub_images = [crop_image]

    logger.info(
        "_coarse_to_fine_vision_extract: %d image(s) à envoyer au LLM page=%d",
        len(sub_images), page_num,
    )

    # ── Étape 5 : passe 2 — extraction de chaque sous-crop ───────────────────
    extraction_prompt = _build_vision_prompt(
        table_name=table_name,
        type_comptes=type_comptes,
        secteur=secteur,
    )

    extracted_dfs: List[pd.DataFrame] = []
    for i, sub_img in enumerate(sub_images):
        if debug_save and len(sub_images) > 1:
            _debug_save_image(
                sub_img, pdf_path, page_num, table_name,
                suffix=f"c2f_subcrop{i + 1}of{len(sub_images)}",
            )

        sub_b64 = _img_to_b64(sub_img)
        if not sub_b64:
            logger.warning(
                "_coarse_to_fine_vision_extract: base64 échoué sous-crop %d/%d",
                i + 1, len(sub_images),
            )
            continue

        logger.info(
            "_coarse_to_fine_vision_extract: extraction sous-crop %d/%d provider=%s",
            i + 1, len(sub_images), provider,
        )

        try:
            raw = _call_vision_llm(
                img_b64=sub_b64,
                prompt=extraction_prompt,
                provider=provider,
                api_key=resolved_key,
            )
        except Exception as exc:
            logger.warning(
                "_coarse_to_fine_vision_extract: LLM échoué sous-crop %d/%d: %s",
                i + 1, len(sub_images), exc,
            )
            continue

        if not raw:
            logger.warning(
                "_coarse_to_fine_vision_extract: réponse vide sous-crop %d/%d",
                i + 1, len(sub_images),
            )
            continue

        df = _vision_parse_response(
            content=raw,
            table_name=table_name,
            pdf_path=pdf_path,
            page_num=page_num,
        )

        if not df.empty:
            extracted_dfs.append(df)
            logger.info(
                "_coarse_to_fine_vision_extract: sous-crop %d/%d → shape=%s",
                i + 1, len(sub_images), df.shape,
            )
        else:
            logger.warning(
                "_coarse_to_fine_vision_extract: parsing vide sous-crop %d/%d",
                i + 1, len(sub_images),
            )

    # ── Étape 6 : fusion ─────────────────────────────────────────────────────
    if not extracted_dfs:
        logger.warning(
            "_coarse_to_fine_vision_extract: aucun sous-crop n'a produit de DataFrame page=%d",
            page_num,
        )
        return pd.DataFrame()

    result_df = _merge_vision_dataframes(extracted_dfs, table_name=table_name)
    logger.info(
        "_coarse_to_fine_vision_extract: résultat final shape=%s page=%d table=%r",
        result_df.shape, page_num, table_name,
    )
    return result_df


# ==============================================================================
# PIPELINE MULTI-TABLEAUX — PHASES 1 → 4
# Objectif : isoler un tableau parmi plusieurs sur la même page PDF, le cropper
# localement puis en extraire les données — le LLM ne lit jamais la page entière.
# ==============================================================================


def _build_multi_table_detection_prompt(table_name: str) -> str:
    """Phase 1 — Prompt LLM : détecter tous les tableaux de la page et localiser le tableau cible."""
    return (
        f'You are analyzing a financial report page image that may contain MULTIPLE tables.\n\n'
        f'Your task:\n'
        f'1. Identify ALL visible tables on this page by looking at their titles/headers.\n'
        f'2. Select the table that matches: "{table_name}"\n'
        f'3. Return the bounding box of ONLY that table.\n\n'
        f'IMPORTANT rules:\n'
        f'- This page may have 2, 3 or more tables side-by-side or stacked\n'
        f'  (e.g. BILAN ACTIF, BILAN PASSIF, CPC, HORS BILAN, ÉTAT DES SOLDES DE GESTION).\n'
        f'- Identify each table by its visible title row.\n'
        f'- Select ONLY the table matching "{table_name}".\n'
        f'- Do NOT extract any numbers or cell values.\n'
        f'- Do NOT read data from cells.\n'
        f'- Return ONLY a strict JSON object — no prose, no explanation.\n\n'
        f'Return this exact JSON structure:\n'
        f'{{\n'
        f'  "tables_detected": [\n'
        f'    {{"title": "<TABLE_TITLE>", "location_hint": "<upper_left|upper_right|lower_left|lower_right|upper_half|lower_half|left_column|right_column|full_page>"}},\n'
        f'    ...\n'
        f'  ],\n'
        f'  "selected_table": "{table_name}",\n'
        f'  "table_found": true,\n'
        f'  "bbox_norm": [x0, y0, x1, y1]\n'
        f'}}\n\n'
        f'bbox_norm coordinates:\n'
        f'- All values between 0.0 and 1.0 (origin = top-left, x = horizontal, y = vertical)\n'
        f'- Must cover the ENTIRE selected table from its title row to its last data row\n'
        f'- x1 > x0 and y1 > y0\n'
        f'- Add a small margin (~1%) around the table\n\n'
        f'If the target table is NOT found:\n'
        f'{{\n'
        f'  "tables_detected": [...],\n'
        f'  "selected_table": "{table_name}",\n'
        f'  "table_found": false,\n'
        f'  "bbox_norm": null\n'
        f'}}\n\n'
        f'Return ONLY the JSON. No text before or after.'
    )


def _vision_detect_and_select_table_on_full_page(
    pdf_path: str,
    page_num: int,
    table_name: str,
    api_provider: str = "groq",
    api_key: Optional[str] = None,
    dpi: Optional[int] = None,
    enhance: Optional[bool] = None,
) -> dict:
    """
    Phase 1 — Détection visuelle multi-tableaux sur la page complète.

    Envoie la page entière au Vision LLM une seule fois.
    Le LLM détecte tous les tableaux visibles, sélectionne celui correspondant
    à table_name, et retourne sa bounding box.

    Ne fait AUCUNE extraction de chiffres. Ne fait AUCUN crop. Ne sauvegarde AUCUNE image.

    Returns:
        {
            "tables_detected": [{"title": str, "location_hint": str}, ...],
            "selected_table": str,
            "table_found": bool,
            "bbox_norm": [x0, y0, x1, y1] normalisé 0–1 | None,
            "bbox_px":   [x0_px, y0_px, x1_px, y1_px] pixels | None,
        }
    """
    _FAILED: dict = {
        "tables_detected": [],
        "selected_table": table_name,
        "table_found": False,
        "bbox_norm": None,
        "bbox_px": None,
    }

    provider = _normalize_api_provider(api_provider)
    resolved_key = _provider_api_key(provider, api_key=api_key)
    if not resolved_key:
        logger.warning("[MULTI_TABLE] Phase1: API key manquante provider=%s", provider)
        return _FAILED

    if dpi is None or enhance is None:
        auto_dpi, auto_enhance = _choose_dpi(pdf_path, page_num)
        if dpi is None:
            dpi = auto_dpi
        if enhance is None:
            enhance = auto_enhance

    full_image = _render_pdf_page_high_quality(
        pdf_path=pdf_path,
        page_num=page_num,
        dpi=dpi,
        enhance=enhance,
    )
    if full_image is None:
        logger.warning(
            "[MULTI_TABLE] Phase1: rendu page échoué page=%d table=%r", page_num, table_name
        )
        return _FAILED

    img_w, img_h = full_image.size

    img_b64 = _img_to_b64(full_image)
    if not img_b64:
        logger.warning("[MULTI_TABLE] Phase1: conversion base64 échouée page=%d", page_num)
        return _FAILED

    prompt = _build_multi_table_detection_prompt(table_name)
    logger.info(
        "[MULTI_TABLE] requested=%r page=%d provider=%s size=%dx%d",
        table_name, page_num, provider, img_w, img_h,
    )

    try:
        raw = _call_vision_llm(
            img_b64=img_b64,
            prompt=prompt,
            provider=provider,
            api_key=resolved_key,
        )
    except Exception as exc:
        logger.warning("[MULTI_TABLE] Phase1: appel LLM échoué page=%d: %s", page_num, exc)
        return _FAILED

    if not raw:
        logger.warning("[MULTI_TABLE] Phase1: réponse vide page=%d provider=%s", page_num, provider)
        return _FAILED

    try:
        json_str = _extract_json_object(raw)
        if not json_str:
            logger.warning(
                "[MULTI_TABLE] Phase1: aucun JSON — extrait=%r", raw[:300]
            )
            return _FAILED

        payload = _safe_load_llm_json(json_str)
        if payload is None:
            logger.warning(
                "[MULTI_TABLE] Phase1: JSON invalide après normalisation — raw=%r", raw[:300]
            )
            return _FAILED

        tables_detected = payload.get("tables_detected", [])
        if not isinstance(tables_detected, list):
            tables_detected = []

        table_found = bool(payload.get("table_found", False))
        bbox_norm = payload.get("bbox_norm")

        detected_titles = [
            t.get("title", "") for t in tables_detected if isinstance(t, dict)
        ]
        logger.info("[MULTI_TABLE] detected=%s", detected_titles)

        if not table_found or bbox_norm is None:
            logger.info("[MULTI_TABLE] Phase1: tableau %r non trouvé par le LLM", table_name)
            return {
                "tables_detected": tables_detected,
                "selected_table": table_name,
                "table_found": False,
                "bbox_norm": None,
                "bbox_px": None,
            }

        if (
            not isinstance(bbox_norm, list)
            or len(bbox_norm) != 4
            or not all(isinstance(v, (int, float)) for v in bbox_norm)
        ):
            logger.warning(
                "[MULTI_TABLE] Phase1: bbox_norm invalide=%r table=%r", bbox_norm, table_name
            )
            return _FAILED

        x0, y0, x1, y1 = [float(v) for v in bbox_norm]
        x0 = max(0.0, min(1.0, x0))
        y0 = max(0.0, min(1.0, y0))
        x1 = max(0.0, min(1.0, x1))
        y1 = max(0.0, min(1.0, y1))

        if x1 <= x0 or y1 <= y0:
            logger.warning(
                "[MULTI_TABLE] Phase1: bbox incohérente [%.3f,%.3f,%.3f,%.3f] table=%r",
                x0, y0, x1, y1, table_name,
            )
            return _FAILED

        if (x1 - x0) < 0.04 or (y1 - y0) < 0.04:
            logger.warning(
                "[MULTI_TABLE] Phase1: bbox trop petite [%.3f,%.3f,%.3f,%.3f] table=%r",
                x0, y0, x1, y1, table_name,
            )
            return _FAILED

        x0_px = int(x0 * img_w)
        y0_px = int(y0 * img_h)
        x1_px = int(x1 * img_w)
        y1_px = int(y1 * img_h)

        logger.info(
            "[MULTI_TABLE] selected=%r bbox_norm=[%.3f,%.3f,%.3f,%.3f] bbox_px=[%d,%d,%d,%d]",
            table_name, x0, y0, x1, y1, x0_px, y0_px, x1_px, y1_px,
        )
        return {
            "tables_detected": tables_detected,
            "selected_table": table_name,
            "table_found": True,
            "bbox_norm": [x0, y0, x1, y1],
            "bbox_px": [x0_px, y0_px, x1_px, y1_px],
        }

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "[MULTI_TABLE] Phase1: parse JSON échoué table=%r: %s — raw=%r",
            table_name, exc, raw[:400],
        )
        return _FAILED


def _crop_selected_table_from_bbox(
    image: "PIL.Image.Image",  # noqa: F821
    bbox_px: List[int],
    padding_px: int = MULTI_TABLE_CROP_PADDING_PX,
) -> "Optional[PIL.Image.Image]":  # noqa: F821
    """
    Phase 2 — Crop local du tableau sélectionné.

    Applique un padding configurable, clamp aux limites de l'image, retourne un
    crop PIL propre. Le crop est entièrement local — aucun LLM, aucune sauvegarde.

    Args:
        image:      Image PIL de la page complète.
        bbox_px:    [x0_px, y0_px, x1_px, y1_px] en pixels absolus.
        padding_px: Nombre de pixels de marge ajouté de chaque côté.

    Returns:
        Image PIL croppée, ou None si les coordonnées sont invalides / crop trop petit.
    """
    try:
        img_w, img_h = image.size
        x0, y0, x1, y1 = [int(v) for v in bbox_px]

        x0 = max(0, x0 - padding_px)
        y0 = max(0, y0 - padding_px)
        x1 = min(img_w, x1 + padding_px)
        y1 = min(img_h, y1 + padding_px)

        crop_w = x1 - x0
        crop_h = y1 - y0

        if crop_w <= 0 or crop_h <= 0:
            logger.warning(
                "[MULTI_TABLE] Phase2: crop invalide px=(%d,%d,%d,%d) img=%dx%d",
                x0, y0, x1, y1, img_w, img_h,
            )
            return None

        if crop_w < 50 or crop_h < 50:
            logger.warning(
                "[MULTI_TABLE] Phase2: crop trop petit (%dx%d) — ignoré", crop_w, crop_h
            )
            return None

        cropped = image.crop((x0, y0, x1, y1))
        logger.info(
            "[MULTI_TABLE] crop_size=(%dx%d) px=(%d,%d,%d,%d) padding=%d",
            crop_w, crop_h, x0, y0, x1, y1, padding_px,
        )
        return cropped

    except Exception as exc:
        logger.warning("[MULTI_TABLE] Phase2: crop échoué: %s", exc)
        return None


def _should_split_table_crop(
    crop_img: "PIL.Image.Image",  # noqa: F821
    full_page_img: "PIL.Image.Image",  # noqa: F821
    height_ratio_threshold: float = MULTI_TABLE_SPLIT_HEIGHT_RATIO,
    pixel_threshold: int = MULTI_TABLE_SPLIT_PIXEL_THRESHOLD,
) -> bool:
    """
    Phase 3 — Détermine si le crop du tableau doit être découpé verticalement.

    Heuristiques (OU logique) :
    - crop_height / page_height > height_ratio_threshold
    - crop_w × crop_h > pixel_threshold
    """
    if not MULTI_TABLE_SPLIT_ENABLED:
        return False

    crop_w, crop_h = crop_img.size
    _, full_h = full_page_img.size

    surface = crop_w * crop_h
    height_ratio = crop_h / full_h if full_h > 0 else 1.0

    decision = surface > pixel_threshold or height_ratio > height_ratio_threshold
    logger.info(
        "[MULTI_TABLE] split_check: surface=%d (seuil=%d) height_ratio=%.2f (seuil=%.2f) → split=%s",
        surface, pixel_threshold, height_ratio, height_ratio_threshold, decision,
    )
    return decision


def _split_table_crop_vertically(
    crop_img: "PIL.Image.Image",  # noqa: F821
    n: int = MULTI_TABLE_SPLIT_DEFAULT_N,
    overlap_px: int = MULTI_TABLE_SPLIT_OVERLAP_PX,
) -> "List[PIL.Image.Image]":  # noqa: F821
    """
    Phase 3 — Découpage vertical du crop en n sous-crops avec chevauchement.

    Conserve l'ordre top → bottom. Ne sauvegarde aucune image.

    Args:
        crop_img:   Image PIL du crop du tableau sélectionné.
        n:          Nombre de sous-crops (2 défaut ; 3 si crop très haut).
        overlap_px: Chevauchement en pixels entre tranches adjacentes.

    Returns:
        Liste ordonnée [top, ..., bottom] de n images PIL.
    """
    if n <= 1:
        return [crop_img]

    img_w, img_h = crop_img.size

    if img_h < n * 40:
        logger.warning(
            "[MULTI_TABLE] Phase3: crop trop court (%dpx) pour %d splits → retour entier",
            img_h, n,
        )
        return [crop_img]

    slice_h = img_h // n
    parts = []
    for i in range(n):
        y0 = max(0, i * slice_h - (overlap_px if i > 0 else 0))
        y1 = min(img_h, (i + 1) * slice_h + (overlap_px if i < n - 1 else 0))
        part = crop_img.crop((0, y0, img_w, y1))
        logger.info(
            "[MULTI_TABLE] Phase3: sous-crop %d/%d y=(%d,%d) size=%dx%d",
            i + 1, n, y0, y1, img_w, y1 - y0,
        )
        parts.append(part)

    return parts


def _get_table_stop_markers(table_name: str) -> List[str]:
    """
    Retourne les stop markers pour un type de tableau.

    Lors de la fusion, si une ligne correspond à un stop marker, les lignes
    suivantes sont supprimées (le marker lui-même est conservé).
    """
    name = (table_name or "").lower().replace("\n", " ")

    if "actif" in name and "passif" not in name:
        return [
            "total actif",
            "total de l'actif",
            "total de lactif",
        ]

    if "passif" in name and "hors bilan" not in name:
        return [
            "total passif",
            "total du passif",
            "total des passifs",
        ]

    if (
        "cpc" in name
        or "compte de produits" in name
        or "produits et charges" in name
        or "compte de résultat" in name
        or "compte de resultat" in name
    ):
        return [
            "resultat net de l'exercice",
            "résultat net de l'exercice",
            "résultat net",
            "resultat net",
            "résultat de l'exercice",
            "resultat de l exercice",
        ]

    return []


def _merge_partial_table_dataframes(
    dfs: "List[pd.DataFrame]",
    table_name: str,
) -> pd.DataFrame:
    """
    Phase 4 — Fusion de DataFrames partiels extraits de sous-crops verticaux.

    Règles :
    - Concat dans l'ordre top → bottom.
    - Ignore les DataFrames vides.
    - Normalise les labels (trim, collapse espaces multiples).
    - Déduplication prudente sur l'overlap : deux lignes consécutives avec même
      label normalisé ET mêmes valeurs → une seule gardée.
    - Applique les stop markers : conserve le marker, coupe tout ce qui suit.
    - Préserve l'ordre d'origine.
    """
    valid_dfs = [df for df in dfs if df is not None and not df.empty]
    if not valid_dfs:
        logger.warning("[MULTI_TABLE] merge: tous les DataFrames sont vides table=%r", table_name)
        return pd.DataFrame()

    ref_cols = list(valid_dfs[0].columns)

    aligned = []
    for i, df in enumerate(valid_dfs):
        df_aligned = df.reindex(columns=ref_cols, fill_value="")
        aligned.append(df_aligned)

    merged = pd.concat(aligned, ignore_index=True)
    rows_before_dedup = len(merged)

    def _norm_label(s: str) -> str:
        return re.sub(r"\s+", " ", str(s or "").strip().lower())

    if not merged.empty and len(merged.columns) > 0:
        first_col = merged.columns[0]
        merged = merged.copy()
        merged["_norm_label"] = merged[first_col].apply(_norm_label)

        # Déduplication prudente sur les lignes consécutives identiques (zone overlap)
        keep_mask = [True] * len(merged)
        for i in range(1, len(merged)):
            prev_label = merged["_norm_label"].iloc[i - 1]
            curr_label = merged["_norm_label"].iloc[i]
            if prev_label and curr_label and prev_label == curr_label:
                prev_vals = merged.iloc[i - 1, 1:-1].astype(str).tolist()
                curr_vals = merged.iloc[i, 1:-1].astype(str).tolist()
                if prev_vals == curr_vals:
                    keep_mask[i] = False

        merged = merged[keep_mask].reset_index(drop=True)
        rows_after_dedup = len(merged)
        if rows_before_dedup != rows_after_dedup:
            logger.info(
                "[MULTI_TABLE] merge: %d doublons overlap supprimés (%d → %d)",
                rows_before_dedup - rows_after_dedup, rows_before_dedup, rows_after_dedup,
            )

        # Stop markers
        stop_markers = _get_table_stop_markers(table_name)
        stop_found = False
        stop_idx = len(merged)
        if stop_markers:
            for i, label_norm in enumerate(merged["_norm_label"]):
                for marker in stop_markers:
                    if _norm_label(marker) in label_norm or label_norm in _norm_label(marker):
                        stop_idx = i + 1  # Conserver le marker lui-même
                        stop_found = True
                        logger.info(
                            "[MULTI_TABLE] stop_marker=%r found=True at row=%d", marker, i
                        )
                        break
                if stop_found:
                    break

            if not stop_found:
                logger.info("[MULTI_TABLE] stop_marker=None found=False")

            merged = merged.iloc[:stop_idx].reset_index(drop=True)

        if "_norm_label" in merged.columns:
            merged = merged.drop(columns=["_norm_label"])

    logger.info(
        "[MULTI_TABLE] merged_rows_before_stop=%d final_rows=%d",
        rows_before_dedup, len(merged),
    )
    return merged


def _extract_table_from_selected_crop(
    pdf_path: str,
    page_num: int,
    table_name: str,
    api_provider: str = "groq",
    api_key: Optional[str] = None,
    type_comptes: Optional[str] = None,
    secteur: Optional[str] = None,
) -> pd.DataFrame:
    """
    Orchestrateur du pipeline multi-tableaux Phases 1 → 4.

    Pipeline :
    1. Rendu page complète.
    2. Phase 1 — Détection LLM : identifier les tableaux visibles, sélectionner
       le tableau cible, obtenir sa bbox. (LLM = localisation seulement, pas d'extraction)
    3. Phase 2 — Crop local à partir de la bbox (sans LLM).
    4. Phase 3 — Split vertical si le crop est trop grand (sans LLM).
    5. Phase 4 — Extraction sur chaque (sous-)crop uniquement — jamais page complète.
    6. Fusion + stop markers → DataFrame final.

    Returns:
        DataFrame extrait, ou DataFrame vide si tableau non trouvé / extraction échoue.
    """
    provider = _normalize_api_provider(api_provider)
    resolved_key = _provider_api_key(provider, api_key=api_key)
    if not resolved_key:
        logger.warning("[MULTI_TABLE] Phase4: API key manquante provider=%s", provider)
        return pd.DataFrame()

    logger.info("[MULTI_TABLE] requested=%r page=%d provider=%s", table_name, page_num, provider)

    dpi, enhance = _choose_dpi(pdf_path, page_num)

    # ── Rendu page complète ───────────────────────────────────────────────────
    full_image = _render_pdf_page_high_quality(
        pdf_path=pdf_path,
        page_num=page_num,
        dpi=dpi,
        enhance=enhance,
    )
    if full_image is None:
        logger.warning("[MULTI_TABLE] Phase4: rendu page échoué page=%d → fallback", page_num)
        return pd.DataFrame()

    if MULTI_TABLE_DEBUG_SAVE_IMAGES:
        _debug_save_image(full_image, pdf_path, page_num, table_name, suffix="mt_full")

    # ── Phase 1 : Détection et sélection du tableau ───────────────────────────
    detect_result = _vision_detect_and_select_table_on_full_page(
        pdf_path=pdf_path,
        page_num=page_num,
        table_name=table_name,
        api_provider=provider,
        api_key=resolved_key,
        dpi=dpi,
        enhance=enhance,
    )

    if not detect_result["table_found"] or detect_result["bbox_px"] is None:
        logger.warning(
            "[MULTI_TABLE] Phase4: tableau %r non détecté page=%d → fallback",
            table_name, page_num,
        )
        return pd.DataFrame()

    # ── Phase 2 : Crop local ──────────────────────────────────────────────────
    crop_img = _crop_selected_table_from_bbox(
        image=full_image,
        bbox_px=detect_result["bbox_px"],
        padding_px=MULTI_TABLE_CROP_PADDING_PX,
    )

    if crop_img is None:
        logger.warning(
            "[MULTI_TABLE] Phase4: crop échoué page=%d table=%r → fallback", page_num, table_name
        )
        return pd.DataFrame()

    if MULTI_TABLE_DEBUG_SAVE_IMAGES:
        _debug_save_image(crop_img, pdf_path, page_num, table_name, suffix="mt_crop")

    # ── Phase 3 : Split vertical si crop trop grand ───────────────────────────
    do_split = _should_split_table_crop(crop_img, full_image)
    if do_split:
        n_splits = MULTI_TABLE_SPLIT_DEFAULT_N
        _, crop_h = crop_img.size
        _, full_h = full_image.size
        if full_h > 0 and (crop_h / full_h) > 0.80:
            n_splits = 3
        sub_crops = _split_table_crop_vertically(
            crop_img, n=n_splits, overlap_px=MULTI_TABLE_SPLIT_OVERLAP_PX
        )
        logger.info(
            "[MULTI_TABLE] split=True n=%d overlap=%d", n_splits, MULTI_TABLE_SPLIT_OVERLAP_PX
        )
    else:
        sub_crops = [crop_img]
        logger.info("[MULTI_TABLE] split=False n=1")

    logger.info("[MULTI_TABLE] nombre de sous-crops: %d", len(sub_crops))

    # ── Phase 4 : Extraction sur chaque sous-crop (jamais page complète) ──────
    extraction_prompt = _build_vision_prompt(
        table_name=table_name,
        type_comptes=type_comptes,
        secteur=secteur,
    )

    extracted_dfs: List[pd.DataFrame] = []
    for i, sub_img in enumerate(sub_crops):
        if MULTI_TABLE_DEBUG_SAVE_IMAGES and len(sub_crops) > 1:
            _debug_save_image(
                sub_img, pdf_path, page_num, table_name,
                suffix=f"mt_subcrop{i + 1}of{len(sub_crops)}",
            )

        sub_b64 = _img_to_b64(sub_img)
        if not sub_b64:
            logger.warning(
                "[MULTI_TABLE] base64 échoué sous-crop %d/%d", i + 1, len(sub_crops)
            )
            continue

        try:
            raw = _call_vision_llm(
                img_b64=sub_b64,
                prompt=extraction_prompt,
                provider=provider,
                api_key=resolved_key,
            )
        except Exception as exc:
            logger.warning(
                "[MULTI_TABLE] LLM échoué sous-crop %d/%d: %s", i + 1, len(sub_crops), exc
            )
            continue

        if not raw:
            logger.warning(
                "[MULTI_TABLE] réponse vide sous-crop %d/%d", i + 1, len(sub_crops)
            )
            continue

        df = _vision_parse_response(
            content=raw,
            table_name=table_name,
            pdf_path=pdf_path,
            page_num=page_num,
        )

        if df is not None and not df.empty:
            extracted_dfs.append(df)
            logger.info("[MULTI_TABLE] segment=%d rows=%d", i + 1, len(df))
        else:
            logger.warning(
                "[MULTI_TABLE] extraction vide sous-crop %d/%d", i + 1, len(sub_crops)
            )

    if not extracted_dfs:
        logger.warning(
            "[MULTI_TABLE] Phase4: aucun sous-crop n'a produit de DataFrame page=%d table=%r",
            page_num, table_name,
        )
        return pd.DataFrame()

    # ── Fusion + stop markers ─────────────────────────────────────────────────
    result_df = _merge_partial_table_dataframes(extracted_dfs, table_name)
    logger.info(
        "[MULTI_TABLE] résultat final shape=%s page=%d table=%r",
        result_df.shape, page_num, table_name,
    )
    return result_df


# ══════════════════════════════════════════════════════════════════════════════
#  Pipeline multi-tableaux simplifié — Phase 1 → Phase 2 → Phase 3 → Phase 4
#  Flux : page complète → détection de toutes les zones → crop local →
#         classification de chaque crop → sélection → extraction sur le crop
# ══════════════════════════════════════════════════════════════════════════════


def _build_detect_all_tables_prompt() -> str:
    """Phase 1 — Prompt : détecter toutes les zones de tableaux sans lire aucune donnée."""
    return (
        "You are analyzing a financial report page image.\n\n"
        "Your ONLY task: visually locate ALL table structures visible on this page.\n\n"
        "STRICT rules:\n"
        "- Do NOT read any numbers, amounts, or cell values.\n"
        "- Do NOT invent or guess table titles.\n"
        "- Do NOT classify the tables.\n"
        "- ONLY detect where table grid structures are visually present.\n"
        "- A table is a grid of rows and columns of data.\n\n"
        "Return a strict JSON object with ONLY this structure:\n"
        "{\n"
        '  "tables": [\n'
        '    {"bbox_norm": [x0, y0, x1, y1]},\n'
        '    {"bbox_norm": [x0, y0, x1, y1]}\n'
        "  ]\n"
        "}\n\n"
        "bbox_norm coordinates:\n"
        "- All values between 0.0 and 1.0 (origin = top-left, x = horizontal, y = vertical)\n"
        "- Must cover the complete table from its first row to its last data row\n"
        "- x1 > x0 and y1 > y0\n\n"
        "Return ONLY the JSON. No text before or after."
    )


def _build_classify_table_prompt() -> str:
    """Phase 3 — Prompt : classifier un tableau cropé sans extraire de données."""
    return (
        "You are analyzing a single financial table image from a Moroccan financial report.\n\n"
        "Your ONLY task: identify the TYPE of this table based on its title/header.\n\n"
        "The only possible types are:\n"
        "- BILAN ACTIF\n"
        "- BILAN PASSIF\n"
        "- COMPTE DE PRODUITS ET CHARGES\n"
        "- AUTRE\n\n"
        "Rules:\n"
        "- Do NOT read numbers or extract any data values.\n"
        "- Look at the table title or header row to determine the type.\n"
        "- If uncertain or none of the above match, return AUTRE.\n\n"
        "Return a strict JSON:\n"
        '{"table_type": "BILAN ACTIF"}\n\n'
        "Return ONLY the JSON. No text before or after."
    )


def _build_detect_all_tables_prompt() -> str:
    """Phase 1: locate only ACTIF, PASSIF and CPC sections on the full page."""
    return """You are a highly precise financial document vision analyst specialized in Moroccan annual reports, AMMC-style filings, and bank financial statements.

YOUR TASK:
Analyze the FULL PAGE image of a financial report and locate ONLY the following target sections if they are visible on the page:

- BILAN_ACTIF
- BILAN_PASSIF
- CPC

IMPORTANT BUSINESS CONTEXT:
This is a Moroccan financial statement page that may contain multiple financial sections on the same page.
Typical layouts may include ACTIF, PASSIF, HORS BILAN, Compte de produits et charges, headers, footers, page numbers, notes, and summary labels.

YOU MUST IGNORE COMPLETELY:
- HORS BILAN
- Sommaire / Summary
- Page headers / report titles / page numbers
- Footers
- Notes
- Decorative colored bars not belonging to the target table
- Any table that is not ACTIF, PASSIF, or Compte de produits et charges

SECTION DEFINITIONS:

1) BILAN_ACTIF
- This is the ACTIF subsection inside a BILAN page.
- It usually starts at a red/orange header bar containing the text "ACTIF".
- It ends EXACTLY at the row "Total de l'Actif".
- The bounding box MUST include the ACTIF header row, all visible ACTIF rows, and "Total de l'Actif".
- The bounding box MUST NOT include PASSIF, HORS BILAN, footer, or summary.

2) BILAN_PASSIF
- This is the PASSIF subsection inside a BILAN page.
- It usually starts at a red/orange header bar containing the text "PASSIF".
- It ends EXACTLY at the row "Total du Passif".
- The bounding box MUST include the PASSIF header row, all visible PASSIF rows, and "Total du Passif".
- The bounding box MUST NOT include HORS BILAN title, HORS BILAN rows, footer, or any following section.

3) CPC
- This is the "Compte de produits et charges" section.
- It usually starts at a visible title such as "Compte de produits et charges au 31 decembre 2024" or equivalent French wording.
- In Moroccan bank consolidated statements, the same target section may be titled "COMPTE DE RESULTAT CONSOLIDE".
- Bank consolidated CPC rows may include Interets et produits assimiles, MARGE D'INTERET, MARGE SUR COMMISSIONS, PRODUIT NET BANCAIRE, RESULTAT NET PART DU GROUPE.
- The bounding box MUST include the CPC title, the table header with date columns, and all visible CPC rows belonging to the section.
- The bounding box MUST NOT include previous unrelated section, next unrelated section, notes, summary, or footer.

STRICT CROPPING RULES:
1. NEVER merge two target sections into one box.
2. NEVER include ignored sections in a target box.
3. If BILAN_ACTIF is visible, start at ACTIF and stop exactly at "Total de l'Actif" if visible.
4. If BILAN_PASSIF is visible, start at PASSIF and stop exactly at "Total du Passif" if visible.
5. If CPC is visible, include the CPC title and CPC table only.
6. Bounding boxes must be tight and semantic, but not so tight that rows are cut.
7. If a target section is partially visible, still return it and set visibility_status accordingly.

BOUNDING BOX FORMAT:
Use NORMALIZED coordinates in [0,1]: x0 left, y0 top, x1 right, y1 bottom.
Constraints: 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1.

OUTPUT FORMAT (STRICT JSON ONLY):
Return exactly this schema:
{
  "page_analysis": {
    "has_target_sections": true,
    "detected_section_types": ["BILAN_ACTIF", "BILAN_PASSIF"],
    "notes": "short internal note"
  },
  "sections": [
    {
      "section_type": "BILAN_ACTIF",
      "bbox": {"x0": 0.03, "y0": 0.14, "x1": 0.97, "y1": 0.56},
      "confidence": 0.98,
      "header_text_detected": "ACTIF",
      "must_start_at": "ACTIF",
      "must_stop_at": "Total de l'Actif",
      "starts_on_this_page": true,
      "ends_on_this_page": true,
      "visibility_status": "complete",
      "reason": "ACTIF section clearly visible and ends at Total de l'Actif"
    }
  ]
}

ALLOWED VALUES:
section_type: BILAN_ACTIF, BILAN_PASSIF, CPC
visibility_status: complete, partial_top, partial_bottom, partial_both

FINAL HARD RULES:
- STRICT JSON ONLY
- MAXIMUM 3 sections
- NEVER return HORS_BILAN
- NEVER return AUTRE
- NEVER return generic names like table_1 or table_2
- NEVER merge PASSIF with HORS BILAN
- NEVER merge ACTIF with PASSIF
- If "Total de l'Actif" is visible, BILAN_ACTIF must stop there
- If "Total du Passif" is visible, BILAN_PASSIF must stop there
- If a target section is not visible, do not return it
- If no target section is visible, return an empty sections array"""


def _build_validate_crop_prompt(target_section_type: Optional[str]) -> str:
    """Prompt de validation d'un crop deja selectionne."""
    target = target_section_type or "UNKNOWN"
    return f"""You are validating whether a cropped image correctly isolates ONE target financial section.

TARGET SECTION TYPES:
- BILAN_ACTIF
- BILAN_PASSIF
- CPC

EXPECTED TARGET SECTION:
- {target}

TASK:
Given a cropped image that is supposed to contain exactly one target section, determine if the crop is valid.

VALIDATION RULES:

For BILAN_ACTIF:
- Must contain the "ACTIF" header
- Must contain rows belonging to ACTIF
- If "Total de l'Actif" is visible, crop should end there
- Must NOT contain "PASSIF"
- Must NOT contain "HORS BILAN"

For BILAN_PASSIF:
- Must contain the "PASSIF" header
- Must contain rows belonging to PASSIF
- If "Total du Passif" is visible, crop should end there
- Must NOT contain "HORS BILAN"
- Must NOT contain unrelated following sections

For CPC:
- Must contain "Compte de produits et charges", "Compte de resultat consolide", or clearly be the CPC/result table
- For bank consolidated statements, accept rows such as MARGE D'INTERET, PRODUIT NET BANCAIRE, RESULTAT NET PART DU GROUPE
- Must contain CPC rows
- Must NOT contain unrelated sections

OUTPUT STRICT JSON ONLY:
{{
  "is_valid_crop": true,
  "detected_section_type": "BILAN_PASSIF",
  "contains_expected_header": true,
  "contains_forbidden_content": false,
  "forbidden_content_detected": [],
  "has_total_row_visible": true,
  "detected_total_row": "Total du Passif",
  "crop_quality": "good",
  "recommended_action": "accept"
}}

ALLOWED crop_quality:
- good
- acceptable
- bad

ALLOWED recommended_action:
- accept
- recrop_tighter
- recrop_expand_down
- recrop_expand_up
- reject

STRICT JSON ONLY."""


def _vision_validate_selected_crop(
    crop_image: "PIL.Image.Image",  # noqa: F821
    target_section_type: Optional[str],
    api_provider: str = "groq",
    api_key: Optional[str] = None,
) -> dict:
    """Valide qu'un crop contient bien une seule section cible."""
    fallback = {
        "is_valid_crop": False,
        "detected_section_type": None,
        "contains_expected_header": False,
        "contains_forbidden_content": True,
        "forbidden_content_detected": ["validation_failed"],
        "has_total_row_visible": False,
        "detected_total_row": "",
        "crop_quality": "bad",
        "recommended_action": "reject",
    }
    provider = _normalize_api_provider(api_provider)
    resolved_key = _provider_api_key(provider, api_key=api_key)
    if not resolved_key:
        return fallback
    img_b64 = _img_to_b64(crop_image)
    if not img_b64:
        return fallback
    try:
        raw = _call_vision_llm(
            img_b64=img_b64,
            prompt=_build_validate_crop_prompt(target_section_type),
            provider=provider,
            api_key=resolved_key,
        )
        json_str = _extract_json_object(raw or "")
        if not json_str:
            return fallback
        payload = _safe_load_llm_json(json_str)
        if not isinstance(payload, dict):
            return fallback
        return payload
    except Exception as exc:
        logger.warning("[MULTI_TABLE] crop validation failed: %s", exc)
        return fallback


def _vision_detect_tables_on_full_page(
    pdf_path: str,
    page_num: int,
    api_provider: str = "groq",
    api_key: Optional[str] = None,
    full_image: "Optional[PIL.Image.Image]" = None,  # noqa: F821
) -> dict:
    """
    Phase 1 — Détection visuelle de tous les tableaux sur la page complète.

    Envoie la page entière au Vision LLM.  Le LLM retourne uniquement la liste
    des bounding boxes de tous les tableaux visibles — aucune extraction de
    chiffres, aucune lecture de contenu, aucun titre inventé.

    Args:
        full_image: Image PIL déjà rendue (optionnel). Si fournie, le rendu PDF
                    est ignoré — évite un double rendu depuis l'orchestrateur.

    Returns:
        {"tables": [{"bbox_norm": [x0, y0, x1, y1]}, ...]}
        ou {"tables": []} si aucun tableau détecté ou en cas d'erreur.
    """
    _EMPTY: dict = {"tables": []}

    provider = _normalize_api_provider(api_provider)
    resolved_key = _provider_api_key(provider, api_key=api_key)
    if not resolved_key:
        logger.warning("[MULTI_TABLE] Phase1: API key manquante provider=%s", provider)
        return _EMPTY

    if full_image is None:
        dpi, enhance = _choose_dpi(pdf_path, page_num)
        full_image = _render_pdf_page_high_quality(
            pdf_path=pdf_path,
            page_num=page_num,
            dpi=dpi,
            enhance=enhance,
        )
    if full_image is None:
        logger.warning("[MULTI_TABLE] Phase1: rendu page échoué page=%d", page_num)
        return _EMPTY

    img_b64 = _img_to_b64(full_image)
    if not img_b64:
        logger.warning("[MULTI_TABLE] Phase1: conversion base64 échouée page=%d", page_num)
        return _EMPTY

    logger.info(
        "[MULTI_TABLE] full page detection started page=%d provider=%s size=%dx%d",
        page_num, provider, full_image.width, full_image.height,
    )

    prompt = _build_detect_all_tables_prompt()
    try:
        raw = _call_vision_llm(
            img_b64=img_b64, prompt=prompt, provider=provider, api_key=resolved_key
        )
    except Exception as exc:
        logger.warning("[MULTI_TABLE] Phase1: appel LLM échoué page=%d: %s", page_num, exc)
        return _EMPTY

    if not raw:
        logger.warning("[MULTI_TABLE] Phase1: réponse vide page=%d", page_num)
        return _EMPTY

    try:
        json_str = _extract_json_object(raw)
        if not json_str:
            logger.warning("[MULTI_TABLE] Phase1: aucun JSON — extrait=%r", raw[:300])
            return _EMPTY

        payload = _safe_load_llm_json(json_str)
        if payload is None:
            logger.warning("[MULTI_TABLE] Phase1: JSON invalide — raw=%r", raw[:300])
            return _EMPTY

        tables_raw = payload.get("tables", [])
        if not isinstance(tables_raw, list):
            tables_raw = []
        sections_raw = payload.get("sections", [])
        if not isinstance(sections_raw, list):
            sections_raw = []

        valid_tables = []
        # New semantic schema: sections[].bbox = {x0,y0,x1,y1}
        for sec in sections_raw[:3]:
            if not isinstance(sec, dict):
                continue
            section_type = str(sec.get("section_type", "")).strip().upper()
            if section_type not in {"BILAN_ACTIF", "BILAN_PASSIF", "CPC"}:
                continue
            bbox_obj = sec.get("bbox")
            if not isinstance(bbox_obj, dict):
                continue
            bbox = [bbox_obj.get("x0"), bbox_obj.get("y0"), bbox_obj.get("x1"), bbox_obj.get("y1")]
            if (
                len(bbox) == 4
                and all(isinstance(v, (int, float)) for v in bbox)
            ):
                x0, y0, x1, y1 = [max(0.0, min(1.0, float(v))) for v in bbox]
                if x1 > x0 and y1 > y0 and (x1 - x0) >= 0.04 and (y1 - y0) >= 0.04:
                    valid_tables.append({
                        "bbox_norm": [x0, y0, x1, y1],
                        "section_type": section_type,
                        "confidence": sec.get("confidence"),
                        "header_text_detected": sec.get("header_text_detected", ""),
                        "visibility_status": sec.get("visibility_status", ""),
                    })

        # Backward compatibility with the previous generic schema.
        for t in tables_raw:
            if not isinstance(t, dict):
                continue
            bbox = t.get("bbox_norm")
            if (
                isinstance(bbox, list)
                and len(bbox) == 4
                and all(isinstance(v, (int, float)) for v in bbox)
            ):
                x0, y0, x1, y1 = [max(0.0, min(1.0, float(v))) for v in bbox]
                if x1 > x0 and y1 > y0 and (x1 - x0) >= 0.04 and (y1 - y0) >= 0.04:
                    valid_tables.append({"bbox_norm": [x0, y0, x1, y1]})

        logger.info(
            "[MULTI_TABLE] detected_sections=%s generic_tables=%d valid=%d page=%d",
            [t.get("section_type") for t in valid_tables if t.get("section_type")],
            len(tables_raw),
            len(valid_tables),
            page_num,
        )
        return {"tables": valid_tables}

    except Exception as exc:
        logger.warning(
            "[MULTI_TABLE] Phase1: parse échoué page=%d: %s — raw=%r", page_num, exc, raw[:300]
        )
        return _EMPTY


def _crop_detected_tables(
    image: "PIL.Image.Image",  # noqa: F821
    table_bboxes: List[dict],
    output_dir: Optional[str] = None,
) -> List[dict]:
    """
    Phase 2 — Crop local de chaque tableau détecté.

    Convertit les bbox normalisées en pixels, applique un padding
    (MULTI_TABLE_CROP_PADDING_PX) et croppe chaque tableau localement.
    Aucun LLM, aucune lecture de données.

    Args:
        image:        Image PIL de la page complète.
        table_bboxes: Liste de {"bbox_norm": [x0, y0, x1, y1]}.
        output_dir:   Dossier de sauvegarde debug (None = pas de sauvegarde).

    Returns:
        [
            {
                "bbox_norm":  [x0, y0, x1, y1],
                "bbox_px":    [x0_px, y0_px, x1_px, y1_px],
                "image":      PIL.Image,
                "debug_name": "table_01",
            },
            ...
        ]
    """
    img_w, img_h = image.size
    crops: List[dict] = []

    for i, entry in enumerate(table_bboxes):
        bbox_norm = entry.get("bbox_norm")
        if not bbox_norm or len(bbox_norm) != 4:
            continue

        x0, y0, x1, y1 = [float(v) for v in bbox_norm]
        pad = MULTI_TABLE_CROP_PADDING_PX
        x0_px = max(0, int(x0 * img_w) - pad)
        y0_px = max(0, int(y0 * img_h) - pad)
        x1_px = min(img_w, int(x1 * img_w) + pad)
        y1_px = min(img_h, int(y1 * img_h) + pad)

        if x1_px <= x0_px or y1_px <= y0_px:
            logger.warning("[MULTI_TABLE] Phase2: bbox invalide crop_%02d — ignoré", i + 1)
            continue

        crop_img = image.crop((x0_px, y0_px, x1_px, y1_px))
        debug_name = f"table_{i + 1:02d}"

        if output_dir and MULTI_TABLE_DEBUG_SAVE_CROPS:
            import os as _os
            _os.makedirs(output_dir, exist_ok=True)
            out_path = _os.path.join(output_dir, f"{debug_name}.png")
            try:
                crop_img.save(out_path)
                logger.debug("[MULTI_TABLE] Phase2: crop sauvegardé → %s", out_path)
            except Exception as e:
                logger.warning("[MULTI_TABLE] Phase2: sauvegarde échouée %s: %s", out_path, e)

        crops.append({
            "bbox_norm": [x0, y0, x1, y1],
            "bbox_px": [x0_px, y0_px, x1_px, y1_px],
            "image": crop_img,
            "debug_name": debug_name,
            "section_type": entry.get("section_type"),
            "detector_confidence": entry.get("confidence"),
            "header_text_detected": entry.get("header_text_detected", ""),
            "visibility_status": entry.get("visibility_status", ""),
        })

    return crops


def _vision_classify_cropped_table(
    crop_image: "PIL.Image.Image",  # noqa: F821
    api_provider: str = "groq",
    api_key: Optional[str] = None,
) -> dict:
    """
    Phase 3 — Classification d'un tableau cropé via Vision LLM.

    Envoie uniquement l'image du crop au LLM.
    Retourne le type du tableau — aucune extraction de données.

    Returns:
        {"table_type": "BILAN ACTIF" | "BILAN PASSIF" | "COMPTE DE PRODUITS ET CHARGES" | "AUTRE"}
    """
    _UNKNOWN: dict = {"table_type": "AUTRE"}

    provider = _normalize_api_provider(api_provider)
    resolved_key = _provider_api_key(provider, api_key=api_key)
    if not resolved_key:
        logger.warning("[MULTI_TABLE] Phase3: API key manquante provider=%s", provider)
        return _UNKNOWN

    img_b64 = _img_to_b64(crop_image)
    if not img_b64:
        logger.warning("[MULTI_TABLE] Phase3: conversion base64 échouée")
        return _UNKNOWN

    prompt = _build_classify_table_prompt()
    try:
        raw = _call_vision_llm(
            img_b64=img_b64, prompt=prompt, provider=provider, api_key=resolved_key
        )
    except Exception as exc:
        logger.warning("[MULTI_TABLE] Phase3: appel LLM échoué: %s", exc)
        return _UNKNOWN

    if not raw:
        logger.warning("[MULTI_TABLE] Phase3: réponse vide")
        return _UNKNOWN

    try:
        json_str = _extract_json_object(raw)
        if not json_str:
            logger.warning("[MULTI_TABLE] Phase3: aucun JSON — raw=%r", raw[:200])
            return _UNKNOWN

        payload = _safe_load_llm_json(json_str)
        if payload is None:
            logger.warning("[MULTI_TABLE] Phase3: JSON invalide — raw=%r", raw[:200])
            return _UNKNOWN

        table_type = payload.get("table_type", "AUTRE")
        _VALID_TYPES = {"BILAN ACTIF", "BILAN PASSIF", "COMPTE DE PRODUITS ET CHARGES", "AUTRE"}
        if table_type not in _VALID_TYPES:
            table_type = "AUTRE"

        return {"table_type": table_type}

    except Exception as exc:
        logger.warning("[MULTI_TABLE] Phase3: parse échoué: %s — raw=%r", exc, raw[:200])
        return _UNKNOWN


def _table_name_matches_type(table_name: str, table_type: str) -> bool:
    """Correspondance flexible entre un nom de tableau demandé et un type classifié."""
    name = table_name.upper().strip()
    typ = table_type.upper().strip()
    return name == typ or name in typ or typ in name


def _detect_crop_classify_tables(
    pdf_path: str,
    page_num: int,
    table_name: str,
    api_provider: str = "groq",
    api_key: Optional[str] = None,
) -> dict:
    """
    Phase 4 — Orchestrateur du pipeline simplifié.

    Flux :
    1. Détection visuelle de tous les tableaux (page complète → liste de bboxes).
    2. Crop local de chaque bbox détectée.
    3. Classification de chaque crop (BILAN ACTIF / BILAN PASSIF / CPC / AUTRE).
    4. Sélection du crop correspondant à table_name.

    Returns:
        {
            "table_found":         bool,
            "selected_table_type": str,
            "bbox_norm":           [x0, y0, x1, y1] | None,
            "bbox_px":             [x0_px, y0_px, x1_px, y1_px] | None,
            "crop_image":          PIL.Image | None,
        }
    """
    _NOT_FOUND: dict = {
        "table_found": False,
        "selected_table_type": "AUTRE",
        "bbox_norm": None,
        "bbox_px": None,
        "crop_image": None,
    }

    provider = _normalize_api_provider(api_provider)
    resolved_key = _provider_api_key(provider, api_key=api_key)

    logger.info(
        "[MULTI_TABLE] full page detection started page=%d table=%r provider=%s",
        page_num, table_name, provider,
    )

    # ── Rendu unique de la page complète ─────────────────────────────────────
    # Rendu en premier, avant Phase 1, pour pouvoir sauvegarder l'image
    # quelle que soit la suite (même si Phase 1 ne trouve rien).
    dpi, enhance = _choose_dpi(pdf_path, page_num)
    full_image = _render_pdf_page_high_quality(
        pdf_path=pdf_path,
        page_num=page_num,
        dpi=dpi,
        enhance=enhance,
    )
    if full_image is None:
        logger.warning("[MULTI_TABLE] rendu page échoué page=%d → fallback", page_num)
        return _NOT_FOUND

    # Sauvegarde immédiate de la page complète — utilise _debug_save_image
    # (même mécanisme que VISION_DEBUG_SAVE_INPUTS, chemin connu dans trash/).
    import pathlib as _pathlib
    _debug_dir = _pathlib.Path(VISION_DEBUG_OUTPUT_DIR).resolve()
    _debug_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[MULTI_TABLE] dossier debug → %s", _debug_dir)

    _full_path = _debug_dir / f"page_{page_num:03d}_full.png"
    try:
        if MULTI_TABLE_DEBUG_SAVE_IMAGES:
            full_image.save(str(_full_path), format="PNG")
        logger.info("[MULTI_TABLE] page complète sauvegardée → %s", _full_path)
    except Exception as _e:
        logger.warning("[MULTI_TABLE] sauvegarde page complète échouée: %s", _e)

    # Phase 1 — Détection de toutes les zones de tableaux
    # On passe full_image déjà rendue pour éviter un second rendu PDF.
    detection_result = _vision_detect_tables_on_full_page(
        pdf_path=pdf_path,
        page_num=page_num,
        api_provider=provider,
        api_key=resolved_key,
        full_image=full_image,
    )
    table_bboxes = detection_result.get("tables", [])

    if not table_bboxes:
        logger.warning(
            "[MULTI_TABLE] Phase1: aucun tableau détecté page=%d table=%r",
            page_num, table_name,
        )
        return _NOT_FOUND

    logger.info(
        "[MULTI_TABLE] detected_tables_count=%d page=%d", len(table_bboxes), page_num
    )

    # Phase 2 — Crop local de chaque tableau détecté
    crops = _crop_detected_tables(
        image=full_image,
        table_bboxes=table_bboxes,
        output_dir=None,
    )

    if not crops:
        logger.warning("[MULTI_TABLE] Phase2: aucun crop valide page=%d", page_num)
        return _NOT_FOUND

    # Sauvegarde des crops bruts (avant classification)
    for _ci, _crop in enumerate(crops):
        _crop_path = _debug_dir / f"page_{page_num:03d}_crop_{_ci + 1:02d}.png"
        try:
            if MULTI_TABLE_DEBUG_SAVE_CROPS:
                _crop["image"].save(str(_crop_path), format="PNG")
            logger.info("[MULTI_TABLE] crop_%02d sauvegardé → %s", _ci + 1, _crop_path)
        except Exception as _e:
            logger.warning("[MULTI_TABLE] sauvegarde crop_%02d échouée: %s", _ci + 1, _e)

    # Phase 3 — Classification de chaque crop
    classified: List[dict] = []
    for crop_info in crops:
        section_type = crop_info.get("section_type")
        section_to_label = {
            "BILAN_ACTIF": "BILAN ACTIF",
            "BILAN_PASSIF": "BILAN PASSIF",
            "CPC": "COMPTE DE PRODUITS ET CHARGES",
        }
        if section_type in section_to_label:
            table_type = section_to_label[str(section_type)]
        else:
            type_result = _vision_classify_cropped_table(
                crop_image=crop_info["image"],
                api_provider=provider,
                api_key=resolved_key,
            )
            table_type = type_result["table_type"]
        crop_text = _extract_text_from_bbox_px(
            pdf_path=pdf_path,
            page_num=page_num,
            bbox_px=crop_info["bbox_px"],
            image_size=full_image.size,
            padding_px=MULTI_TABLE_CROP_PADDING_PX * 2,
        )
        anchor_score, anchor_details = _score_text_for_table(crop_text, table_name)
        type_match = _table_name_matches_type(table_name, table_type)
        target_section = _target_section_type(table_name)
        section_match = bool(section_type and target_section and section_type == target_section)
        selection_score = anchor_score + (75.0 if type_match else 0.0) + (120.0 if section_match else 0.0)
        classified.append({
            **crop_info,
            "table_type": table_type,
            "section_match": section_match,
            "anchor_score": anchor_score,
            "selection_score": selection_score,
            "anchor_details": anchor_details,
            "text_sample": crop_text[:500],
        })
        logger.info(
            "[MULTI_TABLE] %s section=%s classified=%s section_match=%s type_match=%s anchor_score=%.1f selection_score=%.1f hits=%s neg=%s",
            crop_info["debug_name"],
            section_type,
            table_type,
            section_match,
            type_match,
            anchor_score,
            selection_score,
            anchor_details.get("positive_hits"),
            anchor_details.get("negative_hits"),
        )

    # Sauvegarde des crops avec leur label de classification
    for _ci, _crop in enumerate(classified):
        _label = _crop["table_type"].replace(" ", "_")
        _crop_labeled_path = _debug_dir / f"page_{page_num:03d}_crop_{_ci + 1:02d}_{_label}.png"
        try:
            if MULTI_TABLE_DEBUG_SAVE_CROPS:
                _crop["image"].save(str(_crop_labeled_path), format="PNG")
            logger.info(
                "[MULTI_TABLE] crop_%02d_%s → %s", _ci + 1, _label, _crop_labeled_path
            )
        except Exception as _e:
            logger.warning(
                "[MULTI_TABLE] sauvegarde crop_%02d_%s échouée: %s", _ci + 1, _label, _e
            )

    # Sélection du crop correspondant à table_name
    try:
        candidates_path = _debug_dir / f"page_{page_num:03d}_crop_candidates.json"
        candidates_payload = [
            {
                "debug_name": c.get("debug_name"),
                "section_type": c.get("section_type"),
                "section_match": c.get("section_match"),
                "detector_confidence": c.get("detector_confidence"),
                "visibility_status": c.get("visibility_status"),
                "table_type": c.get("table_type"),
                "bbox_px": c.get("bbox_px"),
                "anchor_score": c.get("anchor_score"),
                "selection_score": c.get("selection_score"),
                "anchor_details": c.get("anchor_details"),
                "text_sample": c.get("text_sample"),
            }
            for c in classified
        ]
        if MULTI_TABLE_DEBUG_SAVE_CROPS:
            candidates_path.write_text(
                json.dumps(candidates_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    except Exception as _e:
        logger.debug("[MULTI_TABLE] sauvegarde crop_candidates impossible: %s", _e)

    selected = max(
        classified,
        key=lambda c: float(c.get("selection_score") or -999.0),
        default=None,
    )

    if selected is None or float(selected.get("selection_score") or -999.0) < 20.0:
        logger.warning(
            "[MULTI_TABLE] aucun crop correspondant à table=%r parmi %s",
            table_name, [c["table_type"] for c in classified],
        )
        return _NOT_FOUND

    logger.info(
        "[MULTI_TABLE] selected crop type=%s section=%s bbox_px=%s score=%.1f",
        selected["table_type"],
        selected.get("section_type"),
        selected["bbox_px"],
        float(selected.get("selection_score") or 0.0),
    )

    target_section = _target_section_type(table_name)
    anchor_details = selected.get("anchor_details") or {}
    should_validate = bool(MULTI_TABLE_VALIDATE_SELECTED_CROP)
    if MULTI_TABLE_VALIDATE_ONLY_AMBIGUOUS:
        strong_semantic_crop = (
            bool(selected.get("section_match"))
            and float(selected.get("selection_score") or 0.0) >= 150.0
        )
        should_validate = False if strong_semantic_crop else (
            not bool(selected.get("section_match"))
            or float(selected.get("anchor_score") or 0.0) < 45.0
            or bool(anchor_details.get("negative_hits"))
        )

    validation = None
    if should_validate:
        validation = _vision_validate_selected_crop(
            crop_image=selected["image"],
            target_section_type=target_section,
            api_provider=provider,
            api_key=resolved_key,
        )
        try:
            validation_path = _debug_dir / f"page_{page_num:03d}_selected_crop_validation.json"
            validation_path.write_text(
                json.dumps(validation, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as _e:
            logger.debug("[MULTI_TABLE] sauvegarde crop validation impossible: %s", _e)

        detected_section = str(validation.get("detected_section_type") or "").strip().upper()
        recommended_action = str(validation.get("recommended_action") or "").strip().lower()
        crop_quality = str(validation.get("crop_quality") or "").strip().lower()
        contains_forbidden = bool(validation.get("contains_forbidden_content"))
        is_valid_crop = bool(validation.get("is_valid_crop"))
        if (
            not is_valid_crop
            or contains_forbidden
            or recommended_action == "reject"
            or crop_quality == "bad"
            or (target_section and detected_section and detected_section != target_section)
        ):
            logger.warning(
                "[MULTI_TABLE] selected crop rejected by validation target=%s detected=%s valid=%s forbidden=%s quality=%s action=%s details=%s",
                target_section,
                detected_section,
                is_valid_crop,
                contains_forbidden,
                crop_quality,
                recommended_action,
                validation,
            )
            return _NOT_FOUND
    else:
        logger.info(
            "[MULTI_TABLE] crop validation skipped (strong semantic detection) section=%s score=%.1f",
            selected.get("section_type"),
            float(selected.get("selection_score") or 0.0),
        )

    return {
        "table_found": True,
        "selected_table_type": selected["table_type"],
        "bbox_norm": selected["bbox_norm"],
        "bbox_px": selected["bbox_px"],
        "crop_image": selected["image"],
        "crop_validation": validation,
    }


def _simple_vision_extract(
    pdf_path: str,
    page_num: int,
    table_name: str,
    api_provider: str = "groq",
    api_key: Optional[str] = None,
    type_comptes: Optional[str] = None,
    secteur: Optional[str] = None,
) -> pd.DataFrame:
    """
    Extraction directe : page PDF → image PNG (HQ adaptatif) → LLM vision → DataFrame.

    Prompt JSON unique intelligent (via _build_vision_prompt) avec fallback Markdown.
    Providers supportés : groq (llama-4-maverick), openai (gpt-4o), gemini (2.0-flash).
    """
    provider = _normalize_api_provider(api_provider)
    api_key = _provider_api_key(provider, api_key=api_key)
    if not api_key:
        logger.warning("_simple_vision_extract: API key manquante pour provider=%s", provider)
        return pd.DataFrame()

    # ── Pipeline multi-tableaux simplifié (MULTI_TABLE_PAGE_MODE) ────────────
    # Flux : page complète → détection de toutes les zones → crop local →
    # classification de chaque crop → sélection du bon crop → extraction
    # uniquement sur l'image cropée. Jamais sur la page complète.
    if MULTI_TABLE_PAGE_MODE:
        try:
            logger.info(
                "[MULTI_TABLE] full page detection started page=%d table=%r",
                page_num, table_name,
            )
            detect_result = _detect_crop_classify_tables(
                pdf_path=pdf_path,
                page_num=page_num,
                table_name=table_name,
                api_provider=provider,
                api_key=api_key,
            )
            if detect_result["table_found"] and detect_result["crop_image"] is not None:
                crop_b64 = _img_to_b64(detect_result["crop_image"])
                if crop_b64:
                    logger.info(
                        "[MULTI_TABLE] extraction launched on selected crop "
                        "type=%s bbox_px=%s",
                        detect_result["selected_table_type"], detect_result["bbox_px"],
                    )
                    extraction_prompt = _build_vision_prompt(
                        table_name=table_name,
                        type_comptes=type_comptes,
                        secteur=secteur,
                    )
                    raw_mt = _call_vision_llm(
                        img_b64=crop_b64,
                        prompt=extraction_prompt,
                        provider=provider,
                        api_key=api_key,
                    )
                    if raw_mt:
                        df_mt = _vision_parse_response(
                            content=raw_mt,
                            table_name=table_name,
                            pdf_path=pdf_path,
                            page_num=page_num,
                        )
                        ok_mt, completeness_mt = _is_acceptable_extraction(df_mt, table_name)
                        if df_mt is not None and not df_mt.empty and ok_mt:
                            logger.info(
                                "_simple_vision_extract [MULTI_TABLE]: OK page=%d shape=%s completeness=%.2f",
                                page_num,
                                df_mt.shape,
                                float(completeness_mt.get("completeness_score") or 0.0),
                            )
                            return df_mt
                        if df_mt is not None and not df_mt.empty:
                            logger.warning(
                                "_simple_vision_extract [MULTI_TABLE]: crop extrait mais incomplet page=%d shape=%s completeness=%s missing=%s",
                                page_num,
                                df_mt.shape,
                                completeness_mt.get("completeness_score"),
                                completeness_mt.get("missing_anchors"),
                            )
            logger.warning(
                "[MULTI_TABLE] fallback on standard pipeline page=%d table=%r",
                page_num, table_name,
            )
        except Exception as _exc_mt:
            logger.warning(
                "[MULTI_TABLE] fallback on standard pipeline page=%d table=%r: %s",
                page_num, table_name, _exc_mt,
            )

    # ── Mode localisation seule (VISION_LOCALIZATION_ONLY=True) ─────────────────
    if VISION_LOCALIZATION_ONLY:
        loc = _vision_locate_table_only(
            pdf_path=pdf_path,
            page_num=page_num,
            table_name=table_name,
            api_provider=provider,
            api_key=api_key,
        )
        if loc["table_found"]:
            logger.info(
                "_simple_vision_extract [localization_only]: table=%r "
                "bbox_norm=%s bbox_px=%s",
                table_name, loc["bbox_norm"], loc["bbox_px"],
            )
        else:
            logger.warning(
                "_simple_vision_extract [localization_only]: tableau non localisé — "
                "page=%d table=%r — fallback: bbox_norm=None bbox_px=None",
                page_num, table_name,
            )
        return pd.DataFrame()  # aucune extraction, aucun crop, aucun split

    # ── Pipeline coarse-to-fine (désactivé par défaut, VISION_COARSE_TO_FINE=True pour activer) ──
    if VISION_COARSE_TO_FINE:
        logger.info(
            "_simple_vision_extract: mode coarse-to-fine activé page=%d table=%r",
            page_num, table_name,
        )
        df_c2f = _coarse_to_fine_vision_extract(
            pdf_path=pdf_path,
            page_num=page_num,
            table_name=table_name,
            api_provider=provider,
            api_key=api_key,
            type_comptes=type_comptes,
            secteur=secteur,
        )
        if _is_valid_table(df_c2f) or _is_usable_llm_table(df_c2f):
            logger.info(
                "_simple_vision_extract: coarse-to-fine OK page=%d shape=%s",
                page_num, df_c2f.shape,
            )
            return df_c2f
        logger.warning(
            "_simple_vision_extract: coarse-to-fine échoué page=%d → fallback isolation standard",
            page_num,
        )
        # Fallthrough vers le pipeline d'isolation standard ci-dessous

    # Pipeline d'isolation : si la page contient plusieurs tableaux, extrait uniquement
    # la zone du tableau cible avant d'envoyer l'image au Vision LLM.
    # Fallback vers page entière si isolation impossible (titre non trouvé, mono-tableau…).
    # Default production path: send the full page, not a crop. Crops can miss
    # narrow columns or bottom rows; the prompt isolates the requested section.
    if VISION_MULTI_TABLE_ISOLATION:
        img_b64, _isolation_info = extract_target_table_only(
            pdf_path=pdf_path,
            page_num=page_num,
            table_name=table_name,
            debug_save=VISION_DEBUG_MULTI_TABLE_ISOLATION,
        )
        logger.info(
            "_simple_vision_extract: isolation=%s multi=%s target_found=%s method=%s",
            VISION_MULTI_TABLE_ISOLATION,
            _isolation_info.get("is_multi_table"),
            _isolation_info.get("target_found"),
            _isolation_info.get("method"),
        )
    else:
        # Comportement legacy : pipeline haute qualité classique sans isolation.
        img_b64 = _prepare_vision_image(
            pdf_path=pdf_path,
            page_num=page_num,
            table_name=table_name,
            enable_crop=False,
        )

    if not img_b64:
        logger.warning(
            "_simple_vision_extract: impossible de préparer l'image pour la page %d", page_num
        )
        return pd.DataFrame()

    # Prompt unique intelligent (famille ACTIF/PASSIF/CPC + contexte type_comptes/secteur)
    prompt = _build_vision_prompt(
        table_name=table_name,
        type_comptes=type_comptes,
        secteur=secteur,
    )

    try:
        # ── Passe 1 : extraction complète du tableau ──────────────────────────
        content = _call_vision_llm(img_b64=img_b64, prompt=prompt,
                                   provider=provider, api_key=api_key)
        if not content:
            logger.warning("_simple_vision_extract: réponse vide du provider=%s", provider)
            return pd.DataFrame()

        logger.debug(
            "_simple_vision_extract: réponse brute provider=%s longueur=%d extrait=%r",
            provider, len(content), content[:120],
        )

        df = _vision_parse_response(
            content=content,
            table_name=table_name,
            pdf_path=pdf_path,
            page_num=page_num,
        )
        df = _correct_values_from_pdf_source(
            df,
            pdf_path=pdf_path,
            page_num=page_num,
            table_name=table_name,
        )
        ok_df, completeness_df = _is_acceptable_extraction(df, table_name)
        if not ok_df:
            logger.warning(
                "_simple_vision_extract: parsing échoué provider=%s page=%d — "
                "réponse brute LLM (500 premiers chars): %r",
                provider, page_num, content[:500],
            )
            return pd.DataFrame()

        logger.info(
            "_simple_vision_extract passe-1 OK provider=%s page=%d shape=%s",
            provider, page_num, df.shape,
        )

        # Passe 2 désactivée : le LLM confondait les lignes adjacentes et déplaçait
        # des valeurs vers des lignes vides voisines (hallucination de row-shift).
        # Le prompt de la 1ère passe (règle 2 : "garde toutes les lignes même vides")
        # est suffisant — une cellule vide reste vide, on ne tente pas de la remplir.

        logger.info(
            "_simple_vision_extract OK provider=%s page=%d shape_finale=%s",
            provider, page_num, df.shape,
        )
        return df

    except Exception as e:
        logger.warning("_simple_vision_extract failed provider=%s page=%d: %s", provider, page_num, e)
        return pd.DataFrame()


@dataclass
class ExtractionResult:
    """Résultat d'une extraction de tableau."""

    df: pd.DataFrame
    method: str  # "vision_groq" | "vision_openai" | "vision_gemini" | "pdfplumber_fallback" | "camelot_fallback" | "tabula_fallback" | "failed"
    confidence: float  # 0.0 -> 1.0
    success: bool
    warnings: List[str] = field(default_factory=list)
    completeness_score: float = 0.0
    missing_anchors: List[str] = field(default_factory=list)
    extraction_strategy_used: str = ""
    fallback_used: bool = False
    extraction_warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        rows, cols = self.df.shape if self.df is not None and not self.df.empty else (0, 0)
        return f"Extraction {self.method} -> {rows}x{cols} (confiance={self.confidence:.2f})"


class ExtractionAgent:
    """
    Extraction par vision LLM : page PDF → image → LLM → DataFrame.
    Provider choisi par l'utilisateur (groq / openai / gemini).
    Fallback technique (pdfplumber → camelot → tabula) si la vision échoue.
    """

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    def extract(
        self,
        pdf_path: str,
        page_num: int,
        table_name: str,
        api_provider: str = "groq",
        type_comptes: Optional[str] = None,
        secteur: Optional[str] = None,
    ) -> ExtractionResult:
        """
        Extrait le tableau `table_name` depuis la page `page_num` du PDF.

        Stratégie :
          1. Page → image PNG → LLM vision (provider choisi par l'utilisateur)
          2. Si échec : fallback technique sans LLM (pdfplumber → camelot → tabula)

        Returns:
            ExtractionResult avec .df, .method, .confidence, .success, .summary()
        """
        warnings: List[str] = []

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass

        provider = _normalize_api_provider(api_provider)
        method_label = f"vision_{provider}"

        # ── Pipeline multi-tableaux simplifié (MULTI_TABLE_PAGE_MODE) ──────────
        # Flux : page complète → détection visuelle → crop local → classification
        # → sélection du bon crop → extraction uniquement sur le crop.
        # Jamais d'extraction sur la page complète.
        if MULTI_TABLE_EAGER_EXTRACT and MULTI_TABLE_PAGE_MODE and _provider_api_key(provider):
            try:
                api_key_mt = _provider_api_key(provider)
                detect_result = _detect_crop_classify_tables(
                    pdf_path=pdf_path,
                    page_num=page_num,
                    table_name=table_name,
                    api_provider=provider,
                    api_key=api_key_mt,
                )
                if detect_result["table_found"] and detect_result["crop_image"] is not None:
                    crop_b64 = _img_to_b64(detect_result["crop_image"])
                    if crop_b64:
                        logger.info(
                            "[MULTI_TABLE] extraction launched on selected crop "
                            "type=%s bbox_px=%s",
                            detect_result["selected_table_type"], detect_result["bbox_px"],
                        )
                        extraction_prompt = _build_vision_prompt(
                            table_name=table_name,
                            type_comptes=type_comptes,
                            secteur=secteur,
                        )
                        raw_mt = _call_vision_llm(
                            img_b64=crop_b64,
                            prompt=extraction_prompt,
                            provider=provider,
                            api_key=api_key_mt,
                        )
                        if raw_mt:
                            df_mt = _vision_parse_response(
                                content=raw_mt,
                                table_name=table_name,
                                pdf_path=pdf_path,
                                page_num=page_num,
                            )
                            ok_mt, completeness_mt = _is_acceptable_extraction(df_mt, table_name)
                            if df_mt is not None and not df_mt.empty and ok_mt:
                                logger.info(
                                    "ExtractionAgent.extract [MULTI_TABLE]: OK page=%d shape=%s completeness=%.2f",
                                    page_num,
                                    df_mt.shape,
                                    float(completeness_mt.get("completeness_score") or 0.0),
                                )
                                confidence = 0.92 if _is_valid_table(df_mt) else 0.72
                                return ExtractionResult(
                                    df=df_mt,
                                    method=f"multi_table_{provider}",
                                    confidence=confidence,
                                    success=True,
                                    warnings=[],
                                    completeness_score=float(completeness_mt.get("completeness_score") or 0.0),
                                    missing_anchors=list(completeness_mt.get("missing_anchors") or []),
                                    extraction_strategy_used="multi_table_page_mode",
                                    fallback_used=False,
                                    extraction_warnings=[],
                                )
                            if df_mt is not None and not df_mt.empty:
                                warnings.append(
                                    f"Multi-table crop rejeté: complétude={completeness_mt.get('completeness_score')} "
                                    f"ancres_manquantes={completeness_mt.get('missing_anchors')}"
                                )
                logger.warning(
                    "[MULTI_TABLE] fallback on standard pipeline page=%d table=%r",
                    page_num, table_name,
                )
            except Exception as _exc_mt:
                logger.warning(
                    "[MULTI_TABLE] fallback on standard pipeline page=%d table=%r: %s",
                    page_num, table_name, _exc_mt,
                )

        # ── Mode localisation seule — aucune extraction, aucun fallback ──────
        if VISION_LOCALIZATION_ONLY:
            loc = _vision_locate_table_only(
                pdf_path=pdf_path,
                page_num=page_num,
                table_name=table_name,
                api_provider=provider,
            )
            logger.info(
                "ExtractionAgent.extract [localization_only]: table=%r found=%s "
                "bbox_norm=%s bbox_px=%s",
                table_name, loc["table_found"], loc["bbox_norm"], loc["bbox_px"],
            )
            return ExtractionResult(
                df=pd.DataFrame(),
                method="localization_only",
                confidence=0.0,
                success=False,
                warnings=[
                    f"VISION_LOCALIZATION_ONLY=True — aucune extraction effectuée. "
                    f"bbox_norm={loc['bbox_norm']} bbox_px={loc['bbox_px']}"
                ],
                extraction_strategy_used="localization_only",
                fallback_used=False,
                extraction_warnings=[],
            )

        # ── Étape 1 : Vision LLM ─────────────────────────────────────────────
        if self.use_llm and _provider_api_key(provider):
            df = _simple_vision_extract(
                pdf_path=pdf_path,
                page_num=page_num,
                table_name=table_name,
                api_provider=provider,
                type_comptes=type_comptes,
                secteur=secteur,
            )
            ok_vision, completeness_vision = _is_acceptable_extraction(df, table_name)
            if ok_vision:
                confidence = 0.92 if _is_valid_table(df) else 0.72
                if not _is_valid_table(df):
                    warnings.append(
                        f"Vision LLM ({provider}) a produit un tableau partiel accepté ({df.shape[0]} ligne(s))."
                    )
                logger.info(
                    "ExtractionAgent.extract: vision OK provider=%s page=%d shape=%s",
                    provider, page_num, df.shape,
                )
                return ExtractionResult(
                    df=df,
                    method=method_label,
                    confidence=confidence,
                    success=True,
                    warnings=warnings,
                    completeness_score=float(completeness_vision.get("completeness_score") or 0.0),
                    missing_anchors=list(completeness_vision.get("missing_anchors") or []),
                    extraction_strategy_used=method_label,
                    fallback_used=False,
                    extraction_warnings=warnings.copy(),
                )
            if df is not None and not df.empty:
                warnings.append(
                    f"Vision LLM ({provider}) rejetée: complétude={completeness_vision.get('completeness_score')} "
                    f"ancres_manquantes={completeness_vision.get('missing_anchors')}."
                )
            warnings.append(f"Vision LLM ({provider}) n'a pas produit de tableau valide. Passage au fallback technique.")
            logger.warning(
                "ExtractionAgent.extract: vision failed provider=%s page=%d — fallback technique",
                provider, page_num,
            )
        elif self.use_llm:
            warnings.append(f"API key manquante pour provider={provider}. Passage au fallback technique.")

        if self.use_llm and _provider_api_key(provider):
            logger.info("ExtractionAgent.extract: essai LLM texte structurÃ© provider=%s page=%d", provider, page_num)
            df = _llm_structured_extract(
                pdf_path=pdf_path,
                page_num=page_num,
                table_name=table_name,
                api_provider=provider,
            )
            ok_text, completeness_text = _is_acceptable_extraction(df, table_name)
            if ok_text:
                warnings.append("Fallback LLM texte structurÃ© utilisÃ©.")
                return ExtractionResult(
                    df=df,
                    method=f"structured_text_{provider}",
                    confidence=0.78,
                    success=True,
                    warnings=warnings,
                    completeness_score=float(completeness_text.get("completeness_score") or 0.0),
                    missing_anchors=list(completeness_text.get("missing_anchors") or []),
                    extraction_strategy_used=f"structured_text_{provider}",
                    fallback_used=True,
                    extraction_warnings=warnings.copy(),
                )
            if df is not None and not df.empty:
                warnings.append(
                    f"LLM texte structurÃ© rejetÃ©: complÃ©tude={completeness_text.get('completeness_score')} "
                    f"ancres_manquantes={completeness_text.get('missing_anchors')}."
                )

        # ── Étape 2 : Fallbacks techniques (sans LLM) ────────────────────────
        logger.info("ExtractionAgent.extract: essai pdfplumber page=%d", page_num)
        df = _pdfplumber_extract(pdf_path, page_num)
        if _is_valid_table(df):
            warnings.append("Fallback technique utilisé : pdfplumber.")
            logger.info("ExtractionAgent.extract: pdfplumber OK shape=%s", df.shape)
            return ExtractionResult(
                df=df,
                method="pdfplumber_fallback",
                confidence=0.55,
                success=True,
                warnings=warnings,
                extraction_strategy_used="pdfplumber_fallback",
                fallback_used=True,
                extraction_warnings=warnings.copy(),
            )

        logger.info("ExtractionAgent.extract: essai camelot page=%d", page_num)
        df = _camelot_extract(pdf_path, page_num, table_name)
        if _is_valid_table(df):
            warnings.append("Fallback technique utilisé : camelot.")
            logger.info("ExtractionAgent.extract: camelot OK shape=%s", df.shape)
            return ExtractionResult(
                df=df,
                method="camelot_fallback",
                confidence=0.60,
                success=True,
                warnings=warnings,
                extraction_strategy_used="camelot_fallback",
                fallback_used=True,
                extraction_warnings=warnings.copy(),
            )

        logger.info("ExtractionAgent.extract: essai tabula page=%d", page_num)
        df = _tabula_extract(pdf_path, page_num)
        if _is_valid_table(df):
            warnings.append("Fallback technique utilisé : tabula.")
            logger.info("ExtractionAgent.extract: tabula OK shape=%s", df.shape)
            return ExtractionResult(
                df=df,
                method="tabula_fallback",
                confidence=0.58,
                success=True,
                warnings=warnings,
                extraction_strategy_used="tabula_fallback",
                fallback_used=True,
                extraction_warnings=warnings.copy(),
            )

        logger.error("ExtractionAgent.extract: toutes les méthodes ont échoué page=%d", page_num)
        return ExtractionResult(
            df=pd.DataFrame(),
            method="failed",
            confidence=0.0,
            success=False,
            warnings=warnings + ["Aucune méthode n'a produit de tableau valide."],
            extraction_strategy_used="failed",
        )


def find_page_for_table(
    pdf_name: str,
    table_name: str,
    type_rapport: Optional[str] = None,
) -> Optional[int]:
    """
    Localise la page d'un tableau financier.

    Stratégie :
    1. Localiseur déterministe PyMuPDF (page_locator) — prioritaire, sans RAG.
    2. Fallback RAG (_find_page_by_rag) si le score déterministe est trop faible.
    """
    # --- 1. Résolution du chemin PDF ---
    try:
        from .config import RAGConfig
        cfg = RAGConfig()
        pdf_abs = _resolve_pdf_absolute_path(pdf_name, cfg)
    except Exception:
        pdf_abs = None

    # --- 2. Localiseur déterministe ---
    if pdf_abs and os.path.isfile(pdf_abs):
        try:
            from .page_locator import find_page_by_keywords

            name_l = _norm(table_name).replace("\n", " ")
            is_actif  = "actif" in name_l and "passif" not in name_l
            is_passif = "passif" in name_l
            is_cpc    = (
                "cpc" in name_l
                or "compte de produits" in name_l
                or "produits et charges" in name_l
                or "compte de r\u00e9sultat" in name_l
                or "compte de resultat" in name_l
            )
            is_consolid = "consolid" in name_l

            if is_actif:
                ttype = "ACTIF"
            elif is_passif:
                ttype = "PASSIF"
            elif is_cpc:
                ttype = "CPC"
            else:
                ttype = None

            if ttype:
                rtype_arg = type_rapport or ("consolidés" if is_consolid else "auto")
                rtype_norm = _norm(rtype_arg or "")
                if "consolid" in rtype_norm:
                    rtype: str = "consolidés"
                elif "sociaux" in rtype_norm or "social" in rtype_norm:
                    rtype = "sociaux"
                else:
                    rtype = "auto"

                result = find_page_by_keywords(pdf_abs, ttype, rtype, top_k=1)  # type: ignore
                if result.best_page is not None and result.score >= 30:
                    logger.info(
                        "find_page_for_table: localiseur déterministe → page %d (score=%d, table=%s)",
                        result.best_page, result.score, ttype,
                    )
                    return result.best_page
        except Exception as exc:
            logger.warning("find_page_for_table: localiseur déterministe échoué (%s), fallback RAG", exc)

    # --- 3. Fallback RAG ---
    logger.info("find_page_for_table: fallback RAG pour %r", table_name)
    return _find_page_by_rag(pdf_name, table_name, type_rapport=type_rapport)


def _FINANCIAL_SIGNATURES_DICT() -> dict:
    """Signatures financieres exhaustives par secteur (lazy init)."""
    return {
        # CGNC Comptes Sociaux
        "bilan_actif_cgnc": {
            "anchors": [
                ("immobilisations en non-valeur", 20),
                ("immobilisation en non valeurs", 22),
                ("actif circulant (hors trésorerie)", 16),
                ("trésorerie - actif", 14),
                ("tresorerie actif", 14),
                ("actif immobilisé", 10),
                ("actif circulant", 12),
                ("total de l'actif", 10),
                ("créances de l'actif circulant", 12),
                ("stocks", 5),
                # Formulations fréquentes RFA marocaines / PDF sans ligatures exactes CGNC
                ("actif (en mad)", 18),
                ("créances actif circulant", 14),
                ("total général actif", 16),
                # Lignes CGNC standard (A) à (I)
                ("immobilisations incorporelles", 10),
                ("immobilisations corporelles", 10),
                ("immobilisations financières", 10),
                ("immobilisations financieres", 10),
                ("écarts de conversion - actif", 14),
                ("ecarts de conversion - actif", 14),
                ("ecart de conversion actif", 14),
                ("titres et valeurs de placement", 10),
            ],
            "titles": [("bilan actif", 16), ("actif au 31", 8)],
            "context": [
                ("brut", 4), ("amortissements et provisions", 6), ("net", 3),
                ("31/12", 5), ("milliers de dirhams", 4), ("kdh", 4), ("comptes sociaux", 5),
                ("total i", 8), ("total ii", 8), ("total iii", 8), ("total general", 10),
                ("autres", 4),
            ],
            "penalties": [
                ("goodwill", -20), ("total actif non courant", -20),
                ("produit net bancaire", -18), ("intérêts et produits assimilés", -18),
                ("valeurs en caisse, banques centrales", -18),
                ("dettes de financement", -16), ("passif circulant", -14),
                ("résultat non courant", -14), ("produits d'exploitation", -10),
                ("charges de sinistres", -18), ("primes émises", -18),
                ("capitaux propres - part du groupe", -20),
                ("intérêts minoritaires", -20), ("résultat net - part du groupe", -20),
                # Tableaux d'amortissements (souvent confondus avec le bilan actif par similarité vectorielle)
                ("montant brut debut", -25),
                ("montant brut début", -25),
                ("dotations de l'exercice", -22),
                ("cumul d'amortissement", -25),
                ("amortissements sur immobilisations sorties", -25),
            ],
        },
        "bilan_passif_cgnc": {
            "anchors": [
                ("capitaux propres", 16),
                ("capitaux propres assimiles", 16),
                ("dettes de financement", 18),
                ("passif circulant (hors trésorerie)", 16),
                ("passif circulant", 14),
                ("report à nouveau", 10),
                ("total du passif", 10),
                ("capitaux propres assimilés", 12),
                ("primes d'émission, de fusion, d'apport", 10),
                ("trésorerie - passif", 12),
                ("tresorerie passif", 12),
                # Lignes CGNC standard (A) à (F) + trésorerie
                ("total des capitaux propres", 14),
                ("provisions durables pour risques et charges", 16),
                ("écarts de conversion - passif", 14),
                ("ecarts de conversion - passif", 14),
                ("ecart de conversion passif", 14),
                ("dettes du passif circulant", 12),
                ("autres provisions pour risques et charges", 14),
                ("tresorerie - passif", 12),
            ],
            "titles": [
                ("bilan passif", 16),
                ("passif au 31", 8),
                ("passif (en mad)", 14),
                ("passif", 6),
            ],
            "context": [
                ("capitaux propres", 5), ("31/12", 5),
                ("milliers de dirhams", 4), ("kdh", 4), ("comptes sociaux", 5),
                ("total i", 8), ("total ii", 8), ("total iii", 8), ("total general", 10),
                ("autres", 4),
            ],
            "penalties": [
                ("goodwill", -20), ("immobilisations en non-valeur", -20),
                ("actif circulant (hors trésorerie)", -18),
                ("valeurs en caisse, banques centrales", -18),
                ("produit net bancaire", -18), ("intérêts et produits assimilés", -18),
                ("dettes envers les établissements de crédit et assimilés", -14),
                ("résultat non courant", -12), ("produits d'exploitation", -10),
                ("charges de sinistres", -18), ("primes émises", -18),
                ("capitaux propres - part du groupe", -18),
                ("intérêts minoritaires", -18), ("résultat net - part du groupe", -18),
                # Termes consolidés — ne doivent pas apparaître dans un bilan social CGNC
                ("reserves consolidees", -30),
                ("resultats consolides", -30),
                ("capitaux propres d'ensemble", -30),
                ("capitaux propres part des minoritaires", -28),
                ("capitaux propres part du groupe", -25),
                ("reserves minoritaires", -25),
                ("resultat minoritaire", -25),
                ("dettes financieres non courantes", -20),
                ("total des passifs non courants", -20),
            ],
        },
        "cpc_cgnc": {
            "anchors": [
                ("résultat non courant", 18),
                ("produits d'exploitation", 14),
                ("charges d'exploitation", 14),
                ("résultat d'exploitation", 12),
                ("resultat d'exploitation", 12),
                ("resultat d exploitation", 12),
                ("impôts sur les résultats", 12),
                ("impot sur les societes", 14),
                ("résultat financier", 10),
                ("resultat financier", 10),
                ("chiffre d'affaires", 8),
                ("résultat courant", 8),
                ("resultat courant", 8),
                # Lignes CGNC standard du CPC
                ("produits financiers", 10),
                ("charges financières", 10),
                ("charges financieres", 10),
                ("produits non courants", 12),
                ("charges non courantes", 12),
                ("résultat avant impôts", 12),
                ("resultat avant impots", 12),
                ("total des produits", 10),
                ("total des charges", 10),
                ("resultat net", 14),
            ],
            "titles": [
                ("comptes de produits et charges", 16),
                ("compte de produits et charges", 12),
                ("compte de produits et de charges", 12),
                ("cpc", 14),
            ],
            "context": [
                ("résultat net", 5), ("31/12", 5),
                ("milliers de dirhams", 4), ("kdh", 4), ("comptes sociaux", 5),
                ("total des produits", 8),
            ],
            "penalties": [
                ("goodwill", -18), ("immobilisations en non-valeur", -18),
                ("actif circulant", -14), ("total de l'actif", -18), ("total du passif", -18),
                ("dettes de financement", -14),
                ("valeurs en caisse, banques centrales", -18),
                ("produit net bancaire", -18), ("intérêts et produits assimilés", -18),
                ("capitaux propres - part du groupe", -18),
                ("résultat net - part du groupe", -18),
                ("résultat net - part des minoritaires", -18),
                ("charges de sinistres", -18), ("primes émises", -18),
            ],
        },
        # PCEC Bancaire
        "bilan_actif_pcec": {
            "anchors": [
                # Consolidés IFRS / banques — ancres fortes (cible page exacte)
                (
                    "valeurs en caisse, banques centrales, trésor public, ccp",
                    40,
                ),
                (
                    "valeurs en caisse, banques centrales, trésor public, service des chèques postaux",
                    40,
                ),
                ("valeurs en caisse, banques centrales", 22),
                ("actifs financiers à la juste valeur par résultat", 34),
                ("actifs financiers à la juste valeur par capitaux propres", 34),
                ("titres au coût amorti", 32),
                (
                    "prêts et créances sur les établissements de crédit et assimilés, au coût amorti",
                    36,
                ),
                ("prêts et créances sur la clientèle, au coût amorti", 34),
                ("actifs d'impôt exigible", 28),
                ("actifs d'impôt différé", 28),
                ("actifs d'impôts différés", 28),
                ("immobilisations corporelles", 24),
                ("immobilisations incorporelles", 24),
                ("écart d'acquisition", 30),
                ("total actif", 42),
                ("total de l'actif", 50),
                # PCEC — lignes classiques (complément)
                ("créances sur les établissements de crédit et assimilés", 22),
                ("créances sur la clientèle", 14),
                ("titres de transaction", 12),
                ("titres de placement", 12),
                ("titres d'investissement", 10),
                ("immobilisations données en crédit-bail", 10),
                ("créances acquises par affacturage", 16),
                ("titres de transaction et de placement", 14),
                ("autres actifs", 26),
                ("titres de participation et emplois assimilés", 16),
                ("créances subordonnées", 14),
                ("dépôts d'investissement et wakala bil istithmar placés", 20),
                ("dépôts d'investissement placés", 30),
                ("immobilisations données en crédit-bail et en location", 16),
                ("immobilisations données en ijara", 18),
            ],
            "titles": [("bilan actif", 8), ("actif au 31", 8)],
            "context": [
                ("brut", 4), ("amortissements et provisions", 6), ("net", 3),
                ("31/12", 5), ("milliers de dirhams", 4), ("kdh", 4),
            ],
            "penalties": [
                ("goodwill", -20), ("immobilisations en non-valeur", -18),
                ("actif circulant (hors trésorerie)", -18),
                ("produit net bancaire", -16),
                ("dettes envers les établissements de crédit et assimilés", -18),
                ("résultat non courant", -14),
                ("charges de sinistres", -18), ("primes émises", -18),
                ("capitaux propres - part du groupe", -20),
                ("total actif non courant", -20),
                ("résultat net - part du groupe", -20),
            ],
        },
        "bilan_passif_pcec": {
            "anchors": [
                ("banques centrales, trésor public, service des chèques postaux", 30),
                ("dettes envers les établissements de crédit et assimilés", 38),
                ("dépôts de la clientèle", 36),
                ("dettes envers la clientèle", 34),
                ("dettes envers la clientèle sur produits participatifs", 34),
                ("titres de créance émis", 36),
                ("autres passifs", 30),
                ("provisions pour risques et charges", 30),
                ("provisions réglementées", 30),
                ("subventions, fonds publics affectés et fonds spéciaux de garantie", 34),
                ("dettes subordonnées", 12),
                ("dépôts d'investissement reçus", 34),
                ("écarts de réévaluation", 30),
                ("réserves et primes liées au capital", 30),
                ("capital", 26),
                ("actionnaires capital non versé", 28),
                ("report à nouveau", 28),
                ("résultats nets en instance d'affectation", 30),
                ("passifs financiers à la juste valeur par résultat sur option", 34),
                ("passifs financiers à la juste valeur par résultat", 32),
                ("passifs financiers détenus à des fins de transactions", 34),
                ("instruments dérivés de couverture", 32),
                ("écart de réévaluation passif des portefeuilles couverts en taux", 32),
                ("passifs d'impôt exigible", 28),
                ("passifs d'impôts différés", 28),
                ("comptes de régularisation et autres passifs", 32),
                ("dettes liées aux actifs non courants destinés à être cédés", 32),
                ("passifs des contrats d'assurance", 30),
                ("subventions et fonds assimilés", 30),
                ("dettes subordonnées et fonds spéciaux de garantie", 34),
                ("capitaux propres", 26),
                ("réserves consolidées", 30),
                (
                    "gains et pertes comptabilisés directement en capitaux propres",
                    34,
                ),
                ("résultat net de l'exercice", 30),
                ("total du passif", 50),
                ("total passif", 42),
            ],
            "titles": [("bilan passif", 8), ("passif au 31", 8)],
            "context": [
                ("capitaux propres", 5), ("31/12", 5),
                ("milliers de dirhams", 4), ("kdh", 4),
            ],
            "penalties": [
                ("goodwill", -20), ("immobilisations en non-valeur", -18),
                ("actif circulant (hors trésorerie)", -18),
                ("valeurs en caisse, banques centrales", -18),
                ("produit net bancaire", -16),
                ("résultat non courant", -14),
                ("charges de sinistres", -18), ("primes émises", -18),
                ("capitaux propres - part du groupe", -20),
                ("total actif non courant", -20),
                ("résultat net - part du groupe", -20),
            ],
        },
        "cpc_pcec": {
            "anchors": [
                # Uniquement les lignes STRUCTURELLES du CPC principal (totaux et en-têtes de section)
                # — PAS les sous-détails qui apparaissent sur les pages d'annexes.
                ("produits d'exploitation bancaire", 22),
                ("charges d'exploitation bancaire", 22),
                ("produit net bancaire", 24),
                ("résultat brut d'exploitation", 18),
                ("charges générales d'exploitation", 16),
                ("dotations aux provisions et pertes sur créances irrécouvrables", 20),
                ("dotations aux provisions pour créances et engagements par signature en souffrance", 22),
                ("reprises de provisions et récupérations sur créances amorties", 20),
                ("résultat courant", 14),
                ("résultat avant impôts", 14),
                ("impôts sur les résultats", 14),
                ("résultat net de l'exercice", 16),
                # Lignes d'intérêts : très discriminantes pour les banques (PCEC)
                ("intérêts et produits assimilés", 22),
                (
                    "intérêts, rémunérations et produits assimilés sur opérations avec les établissements de crédit",
                    32,
                ),
                (
                    "intérêts et produits assimilés sur opérations avec la clientèle",
                    32,
                ),
                (
                    "intérêts et produits assimilés sur titres de créance",
                    32,
                ),
                ("intérêts et charges assimilées", 16),
                ("coefficient d'exploitation", 14),
                # Finance islamique — lignes structurelles uniquement
                ("produits sur titres de moudaraba et moucharaka", 20),
                ("charges sur titres de moudaraba et moucharaka", 20),
                ("produits sur immobilisations données en ijara", 20),
                ("charges sur immobilisations données en ijara", 20),
                ("transfert de charges sur dépôts d'investissement reçus", 20),
                ("transfert de produits sur dépôts d'investissement reçus", 20),
            ],
            "titles": [("compte de produits et charges", 10)],
            "context": [
                ("résultat net", 5), ("31/12", 5),
                ("milliers de dirhams", 4), ("kdh", 4),
            ],
            "penalties": [
                ("goodwill", -18), ("immobilisations en non-valeur", -18),
                ("actif circulant", -14), ("total de l'actif", -18), ("total du passif", -18),
                ("valeurs en caisse, banques centrales", -18),
                ("dettes envers les établissements de crédit et assimilés", -14),
                ("résultat non courant", -14),
                ("capitaux propres - part du groupe", -18),
                ("résultat net - part du groupe", -18),
                ("résultat net - part des minoritaires", -18),
                ("charges de sinistres", -18), ("primes émises", -18),
                # Pénalités pages de détail/annexes (sous-lignes qui n'apparaissent pas sur le CPC principal)
                ("divers autres produits bancaires", -20),
                ("divers autres charges bancaires", -20),
                ("divers charges sur titres de propriété", -20),
                ("frais d'émission des emprunts", -18),
                ("cotisation au fonds de garantie des dépôts", -18),
                ("reprise de provisions pour dépréciation des titres de placement", -18),
                ("dotations aux provisions pour dépréciation des titres de placement", -18),
                ("produits rétrocédés", -16),
                ("charges des exercices antérieurs", -16),
                ("produits des exercices antérieurs", -16),
                ("quote-part sur opérations bancaires faite en commun", -18),
            ],
        },
        # IFRS Consolide
        "bilan_actif_ifrs": {
            "anchors": [
                # Banques & SDF — bilan actif consolidé (libellés fréquents)
                (
                    "valeurs en caisse, banques centrales, trésor public, ccp",
                    36,
                ),
                (
                    "valeurs en caisse, banques centrales, trésor public, service des chèques postaux",
                    36,
                ),
                ("actifs financiers à la juste valeur par résultat", 32),
                ("actifs financiers à la juste valeur par capitaux propres", 32),
                ("titres au coût amorti", 30),
                (
                    "prêts et créances sur les établissements de crédit et assimilés, au coût amorti",
                    34,
                ),
                ("prêts et créances sur la clientèle, au coût amorti", 32),
                ("actifs d'impôt exigible", 26),
                ("actifs d'impôt différé", 26),
                ("actifs d'impôts différés", 26),
                ("immobilisations corporelles", 22),
                ("immobilisations incorporelles", 22),
                ("écart d'acquisition", 28),
                ("total actif", 34),
                ("goodwill", 18),
                ("total actif non courant", 18),
                ("total actif courant", 18),
                ("participations dans les sociétés mises en équivalence", 14),
                ("immeubles de placement", 12),
                ("autres actifs financiers non courants", 12),
                ("autres actifs financiers", 10),
                ("total actifs non courants", 16),
                ("créances clients", 12),
                ("stocks et encours", 12),
                ("total actifs courants", 16),
                ("total de l'actif", 12),
            ],
            "titles": [
                ("bilan actif consolidé", 12),
                ("état de la situation financière", 10),
                ("actif au 31", 8),
            ],
            "context": [("31/12", 5), ("notes", 4), ("consolid", 6)],
            "penalties": [
                ("immobilisations en non-valeur", -20),
                ("actif circulant (hors trésorerie)", -20),
                ("valeurs en caisse, banques centrales", -18),
                ("produit net bancaire", -18),
                ("résultat non courant", -18),
                ("dettes de financement", -20),
                ("passif circulant", -14),
                ("charges de sinistres", -18), ("primes émises", -18),
                ("capitaux propres - part du groupe", -12),
                ("intérêts minoritaires", -12),
            ],
        },
        "bilan_passif_ifrs": {
            "anchors": [
                # Assurance — Bilan Passif Consolidé (mots-clés à très fort signal)
                ("capital", 18),
                ("réserves consolidées", 28),
                ("résultat consolidé", 30),
                ("capitaux propres de l'ensemble consolidé", 34),
                ("dont : capitaux propres part du groupe", 34),
                ("capitaux propres - part du groupe", 22),
                ("intérêts minoritaires", 18),
                ("résultat net - part du groupe", 16),
                ("total passif non courant", 16),
                ("total passif courant", 16),
                ("dettes financières non courantes", 14),
                ("dettes de financement", 22),
                ("provisions techniques", 26),
                ("passif circulant", 24),
                ("dettes pour les espèces remises par les cessionnaires", 30),
                ("cessionnaires, cédants coassureurs et comptes rattachés créditeurs", 34),
                ("assurés, intermédiaires et comptes rattachés créditeurs", 34),
                ("autres dettes du passif circulant", 28),
                ("trésorerie - passif", 26),
                ("tresorerie - passif", 26),
                ("total capitaux propres", 12),
                # Lignes IFRS standard du bilan passif consolidé
                ("capital", 5),
                ("primes d'émission et de fusion", 12),
                ("réserves consolidées", 18),
                ("capitaux propres part du groupe", 18),
                ("capitaux propres part des minoritaires", 16),
                ("capitaux propres d'ensemble", 18),
                ("total des passifs non courants", 14),
                ("total dettes courantes", 14),
                ("total passif", 8),
            ],
            "titles": [("bilan passif consolidé", 12), ("passif au 31", 8)],
            "context": [("31/12", 5), ("notes", 4), ("consolid", 6)],
            "penalties": [
                ("goodwill", -16),
                ("total actif non courant", -20),
                ("immobilisations en non-valeur", -20),
                ("actif circulant (hors trésorerie)", -20),
                ("valeurs en caisse, banques centrales", -18),
                ("produit net bancaire", -18),
                ("résultat non courant", -18),
                ("charges de sinistres", -18), ("primes émises", -18),
                # Pénalités CGNC social
                ("financement permanent", -25),
                ("dettes du passif circulant", -22),
                ("provisions durables pour risques et charges", -20),
                ("passif circulant hors tresorerie", -18),
            ],
        },
        "bilan_passif_cgnc_consolide": {
            # Bilan Passif Consolidé — format CGNC adapté (sociétés non financières marocaines)
            # Ex: Addoha, Alliance Darna — présentent des CP consolidés avec part groupe / minoritaires
            # mais utilisent la nomenclature CGNC (pas IFRS pur, pas PCEC bancaire).
            "anchors": [
                ("reserves consolidees", 30),
                ("resultats consolides", 30),
                ("capitaux propres d'ensemble", 32),
                ("capitaux propres part du groupe", 28),
                ("capitaux propres part des minoritaires", 28),
                ("reserves minoritaires", 26),
                ("resultat minoritaire", 24),
                ("primes d'emission et de fusion", 22),
                ("dettes financieres non courantes", 22),
                ("total des passifs non courants", 22),
                ("dettes financieres courantes", 22),
                ("total dettes courantes", 22),
                ("provisions non courantes", 20),
                ("impot differe passif", 20),
                ("avantages au personnel", 18),
                ("autres passifs non courants", 18),
                ("provisions courantes", 16),
                ("autres passifs courants", 16),
                ("dettes fournisseurs", 14),
                ("total passif", 14),
                ("capital", 12),
                ("ecarts de conversion", 10),
            ],
            "titles": [
                ("bilan passif consolide", 20),
                ("passif consolide", 16),
                ("passif au 31", 8),
            ],
            "context": [
                ("consolid", 8), ("31/12", 5), ("30/06", 5),
                ("kdh", 4), ("mad", 3),
            ],
            "penalties": [
                # Termes purement sociaux CGNC
                ("financement permanent", -30),
                ("passif circulant hors tresorerie", -28),
                ("dettes du passif circulant", -25),
                ("provisions durables pour risques et charges", -25),
                ("ecarts de conversion -passif", -20),
                ("total general", -20),
                # Termes bancaires
                ("valeurs en caisse, banques centrales", -25),
                ("produit net bancaire", -25),
                ("dettes envers les etablissements de credit", -20),
                # Termes assurance
                ("provisions techniques", -20),
                ("primes emises", -20),
                ("charges de sinistres", -20),
                # Termes actif
                ("immobilisations en non-valeur", -25),
                ("actif circulant", -20),
                ("total actif non courant", -25),
                ("resultat non courant", -20),
            ],
        },
        "cpc_ifrs": {
            "anchors": [
                ("résultat net - part du groupe", 22),
                ("résultat net - part des minoritaires", 20),
                ("résultat opérationnel", 16),
                ("coût des ventes", 14),
                ("marge brute", 12),
                ("charge d'impôt sur le résultat", 12),
                ("résultat des activités poursuivies", 10),
                # Lignes IFRS standard du CPC consolidé
                ("chiffre d'affaires", 8),
                ("produits des activités ordinaires", 16),
                ("charges d'exploitation courantes", 14),
                ("résultat d'exploitation courant", 14),
                ("autres produits et charges d'exploitation", 14),
                ("résultat des activités opérationnelles", 16),
                ("résultat financier", 10),
                ("résultat avant impôts des entreprises intégrées", 18),
                ("résultat net des entreprises intégrées", 18),
                ("résultat net des activités poursuivies", 16),
                ("résultat de l'ensemble consolidé", 18),
            ],
            "titles": [
                ("compte de résultat consolidé", 14),
                ("compte de produits et charges consolidé", 12),
                ("état du résultat global", 12),
            ],
            "context": [("résultat net", 5), ("31/12", 5), ("consolid", 6)],
            "penalties": [
                ("total actif non courant", -20), ("immobilisations en non-valeur", -20),
                ("actif circulant", -16), ("total de l'actif", -20), ("total du passif", -20),
                ("valeurs en caisse, banques centrales", -18),
                ("dettes de financement", -14),
                ("résultat non courant", -12),
                ("charges de sinistres", -18), ("primes émises", -18),
            ],
        },
        # CPC IFRS Bancaire (Banques & SDF — comptes consolidés IFRS)
        # Format : Compte de résultat consolidé type Attijariwafa / BOA / CIH / BMCI
        "cpc_ifrs_bancaire": {
            "anchors": [
                # Marges structurelles (libellés en majuscules = titres de section → très discriminants)
                ("marge d'interet", 30),
                ("marge d'intérêt", 30),
                ("marge sur commissions", 30),
                ("produit net bancaire", 28),
                ("resultat brut d'exploitation", 22),
                # Variantes de "coût du risque" selon l'émetteur
                ("cout du risque de credit", 28),
                ("coût du risque de crédit", 28),
                ("cout du risque", 26),           # BOA, CIH, CDM : sans "de crédit"
                # Variantes d'impôts selon le plan comptable
                ("resultat avant impot", 18),     # match "avant impôt" ET "avant impôts"
                ("impots sur les benefices", 20),
                ("impôts sur les bénéfices", 20),
                ("impot sur les resultats", 18),   # PCEC sociaux / S1 BOA
                # Résultat net — plusieurs formulations selon émetteur
                ("resultat net part du groupe", 28),
                ("résultat net part du groupe", 28),
                ("resultat net de l'ensemble consolide", 24),  # variante IFRS
                ("part du groupe", 20),            # fallback court mais discriminant
                # Lignes de détail — très spécifiques au CPC bancaire IFRS
                ("interets et produits assimiles", 24),
                ("intérêts et produits assimilés", 24),
                ("interets et charges assimilees", 22),
                ("intérêts et charges assimilées", 22),
                ("commissions (produits)", 24),
                ("commissions percues", 20),       # variante BOA : "commissions perçues"
                ("commissions (charges)", 22),
                ("commissions servies", 18),        # variante BOA
                # "gains ou pertes nets" — accept both nets/nettes
                ("gains ou pertes net", 28),        # substring : couvre "nets" et "nettes"
                ("produits des autres activites", 22),
                ("charges des autres activites", 22),
                ("produits nets des activites d'assurance", 26),
                ("charges generales d'exploitation", 18),
                ("interets minoritaires", 20),
                ("intérêts minoritaires", 20),
                ("resultat de base par action", 24),
                ("résultat de base par action", 24),
            ],
            "titles": [
                ("compte de resultat consolide", 16),
                ("compte de résultat consolidé", 16),
                ("etat du resultat global", 12),
            ],
            "context": [
                ("resultat net", 5), ("31/12", 5), ("consolid", 6),
                ("kdh", 4), ("milliers de dirhams", 4),
            ],
            "penalties": [
                ("total actif non courant", -20), ("immobilisations en non-valeur", -20),
                ("actif circulant", -16), ("total de l'actif", -20), ("total du passif", -20),
                ("valeurs en caisse, banques centrales", -18),
                ("dettes de financement", -14),
                ("résultat non courant", -12),
                ("charges de sinistres", -18), ("primes emises", -18),
                ("placements affectes aux operations d'assurance", -20),
            ],
        },
        # Assurance
        "bilan_actif_assurance": {
            "anchors": [
                # Mots-clés très discriminants Assurance (Bilan Actif - Comptes Sociaux)
                ("placements affectes aux operations d'assurance", 42),
                ("placements affectes aux operations d assurance", 42),
                ("placements affectes aux operations dassurance", 42),
                ("part des cessionnaires dans les provisions techniques", 44),
                ("tresorerie-actif", 30),
                ("tresorerie", 22),
                ("immobilisation en non-valeurs", 30),
                ("immobilisations en non-valeur", 30),
                ("immobilisations incorporelles", 24),
                ("immobilisations corporelles", 24),
                ("immobilisations financieres autres que placements", 28),
                ("immobilisations financieres (autres que placements)", 28),
                ("ecarts de conversion - actif", 20),
                ("ecarts de conversion -actif", 20),
                ("ecarts de conversion -actif elements circulants", 22),
                ("actif circulant (hors tresorerie)", 22),
                ("actif circulant hors tresorerie", 22),
                ("creances de l'actif circulant", 18),
                ("creances de l actif circulant", 18),
                ("titres et valeurs de placement", 18),
                ("part des cessionnaires dans les provisions techniques", 34),
                ("non affectes aux opérations d'assurance", 24),
                ("non affectes aux operations d'assurance", 24),
                ("non affectes aux operations dassurance", 24),
                ("non affectes aux opdass", 24),
                ("non affectes aux op dassurance", 24),
                ("non affectes", 10),
                ("actif immobilise", 20),
                ("immobilisations incorporelles", 24),
                ("immobilisations corporelles", 24),
                ("immobilisations financieres autres que placements", 28),
                ("total general", 36),
                # Ancres existantes (compatibilité)
                ("placements representatifs des provisions techniques", 26),
                ("creances nees d operations d'assurance", 18),
                ("creances nees d operations dassurance", 18),
                ("placements des entreprises d'assurance", 22),
                ("actifs incorporels", 8),
            ],
            "titles": [("bilan actif", 8)],
            "context": [("31/12", 5), ("kdh", 4), ("milliers de dirhams", 4)],
            "penalties": [
                ("goodwill", -18), ("immobilisations en non-valeur", -18),
                ("valeurs en caisse, banques centrales", -18),
                ("produit net bancaire", -18), ("résultat non courant", -14),
                ("capitaux propres - part du groupe", -18),
            ],
        },
        "bilan_passif_assurance": {
            "anchors": [
                # Mots-clés très discriminants Assurance (Bilan Passif - Comptes Sociaux)
                ("financement permanent", 34),
                ("capitaux propres", 30),
                ("capitaux propres assimiles", 32),
                ("dettes de financement", 30),
                ("provisions durables pour risques et charges", 32),
                ("provisions techniques brutes", 40),
                ("ecarts de conversion -passif", 24),
                ("ecarts de conversion - passif", 24),
                ("passif circulant (hors tresorerie)", 28),
                ("passif circulant hors tresorerie", 28),
                ("dettes pour especes remises par les cessionnaires", 34),
                ("dettes de passif circulant", 30),
                ("autres provisions pour risques et charges", 28),
                ("ecarts de conversion -passif (elements circulants)", 28),
                ("ecarts de conversion -passif elements circulants", 28),
                ("tresorerie-passif", 30),
                ("tresorerie", 20),
                ("total general", 36),
                # Ancres existantes (compatibilité)
                ("provisions techniques", 18),
                ("primes non acquises", 16),
                ("provisions pour sinistres", 16),
                ("dettes nées d'opérations d'assurance", 16),
            ],
            "titles": [("bilan passif", 8)],
            "context": [("31/12", 5), ("kdh", 4), ("milliers de dirhams", 4)],
            "penalties": [
                ("goodwill", -18), ("immobilisations en non-valeur", -18),
                ("valeurs en caisse, banques centrales", -18),
                ("produit net bancaire", -18), ("résultat non courant", -14),
                ("capitaux propres - part du groupe", -18),
            ],
        },
        "cpc_assurance": {
            "anchors": [
                # Assurance consolidé : Compte technique Vie
                ("compte technique assurance vie", 34),
                ("primes emises brutes", 30),
                ("primes emises cedees", 30),
                ("produits techniques d'exploitation", 26),
                ("prestations et frais", 22),
                ("prestations et frais cedes", 26),
                ("charges techniques d'exploitation", 26),
                ("produits des placements affectes aux operations d'assurance", 30),
                ("charges des placements affectes aux operations d'assurance", 30),
                ("resultat technique vie", 32),
                # Assurance consolidé : Compte technique Non Vie
                ("compte technique assurance non vie", 34),
                ("variation des provisions pour primes non acquises brutes", 28),
                ("variation des provisions pour primes non acquises cedees", 28),
                ("resultat technique non vie", 32),
                ("resultat technique (c = a + b)", 34),
                # Compte non technique
                ("compte non technique", 32),
                ("produits non techniques courants", 28),
                ("charges non techniques courantes", 28),
                ("resultat non technique courant", 30),
                ("produits non techniques non courants", 28),
                ("charges non techniques non courantes", 28),
                ("resultat non technique non courant", 30),
                ("resultat non technique (d)", 32),
                ("resultat avant impot (c + d)", 34),
                ("impot sur le resultat", 26),
                ("dotations d'amortissement des ecarts d'acquisition", 30),
                ("quote-part des societes mises en equivalence", 28),
                ("resultat net", 24),
                # Compatibilité historique
                ("primes émises", 20),
                ("charges de sinistres", 18),
                ("résultat technique", 18),
                ("primes acquises", 14),
                ("résultat non technique", 14),
                ("frais de gestion des sinistres", 12),
            ],
            "titles": [("compte de produits et charges", 10)],
            "context": [("résultat net", 5), ("31/12", 5), ("kdh", 4)],
            "penalties": [
                ("goodwill", -18), ("immobilisations en non-valeur", -18),
                ("actif circulant", -14), ("total de l'actif", -18), ("total du passif", -18),
                ("valeurs en caisse, banques centrales", -18),
                ("produit net bancaire", -18), ("résultat non courant", -10),
                ("résultat net - part du groupe", -18),
            ],
        },
    }


def _norm(text: str) -> str:
    """
    Normalise le texte OCR:
    - minuscules + suppression des accents
    - espaces/tirets homogènes
    - reconstruction des mots éclatés lettre par lettre ("P A S S I F" -> "passif")
    """
    if not text:
        return ""
    norm = unicodedata.normalize("NFD", text.lower()).encode("ascii", "ignore").decode("ascii")
    norm = norm.replace("\n", " ").replace("\t", " ").replace("–", "-").replace("—", "-")
    # Recolle les séquences OCR de lettres isolées (ex: "c a p i t a u x")
    norm = re.sub(
        r"\b(?:[a-z]\s+){2,}[a-z]\b",
        lambda m: m.group(0).replace(" ", ""),
        norm,
    )
    norm = re.sub(r"\s+", " ", norm).strip()
    return norm


def _contains_kw(text_norm: str, kw_norm: str, text_no_space: Optional[str] = None) -> bool:
    """
    Matching robuste pour OCR dégradé:
    - match standard
    - fallback sans espaces (capture les mots cassés/collés)
    """
    if not text_norm or not kw_norm:
        return False
    if kw_norm in text_norm:
        return True
    txt_ns = text_no_space if text_no_space is not None else re.sub(r"\s+", "", text_norm)
    kw_ns = re.sub(r"\s+", "", kw_norm)
    return bool(kw_ns) and kw_ns in txt_ns


def _normalize_signatures(raw: dict) -> dict:
    """Pré-normalise tous les mots-clés des signatures financières."""
    result = {}
    for sig_name, sig in raw.items():
        result[sig_name] = {
            "anchors":  [(_norm(kw), w) for kw, w in sig["anchors"]],
            "titles":   [(_norm(kw), w) for kw, w in sig["titles"]],
            "context":  [(_norm(kw), w) for kw, w in sig["context"]],
            "penalties":[(_norm(kw), w) for kw, w in sig["penalties"]],
        }
    return result


_FINANCIAL_SIGNATURES: dict = _normalize_signatures(_FINANCIAL_SIGNATURES_DICT())


def _is_toc_chunk(content: str) -> bool:
    """
    Detecte si un chunk est une page de sommaire/table des matieres.
    Heuristique : nombreuses lignes finissant par un numero de page.
    """
    lines = content.split("\n")
    dot_end   = sum(1 for l in lines if re.search(r"\.{3,}\s*\d{1,3}\s*$", l.strip()))
    space_end = sum(1 for l in lines if re.search(r"\s{3,}\d{1,3}\s*$", l.strip()))
    toc_kw    = any(kw in content.lower() for kw in ["sommaire", "table des mati\u00e8res", "table of contents"])
    return (dot_end + space_end) >= 6 or (toc_kw and (dot_end + space_end) >= 2)


def _get_target_hints(table_name: str) -> List[str]:
    """Retourne la liste des section_hints attendus pour un nom de tableau."""
    name = _norm(table_name).replace("\n", " ")
    is_actif    = "actif" in name and "passif" not in name
    is_passif   = "passif" in name
    wants_autres = "autres" in name
    is_cpc      = (
        "cpc" in name
        or "compte de produits" in name
        or "produits et charges" in name
        or "compte de r\u00e9sultat" in name
        or "compte de resultat" in name
    )
    # Le frontend envoie souvent "COMPTES CONSOLIDES" (pas "consolidés").
    # On considère comme consolidé tout ce qui contient "consolid" ou "consolides"/"consolide".
    is_consolid = ("consolid" in name) or ("consolide" in name) or ("consolides" in name)

    if is_consolid:
        # Inclure tous les formats (IFRS + CGNC + PCEC + Assurance) car les consolidés
        # peuvent utiliser n'importe quel format selon l'émetteur.
        if is_actif:  return ["bilan_actif_ifrs", "bilan_actif_cgnc", "bilan_actif_pcec", "bilan_actif_assurance"]
        if is_passif: return ["bilan_passif_cgnc_consolide", "bilan_passif_ifrs", "bilan_passif_cgnc", "bilan_passif_pcec", "bilan_passif_assurance"]
        if is_cpc:    return ["cpc_ifrs", "cpc_pcec", "cpc_cgnc", "cpc_assurance"]
    else:
        if is_actif:  return ["bilan_actif_cgnc", "bilan_actif_pcec", "bilan_actif_assurance"]
        if is_passif: return ["bilan_passif_cgnc", "bilan_passif_pcec", "bilan_passif_assurance"]
        if is_cpc:    return ["cpc_cgnc", "cpc_pcec", "cpc_assurance"]
    return []


def _find_page_by_section_hint(doc_filter: str, target_hints: List[str], index_dir: str) -> Optional[int]:
    """
    Chemin rapide : lecture directe de metadata.json pour trouver les chunks
    annotes avec un section_hint correspondant a la cible.
    Vote majoritaire si plusieurs pages candidates.
    """
    import json
    from collections import Counter

    meta_path = os.path.join(index_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except Exception:
        return None

    page_votes: Counter = Counter()
    for m in meta:
        if not m.get("doc_id") or doc_filter not in m["doc_id"].lower():
            continue
        hint = m.get("section_hint")
        if hint and hint in target_hints:
            try:
                page_votes[int(m["page"])] += 1
            except (TypeError, ValueError, KeyError):
                pass

    if page_votes:
        best_page, best_count = page_votes.most_common(1)[0]
        logger.info("section_hint: page %d (%d chunks, hints=%s)", best_page, best_count, target_hints)
        return best_page
    return None


def _resolve_pdf_absolute_path(pdf_name: str, cfg) -> Optional[str]:
    """Chemin absolu vers le PDF (dossier data/pdf ou chemin déjà complet)."""
    doc_id = os.path.basename(pdf_name)
    if not doc_id.lower().endswith(".pdf"):
        doc_id += ".pdf"
    candidate = os.path.join(cfg.raw_pdf_dir, doc_id)
    if os.path.isfile(candidate):
        return os.path.abspath(candidate)
    if os.path.isfile(pdf_name):
        return os.path.abspath(pdf_name)
    return None


def _infer_index_dir_from_pdf_name(pdf_name: str, cfg, type_rapport: Optional[str] = None) -> str:
    """
    Dossier d'index pour la lecture RAG : .../{emetteur}/{year}/{annuel|s1}/index/
    ou ancien .../{emetteur}/{year}/index/ si déjà indexé ainsi.
    """
    from .index_paths import normalize_report_code, resolve_index_dir_for_read

    tr = normalize_report_code(type_rapport) if type_rapport is not None else None
    return resolve_index_dir_for_read(pdf_name, cfg.index_dir, tr)


def _find_page_by_pdf_title_scan(pdf_path: str, table_name: str) -> Optional[int]:
    """
    Localise la page par titres/mots-clés structurels présents dans le PDF.
    Indépendant de l'index vectoriel : évite les erreurs de similarité.
    Pour les consolidés : scan des mots-clés structurels (≥ 2 requis) si les titres échouent.
    """
    name = (table_name or "").lower().replace("\n", " ")
    is_consolid = ("consolid" in name) or ("consolide" in name) or ("consolides" in name)
    want_sociaux = ("sociaux" in name or "social" in name) and not is_consolid

    is_actif = "actif" in name and "passif" not in name
    is_passif = "passif" in name
    is_cpc = (
        "cpc" in name
        or "compte de produits" in name
        or "produits et charges" in name
        or "compte de résultat" in name
        or "compte de resultat" in name
    )
    if not (is_actif or is_passif or is_cpc):
        return None

    try:
        import pymupdf
    except ImportError:
        return None

    # Mots-clés structurels du passif (n'apparaissent pas sur la page actif)
    _PASSIF_STRUCTURAL = (
        # IFRS PCEC bancaire marocain (Attijariwafa, CIH, etc.)
        "passifs financiers a la juste valeur par resultat",
        "dettes envers les etablissements de credit et assimiles",
        "dettes envers la clientele",
        "dettes envers les etablissements de credit",
        # IFRS générique
        "capitaux propres - part du groupe",
        "capitaux propres part du groupe",
        "int\u00e9r\u00eats minoritaires",
        "interets minoritaires",
        "total passif non courant",
        "total passif courant",
        "dettes financi\u00e8res non courantes",
        "dettes financieres non courantes",
        "dettes de financement",
        "total du passif",
        "d\u00e9p\u00f4ts de la client\u00e8le",
        "depots de la clientele",
        "provisions techniques",
    )

    doc = pymupdf.open(pdf_path)
    try:

        def page_lower(pi: int) -> str:
            # Utiliser la normalisation OCR robuste (accents + mots espacés lettre à lettre).
            return _norm(doc[pi].get_text() or "")

        def section_markers(page_text: str) -> tuple[bool, bool]:
            """Retourne (has_social, has_consolid) à partir du texte de page."""
            has_social = (
                "comptes sociaux" in page_text
                or "etats financiers sociaux" in page_text
                or "états financiers sociaux" in page_text
            )
            has_consolid = (
                "comptes consolid" in page_text
                or "etats financiers consolides" in page_text
                or "états financiers consolidés" in page_text
                or "consolid" in page_text
            )
            return has_social, has_consolid

        def section_is_compatible(page_text: str) -> bool:
            """Filtre strict pour éviter confusion sociaux/consolidés."""
            has_social, has_consolid = section_markers(page_text)
            if is_consolid and has_social and not has_consolid:
                return False
            if want_sociaux and has_consolid and not has_social:
                return False
            return True

        if is_actif and not is_passif:
            if not is_consolid:
                # Comptes sociaux : recherche du titre exact
                for i in range(len(doc)):
                    t = page_lower(i)
                    if "bilan actif" in t and section_is_compatible(t):
                        logger.info("pdf_title_scan: bilan actif -> page %d", i + 1)
                        return i + 1
                return None
            else:
                # Comptes consolidés : scan structurel (évite de matcher les pages sociaux)
                _ACTIF_STRUCTURAL = (
                    # IFRS PCEC bancaire marocain
                    "valeurs en caisse, banques centrales",
                    "actifs financiers a la juste valeur par resultat",
                    "prets et creances sur la clientele",
                    "prets et creances sur les etablissements de credit",
                    "bilan consolide",
                    # IFRS générique
                    "total actif non courant", "total actif courant", "goodwill",
                    "actifs d'imp\u00f4ts diff\u00e9r\u00e9s",
                    "\u00e9tat de la situation financi\u00e8re",
                    "etat de la situation financiere",
                    "participations dans les soci\u00e9t\u00e9s mises en \u00e9quivalence",
                    "cr\u00e9ances sur les \u00e9tablissements de cr\u00e9dit",
                )
                page_actif_kw: dict = {}
                for i in range(len(doc)):
                    t = page_lower(i)
                    if not section_is_compatible(t):
                        continue
                    count = sum(1 for kw in _ACTIF_STRUCTURAL if kw in t)
                    if count >= 2:
                        page_actif_kw[i] = count
                if page_actif_kw:
                    best_i = max(page_actif_kw, key=lambda k: page_actif_kw[k])
                    logger.info("pdf_title_scan: bilan actif consolide (structurel, %d kw) -> page %d", page_actif_kw[best_i], best_i + 1)
                    return best_i + 1
                return None

        if is_passif:
            if not is_consolid:
                # Comptes sociaux : recherche du titre exact
                for i in range(len(doc)):
                    t = page_lower(i)
                    if "bilan passif" in t and section_is_compatible(t):
                        logger.info("pdf_title_scan: bilan passif -> page %d", i + 1)
                        return i + 1
                # Format PCEC : actif et passif sur la même page
                for i in range(len(doc)):
                    t = page_lower(i)
                    if (
                        ("bilan au 31" in t or "bilan actif" in t)
                        and "passif" in t
                        and section_is_compatible(t)
                    ):
                        logger.info("pdf_title_scan: bilan passif PCEC (format combiné) -> page %d", i + 1)
                        return i + 1
                # Fallback structurel PCEC sociaux (banques) : mêmes lignes exclusives
                # au tableau passif — absentes des annexes qui n'en citent qu'une ou deux.
                _PCEC_PASSIF_SOCIAUX_STRUCTURAL = (
                    "banques centrales, tresor public, service des cheques postaux",
                    "dettes envers les etablissements de credit et assimiles",
                    "depots de la clientele",
                    "dettes envers la clientele sur produits participatifs",
                    "titres de creance emis",
                    "autres passifs",
                    "provisions pour risques et charges",
                    "provisions reglementees",
                    "subventions, fonds publics affectes et fonds speciaux de garantie",
                    "dettes subordonnees",
                    "depots d'investissement recus",
                    "ecarts de reevaluation",
                    "reserves et primes liees au capital",
                    "actionnaires.capital non verse",
                    "actionnaires capital non verse",
                    "report a nouveau",
                    "resultats nets en instance d'affectation",
                    "resultat net de l'exercice",
                )
                page_pcec_kw: dict = {}
                for i in range(len(doc)):
                    t = page_lower(i)
                    if not section_is_compatible(t):
                        continue
                    count = sum(1 for kw in _PCEC_PASSIF_SOCIAUX_STRUCTURAL if kw in t)
                    if count >= 3:
                        page_pcec_kw[i] = count
                if page_pcec_kw:
                    best_i = max(page_pcec_kw, key=lambda k: page_pcec_kw[k])
                    logger.info(
                        "pdf_title_scan: bilan passif PCEC sociaux structurel (%d kw) -> page %d",
                        page_pcec_kw[best_i], best_i + 1,
                    )
                    return best_i + 1
                return None
            else:
                # Comptes consolidés : scan structurel (évite de matcher les pages sociaux)
                page_kw_count: dict = {}
                for i in range(len(doc)):
                    t = page_lower(i)
                    if not section_is_compatible(t):
                        continue
                    count = sum(1 for kw in _PASSIF_STRUCTURAL if kw in t)
                    if count >= 2:
                        page_kw_count[i] = count
                if page_kw_count:
                    best_i = max(page_kw_count, key=lambda k: page_kw_count[k])
                    logger.info(
                        "pdf_title_scan: bilan passif consolide structurel (%d kw) -> page %d",
                        page_kw_count[best_i], best_i + 1,
                    )
                    return best_i + 1
                return None

        if is_cpc:
            if not is_consolid:
                # Comptes sociaux : le titre CPC doit figurer EN TÊTE de page (250 premiers chars).
                # Évite les pages de rapport des commissaires aux comptes qui mentionnent
                # "compte de produits et charges" dans le corps du texte.
                needles = (
                    "comptes de produits et charges",
                    "compte de produits et charges",
                    "compte de produits et de charges",
                )
                for i in range(len(doc)):
                    t_head = page_lower(i)[:250]
                    t_full = page_lower(i)
                    if any(n in t_head for n in needles) and section_is_compatible(t_full):
                        logger.info("pdf_title_scan: CPC sociaux -> page %d", i + 1)
                        return i + 1
                return None
            else:
                # Comptes consolidés : chercher l'état du résultat global (IFRS)
                # ou les pages CPC avec marqueur consolidé explicite
                ifrs_needles = (
                    # Titre direct (Attijariwafa, CIH, banques PCEC IFRS)
                    "compte de resultat consolide",
                    "compte de resultats consolide",
                    # IFRS générique
                    "\u00e9tat du r\u00e9sultat global",
                    "etat du resultat global",
                    "r\u00e9sultat net - part du groupe",
                    "resultat net - part du groupe",
                    "r\u00e9sultat op\u00e9rationnel",
                    "resultat operationnel",
                    # PCEC IFRS bancaire marocain (premières lignes du CPC)
                    "interets et produits assimiles",
                    "interets et charges assimiles",
                )
                # Score-based : retourne la page avec le plus de needles (≥2)
                # évite les pages narratives qui mentionnent le titre une seule fois
                page_cpc_kw: dict = {}
                for i in range(len(doc)):
                    t = page_lower(i)
                    if not section_is_compatible(t):
                        continue
                    count = sum(1 for n in ifrs_needles if n in t)
                    if count >= 2:
                        page_cpc_kw[i] = count
                if page_cpc_kw:
                    best_i = max(page_cpc_kw, key=lambda k: page_cpc_kw[k])
                    logger.info("pdf_title_scan: CPC consolide IFRS (%d kw) -> page %d", page_cpc_kw[best_i], best_i + 1)
                    return best_i + 1
                # Fallback : première page avec au moins 1 needle
                for i in range(len(doc)):
                    t = page_lower(i)
                    if any(n in t for n in ifrs_needles) and section_is_compatible(t):
                        logger.info("pdf_title_scan: CPC consolide IFRS (fallback 1kw) -> page %d", i + 1)
                        return i + 1
                # Fallback : CPC CGNC consolide (page avec marqueur "comptes consolid")
                cpc_needles = (
                    "comptes de produits et charges",
                    "compte de produits et charges",
                    "compte de produits et de charges",
                )
                for i in range(len(doc)):
                    t = page_lower(i)
                    if (
                        any(n in t for n in cpc_needles)
                        and ("comptes consolid" in t or "consolid" in t)
                        and section_is_compatible(t)
                    ):
                        logger.info("pdf_title_scan: CPC consolide CGNC -> page %d", i + 1)
                        return i + 1
                return None

    finally:
        doc.close()

    return None


def _score_chunk_for_table(chunk_content: str, table_name: str) -> int:
    """
    Score un chunk selon sa pertinence pour le tableau demande.

    Utilise _FINANCIAL_SIGNATURES couvrant tous les secteurs :
    CGNC (comptes sociaux), PCEC (bancaire), IFRS (consolide), Assurance.

    Logique :
      1. Identifie les signatures primaires (actif/passif/cpc x consolid/sociaux)
      2. Prend le MEILLEUR score positif parmi les signatures primaires
      3. Cumule les penalites (keywords d'autres tableaux)
      4. Bonus : date de cloture 31/12
      5. Penalite forte si page de sommaire (TOC)
      6. Legere penalite si texte trop narratif
    """
    if not chunk_content:
        return 0
    text = _norm(chunk_content)
    text_no_space = re.sub(r"\s+", "", text)
    name = _norm(table_name).replace("\n", " ")
    wants_autres = "autres" in name

    is_actif    = "actif" in name and "passif" not in name
    is_passif   = "passif" in name
    is_cpc      = (
        "cpc" in name
        or "compte de produits" in name
        or "produits et charges" in name
        or "compte de r\u00e9sultat" in name
        or "compte de resultat" in name
    )
    # Même logique que _get_target_hints : accepter "consolides".
    is_consolid = ("consolid" in name) or ("consolide" in name) or ("consolides" in name)

    if is_consolid:
        # Inclure toutes les signatures (IFRS + CGNC + PCEC + Assurance) car les consolidés
        # peuvent utiliser n'importe quel format selon l'émetteur.
        if is_actif:    primaries = ["bilan_actif_ifrs", "bilan_actif_cgnc", "bilan_actif_pcec", "bilan_actif_assurance"]
        elif is_passif: primaries = ["bilan_passif_cgnc_consolide", "bilan_passif_ifrs", "bilan_passif_cgnc", "bilan_passif_pcec", "bilan_passif_assurance"]
        elif is_cpc:    primaries = ["cpc_ifrs_bancaire", "cpc_ifrs", "cpc_pcec", "cpc_cgnc", "cpc_assurance"]
        else:           primaries = ["bilan_actif_ifrs", "bilan_passif_ifrs", "cpc_ifrs"]
    else:
        if is_actif:    primaries = ["bilan_actif_cgnc", "bilan_actif_pcec", "bilan_actif_assurance"]
        elif is_passif: primaries = ["bilan_passif_cgnc", "bilan_passif_pcec", "bilan_passif_assurance"]
        elif is_cpc:    primaries = ["cpc_cgnc", "cpc_pcec", "cpc_assurance"]
        else:           primaries = list(_FINANCIAL_SIGNATURES.keys())

    if _is_toc_chunk(chunk_content):
        return -30

    best_positive = 0
    for sig_name in primaries:
        sig = _FINANCIAL_SIGNATURES.get(sig_name)
        if not sig:
            continue
        s = 0
        anchor_matches = 0
        for kw, pts in sig["anchors"]:
            if _contains_kw(text, kw, text_no_space):
                s += pts
                anchor_matches += 1
        for kw, pts in sig["titles"]:
            if _contains_kw(text, kw, text_no_space):
                s += pts
        for kw, pts in sig["context"]:
            if _contains_kw(text, kw, text_no_space):
                s += pts
        # Exiger un nombre minimum d'anchors trouvés pour valider la signature.
        # La formule s'adapte à la taille de la liste :
        #   ≤10 anchors → 3 min  |  11-20 → 4 min  |  21-35 → 5 min  |  36+ → 7 min
        na = len(sig["anchors"])
        min_anchors = 3 if na <= 10 else 4 if na <= 20 else 5 if na <= 35 else 7
        if anchor_matches >= min_anchors:
            best_positive = max(best_positive, s)

    score = best_positive

    penalized: set = set()
    for sig_name in primaries:
        sig = _FINANCIAL_SIGNATURES.get(sig_name)
        if not sig:
            continue
        for kw, pts in sig["penalties"]:
            if _contains_kw(text, kw, text_no_space) and kw not in penalized:
                score += pts
                penalized.add(kw)

    if re.search(r"31/12|31-12", chunk_content):
        score += 5

    # Requêtes métier "catégorie autres": bonus léger uniquement sur chunks déjà compatibles
    # pour éviter de surclasser des pages non financières contenant le mot "autres".
    if wants_autres and score > 0 and _contains_kw(text, "autres", text_no_space):
        score += 8

    words = chunk_content.split()
    if len(words) > 200:
        digit_ratio = sum(1 for w in words if re.search(r"\d", w)) / len(words)
        if digit_ratio < 0.05:
            score -= 5

    return score


def _find_page_by_rag(
    pdf_name: str,
    table_name: str,
    type_rapport: Optional[str] = None,
) -> Optional[int]:
    """
    Localise la page exacte du tableau via une strategie en couches :

    Couche 0 - scan texte du PDF (titres CGNC / RFA) :
        Repère « bilan actif », « bilan passif », « comptes de produits et charges », etc.
        Sans dépendre de l'ingestion ni du vecteur.

    Couche 1 - section_hint (metadata.json) :
        Annotation produite à l'ingest.

    Couche 2 - vote multi-chunk par page (vectoriel + scoring metier) :
        k=100, regroupement par page, score = max + 30%% des autres + bonus.

    Couche 3 - seuil de confiance :
        Si score < -5, retourne None.
    """
    try:
        from collections import Counter, defaultdict

        from .config import RAGConfig, ensure_dirs
        from .retriever import SimpleVectorRetriever

        cfg = RAGConfig()
        ensure_dirs(cfg)

        doc_id = os.path.basename(pdf_name)
        if not doc_id.lower().endswith(".pdf"):
            doc_id += ".pdf"
        doc_filter      = doc_id.lower().replace(".pdf", "")
        table_name_norm = table_name.replace("\n", " ").strip()
        target_hints    = _get_target_hints(table_name_norm)
        name_l = table_name_norm.lower()
        want_consolid = ("consolid" in name_l) or ("consolide" in name_l) or ("consolides" in name_l)
        want_sociaux = ("sociaux" in name_l or "social" in name_l) and not want_consolid

        pdf_abs = _resolve_pdf_absolute_path(pdf_name, cfg)

        def _detect_section_ranges(pdf_path: str) -> tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
            """
            Détecte les plages de pages (1-based) pour:
            - comptes sociaux
            - comptes consolidés
            """
            social_pages: List[int] = []
            consolid_pages: List[int] = []
            try:
                import pymupdf  # type: ignore
                doc = pymupdf.open(pdf_path)
                try:
                    for i in range(len(doc)):
                        t = (doc[i].get_text() or "").lower()
                        has_social = (
                            "comptes sociaux" in t
                            or "etats financiers sociaux" in t
                            or "états financiers sociaux" in t
                        )
                        has_consolid = (
                            "comptes consolid" in t
                            or "etats financiers consolides" in t
                            or "états financiers consolidés" in t
                            or "consolid" in t
                        )
                        p = i + 1
                        if has_social:
                            social_pages.append(p)
                        if has_consolid:
                            consolid_pages.append(p)
                finally:
                    doc.close()
            except Exception:
                return ([], [])

            def _to_ranges(pages: List[int], gap: int = 2) -> List[Tuple[int, int]]:
                if not pages:
                    return []
                pages_sorted = sorted(set(pages))
                ranges: List[Tuple[int, int]] = []
                start = pages_sorted[0]
                prev = start
                for p in pages_sorted[1:]:
                    if p - prev <= gap:
                        prev = p
                        continue
                    ranges.append((start, prev))
                    start = p
                    prev = p
                ranges.append((start, prev))
                return ranges

            return (_to_ranges(social_pages), _to_ranges(consolid_pages))

        def _page_in_ranges(page: int, ranges: List[Tuple[int, int]], margin: int = 1) -> bool:
            if not ranges:
                return True
            for a, b in ranges:
                if (a - margin) <= page <= (b + margin):
                    return True
            return False
        # Use the per-emetteur/year index folder for this PDF.
        index_dir = _infer_index_dir_from_pdf_name(pdf_abs or pdf_name, cfg, type_rapport)
        retriever = SimpleVectorRetriever(cfg, index_dir_override=index_dir)
        social_ranges: List[Tuple[int, int]] = []
        consolid_ranges: List[Tuple[int, int]] = []
        if pdf_abs:
            social_ranges, consolid_ranges = _detect_section_ranges(pdf_abs)

        _page_text_cache: dict[int, str] = {}

        def _page_text_norm(page: int) -> str:
            if page in _page_text_cache:
                return _page_text_cache[page]
            if not pdf_abs:
                _page_text_cache[page] = ""
                return ""
            try:
                import pymupdf  # type: ignore
                doc = pymupdf.open(pdf_abs)
                try:
                    if page < 1 or page > len(doc):
                        _page_text_cache[page] = ""
                        return ""
                    txt = _norm(doc[page - 1].get_text() or "")
                    _page_text_cache[page] = txt
                    return txt
                finally:
                    doc.close()
            except Exception:
                _page_text_cache[page] = ""
                return ""

        def _explicit_page_markers(page: int) -> tuple[bool, bool]:
            txt = _page_text_norm(page)
            has_social = (
                "comptes sociaux" in txt
                or "etats financiers sociaux" in txt
                or "etats financiers sociaux resumes" in txt
            )
            has_consolid = (
                "comptes consolides" in txt
                or "etats financiers consolides" in txt
                or "consolide" in txt
            )
            return has_social, has_consolid

        def _is_page_allowed_by_section(page: int) -> bool:
            # Règle stricte mais non fragile:
            # on rejette uniquement s'il y a contradiction explicite sur la page.
            has_social, has_consolid = _explicit_page_markers(page)
            if want_consolid and has_social and not has_consolid:
                return False
            if want_sociaux and has_consolid and not has_social:
                return False
            return True

        def _primary_signatures_for_name(name: str) -> List[str]:
            n = _norm(name).replace("\n", " ")
            is_actif = "actif" in n and "passif" not in n
            is_passif = "passif" in n
            is_cpc = (
                "cpc" in n
                or "compte de produits" in n
                or "produits et charges" in n
                or "compte de resultat" in n
            )
            is_consol = ("consolid" in n) or ("consolide" in n) or ("consolides" in n)
            if is_consol:
                if is_actif:
                    return ["bilan_actif_ifrs", "bilan_actif_cgnc", "bilan_actif_pcec", "bilan_actif_assurance"]
                if is_passif:
                    return ["bilan_passif_cgnc_consolide", "bilan_passif_ifrs", "bilan_passif_cgnc", "bilan_passif_pcec", "bilan_passif_assurance"]
                if is_cpc:
                    return ["cpc_ifrs", "cpc_pcec", "cpc_cgnc", "cpc_assurance"]
                return ["bilan_actif_ifrs", "bilan_passif_ifrs", "cpc_ifrs"]
            if is_actif:
                return ["bilan_actif_cgnc", "bilan_actif_pcec", "bilan_actif_assurance"]
            if is_passif:
                return ["bilan_passif_cgnc", "bilan_passif_pcec", "bilan_passif_assurance"]
            if is_cpc:
                return ["cpc_cgnc", "cpc_pcec", "cpc_assurance"]
            return list(_FINANCIAL_SIGNATURES.keys())

        def _page_anchor_hits(page: int, table_name: str) -> tuple[int, int]:
            """
            Retourne (hits, min_required) sur la meilleure signature primaire.
            La règle min_required respecte la demande utilisateur: viser 10 mots-clés si possible.
            """
            txt = _page_text_norm(page)
            txt_no_space = re.sub(r"\s+", "", txt)
            primaries = _primary_signatures_for_name(table_name)
            best_hits = 0
            best_req = 10
            for sig_name in primaries:
                sig = _FINANCIAL_SIGNATURES.get(sig_name)
                if not sig:
                    continue
                anchors = [kw for kw, _pts in sig["anchors"]]
                hits = sum(1 for kw in anchors if _contains_kw(txt, kw, txt_no_space))
                req = min(10, max(4, len(anchors) // 3))
                if hits > best_hits:
                    best_hits = hits
                    best_req = req
            return best_hits, best_req

        title_scan_candidate: Optional[int] = None
        if pdf_abs:
            scanned = _find_page_by_pdf_title_scan(pdf_abs, table_name_norm)
            if scanned is not None and _is_page_allowed_by_section(scanned):
                title_scan_candidate = scanned

        hint_page: Optional[int] = None
        if target_hints:
            # Ne pas retourner directement `hint_page`: certains PDFs attribuent un
            # `section_hint` à une page narrative/annexe. On le garde uniquement
            # comme indice (bonus) via `hint_match` pendant la recherche vectorielle.
            hint_page = _find_page_by_section_hint(doc_filter, target_hints, index_dir)
            if hint_page is not None and not _is_page_allowed_by_section(hint_page):
                logger.info(
                    "section_hint ignore (hors section demandee): page %d pour '%s'",
                    hint_page, table_name_norm,
                )
                hint_page = None

        chunks = retriever.search(table_name_norm, k=100, kind_filter=None)
        doc_chunks = [c for c in chunks if doc_filter in c.doc_id.lower()]
        if not doc_chunks:
            return None

        # Passe 1 : collecter les marqueurs sociaux/consolidés au niveau de la PAGE
        # (les chunks de tableaux n'ont pas d'en-tête → utiliser les chunks texte pour tagger la page)
        page_markers: dict = {}  # page -> (has_social, has_consol)
        for chunk in doc_chunks:
            p = chunk.page
            content_l = (chunk.content or "").lower()
            has_social = ("comptes sociaux" in content_l) or ("etats financiers sociaux" in content_l)
            has_consol = ("comptes consolid" in content_l) or ("consolid" in content_l)
            if p not in page_markers:
                page_markers[p] = (False, False)
            old_s, old_c = page_markers[p]
            page_markers[p] = (old_s or has_social, old_c or has_consol)

        # Passe 2 : scorer les chunks en filtrant les pages de la mauvaise section
        page_data: dict = defaultdict(lambda: {
            "metier_scores": [],
            "best_vector":   0.0,
            "hint_match":    False,
            "title_scan_hit": False,
        })
        for chunk in doc_chunks:
            p = chunk.page
            if not _is_page_allowed_by_section(p):
                continue
            page_has_social, page_has_consol = page_markers.get(p, (False, False))

            # Éliminer les pages de la mauvaise section dans un PDF mixte
            if want_consolid and page_has_social and not page_has_consol:
                continue  # Page clairement sociaux → ignorer pour une recherche consolidée
            if want_sociaux and page_has_consol and not page_has_social:
                continue  # Page clairement consolidée → ignorer pour une recherche sociaux

            metier = _score_chunk_for_table(chunk.content or "", table_name_norm)
            page_data[p]["metier_scores"].append(metier)
            page_data[p]["best_vector"] = max(
                page_data[p]["best_vector"], float(chunk.score)
            )
            if (
                target_hints
                and getattr(chunk, "section_hint", None)
                and chunk.section_hint in target_hints
            ):
                page_data[p]["hint_match"] = True
            if title_scan_candidate is not None and p == title_scan_candidate:
                page_data[p]["title_scan_hit"] = True

        if not page_data:
            # Aucun candidat vectoriel valide après filtrage : on retombe sur le scan titre s'il existe.
            if title_scan_candidate is not None:
                logger.info(
                    "RAG: fallback sur title_scan page %d pour '%s' (apres filtrage section)",
                    title_scan_candidate, table_name_norm,
                )
                return title_scan_candidate
            return None

        def _page_final_score(p: int):
            d      = page_data[p]
            scores = d["metier_scores"]
            total  = max(scores) + sum(s for s in sorted(scores, reverse=True)[1:]) * 0.3
            if d["hint_match"]:
                total += 30
            if d["title_scan_hit"]:
                total += 20
            total += 5 * min(len(scores) - 1, 3)
            return (total, d["best_vector"])

        ranked_pages = sorted(page_data.keys(), key=_page_final_score, reverse=True)
        best_page = ranked_pages[0]
        best_score, best_v = _page_final_score(best_page)

        # Un titre explicite détecté dans le PDF est plus fiable qu'un score vectoriel
        # sur une page narrative — priorité pour sociaux ET consolidés.
        if title_scan_candidate is not None and (want_sociaux or want_consolid):
            if want_consolid:
                # Pour les consolidés, _find_page_by_pdf_title_scan utilise déjà
                # une validation structurelle forte (≥2 keywords PDF) → confiance directe.
                logger.info(
                    "RAG: priorite title_scan consolide page %d pour '%s'",
                    title_scan_candidate, table_name_norm,
                )
                return title_scan_candidate
            ts_hits, ts_req = _page_anchor_hits(title_scan_candidate, table_name_norm)
            if ts_hits >= max(4, ts_req - 2):
                logger.info(
                    "RAG: priorite title_scan sociaux page %d (hits=%d/%d) pour '%s'",
                    title_scan_candidate, ts_hits, ts_req, table_name_norm,
                )
                return title_scan_candidate

        # Validation forte par ancres sur la page finale.
        # On n'accepte pas une page qui ne contient pas assez de mots-clés métier.
        selected_page: Optional[int] = None
        for p in ranked_pages[:8]:
            hits, req = _page_anchor_hits(p, table_name_norm)
            if hits >= req:
                selected_page = p
                logger.info(
                    "RAG keyword validation: page %d retenue (hits=%d/%d) pour '%s'",
                    p, hits, req, table_name_norm,
                )
                break

        # Fiabilité: éviter une décision rapide sur un score faible quand un section_hint
        # cohérent existe. Dans ce cas on préfère le hint_page.
        if hint_page is not None:
            if best_score < 25:
                h_hits, h_req = _page_anchor_hits(hint_page, table_name_norm)
                if h_hits >= h_req:
                    logger.info(
                        "RAG: score faible (%.1f), priorite section_hint page %d (hits=%d/%d) pour '%s'",
                        best_score, hint_page, h_hits, h_req, table_name_norm,
                    )
                    return hint_page
                logger.info(
                    "RAG: score faible mais section_hint insuffisant en mots-cles (page=%d hits=%d/%d)",
                    hint_page, h_hits, h_req,
                )
            if hint_page in page_data:
                hint_score, hint_v = _page_final_score(hint_page)
                if (best_score - hint_score) <= 12:
                    h_hits, h_req = _page_anchor_hits(hint_page, table_name_norm)
                    if h_hits < h_req:
                        logger.info(
                            "RAG: section_hint proche mais invalide mots-cles page %d (hits=%d/%d)",
                            hint_page, h_hits, h_req,
                        )
                    else:
                        logger.info(
                            "RAG: arbitration section_hint page %d (hint_score=%.1f, best=%d score=%.1f, vecteur=%.3f)",
                            hint_page, hint_score, best_page, best_score, hint_v,
                        )
                        return hint_page

        if selected_page is not None:
            s_score, s_v = _page_final_score(selected_page)
            logger.info(
                "RAG: page %d selectionnee (score=%.1f, vecteur=%.3f, chunks=%d) pour '%s'",
                selected_page, s_score, s_v,
                len(page_data[selected_page]["metier_scores"]), table_name_norm,
            )
            return selected_page

        if best_score < -5:
            logger.warning(
                "RAG: aucune page suffisamment confiante (score=%.1f) pour '%s'",
                best_score, table_name_norm,
            )
            return None

        logger.info(
            "RAG: page %d selectionnee (score=%.1f, vecteur=%.3f, chunks=%d) pour '%s'",
            best_page, best_score, best_v,
            len(page_data[best_page]["metier_scores"]), table_name_norm,
        )
        return best_page

    except Exception as e:
        logger.warning("RAG find page failed: %s", e)
        return None


def _main() -> None:
    """CLI : python -m rag_agent.extraction_agent <pdf_name> [page] <table_name>
    Si page est omis, la page est trouvée automatiquement via le RAG."""
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    if len(sys.argv) < 3:
        print("Usage: python -m rag_agent.extraction_agent <pdf_name> [page] <table_name>")
        print("       (sans page -> recherche automatique via RAG)")
        print("Ex:    python -m rag_agent.extraction_agent Addoha_RFA_2024.pdf \"BILAN ACTIF\"")
        print("Ex:    python -m rag_agent.extraction_agent Addoha_RFA_2024.pdf 21 \"BILAN ACTIF\"")
        sys.exit(1)

    pdf_name = sys.argv[1]
    table_name = sys.argv[-1]
    page_num: Optional[int] = None

    if len(sys.argv) == 4:
        try:
            page_num = int(sys.argv[2])
        except ValueError:
            page_num = None
    if page_num is None:
        page_num = find_page_for_table(pdf_name, table_name)
        if page_num is None:
            print("Erreur: page non fournie et impossible de la trouver via le RAG (lancez l'ingestion: python -m rag_agent.ingest).")
            sys.exit(1)
        print(f"Page trouvee automatiquement: {page_num}")

    # Chercher le PDF dans data/pdf si chemin relatif
    if not os.path.isabs(pdf_name) and not os.path.exists(pdf_name):
        data_pdf = os.path.join("data", "pdf", pdf_name)
        if os.path.exists(data_pdf):
            pdf_name = data_pdf

    agent = ExtractionAgent(use_llm=True)
    result = agent.extract(pdf_name, page_num, table_name)
    print(result.summary())
    if result.warnings:
        for w in result.warnings:
            print("  [!]", w)
    if not result.df.empty:
        print(result.df.to_string())
    else:
        print("(DataFrame vide)")
    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    _main()
