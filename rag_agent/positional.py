"""
Module d'extraction positionnelle X/Y pour tableaux financiers.

Point d'entrée unique :
    df = extract_table_positionnel(pdf_path, page_num)
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd
import pdfplumber

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  Paramètres (ajuste si besoin selon ton PDF)
# ─────────────────────────────────────────────────────────────
Y_TOL = 4  # tolérance verticale
X_TOL = 8  # tolérance horizontale
MIN_ROWS = 3
MIN_COLS = 2


def extract_table_positionnel(
    pdf_path: str,
    page_num: int,  # 1-indexé
    y_tol: int = Y_TOL,
    x_tol: int = X_TOL,
    prefer_xy: bool = False,
) -> pd.DataFrame:
    """
    Extrait le tableau principal de la page `page_num` en utilisant
    les coordonnées X/Y de chaque mot (pas de regex).

    Stratégies:
      1. pdfplumber natif (bordures)
      2. Clustering X/Y (sans bordures)
      3. Texte brut + espaces multiples (fallback)

    Si prefer_xy=True (ex. comptes consolidés), on essaie la stratégie XY en premier
    pour mieux détecter les 4 colonnes (ACTIF, Notes, date1, date2).
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_num < 1 or page_num > len(pdf.pages):
                logger.warning("positional: page %s hors limites pour %s", page_num, pdf_path)
                return pd.DataFrame()

            page = pdf.pages[page_num - 1]

            # Pour tableaux à 5 colonnes (ex. consolidés) : XY d'abord avec x_tol plus fin
            if prefer_xy:
                xy_tol = min(x_tol, 4)
                df = _strategy_xy(page, y_tol, xy_tol, prefer_more_columns=True)
                if _is_valid(df):
                    logger.debug("positional[xy] p.%s -> %s x %s", page_num, df.shape[0], df.shape[1])
                    return df

            # Stratégie 1
            df = _strategy_native(page)
            if _is_valid(df):
                logger.debug("positional[natif] p.%s -> %s x %s", page_num, df.shape[0], df.shape[1])
                return df

            # Stratégie 2
            df = _strategy_xy(page, y_tol, x_tol)
            if _is_valid(df):
                logger.debug("positional[xy] p.%s -> %s x %s", page_num, df.shape[0], df.shape[1])
                return df

            # Stratégie 3
            df = _strategy_spaces(page)
            if _is_valid(df):
                logger.debug("positional[spaces] p.%s -> %s x %s", page_num, df.shape[0], df.shape[1])
                return df

            logger.warning("positional: aucune stratégie n'a fonctionné pour %s p.%s", pdf_path, page_num)
            return pd.DataFrame()
    except Exception as e:
        logger.error("positional: erreur sur %s p.%s -> %s", pdf_path, page_num, e)
        return pd.DataFrame()


def _strategy_native(page) -> pd.DataFrame:
    """Tentative avec pdfplumber.extract_tables et paramètres de grille."""
    try:
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
        return _to_dataframe(best)
    except Exception:
        return pd.DataFrame()


def _strategy_xy(page, y_tol: int, x_tol: int, prefer_more_columns: bool = False) -> pd.DataFrame:
    """Reconstruction de tableau par clustering des positions X/Y."""
    try:
        words = page.extract_words(
            x_tolerance=x_tol,
            y_tolerance=y_tol,
            keep_blank_chars=False,
            use_text_flow=False,
        )
        if not words:
            return pd.DataFrame()

        lines = _group_by_y(words, y_tol)
        if len(lines) < MIN_ROWS:
            return pd.DataFrame()

        col_bounds = _detect_columns(lines, x_tol, prefer_more_columns)
        if len(col_bounds) < MIN_COLS:
            return pd.DataFrame()

        grid = _build_grid(lines, col_bounds)
        if len(grid) < MIN_ROWS:
            return pd.DataFrame()

        return _to_dataframe(grid)
    except Exception:
        return pd.DataFrame()


def _group_by_y(words: list, y_tol: int) -> list:
    """Regroupe les mots par ligne selon Y."""
    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: list[list] = []
    current = [sorted_words[0]]
    current_y = sorted_words[0]["top"]

    for w in sorted_words[1:]:
        if abs(w["top"] - current_y) <= y_tol:
            current.append(w)
        else:
            lines.append(sorted(current, key=lambda w: w["x0"]))
            current = [w]
            current_y = w["top"]

    if current:
        lines.append(sorted(current, key=lambda w: w["x0"]))
    return lines


def _detect_columns(lines: list, x_tol: int, prefer_more_columns: bool = False) -> list[tuple[float, float]]:
    """Clusterise les positions X pour trouver les colonnes. Si prefer_more_columns, garde plus de colonnes (ex. 5)."""
    all_x0 = sorted(w["x0"] for line in lines for w in line)
    all_x1 = [w["x1"] for line in lines for w in line]
    if not all_x0 or not all_x1:
        return []

    # prefer_more_columns : seuil plus serré pour ne pas fusionner (ex. 5 colonnes au lieu de 4)
    if prefer_more_columns:
        merge_dist = x_tol * 1.5
        min_presence = max(1, int(len(lines) * 0.05))
    else:
        merge_dist = x_tol * 2
        min_presence = max(1, int(len(lines) * 0.10))
    clusters: list[list[float]] = []
    current = [all_x0[0]]
    for x in all_x0[1:]:
        if x - current[-1] <= merge_dist:
            current.append(x)
        else:
            clusters.append(current)
            current = [x]
    clusters.append(current)

    valid = [c for c in clusters if len(c) >= min_presence] or clusters

    max_x1 = max(all_x1)
    bounds: list[tuple[float, float]] = []
    for i, c in enumerate(valid):
        x_start = min(c) - x_tol
        x_end = min(valid[i + 1]) - 1 if i + 1 < len(valid) else max_x1 + x_tol
        bounds.append((x_start, x_end))
    return bounds


def _build_grid(lines: list, col_bounds: list[tuple[float, float]]) -> list[list[str]]:
    """Assigne chaque mot à une colonne et construit la grille."""
    grid: list[list[str]] = []
    n = len(col_bounds)
    for line in lines:
        row = [""] * n
        for w in line:
            cx = (w["x0"] + w["x1"]) / 2
            for i, (xs, xe) in enumerate(col_bounds):
                if xs <= cx <= xe:
                    row[i] = (row[i] + " " + w["text"]).strip()
                    break
        if any(c.strip() for c in row):
            grid.append(row)
    return grid


def _strategy_spaces(page) -> pd.DataFrame:
    """Fallback: découpe du texte brut sur espaces multiples."""
    try:
        text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
        lines = [l.rstrip() for l in text.split("\n") if l.strip()]
        if len(lines) < MIN_ROWS:
            return pd.DataFrame()

        parsed: list[list[str]] = []
        for line in lines:
            cells = [c.strip() for c in re.split(r" {2,}", line.strip()) if c.strip()]
            if cells:
                parsed.append(cells)

        if len(parsed) < MIN_ROWS:
            return pd.DataFrame()

        max_c = max(len(r) for r in parsed)
        if max_c < MIN_COLS:
            return pd.DataFrame()

        normed = [(r + [""] * max_c)[:max_c] for r in parsed]
        return _to_dataframe(normed)
    except Exception:
        return pd.DataFrame()


def _to_dataframe(grid: list[list[str]]) -> pd.DataFrame:
    """Convertit une grille en DataFrame propre."""
    if not grid or len(grid) < 2:
        return pd.DataFrame()

    max_c = max(len(r) for r in grid)
    normed = [(r + [""] * max_c)[:max_c] for r in grid]

    raw_headers = [str(h or "").strip() for h in normed[0]]
    headers = _deduplicate_headers(raw_headers)
    rows = [[str(c or "").strip() for c in row] for row in normed[1:]]

    df = pd.DataFrame(rows, columns=headers)

    # Supprimer lignes et colonnes totalement vides
    df = df[df.apply(lambda r: any(str(v).strip() for v in r), axis=1)]
    df = df.loc[:, df.apply(lambda c: any(str(v).strip() for v in c))]
    return df.reset_index(drop=True)


def _deduplicate_headers(headers: list[str]) -> list[str]:
    """Évite les doublons dans les noms de colonnes."""
    seen: dict[str, int] = {}
    result: list[str] = []
    for h in headers:
        h = h or "col"
        if h in seen:
            seen[h] += 1
            result.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            result.append(h)
    return result


def _is_valid(df: pd.DataFrame) -> bool:
    """Vérifie qu'un DataFrame ressemble à un tableau financier."""
    if df is None or df.empty:
        return False
    if len(df) < MIN_ROWS or len(df.columns) < MIN_COLS:
        return False

    # Accès par position pour éviter df[col] = DataFrame quand colonnes dupliquées
    has_numbers = any(
        df.iloc[:, i].astype(str).str.contains(r"\d{3,}", regex=True).any()
        for i in range(len(df.columns))
    )
    return has_numbers

