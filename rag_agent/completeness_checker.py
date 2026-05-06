"""
Contrôle de complétude des tableaux financiers extraits.

Vérifie :
1. Présence des ancres critiques (ex: TOTAL DE L'ACTIF)
2. Nombre minimal de lignes selon type de tableau
3. Cohérence des colonnes
4. Score global de complétude (0.0 → 1.0)

Utilisé par ExtractionAgent pour décider si l'extraction est complète
ou si une stratégie de niveau supérieur doit être tentée.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ── Règles de complétude par type de tableau ─────────────────────────────────
#
# critical  : au moins UNE ancre doit être présente (sinon score = 0 sur ce critère)
# important : idéalement présentes (pondération 20%)
# min_rows  : nombre minimal de lignes attendu (hors en-tête)
# min_cols  : nombre minimal de colonnes de données (hors Rubrique)

_COMPLETENESS_RULES: Dict[str, Dict] = {
    "bilan_actif": {
        "min_rows": 10,
        "min_cols": 2,
        "critical": [
            "total de l'actif",
            "total general",
            "total actif",
            "total de lactif",
            "total general actif",
        ],
        "important": [
            "actif immobilise",
            "actif circulant",
            "tresorerie",
            "immobilisations",
            "actif non courant",
        ],
    },
    "bilan_passif": {
        "min_rows": 8,
        "min_cols": 2,
        "critical": [
            "total du passif",
            "total passif",
            "total general",
            "total du bilan",
        ],
        "important": [
            "capitaux propres",
            "dettes de financement",
            "passif circulant",
            "financement permanent",
            "dettes financieres",
        ],
    },
    "cpc": {
        "min_rows": 8,
        "min_cols": 2,
        "critical": [
            "resultat net",
            "resultat net de l'exercice",
            "resultat de l'exercice",
            "resultat net part du groupe",
        ],
        "important": [
            "produits d'exploitation",
            "charges d'exploitation",
            "chiffre d'affaires",
            "resultat d'exploitation",
            "resultat courant",
            "compte de resultat consolide",
            "interets et produits assimiles",
            "marge d'interet",
            "marge sur commissions",
            "produit net bancaire",
            "resultat brut d'exploitation",
            "resultat net part du groupe",
        ],
    },
    "flux_tresorerie": {
        "min_rows": 6,
        "min_cols": 2,
        "critical": [
            "tresorerie nette",
            "variation de tresorerie",
            "tresorerie fin de periode",
        ],
        "important": [
            "activites operationnelles",
            "activites d'investissement",
            "activites de financement",
        ],
    },
}

# ── Table de correspondance nom → clé ────────────────────────────────────────
_TABLE_NAME_MAP: List[Tuple[str, str]] = [
    ("bilan actif", "bilan_actif"),
    ("bilan passif", "bilan_passif"),
    ("compte de produits et charges", "cpc"),
    ("compte de resultat", "cpc"),
    ("cpc", "cpc"),
    ("compte de produits", "cpc"),
    ("flux de tresorerie", "flux_tresorerie"),
    ("tableau des flux", "flux_tresorerie"),
]


def _normalize(text: str) -> str:
    """Supprime accents et met en minuscules."""
    return (
        unicodedata.normalize("NFD", str(text).lower())
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def _resolve_table_key(table_name: str) -> Optional[str]:
    """Résout le nom d'un tableau en clé de règles."""
    n = _normalize(table_name)
    for pattern, key in _TABLE_NAME_MAP:
        if pattern in n:
            return key
    # Fallback par mots-clés
    if "actif" in n and "bilan" in n:
        return "bilan_actif"
    if "passif" in n and "bilan" in n:
        return "bilan_passif"
    if "produits" in n or "charges" in n:
        return "cpc"
    return None


def check_completeness(df: pd.DataFrame, table_name: str) -> Dict:
    """
    Vérifie la complétude d'un tableau financier extrait.

    Args:
        df: DataFrame du tableau extrait
        table_name: Nom du tableau (ex: "BILAN ACTIF COMPTES SOCIAUX")

    Returns:
        {
            "completeness_score": float (0.0-1.0),
            "is_complete": bool,
            "missing_anchors": list[str],
            "row_count": int,
            "col_count": int,
            "details": dict
        }
    """
    base_result: Dict = {
        "completeness_score": 0.0,
        "is_complete": False,
        "missing_anchors": [],
        "row_count": 0,
        "col_count": 0,
        "details": {},
    }

    if df is None or df.empty:
        base_result["missing_anchors"] = ["tableau_vide"]
        return base_result

    row_count = len(df)
    col_count = len(df.columns)
    base_result["row_count"] = row_count
    base_result["col_count"] = col_count

    key = _resolve_table_key(table_name)
    if key is None or key not in _COMPLETENESS_RULES:
        # Type inconnu : score minimal basé sur la taille
        score = min(1.0, row_count / 8.0) * 0.7 + min(1.0, col_count / 3.0) * 0.3
        base_result["completeness_score"] = round(score, 2)
        base_result["is_complete"] = score >= 0.5
        return base_result

    rules = _COMPLETENESS_RULES[key]
    min_rows = rules["min_rows"]
    min_cols = rules["min_cols"]
    critical_anchors = rules["critical"]
    important_anchors = rules["important"]

    # Texte normalisé de la première colonne (Rubrique) pour recherche d'ancres
    rubrique_text = " ".join(_normalize(str(v)) for v in df.iloc[:, 0].tolist())

    # ── 1. Score lignes (poids 30%) ──────────────────────────────────────────
    row_score = min(1.0, row_count / min_rows)

    # ── 2. Score colonnes (poids 15%) ────────────────────────────────────────
    # Colonnes de données = toutes sauf Rubrique
    data_cols = col_count - 1
    col_score = min(1.0, data_cols / min_cols) if data_cols > 0 else 0.0

    # ── 3. Ancres critiques (poids 40%) ──────────────────────────────────────
    critical_found: List[str] = []
    critical_missing: List[str] = []
    for anchor in critical_anchors:
        if _normalize(anchor) in rubrique_text:
            critical_found.append(anchor)
        else:
            critical_missing.append(anchor)

    has_critical = len(critical_found) > 0
    critical_score = 1.0 if has_critical else 0.0

    # ── 4. Ancres importantes (poids 15%) ────────────────────────────────────
    important_found: List[str] = []
    important_missing: List[str] = []
    for anchor in important_anchors:
        if _normalize(anchor) in rubrique_text:
            important_found.append(anchor)
        else:
            important_missing.append(anchor)

    important_score = len(important_found) / max(1, len(important_anchors))

    # ── Score global ──────────────────────────────────────────────────────────
    score = (
        0.30 * row_score
        + 0.15 * col_score
        + 0.40 * critical_score
        + 0.15 * important_score
    )

    # Ancres manquantes à reporter dans l'API
    missing_anchors: List[str] = []
    if not has_critical:
        # Signaler les 2 ancres critiques les plus importantes
        missing_anchors.extend(critical_missing[:2])
    if len(important_found) < 2:
        missing_anchors.extend(important_missing[:2])

    base_result.update(
        {
            "completeness_score": round(score, 2),
            "is_complete": score >= 0.65 and has_critical,
            "missing_anchors": missing_anchors,
            "details": {
                "table_key": key,
                "row_score": round(row_score, 2),
                "col_score": round(col_score, 2),
                "critical_score": critical_score,
                "important_score": round(important_score, 2),
                "critical_found": critical_found,
                "critical_missing": critical_missing,
                "important_found": important_found,
                "important_missing": important_missing,
            },
        }
    )
    return base_result


def should_retry(completeness: Dict, strategy_index: int, max_strategies: int = 5) -> bool:
    """
    Décide si on doit tenter la stratégie suivante.

    Règles :
    - Si tableau vide → toujours retry
    - Si completeness_score < 0.5 → retry si d'autres stratégies disponibles
    - Si ancres critiques manquantes → retry
    - Si score >= 0.65 et ancres présentes → pas de retry
    """
    if strategy_index >= max_strategies - 1:
        return False

    score = completeness.get("completeness_score", 0.0)
    is_complete = completeness.get("is_complete", False)
    missing = completeness.get("missing_anchors", [])

    if completeness.get("row_count", 0) == 0:
        return True  # Tableau vide
    if is_complete and score >= 0.65:
        return False  # Extraction satisfaisante
    if missing:
        return True  # Ancres manquantes → retry
    if score < 0.5:
        return True  # Score trop bas → retry

    return False
