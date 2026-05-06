"""
Tests unitaires — completeness_checker.py

Tester localement :
    python -m pytest tests/test_completeness.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from rag_agent.completeness_checker import (
    check_completeness,
    should_retry,
    _resolve_table_key,
)


# ── Fixtures DataFrames ───────────────────────────────────────────────────────

def _make_bilan_actif_complete() -> pd.DataFrame:
    """Bilan actif complet avec toutes les ancres critiques."""
    rows = [
        ["Immobilisations en non-valeur", "100 000", "90 000"],
        ["Immobilisations incorporelles", "50 000", "45 000"],
        ["Immobilisations corporelles", "500 000", "480 000"],
        ["Immobilisations financières", "200 000", "180 000"],
        ["Actif immobilisé", "850 000", "795 000"],
        ["Stocks", "120 000", "110 000"],
        ["Créances de l'actif circulant", "300 000", "280 000"],
        ["Actif circulant (hors trésorerie)", "420 000", "390 000"],
        ["Trésorerie - Actif", "80 000", "75 000"],
        ["Écarts de conversion - Actif", "5 000", "4 000"],
        ["Total de l'actif", "1 355 000", "1 264 000"],
    ]
    return pd.DataFrame(rows, columns=["Rubrique", "31/12/2024", "31/12/2023"])


def _make_bilan_actif_incomplete() -> pd.DataFrame:
    """Bilan actif incomplet : pas de TOTAL, peu de lignes."""
    rows = [
        ["Immobilisations en non-valeur", "100 000", "90 000"],
        ["Immobilisations incorporelles", "50 000", "45 000"],
        ["Immobilisations corporelles", "500 000", "480 000"],
    ]
    return pd.DataFrame(rows, columns=["Rubrique", "31/12/2024", "31/12/2023"])


def _make_bilan_passif_complete() -> pd.DataFrame:
    rows = [
        ["Capitaux propres", "600 000", "580 000"],
        ["Capitaux propres assimilés", "50 000", "45 000"],
        ["Dettes de financement", "200 000", "210 000"],
        ["Provisions durables", "30 000", "28 000"],
        ["Financement permanent", "880 000", "863 000"],
        ["Passif circulant (hors trésorerie)", "400 000", "380 000"],
        ["Trésorerie - Passif", "75 000", "21 000"],
        ["Écarts de conversion - Passif", "0", "0"],
        ["Total du passif", "1 355 000", "1 264 000"],
    ]
    return pd.DataFrame(rows, columns=["Rubrique", "31/12/2024", "31/12/2023"])


def _make_cpc_complete() -> pd.DataFrame:
    rows = [
        ["Produits d'exploitation", "1 200 000", "1 100 000"],
        ["Chiffre d'affaires", "1 000 000", "950 000"],
        ["Charges d'exploitation", "900 000", "820 000"],
        ["Résultat d'exploitation", "300 000", "280 000"],
        ["Résultat courant", "280 000", "260 000"],
        ["Résultat avant impôts", "280 000", "260 000"],
        ["Impôts sur les bénéfices", "84 000", "78 000"],
        ["Résultat net de l'exercice", "196 000", "182 000"],
    ]
    return pd.DataFrame(rows, columns=["Rubrique", "Exercice N", "Exercice N-1"])


def _make_empty_df() -> pd.DataFrame:
    return pd.DataFrame()


# ── Tests _resolve_table_key ──────────────────────────────────────────────────

def test_resolve_bilan_actif():
    assert _resolve_table_key("BILAN ACTIF COMPTES SOCIAUX") == "bilan_actif"
    assert _resolve_table_key("bilan actif") == "bilan_actif"
    assert _resolve_table_key("Bilan Actif Consolidé") == "bilan_actif"


def test_resolve_bilan_passif():
    assert _resolve_table_key("BILAN PASSIF") == "bilan_passif"
    assert _resolve_table_key("bilan passif comptes sociaux") == "bilan_passif"


def test_resolve_cpc():
    assert _resolve_table_key("COMPTE DE PRODUITS ET CHARGES") == "cpc"
    assert _resolve_table_key("CPC") == "cpc"
    assert _resolve_table_key("Compte de Résultat") == "cpc"


def test_resolve_unknown():
    assert _resolve_table_key("tableau inconnnu xyz") is None


# ── Tests check_completeness ─────────────────────────────────────────────────

def test_completeness_empty_df():
    result = check_completeness(_make_empty_df(), "BILAN ACTIF")
    assert result["completeness_score"] == 0.0
    assert result["is_complete"] is False
    assert "tableau_vide" in result["missing_anchors"]


def test_completeness_bilan_actif_complet():
    df = _make_bilan_actif_complete()
    result = check_completeness(df, "BILAN ACTIF COMPTES SOCIAUX")
    assert result["completeness_score"] >= 0.65
    assert result["is_complete"] is True
    assert result["row_count"] == len(df)
    assert result["col_count"] == 3


def test_completeness_bilan_actif_incomplet():
    df = _make_bilan_actif_incomplete()
    result = check_completeness(df, "BILAN ACTIF")
    # Pas de TOTAL → score critique = 0 → is_complete = False
    assert result["is_complete"] is False
    assert len(result["missing_anchors"]) > 0


def test_completeness_bilan_passif_complet():
    df = _make_bilan_passif_complete()
    result = check_completeness(df, "BILAN PASSIF")
    assert result["completeness_score"] >= 0.65
    assert result["is_complete"] is True


def test_completeness_cpc_complet():
    df = _make_cpc_complete()
    result = check_completeness(df, "COMPTE DE PRODUITS ET CHARGES")
    assert result["completeness_score"] >= 0.65
    assert result["is_complete"] is True


def test_completeness_score_range():
    """Le score doit toujours être dans [0.0, 1.0]."""
    for df, name in [
        (_make_bilan_actif_complete(), "BILAN ACTIF"),
        (_make_bilan_actif_incomplete(), "BILAN ACTIF"),
        (_make_bilan_passif_complete(), "BILAN PASSIF"),
        (_make_cpc_complete(), "CPC"),
        (_make_empty_df(), "BILAN ACTIF"),
    ]:
        result = check_completeness(df, name)
        assert 0.0 <= result["completeness_score"] <= 1.0, (
            f"Score hors limites pour {name}: {result['completeness_score']}"
        )


def test_completeness_details_structure():
    """Les détails doivent avoir la structure attendue."""
    df = _make_bilan_actif_complete()
    result = check_completeness(df, "BILAN ACTIF")
    details = result["details"]
    assert "table_key" in details
    assert "row_score" in details
    assert "critical_found" in details
    assert "important_found" in details


# ── Tests should_retry ────────────────────────────────────────────────────────

def test_should_retry_empty():
    completeness = {"completeness_score": 0.0, "is_complete": False, "missing_anchors": [], "row_count": 0}
    assert should_retry(completeness, 0) is True


def test_should_retry_complete():
    completeness = {"completeness_score": 0.9, "is_complete": True, "missing_anchors": [], "row_count": 15}
    assert should_retry(completeness, 0) is False


def test_should_retry_missing_anchors():
    completeness = {
        "completeness_score": 0.4,
        "is_complete": False,
        "missing_anchors": ["total de l'actif"],
        "row_count": 8,
    }
    assert should_retry(completeness, 1) is True


def test_should_retry_last_strategy():
    """Au dernier niveau de stratégie, ne plus retry."""
    completeness = {"completeness_score": 0.3, "is_complete": False, "missing_anchors": ["total"], "row_count": 5}
    assert should_retry(completeness, 4, max_strategies=5) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
