"""
Tests unitaires — multi_table_detector.py

Tester localement :
    python -m pytest tests/test_multi_table.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from rag_agent.multi_table_detector import (
    TableBlock,
    _detect_table_type_from_text,
    _is_table_boundary,
    find_target_block,
    is_multi_table_page,
)


# ── Tests _detect_table_type_from_text ────────────────────────────────────────

def test_detect_bilan_actif():
    assert _detect_table_type_from_text("BILAN ACTIF") == "bilan_actif"
    assert _detect_table_type_from_text("Actif (en MAD)") == "bilan_actif"
    assert _detect_table_type_from_text("IMMOBILISATIONS EN NON-VALEUR") == "bilan_actif"


def test_detect_bilan_passif():
    assert _detect_table_type_from_text("BILAN PASSIF") == "bilan_passif"
    assert _detect_table_type_from_text("Passif (en MAD)") == "bilan_passif"


def test_detect_cpc():
    assert _detect_table_type_from_text("COMPTE DE PRODUITS ET CHARGES") == "cpc"
    assert _detect_table_type_from_text("Produits d'exploitation bancaire") == "cpc"


def test_detect_unknown():
    assert _detect_table_type_from_text("Texte quelconque sans mots-clés") is None


# ── Tests _is_table_boundary ──────────────────────────────────────────────────

def test_is_table_boundary_true():
    is_b, kw = _is_table_boundary("BILAN ACTIF")
    assert is_b is True
    assert kw == "bilan actif"


def test_is_table_boundary_passif():
    is_b, kw = _is_table_boundary("BILAN PASSIF COMPTES SOCIAUX")
    assert is_b is True


def test_is_table_boundary_false():
    is_b, _ = _is_table_boundary("1 500 000")
    assert is_b is False

    is_b, _ = _is_table_boundary("Notes")
    assert is_b is False


# ── Tests find_target_block ────────────────────────────────────────────────────

def _make_blocks():
    return [
        TableBlock(
            page_num=5,
            y_start=50.0,
            y_end=300.0,
            x_start=0.0,
            x_end=595.0,
            text_sample="BILAN ACTIF Immobilisations en non-valeur",
            table_type_hint="bilan_actif",
            block_index=0,
        ),
        TableBlock(
            page_num=5,
            y_start=320.0,
            y_end=600.0,
            x_start=0.0,
            x_end=595.0,
            text_sample="BILAN PASSIF Capitaux propres Dettes de financement",
            table_type_hint="bilan_passif",
            block_index=1,
        ),
        TableBlock(
            page_num=5,
            y_start=620.0,
            y_end=820.0,
            x_start=0.0,
            x_end=595.0,
            text_sample="COMPTE DE PRODUITS ET CHARGES Produits exploitation",
            table_type_hint="cpc",
            block_index=2,
        ),
    ]


def test_find_target_bilan_actif():
    blocks = _make_blocks()
    target = find_target_block(blocks, "BILAN ACTIF COMPTES SOCIAUX")
    assert target is not None
    assert target.table_type_hint == "bilan_actif"
    assert target.block_index == 0


def test_find_target_bilan_passif():
    blocks = _make_blocks()
    target = find_target_block(blocks, "BILAN PASSIF")
    assert target is not None
    assert target.table_type_hint == "bilan_passif"
    assert target.block_index == 1


def test_find_target_cpc():
    blocks = _make_blocks()
    target = find_target_block(blocks, "COMPTE DE PRODUITS ET CHARGES")
    assert target is not None
    assert target.table_type_hint == "cpc"
    assert target.block_index == 2


def test_find_target_empty_blocks():
    assert find_target_block([], "BILAN ACTIF") is None


def test_find_target_no_match_returns_none():
    """Sans match fiable, retourner None pour forcer un fallback plus sûr."""
    blocks = _make_blocks()
    target = find_target_block(blocks, "Tableau inconnu xyz")
    assert target is None


# ── Tests is_multi_table_page ─────────────────────────────────────────────────

def test_is_multi_table_false():
    assert is_multi_table_page([]) is False
    assert is_multi_table_page([_make_blocks()[0]]) is False


def test_is_multi_table_true():
    assert is_multi_table_page(_make_blocks()) is True
    assert is_multi_table_page(_make_blocks()[:2]) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
