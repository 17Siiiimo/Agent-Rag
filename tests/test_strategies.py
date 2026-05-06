"""
Tests unitaires — extraction_strategies.py

Tester localement :
    python -m pytest tests/test_strategies.py -v

Notes :
- Les tests LLM (groq_vision_extract, groq_structured_json_extract) nécessitent GROQ_API_KEY.
  Ils sont ignorés si la clé est absente.
- Les tests de parsing (markdown→df, json→df) sont toujours exécutés.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest

from rag_agent.extraction_strategies import (
    _extract_markdown_table,
    _is_valid_df,
    _json_payload_to_df,
    _markdown_table_to_df,
    GROQ_TEXT_MODEL,
    GROQ_VISION_MODEL,
    GEMINI_MODELS_2025,
)


# ── Tests _is_valid_df ────────────────────────────────────────────────────────

def test_is_valid_df_empty():
    assert _is_valid_df(pd.DataFrame()) is False


def test_is_valid_df_too_small():
    df = pd.DataFrame([["a", "1"]], columns=["R", "V"])
    assert _is_valid_df(df) is False  # < 3 lignes


def test_is_valid_df_valid():
    df = pd.DataFrame(
        [["Actif", "100 000", "90 000"],
         ["Passif", "200 000", "180 000"],
         ["Total", "300 000", "270 000"]],
        columns=["Rubrique", "N", "N-1"],
    )
    assert _is_valid_df(df) is True


def test_is_valid_df_no_numbers():
    df = pd.DataFrame(
        [["a", "b"], ["c", "d"], ["e", "f"]],
        columns=["R1", "R2"],
    )
    assert _is_valid_df(df) is False


# ── Tests _json_payload_to_df ─────────────────────────────────────────────────

def test_json_payload_to_df_valid():
    payload = {
        "table_name": "BILAN ACTIF",
        "headers": ["Rubrique", "N", "N-1"],
        "rows": [
            ["Immobilisations", "500 000", "480 000"],
            ["Actif circulant", "300 000", "290 000"],
            ["Total", "800 000", "770 000"],
        ],
    }
    df = _json_payload_to_df(payload)
    assert not df.empty
    assert list(df.columns) == ["Rubrique", "N", "N-1"]
    assert len(df) == 3


def test_json_payload_to_df_empty_rows():
    payload = {"table_name": "X", "headers": ["R", "N"], "rows": []}
    df = _json_payload_to_df(payload)
    assert df.empty


def test_json_payload_to_df_missing_headers():
    payload = {"table_name": "X", "rows": [["a", "b"]]}
    df = _json_payload_to_df(payload)
    assert df.empty


def test_json_payload_to_df_row_length_mismatch():
    """Les lignes courtes doivent être complétées avec ''."""
    payload = {
        "headers": ["R", "N", "N-1"],
        "rows": [
            ["Ligne1", "100"],  # Manque N-1
            ["Ligne2", "200", "190"],
            ["Total", "300", "290"],
        ],
    }
    df = _json_payload_to_df(payload)
    assert len(df.columns) == 3
    assert df.iloc[0, 2] == ""  # Colonne manquante complétée


# ── Tests _extract_markdown_table ─────────────────────────────────────────────

def test_extract_markdown_table_clean():
    md = """| Rubrique | N | N-1 |
| --- | --- | --- |
| Actif | 100 | 90 |
| Total | 100 | 90 |"""
    result = _extract_markdown_table(md)
    assert "| Rubrique |" in result
    assert "| Total |" in result


def test_extract_markdown_table_with_preamble():
    content = """Voici le tableau extrait :

| Rubrique | N | N-1 |
| --- | --- | --- |
| Actif | 100 | 90 |
| Total | 100 | 90 |

Fin du tableau."""
    result = _extract_markdown_table(content)
    assert "| Rubrique |" in result


def test_extract_markdown_table_empty():
    assert _extract_markdown_table("") == ""
    assert _extract_markdown_table("Texte sans tableau") == "Texte sans tableau"


# ── Tests _markdown_table_to_df ───────────────────────────────────────────────

def test_markdown_table_to_df_valid():
    md = """| Rubrique | N | N-1 |
| --- | --- | --- |
| Immobilisations | 500 000 | 480 000 |
| Actif circulant | 300 000 | 290 000 |
| Total de l'actif | 800 000 | 770 000 |"""
    df = _markdown_table_to_df(md)
    assert not df.empty
    assert list(df.columns) == ["Rubrique", "N", "N-1"]
    assert len(df) == 3
    assert "Total de l'actif" in df["Rubrique"].values


def test_markdown_table_to_df_empty():
    df = _markdown_table_to_df("")
    assert df.empty


def test_markdown_table_to_df_preserves_numbers():
    """Les nombres avec espaces doivent être préservés."""
    md = """| Rubrique | Montant |
| --- | --- |
| Total | 13 157 683 |
| Autre | 1 000 000 |
| Fin | 14 157 683 |"""
    df = _markdown_table_to_df(md)
    assert "13 157 683" in df["Montant"].values


# ── Tests modèles / configuration ────────────────────────────────────────────

def test_groq_text_model_is_set():
    assert GROQ_TEXT_MODEL == "llama-3.3-70b-versatile"


def test_groq_vision_model_is_maverick():
    # Le compte peut utiliser Scout comme modèle vision stable, avec Maverick en texte.
    assert any(name in GROQ_VISION_MODEL.lower() for name in ("maverick", "scout"))


def test_gemini_models_priority():
    """gemini-2.0-flash doit être en première position (priorité 2025)."""
    assert GEMINI_MODELS_2025[0] == "gemini-2.0-flash"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
