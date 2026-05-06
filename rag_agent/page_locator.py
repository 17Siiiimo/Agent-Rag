"""
Localisation déterministe de pages de tableaux financiers via PyMuPDF.

Stratégie : scan texte page par page + scoring basé sur les signatures
financières déjà définies dans extraction_agent._FINANCIAL_SIGNATURES.
+ filtrage de section (sociaux / consolidés) pour éviter les faux positifs
  quand les deux sections coexistent dans le même PDF.

Aucun RAG, aucun embedding — 100 % déterministe.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from .extraction_agent import (
    _FINANCIAL_SIGNATURES,
    _contains_kw,
    _is_toc_chunk,
    _norm,
)

# ---------------------------------------------------------------------------
# Types publics
# ---------------------------------------------------------------------------

TableType = Literal["ACTIF", "PASSIF", "CPC"]
ReportType = Literal["sociaux", "consolidés", "auto"]

# Marqueurs forts de début de section (cherchés dans les 300 premiers caractères de la page)
_CONSOLID_START_MARKERS = [
    "bilan consolid",
    "comptes consolid",
    "etats financiers consolid",
    "compte de resultat consolid",
    "i. bilan consolid",
    "1.1. bilan consolid",
]
_SOCIAL_START_MARKERS = [
    "comptes sociaux",
    "etats financiers sociaux",
    "etats financiers individuels",
    "attestation d'examen",    # page attestation = début section sociaux
    "attestation des commissaires",
    "bilan social au",
]
# Marqueurs de contexte général (cherchés sur toute la page, poids moins fort)
_CONSOLID_CONTEXT = ["consolid"]
_SOCIAL_CONTEXT   = ["comptes sociaux", "etats financiers sociaux", "etats financiers individuels"]

# Bonus / pénalité appliqués selon l'appartenance de section
_SECTION_BONUS   =  50   # page dans la bonne section
_SECTION_PENALTY = -80   # page dans la mauvaise section (fort, pour écarter les doublons)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PageScore:
    page: int          # 1-based
    score: int
    matched: list[str] = field(default_factory=list)


@dataclass
class LocatorResult:
    best_page: int | None
    score: int
    matched: list[str]
    top_k: list[PageScore]


# ---------------------------------------------------------------------------
# Détection des sections sociaux / consolidés dans le PDF
# ---------------------------------------------------------------------------

def _detect_section_ranges(
    doc,  # pymupdf.Document
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Retourne (social_ranges, consolid_ranges) en numérotation 1-based.

    Stratégie "propagation" :
    1. Détecte les pages de début de section (forte marqueur dans les 300 premiers chars).
    2. Propage le label vers l'avant jusqu'au prochain changement de section.
    3. Les pages sans marqueur explicite héritent du label de la dernière page marquée.
    """
    n = len(doc)
    # label[i] : "social" | "consolid" | None
    labels: list[str | None] = [None] * n

    for i in range(n):
        raw = doc[i].get_text() or ""
        head = _norm(raw[:400])   # cherche dans les 400 premiers caractères
        full = _norm(raw)

        is_consolid_start = any(m in head for m in _CONSOLID_START_MARKERS)
        is_social_start   = any(m in head for m in _SOCIAL_START_MARKERS)

        # Fallback : marqueurs de contexte sur toute la page (moins fort)
        if not is_consolid_start and not is_social_start:
            is_consolid_start = any(m in full for m in _CONSOLID_START_MARKERS)
            is_social_start   = any(m in full for m in _SOCIAL_START_MARKERS)

        if is_consolid_start and not is_social_start:
            labels[i] = "consolid"
        elif is_social_start and not is_consolid_start:
            labels[i] = "social"
        elif is_consolid_start and is_social_start:
            # Page mixte (ex: résumé avec les deux) → regarde le contexte dominant
            c_count = sum(1 for m in _CONSOLID_CONTEXT if m in full)
            s_count = sum(1 for m in _SOCIAL_CONTEXT   if m in full)
            labels[i] = "consolid" if c_count >= s_count else "social"

    # Propagation vers l'avant : chaque page hérite de la dernière section connue
    last_known: str | None = None
    for i in range(n):
        if labels[i] is not None:
            last_known = labels[i]
        elif last_known is not None:
            labels[i] = last_known

    # Construction des ranges à partir des labels propagés
    social_pages   = [i + 1 for i, l in enumerate(labels) if l == "social"]
    consolid_pages = [i + 1 for i, l in enumerate(labels) if l == "consolid"]

    def _to_ranges(pages: list[int]) -> list[tuple[int, int]]:
        if not pages:
            return []
        s = sorted(set(pages))
        ranges: list[tuple[int, int]] = []
        start = prev = s[0]
        for p in s[1:]:
            if p - prev == 1:   # pages strictement consécutives seulement
                prev = p
            else:
                ranges.append((start, prev))
                start = prev = p
        ranges.append((start, prev))
        return ranges

    return _to_ranges(social_pages), _to_ranges(consolid_pages)


def _page_in_ranges(page: int, ranges: list[tuple[int, int]], margin: int = 0) -> bool:
    """True si la page est dans l'un des ranges (marge=0 car propagation déjà faite)."""
    for a, b in ranges:
        if a - margin <= page <= b + margin:
            return True
    return False


# ---------------------------------------------------------------------------
# Helpers de scoring
# ---------------------------------------------------------------------------

def _select_primaries(table_type: TableType, report_type: ReportType) -> list[str]:
    is_consolid = report_type in ("consolidés", "consolides", "consolide")

    if is_consolid:
        if table_type == "ACTIF":
            return ["bilan_actif_ifrs", "bilan_actif_cgnc", "bilan_actif_pcec", "bilan_actif_assurance"]
        if table_type == "PASSIF":
            return ["bilan_passif_cgnc_consolide", "bilan_passif_ifrs", "bilan_passif_cgnc", "bilan_passif_pcec", "bilan_passif_assurance"]
        return ["cpc_ifrs_bancaire", "cpc_ifrs", "cpc_pcec", "cpc_cgnc", "cpc_assurance"]
    else:
        if table_type == "ACTIF":
            return ["bilan_actif_cgnc", "bilan_actif_pcec", "bilan_actif_assurance"]
        if table_type == "PASSIF":
            return ["bilan_passif_cgnc", "bilan_passif_pcec", "bilan_passif_assurance"]
        return ["cpc_cgnc", "cpc_pcec", "cpc_assurance"]


def _score_page(
    page_text: str,
    table_type: TableType,
    report_type: ReportType,
) -> tuple[int, list[str]]:
    """Score une page et retourne (score, mots-clés matchés)."""
    if not page_text or not page_text.strip():
        return 0, []

    if _is_toc_chunk(page_text):
        return -30, ["[TABLE DES MATIÈRES]"]

    text = _norm(page_text)
    text_ns = re.sub(r"\s+", "", text)

    primaries = _select_primaries(table_type, report_type)

    best_positive = 0
    best_matched: list[str] = []

    for sig_name in primaries:
        sig = _FINANCIAL_SIGNATURES.get(sig_name)
        if not sig:
            continue

        s = 0
        matched: list[str] = []
        anchor_hits = 0

        for kw, pts in sig["anchors"]:
            if _contains_kw(text, kw, text_ns):
                s += pts
                anchor_hits += 1
                matched.append(kw)

        for kw, pts in sig["titles"]:
            if _contains_kw(text, kw, text_ns):
                s += pts
                matched.append(f"[titre] {kw}")

        for kw, pts in sig["context"]:
            if _contains_kw(text, kw, text_ns):
                s += pts
                matched.append(f"[ctx] {kw}")

        na = len(sig["anchors"])
        min_anchors = 3 if na <= 10 else 4 if na <= 20 else 5 if na <= 35 else 7

        if anchor_hits >= min_anchors and s > best_positive:
            best_positive = s
            best_matched = matched

    score = best_positive

    # Pénalités (une seule fois par mot-clé)
    penalized: set[str] = set()
    for sig_name in primaries:
        sig = _FINANCIAL_SIGNATURES.get(sig_name)
        if not sig:
            continue
        for kw, pts in sig["penalties"]:
            if _contains_kw(text, kw, text_ns) and kw not in penalized:
                score += pts
                penalized.add(kw)

    # Bonus date de clôture
    if re.search(r"31/12|31-12|30/06|30-06", page_text):
        score += 5

    # Légère pénalité si texte trop narratif (peu de chiffres)
    words = page_text.split()
    if len(words) > 200:
        digit_ratio = sum(1 for w in words if re.search(r"\d", w)) / len(words)
        if digit_ratio < 0.05:
            score -= 5

    return score, best_matched


# ---------------------------------------------------------------------------
# API publique
# ---------------------------------------------------------------------------

def find_page_by_keywords(
    pdf_path: str,
    table_type: TableType,
    report_type: ReportType = "auto",
    top_k: int = 3,
) -> LocatorResult:
    """
    Localise la page la plus probable pour un tableau financier donné.

    Args:
        pdf_path:    Chemin absolu vers le PDF.
        table_type:  "ACTIF", "PASSIF" ou "CPC".
        report_type: "sociaux", "consolidés" ou "auto".
        top_k:       Nombre de candidats à inclure dans le résultat.

    Returns:
        LocatorResult avec best_page (1-based), score, mots-clés matchés, top_k.
    """
    try:
        import pymupdf  # type: ignore
    except ImportError:
        raise ImportError("pymupdf est requis : pip install pymupdf")

    table_type = table_type.upper()  # type: ignore
    if table_type not in ("ACTIF", "PASSIF", "CPC"):
        raise ValueError(f"table_type invalide : {table_type!r}. Valeurs : ACTIF, PASSIF, CPC")

    is_consolid = report_type in ("consolidés", "consolides", "consolide")
    is_sociaux  = report_type in ("sociaux", "social")
    apply_section_filter = is_consolid or is_sociaux

    doc = pymupdf.open(pdf_path)
    try:
        # Détection des sections si le contexte est spécifié
        social_ranges: list[tuple[int, int]] = []
        consolid_ranges: list[tuple[int, int]] = []
        if apply_section_filter:
            social_ranges, consolid_ranges = _detect_section_ranges(doc)

        scores: list[PageScore] = []

        for i in range(len(doc)):
            page_text = doc[i].get_text() or ""
            s, matched = _score_page(page_text, table_type, report_type)  # type: ignore

            # Filtre de section : bonus/pénalité selon appartenance
            if apply_section_filter and s > 0:
                page_num = i + 1
                in_social   = _page_in_ranges(page_num, social_ranges)
                in_consolid = _page_in_ranges(page_num, consolid_ranges)

                if is_sociaux:
                    if in_social and not in_consolid:
                        s += _SECTION_BONUS
                        matched = matched + ["[section] sociaux"]
                    elif in_consolid and not in_social:
                        s += _SECTION_PENALTY
                        matched = matched + ["[section-hors] consolidés"]

                elif is_consolid:
                    if in_consolid and not in_social:
                        s += _SECTION_BONUS
                        matched = matched + ["[section] consolidés"]
                    elif in_social and not in_consolid:
                        s += _SECTION_PENALTY
                        matched = matched + ["[section-hors] sociaux"]

            scores.append(PageScore(page=i + 1, score=s, matched=matched))
    finally:
        doc.close()

    # Tri principal : score décroissant.
    # Tie-breaker : quand deux pages sont à moins de 15 pts d'écart, préférer
    # la première dans le document (la table principale précède les sous-rapports).
    scores.sort(key=lambda x: (-x.score, x.page))

    # Reclasser : si la 2e page est dans les 15 pts du leader ET a un numéro inférieur, elle gagne.
    if len(scores) >= 2 and scores[0].score - scores[1].score <= 15:
        candidates = [s for s in scores if scores[0].score - s.score <= 15]
        candidates.sort(key=lambda x: x.page)
        # Remonter le plus ancien uniquement si dans la même section
        winner = candidates[0]
        # Remet le gagnant en tête
        scores = [winner] + [s for s in scores if s.page != winner.page]

    best = scores[0] if scores else None
    best_page = best.page if (best and best.score > 0) else None

    return LocatorResult(
        best_page=best_page,
        score=best.score if best else 0,
        matched=best.matched if best else [],
        top_k=scores[:top_k],
    )


def score_pages_for_table(
    pdf_path: str,
    table_type: TableType,
    report_type: ReportType = "auto",
) -> list[PageScore]:
    """
    Retourne la liste complète des pages scorées, triée par score décroissant.
    Utile pour le débogage ou l'inspection manuelle.
    """
    result = find_page_by_keywords(pdf_path, table_type, report_type, top_k=9999)
    return result.top_k


# ---------------------------------------------------------------------------
# Localisation groupée avec contrainte de proximité
# ---------------------------------------------------------------------------

_PROXIMITY_BONUS = 35   # bonus si la page est dans la fenêtre de l'ancre


def find_tables_for_section(
    pdf_path: str,
    report_type: ReportType = "auto",
    proximity_window: int = 8,
    top_k: int = 3,
) -> dict[str, LocatorResult]:
    """
    Localise ACTIF, PASSIF et CPC pour une section donnée en une seule passe PDF.

    Exploite la proximité naturelle des tableaux :
    ACTIF et PASSIF sont souvent sur la même page ou à 1-2 pages de distance,
    le CPC suit dans les pages suivantes (décalage de quelques pages max).

    Stratégie :
    1. Score toutes les pages pour les 3 types en une seule lecture.
    2. Utilise PASSIF comme ancre (le mieux discriminé).
    3. Applique un bonus de proximité à ACTIF et CPC si leur meilleur candidat
       est dans la fenêtre [ancre - window, ancre + window].

    Args:
        pdf_path:         Chemin absolu vers le PDF.
        report_type:      "sociaux", "consolidés" ou "auto".
        proximity_window: Fenêtre de pages autour de l'ancre (défaut : 8).
        top_k:            Nombre de candidats dans chaque LocatorResult.

    Returns:
        Dict avec clés "ACTIF", "PASSIF", "CPC" → LocatorResult.
    """
    try:
        import pymupdf  # type: ignore
    except ImportError:
        raise ImportError("pymupdf est requis : pip install pymupdf")

    is_consolid = report_type in ("consolidés", "consolides", "consolide")
    is_sociaux  = report_type in ("sociaux", "social")
    apply_section_filter = is_consolid or is_sociaux

    doc = pymupdf.open(pdf_path)
    try:
        social_ranges: list[tuple[int, int]] = []
        consolid_ranges: list[tuple[int, int]] = []
        if apply_section_filter:
            social_ranges, consolid_ranges = _detect_section_ranges(doc)

        # Lecture unique — score les 3 types simultanément
        raw: dict[str, list[PageScore]] = {"ACTIF": [], "PASSIF": [], "CPC": []}

        for i in range(len(doc)):
            page_text = doc[i].get_text() or ""
            page_num  = i + 1

            for ttype in ("ACTIF", "PASSIF", "CPC"):
                s, matched = _score_page(page_text, ttype, report_type)  # type: ignore

                if apply_section_filter and s > 0:
                    in_social   = _page_in_ranges(page_num, social_ranges)
                    in_consolid = _page_in_ranges(page_num, consolid_ranges)

                    if is_sociaux:
                        if in_social and not in_consolid:
                            s += _SECTION_BONUS
                            matched = matched + ["[section] sociaux"]
                        elif in_consolid and not in_social:
                            s += _SECTION_PENALTY
                            matched = matched + ["[section-hors] consolidés"]
                    elif is_consolid:
                        if in_consolid and not in_social:
                            s += _SECTION_BONUS
                            matched = matched + ["[section] consolidés"]
                        elif in_social and not in_consolid:
                            s += _SECTION_PENALTY
                            matched = matched + ["[section-hors] sociaux"]

                raw[ttype].append(PageScore(page=page_num, score=s, matched=matched))
    finally:
        doc.close()

    def _sort_with_tiebreaker(scores: list[PageScore]) -> list[PageScore]:
        scores.sort(key=lambda x: (-x.score, x.page))
        if len(scores) >= 2 and scores[0].score - scores[1].score <= 15:
            candidates = [s for s in scores if scores[0].score - s.score <= 15]
            candidates.sort(key=lambda x: x.page)
            winner = candidates[0]
            scores = [winner] + [s for s in scores if s.page != winner.page]
        return scores

    # Ancre = PASSIF (meilleure discrimination)
    passif_scores = _sort_with_tiebreaker(raw["PASSIF"])
    anchor_page   = passif_scores[0].page if (passif_scores and passif_scores[0].score > 0) else None

    results: dict[str, LocatorResult] = {}

    for ttype in ("ACTIF", "PASSIF", "CPC"):
        scores = raw[ttype][:]

        # Bonus de proximité par rapport à l'ancre PASSIF
        if anchor_page is not None and ttype != "PASSIF":
            for ps in scores:
                if ps.score > 0 and abs(ps.page - anchor_page) <= proximity_window:
                    ps.score  += _PROXIMITY_BONUS
                    ps.matched = ps.matched + [f"[proximité ancre p{anchor_page}]"]

        scores = _sort_with_tiebreaker(scores)

        best = scores[0] if scores else None
        best_page = best.page if (best and best.score > 0) else None

        results[ttype] = LocatorResult(
            best_page=best_page,
            score=best.score if best else 0,
            matched=best.matched if best else [],
            top_k=scores[:top_k],
        )

    return results
