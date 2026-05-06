from __future__ import annotations

from .keyword_dictionary import FINANCIAL_ANCHORS, SCOPE_KEYWORDS, SECTOR_KEYWORDS, TABLE_TITLES, keyword_hits
from .utils import normalize_text


def detect_scope_candidates(page_text: str) -> list[str]:
    scored: list[tuple[int, str]] = []
    for scope, keywords in SCOPE_KEYWORDS.items():
        hits = keyword_hits(page_text, keywords)
        if hits:
            scored.append((len(hits), scope))
    scored.sort(reverse=True)
    return [scope for _, scope in scored]


def detect_scope_ranges(page_texts: dict[int, str]) -> dict[int, list[str]]:
    """Propagate strong section headings so table pages inherit the right scope."""
    propagated: dict[int, list[str]] = {}
    current_scope: str | None = None
    for page_number in sorted(page_texts):
        direct = detect_scope_candidates(page_texts[page_number])
        if direct:
            current_scope = direct[0]
            propagated[page_number] = direct
        elif current_scope:
            propagated[page_number] = [current_scope]
        else:
            propagated[page_number] = []
    return propagated


def detect_sector(page_texts: dict[int, str], explicit_sector: str | None = None) -> str:
    corpus = "\n".join(page_texts.values())
    if explicit_sector:
        corrected = _correct_explicit_sector(corpus, explicit_sector)
        if corrected:
            return corrected
        return explicit_sector
    scores = {
        sector: len(keyword_hits(corpus, keywords))
        for sector, keywords in SECTOR_KEYWORDS.items()
    }
    return max(scores.items(), key=lambda item: item[1])[0]


def _correct_explicit_sector(corpus: str, explicit_sector: str) -> str | None:
    """Allow PDF-native statement structure to fix a too-broad UI/API sector guess."""
    bancaire_hits = keyword_hits(
        corpus,
        [
            "societe generale marocaine de banques",
            "agrement en qualite d'etablissement de credit",
            "agrement en qualite d etablissement de credit",
            "dettes envers les etablissements de credit",
            "creances sur les etablissements de credit",
            "creances sur la clientele",
            "depots de la clientele",
            "titres de creance emis",
            "produits d'exploitation bancaire",
            "charges d'exploitation bancaire",
            "produit net bancaire",
            "immobilisations donnees en credit-bail",
            "operations de credit-bail",
            "pcec",
            "bank al-maghrib",
            "circulaire n 19/g/2002",
        ],
    )
    assurance_hits = keyword_hits(
        corpus,
        [
            "provisions techniques",
            "primes emises",
            "primes acquises",
            "charges de sinistres",
            "placements representatifs",
            "operations d'assurance",
            "cessionnaires",
        ],
    )

    if explicit_sector != "bancaire_sdf" and len(bancaire_hits) >= 3 and len(bancaire_hits) > len(assurance_hits):
        return "bancaire_sdf"
    if explicit_sector != "assurance" and len(assurance_hits) >= 3 and len(assurance_hits) > len(bancaire_hits):
        return "assurance"
    return None


def detect_titles(page_text: str) -> list[str]:
    titles: list[str] = []
    normalized = "\n" + normalize_text(page_text) + "\n"
    strong_patterns = {
        "BILAN_ACTIF": ["\nbilan actif\n", "\nactif\n", "bilan (actif)", "bilan actif"],
        "BILAN_PASSIF": ["\nbilan passif\n", "\npassif\n", "bilan (passif)", "bilan passif"],
        "CPC": [
            "compte de produits et charges",
            "compte de produits et de charges",
            "compte de resultat",
            "etat du resultat global",
            "cpc consolide",
        ],
    }
    for table_type in TABLE_TITLES:
        if any(pattern in normalized for pattern in strong_patterns.get(table_type, [])):
            titles.append(table_type)
    return titles


def detect_target_candidates(page_text: str, sector: str) -> list[str]:
    candidates: list[str] = []
    for table_type in TABLE_TITLES:
        title_hits = keyword_hits(page_text, TABLE_TITLES[table_type])
        anchor_hits = keyword_hits(page_text, FINANCIAL_ANCHORS.get(sector, {}).get(table_type, []))
        if title_hits or len(anchor_hits) >= 2:
            candidates.append(table_type)
    return candidates


def detect_anchors(page_text: str, sector: str) -> list[str]:
    anchors: list[str] = []
    for table_anchors in FINANCIAL_ANCHORS.get(sector, {}).values():
        anchors.extend(keyword_hits(page_text, table_anchors))
    return sorted(set(anchors))
