from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .keyword_dictionary import FINANCIAL_ANCHORS, NEGATIVE_ANCHORS, SCOPE_KEYWORDS, TABLE_TITLES, keyword_hits, target_query
from .models import PipelineConfig, RetrievedPage
from .page_indexer import PageAwareIndex
from .utils import normalize_text


def retrieve_candidate_pages(
    index: PageAwareIndex,
    target_table: str,
    scope: str,
    sector: str,
    cfg: PipelineConfig,
) -> list[RetrievedPage]:
    query = target_query(target_table, scope, sector)
    bm25_scores = _bm25_scores([p.embedding_text() for p in index.pages], query)
    vector_scores = _vector_scores(index, query, cfg.embedding_model)

    results: list[RetrievedPage] = []
    for i, page in enumerate(index.pages):
        target_anchor_score, matched_anchors = _target_anchor_score(page.page_text, target_table, sector)
        scope_score, scope_evidence = _scope_score(page.page_text, page.scope_candidates, scope)
        sector_score = 1.0 if page.sector == sector else 0.0
        title_score, title_evidence = _title_score(page.page_text, page.detected_titles, target_table)
        bm25 = bm25_scores[i] if i < len(bm25_scores) else 0.0
        vector = vector_scores[i] if i < len(vector_scores) else 0.0
        score = (
            cfg.bm25_weight * bm25
            + cfg.vector_weight * vector
            + cfg.target_anchor_weight * target_anchor_score
            + cfg.scope_weight * scope_score
            + cfg.sector_weight * sector_score
            + cfg.title_weight * title_score
        )
        evidence = scope_evidence + title_evidence
        evidence += [f"anchor:{anchor}" for anchor in matched_anchors[:10]]
        if sector_score:
            evidence.append(f"sector:{sector}")
        results.append(
            RetrievedPage(
                page=page,
                score=float(score),
                bm25_score=float(bm25),
                vector_score=float(vector),
                target_anchor_score=float(target_anchor_score),
                scope_score=float(scope_score),
                sector_score=float(sector_score),
                title_score=float(title_score),
                matched_anchors=matched_anchors,
                evidence=evidence,
            )
        )
    results.sort(key=lambda r: (-r.score, r.page.page_number))
    return results[: cfg.top_k_pages]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", normalize_text(text))


def _bm25_scores(docs: list[str], query: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
    tokenized = [_tokenize(d) for d in docs]
    query_terms = _tokenize(query)
    if not tokenized or not query_terms:
        return [0.0 for _ in docs]
    doc_lens = [len(d) or 1 for d in tokenized]
    avgdl = sum(doc_lens) / len(doc_lens)
    dfs = Counter(term for doc in tokenized for term in set(doc))
    raw_scores: list[float] = []
    for doc, dl in zip(tokenized, doc_lens):
        tf = Counter(doc)
        score = 0.0
        for term in query_terms:
            if term not in tf:
                continue
            idf = math.log(1 + (len(docs) - dfs[term] + 0.5) / (dfs[term] + 0.5))
            denom = tf[term] + k1 * (1 - b + b * dl / avgdl)
            score += idf * (tf[term] * (k1 + 1)) / (denom + 1e-9)
        raw_scores.append(score)
    return _minmax(raw_scores)


def _vector_scores(index: PageAwareIndex, query: str, model_name: str) -> list[float]:
    if index.embeddings is None or len(index.pages) == 0:
        return [0.0 for _ in index.pages]
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        q = model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0].astype("float32")
        if index.faiss_index is not None:
            q_norm = q / (np.linalg.norm(q) + 1e-9)
            scores, positions = index.faiss_index.search(np.asarray([q_norm], dtype="float32"), len(index.pages))
            ordered = [0.0 for _ in index.pages]
            for score, pos in zip(scores[0], positions[0]):
                if 0 <= int(pos) < len(ordered):
                    ordered[int(pos)] = float(score)
            return _minmax(ordered)
        emb = index.embeddings.astype("float32")
        scores = (emb @ q) / ((np.linalg.norm(emb, axis=1) * np.linalg.norm(q)) + 1e-9)
        return _minmax([float(s) for s in scores])
    except Exception:
        return [0.0 for _ in index.pages]


def _target_anchor_score(text: str, target_table: str, sector: str) -> tuple[float, list[str]]:
    norm = normalize_text(text)
    anchors = FINANCIAL_ANCHORS.get(sector, {}).get(target_table, [])
    anchor_hits = keyword_hits(text, anchors)
    negative_hits = keyword_hits(text, NEGATIVE_ANCHORS.get(target_table, []))
    raw = min(len(anchor_hits), 8) / 8.0

    has_total_general = "total general i ii iii" in norm or "total general" in norm
    has_total_actif = "total actif" in norm or "total de l actif" in norm or (sector == "autres_cgnc" and has_total_general)
    has_total_passif = "total passif" in norm or "total du passif" in norm or (sector == "autres_cgnc" and has_total_general)
    if target_table == "BILAN_ACTIF" and not has_total_actif:
        raw *= 0.45
    if target_table == "BILAN_PASSIF" and not has_total_passif:
        raw *= 0.35

    if sector == "autres_cgnc" and target_table in {"BILAN_ACTIF", "BILAN_PASSIF"}:
        is_management_summary = any(
            marker in norm
            for marker in [
                "l actif et ses composantes",
                "le passif et ses composantes",
                "variation 24 23",
                "variation",
                "comptes de bilan",
            ]
        ) and "en millions" in norm
        is_formal_cgnc_statement = any(marker in norm for marker in ["en milliers de mad", "total general i ii iii"])
        if is_management_summary:
            raw *= 0.15
        elif is_formal_cgnc_statement:
            raw = min(1.0, raw + 0.25)
        if _looks_like_note_repeat(norm):
            raw *= 0.25
        if "co-entreprises" in norm[:1800] or "entreprises associees" in norm[:1800]:
            raw *= 0.20
        if "annexe" in norm[:300] and target_table == "BILAN_PASSIF" and "total passif et capitaux propres" in norm:
            raw = min(1.0, raw + 0.20)
        if "annexe" in norm[:300] and target_table == "BILAN_ACTIF" and "total actif" in norm:
            raw = min(1.0, raw + 0.20)

    if "hors bilan" in negative_hits and not (
        (target_table == "BILAN_ACTIF" and has_total_actif)
        or (target_table == "BILAN_PASSIF" and has_total_passif)
    ):
        raw -= 0.35
    if target_table in {"BILAN_ACTIF", "BILAN_PASSIF"} and any(
        marker in norm
        for marker in [
            "tableau des flux de tresorerie",
            "flux de tresorerie nets",
            "capacite d autofinancement",
            "capacite d'autofinancement",
            "etat des derogations",
            "etat des changements de methodes",
            "detail des postes",
            "details des postes",
            "creances sur les etablissements de credit et assimiles",
            "creances sur la clientele",
        ]
    ) and not (
        (target_table == "BILAN_ACTIF" and has_total_actif)
        or (target_table == "BILAN_PASSIF" and has_total_passif)
    ):
        raw -= 0.35
    if target_table == "CPC" and any("flux" in normalize_text(hit) for hit in negative_hits):
        raw -= 0.60
    if target_table == "CPC":
        if sector == "bancaire_sdf" and _looks_like_formal_banking_ifrs_cpc(norm):
            raw = min(1.0, raw + 0.40)
        elif _looks_like_cpc_table(norm):
            raw = min(1.0, raw + 0.25)
        elif _looks_like_financial_narrative(norm):
            raw *= 0.30
        if sector == "bancaire_sdf" and _looks_like_foreign_ifrs_appendix(norm):
            raw *= 0.25
    return max(0.0, min(raw, 1.0)), anchor_hits


def _scope_score(text: str, scope_candidates: list[str], requested_scope: str) -> tuple[float, list[str]]:
    norm = normalize_text(text)
    other_scope = "comptes_sociaux" if requested_scope == "comptes_consolides" else "comptes_consolides"
    requested_hits = keyword_hits(text, SCOPE_KEYWORDS.get(requested_scope, []))
    other_hits = keyword_hits(text, SCOPE_KEYWORDS.get(other_scope, []))
    strong_social = any(
        marker in norm
        for marker in [
            "comptes sociaux ocp sa",
            "comptes sociaux",
            "etats financiers sociaux",
            "etats de synthese sociaux",
            "bilan social",
        ]
    )
    strong_consolidated = any(
        marker in norm
        for marker in [
            "comptes consolides normes ifrs",
            "comptes consolides",
            "etats financiers consolides",
            "situation financiere consolidee",
            "resultat consolide",
        ]
    )
    formal_social_statement = _looks_like_formal_social_statement(norm)

    evidence: list[str] = []
    if requested_scope == "comptes_sociaux" and formal_social_statement and not strong_consolidated:
        evidence.append("scope_formal_social_statement")
        base = 0.90
    elif requested_scope in scope_candidates:
        evidence.append(f"scope:{requested_scope}")
        base = 1.0
    elif requested_hits:
        evidence.append(f"scope_signal:{requested_scope}")
        base = 0.75
    else:
        base = 0.25

    if other_scope in scope_candidates and requested_scope not in scope_candidates and not (
        requested_scope == "comptes_sociaux" and formal_social_statement
    ):
        evidence.append(f"scope_penalty:{other_scope}")
        base = 0.0
    elif other_hits and not requested_hits:
        evidence.append(f"scope_signal_penalty:{other_scope}")
        base = min(base, 0.15)

    if requested_scope == "comptes_consolides" and strong_social:
        evidence.append("scope_strong_penalty:comptes_sociaux")
        base = min(base, 0.05 if not strong_consolidated else 0.20)
    if requested_scope == "comptes_sociaux" and strong_consolidated and not strong_social:
        evidence.append("scope_strong_penalty:comptes_consolides")
        base = min(base, 0.05)

    return max(0.0, min(base, 1.0)), evidence


def _title_score(text: str, detected_titles: list[str], target_table: str) -> tuple[float, list[str]]:
    norm = normalize_text(text)
    title_hits = keyword_hits(text, TABLE_TITLES.get(target_table, []))
    strong_title = _has_strong_table_title(norm, target_table)
    if target_table == "CPC" and _looks_like_foreign_ifrs_appendix(norm):
        if title_hits:
            return 0.10, [f"title_foreign_ifrs_appendix:{h}" for h in title_hits]
        return 0.0, []
    if target_table == "CPC" and strong_title and _looks_like_financial_narrative(norm) and not _looks_like_cpc_table(norm):
        strong_title = False
    note_repeat = _looks_like_note_repeat(norm)
    if note_repeat and strong_title:
        return 0.45, [f"title_note_repeat:{target_table}"] + [f"title_hit:{h}" for h in title_hits]
    if target_table in detected_titles and strong_title:
        return 1.0, [f"title:{target_table}"] + [f"title_hit:{h}" for h in title_hits]
    if target_table in detected_titles:
        return 0.35, [f"title_weak:{target_table}"] + [f"title_hit:{h}" for h in title_hits]
    if strong_title:
        return 0.9, [f"title_strong:{target_table}"] + [f"title_hit:{h}" for h in title_hits]
    if title_hits:
        return 0.25, [f"title_hit:{h}" for h in title_hits]
    return 0.0, []


def _has_strong_table_title(norm_text: str, target_table: str) -> bool:
    if target_table == "BILAN_ACTIF":
        return bool(
            re.search(r"\bbilan\s+actif\b", norm_text)
            or re.search(r"\bactif\b", norm_text[:900])
            or "bilan (actif)" in norm_text
            or "bilan actif" in norm_text[:1200]
            or ("en milliers de mad" in norm_text[:1800] and " actif " in f" {norm_text[:1800]} " and "total general" in norm_text)
        )
    if target_table == "BILAN_PASSIF":
        return bool(
            re.search(r"\bbilan\s+passif\b", norm_text)
            or re.search(r"\bpassif\b", norm_text[:900])
            or "bilan (passif)" in norm_text
            or "bilan passif" in norm_text[:1200]
            or ("en milliers de mad" in norm_text[:1800] and " passif " in f" {norm_text[:1800]} " and "total general" in norm_text)
        )
    if target_table == "CPC":
        has_title = any(
            marker in norm_text[:1800]
            for marker in [
                "compte de produits et charges",
                "compte de produits et de charges",
                "compte de resultat",
                "etat du resultat global",
                "cpc consolide",
            ]
        )
        return has_title and (
            _looks_like_cpc_table(norm_text)
            or _looks_like_formal_banking_ifrs_cpc(norm_text)
            or "compte de resultat consolide" in norm_text[:500]
        )
    return False


def _looks_like_note_repeat(norm_text: str) -> bool:
    early = norm_text[:1400]
    return any(
        marker in early
        for marker in [
            "notes annexes aux etats financiers",
            "note 3",
            "note 4",
            "note 5",
            "note 6",
            "6.2.",
            "6.3.",
        ]
    )


def _looks_like_cpc_table(norm_text: str) -> bool:
    early = norm_text[:2500]
    return any(
        marker in early
        for marker in [
            "designation operations",
            "totaux de l exercice",
            "totaux de l'exercice",
            "propres a l exercice",
            "propres a l'exercice",
            "compte de produits et charges (hors taxes)",
            "compte de produits et charges hors taxes",
            "compte de produits et charges (hors taxes) (suite)",
            "resultat net (xi-xii)",
            "total des produits",
            "total des charges",
        ]
    )


def _looks_like_formal_banking_ifrs_cpc(norm_text: str) -> bool:
    early = norm_text[:3500]
    has_title = "compte de resultat ifrs" in early or "compte de resultat consolide" in early
    has_local_currency = any(
        marker in early
        for marker in [
            "en milliers de dh",
            "en milliers de dirhams",
            "en milliers mad",
            "en milliers de mad",
        ]
    )
    banking_rows = [
        "interets et produits assimiles",
        "interets et charges assimiles",
        "marge d interet",
        "marge sur commissions",
        "produit net bancaire",
        "charges generales d exploitation",
        "resultat brut d exploitation",
        "cout du risque",
        "resultat net part du groupe",
    ]
    hit_count = sum(1 for marker in banking_rows if marker in norm_text)
    return has_title and has_local_currency and hit_count >= 3


def _looks_like_foreign_ifrs_appendix(norm_text: str) -> bool:
    early = norm_text[:3500]
    foreign_markers = [
        "en millions d euros",
        "en millions d'euros",
        "union europeenne",
        "document d enregistrement universel",
        "document d'enregistrement universel",
        "autorite des marches financiers",
        "bnp paribas",
        "euros hors resultat",
    ]
    return sum(1 for marker in foreign_markers if marker in early) >= 2


def _looks_like_financial_narrative(norm_text: str) -> bool:
    early = norm_text[:2600]
    narrative_markers = [
        "communique de presse",
        "resultats economiques et financiers",
        "cette variation s explique",
        "cette evolution s explique",
        "le resultat net s",
        "le resultat d exploitation s",
        "le resultat financier s",
        "au niveau du compte de produits et charges",
    ]
    return any(marker in early for marker in narrative_markers)


def _looks_like_formal_social_statement(norm_text: str) -> bool:
    early = norm_text[:2500]
    return any(
        marker in early
        for marker in [
            "modele normal",
            "bilan (actif)",
            "bilan (passif)",
            "compte de produits et charges (hors taxes)",
            "compte de produits et charges hors taxes",
            "etat des soldes de gestion",
            "tableau de financement",
        ]
    )


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]
