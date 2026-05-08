from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from .keyword_dictionary import (
    FINANCIAL_ANCHORS,
    NEGATIVE_ANCHORS,
    SCOPE_KEYWORDS,
    TABLE_TITLES,
    keyword_hits,
    scoped_table_signature,
    target_query,
)
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
        scope_signature_score, scope_signature_hits, opposite_signature_score = _scope_signature_score(
            page.page_text,
            target_table,
            sector,
            scope,
        )
        if scope_signature_hits:
            target_anchor_score = max(target_anchor_score, 0.55 * target_anchor_score + 0.45 * scope_signature_score)
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
        if scope_score <= 0.05:
            score *= 0.45
        norm_page_text = normalize_text(page.page_text)
        if target_table == "CPC" and _looks_like_cashflow_statement(norm_page_text) and not _has_cpc_title_anywhere(norm_page_text):
            score *= 0.12
        if target_table == "CPC" and _looks_like_foreign_ifrs_appendix(norm_page_text):
            score *= 0.22
        if target_table == "CPC" and _looks_like_note_detail_page(norm_page_text) and title_score < 0.5 and not _has_cpc_title_anywhere(norm_page_text):
            score *= 0.25
        if target_table == "CPC" and _looks_like_financial_narrative(norm_page_text) and title_score < 0.5:
            score *= 0.35
        if target_table == "CPC":
            cpc_layout_score = _cpc_near_balance_layout_score(index.pages, i, norm_page_text)
            if cpc_layout_score >= 0.75 and (title_score >= 0.5 or target_anchor_score >= 0.35):
                score = min(1.0, score + 0.10 * cpc_layout_score)
        if target_table in {"BILAN_ACTIF", "BILAN_PASSIF"} and _looks_like_note_detail_page(norm_page_text) and title_score < 0.5:
            score *= 0.35
        main_balance_statement = False
        balance_layout_score = 0.0
        if target_table in {"BILAN_ACTIF", "BILAN_PASSIF"}:
            main_balance_statement = _looks_like_main_balance_statement(norm_page_text, target_table)
            balance_layout_score = _balance_layout_score(index.pages, i, norm_page_text, target_table)
            if main_balance_statement:
                statement_boost = 0.18
                if scope_signature_score >= 0.55:
                    statement_boost += 0.14
                if balance_layout_score >= 0.75:
                    statement_boost += 0.08
                score = min(1.0, score + statement_boost)
            elif _looks_like_segment_information_page(norm_page_text):
                score *= 0.12
            elif _looks_like_accounting_policy_narrative(norm_page_text):
                score *= 0.10
            elif _looks_like_duration_or_guarantee_breakdown(norm_page_text):
                score *= 0.18
            elif _looks_like_note_detail_page(norm_page_text):
                score *= 0.16
            elif target_anchor_score < 0.12 and title_score < 0.50:
                score *= 0.35
            if (
                not main_balance_statement
                and scope_signature_score < 0.20
                and target_anchor_score < 0.20
                and title_score >= 0.50
            ):
                score *= 0.18
            if balance_layout_score >= 0.75:
                score = min(1.0, score + 0.12 * balance_layout_score)
        score = _apply_scope_signature_dominance(
            score=score,
            scope_signature_score=scope_signature_score,
            opposite_signature_score=opposite_signature_score,
            scope_signature_hits=scope_signature_hits,
            target_table=target_table,
            title_score=title_score,
            scope_score=scope_score,
        )
        if opposite_signature_score > scope_signature_score + 0.18:
            score *= 0.35
        if target_table in {"BILAN_ACTIF", "BILAN_PASSIF"} and not main_balance_statement:
            if (
                _looks_like_note_detail_page(norm_page_text)
                or _looks_like_accounting_policy_narrative(norm_page_text)
                or _looks_like_duration_or_guarantee_breakdown(norm_page_text)
            ):
                score *= 0.28
            elif balance_layout_score <= 0.0 and scope_signature_score < 0.70:
                score *= 0.55
        wrong_table_penalty, wrong_table_evidence = _wrong_table_penalty(norm_page_text, target_table)
        if wrong_table_penalty < 1.0:
            score *= wrong_table_penalty
        evidence = scope_evidence + title_evidence
        evidence += wrong_table_evidence
        evidence.append(f"signature_score:{target_anchor_score:.3f}")
        evidence.append(f"signature_hits:{len(set(normalize_text(anchor) for anchor in matched_anchors))}")
        evidence.append(f"scope_signature_score:{scope_signature_score:.3f}")
        evidence.append(f"scope_signature_hits:{len(set(normalize_text(anchor) for anchor in scope_signature_hits))}")
        evidence.append(f"opposite_scope_signature_score:{opposite_signature_score:.3f}")
        if target_table in {"BILAN_ACTIF", "BILAN_PASSIF"}:
            evidence.append(f"balance_layout_score:{_balance_layout_score(index.pages, i, norm_page_text, target_table):.3f}")
        if target_table == "CPC":
            evidence.append(f"cpc_near_balance_score:{_cpc_near_balance_layout_score(index.pages, i, norm_page_text):.3f}")
        evidence += [f"scope_anchor:{anchor}" for anchor in scope_signature_hits[:10]]
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
    raw = _table_signature_coverage(anchor_hits, anchors, target_table)

    has_total_general = "total general i ii iii" in norm or "total general" in norm
    has_main_actif_title = any(marker in norm for marker in ["actif ifrs", "actif consolide", "bilan actif", "bilan consolide"])
    has_main_passif_title = any(marker in norm for marker in ["passif ifrs", "passif consolide", "bilan passif", "bilan consolide"])
    has_total_actif = "total actif" in norm or "total de l actif" in norm or (has_main_actif_title and "total" in norm) or (sector == "autres_cgnc" and has_total_general)
    has_total_passif = "total passif" in norm or "total du passif" in norm or (has_main_passif_title and "total" in norm) or (sector == "autres_cgnc" and has_total_general)
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
        if _looks_like_cashflow_statement(norm) and not _has_cpc_title_anywhere(norm):
            raw *= 0.05
        if _looks_like_balance_statement(norm) and not _has_cpc_title_anywhere(norm):
            raw *= 0.10
    return max(0.0, min(raw, 1.0)), anchor_hits


def _scope_signature_score(text: str, target_table: str, sector: str, scope: str) -> tuple[float, list[str], float]:
    """Score the requested scope's own row-label signature against the opposite scope.

    FINANCIAL_ANCHORS is intentionally broad per sector/table. This extra layer
    answers the critical question: does the page look more like the requested
    comptes_sociaux or comptes_consolides table?
    """
    other_scope = "comptes_sociaux" if scope == "comptes_consolides" else "comptes_consolides"
    requested_anchors = scoped_table_signature(sector, scope, target_table)
    other_anchors = scoped_table_signature(sector, other_scope, target_table)
    requested_hits = keyword_hits(text, requested_anchors)
    other_hits = keyword_hits(text, other_anchors)
    requested_score = _table_signature_coverage(requested_hits, requested_anchors, target_table)
    other_score = _table_signature_coverage(other_hits, other_anchors, target_table)
    return requested_score, requested_hits, other_score


def _apply_scope_signature_dominance(
    *,
    score: float,
    scope_signature_score: float,
    opposite_signature_score: float,
    scope_signature_hits: list[str],
    target_table: str,
    title_score: float,
    scope_score: float,
) -> float:
    """Make the user's full row-label signatures the main retrieval signal.

    Generic words such as actif, passif, dettes, provisions or resultat can
    appear in notes. The full signature is different: a true table page contains
    a dense group of expected row labels. This function promotes dense matches
    and demotes pages that only match a few isolated labels.
    """
    unique_hit_count = len({normalize_text(hit) for hit in scope_signature_hits if normalize_text(hit)})
    if unique_hit_count == 0:
        return score * 0.65 if title_score >= 0.50 and scope_signature_score < 0.10 else score

    min_dense_hits = {
        "BILAN_ACTIF": 10,
        "BILAN_PASSIF": 10,
        "CPC": 12,
    }.get(target_table, 10)

    # Strong signature pages should dominate BM25/vector noise.
    if unique_hit_count >= min_dense_hits and scope_signature_score >= 0.45:
        score = max(score, 0.62 + 0.34 * scope_signature_score)
    elif scope_signature_score >= 0.55:
        score = max(score, 0.50 + 0.32 * scope_signature_score)
    elif unique_hit_count <= 4 and scope_signature_score < 0.25:
        score *= 0.55

    # If the opposite scope looks more like the page, keep it out.
    if opposite_signature_score > scope_signature_score + 0.12:
        score *= 0.40

    # A title without the user's labels is often a note/detail page, not the
    # target statement.
    if title_score >= 0.70 and scope_signature_score < 0.18 and unique_hit_count <= 5:
        score *= 0.35

    # Scope metadata is still useful, but labels should save a page when the
    # scope detector is weak and the table signature is very clear.
    if scope_score <= 0.20 and scope_signature_score >= 0.70 and unique_hit_count >= min_dense_hits:
        score = max(score, 0.70 + 0.22 * scope_signature_score)

    return max(0.0, min(score, 1.0))


def _table_signature_coverage(anchor_hits: list[str], anchors: list[str], target_table: str) -> float:
    """Score how much of the expected table signature is present on a page.

    A real statement page usually contains a dense set of the row labels the user
    provided. A notes page may repeat important words, but it only covers a small
    part of the full signature, so it should not win only because it has "dettes",
    "provisions", "resultat net", etc.
    """
    unique_hits = {normalize_text(hit) for hit in anchor_hits if normalize_text(hit)}
    unique_anchors = {normalize_text(anchor) for anchor in anchors if normalize_text(anchor)}
    if not unique_anchors:
        return 0.0

    hit_count = len(unique_hits)
    coverage = hit_count / max(len(unique_anchors), 1)
    saturation_targets = {
        "BILAN_ACTIF": 18,
        "BILAN_PASSIF": 18,
        "CPC": 22,
    }
    saturation = min(hit_count, saturation_targets.get(target_table, 18)) / saturation_targets.get(target_table, 18)

    # Coverage rewards the full table shape; saturation keeps compact S1 pages
    # competitive even when labels are abbreviated in the PDF.
    score = 0.65 * saturation + 0.35 * min(coverage * 2.2, 1.0)

    required_groups = _required_signature_groups(target_table)
    if required_groups:
        matched_groups = 0
        for group in required_groups:
            if any(any(marker in hit for hit in unique_hits) for marker in group):
                matched_groups += 1
        group_score = matched_groups / len(required_groups)
        score = 0.75 * score + 0.25 * group_score

    if hit_count <= 3:
        score *= 0.35
    elif hit_count <= 6:
        score *= 0.70
    return max(0.0, min(score, 1.0))


def _required_signature_groups(target_table: str) -> list[list[str]]:
    if target_table == "BILAN_ACTIF":
        return [
            ["actif"],
            ["immobilisations", "actifs financiers", "creances", "stocks"],
            ["tresorerie", "banques centrales", "banque"],
            ["total actif", "total general", "total de l actif"],
        ]
    if target_table == "BILAN_PASSIF":
        return [
            ["passif"],
            ["capitaux propres", "capital"],
            ["dettes", "passifs financiers", "provisions"],
            ["total passif", "total general", "total du passif"],
        ]
    if target_table == "CPC":
        return [
            ["produits", "interets", "primes", "chiffre d affaires"],
            ["charges", "commissions", "cout du risque"],
            ["resultat"],
            ["resultat net", "total des produits", "total des charges"],
        ]
    return []


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
            "bilan consolide",
            "compte de resultat consolide",
        ]
    )
    formal_social_statement = _looks_like_formal_social_statement(norm)
    formal_consolidated_statement = _looks_like_formal_consolidated_statement(norm)

    evidence: list[str] = []
    if requested_scope == "comptes_consolides" and formal_consolidated_statement:
        evidence.append("scope_formal_consolidated_statement")
        base = 0.95
    elif requested_scope == "comptes_sociaux" and formal_social_statement and not strong_consolidated:
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
    if requested_scope == "comptes_consolides" and formal_social_statement and not formal_consolidated_statement:
        evidence.append("scope_strong_penalty:formal_social_statement")
        base = min(base, 0.02)
    if requested_scope == "comptes_sociaux" and strong_consolidated and not strong_social:
        evidence.append("scope_strong_penalty:comptes_consolides")
        base = min(base, 0.05)
    if requested_scope == "comptes_sociaux" and formal_consolidated_statement and not formal_social_statement:
        evidence.append("scope_strong_penalty:formal_consolidated_statement")
        base = min(base, 0.02)

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
    if target_table in {"BILAN_ACTIF", "BILAN_PASSIF"} and strong_title and _looks_like_main_balance_statement(norm, target_table):
        return 1.0, [f"title_strong:{target_table}"] + [f"title_hit:{h}" for h in title_hits]
    if target_table == "CPC" and strong_title and _looks_like_formal_banking_ifrs_cpc(norm):
        return 1.0, [f"title_strong:{target_table}"] + [f"title_hit:{h}" for h in title_hits]
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
            _looks_like_main_balance_statement(norm_text, target_table)
            or re.search(r"\bbilan\s+actif\b", norm_text)
            or re.search(r"\bactif\b", norm_text[:900])
            or "actif ifrs" in norm_text[:3000]
            or "actif consolide" in norm_text[:3000]
            or "bilan (actif)" in norm_text
            or "bilan actif" in norm_text[:1200]
            or ("en milliers de mad" in norm_text[:1800] and " actif " in f" {norm_text[:1800]} " and "total general" in norm_text)
        )
    if target_table == "BILAN_PASSIF":
        return bool(
            _looks_like_main_balance_statement(norm_text, target_table)
            or re.search(r"\bbilan\s+passif\b", norm_text)
            or re.search(r"\bpassif\b", norm_text[:900])
            or "passif ifrs" in norm_text[:3000]
            or "passif consolide" in norm_text[:3000]
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
        if not has_title:
            has_title = any(
                marker in norm_text[:7000]
                for marker in [
                    "compte de resultat consolide",
                    "compte de produits et charges consolide",
                    "compte de resultat ifrs",
                ]
            )
        return has_title and (
            _looks_like_cpc_table(norm_text)
            or _looks_like_formal_banking_ifrs_cpc(norm_text)
            or "compte de produits et charges consolide" in norm_text
            or "comptes de produits et charges consolides" in norm_text
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


def _wrong_table_penalty(norm_text: str, target_table: str) -> tuple[float, list[str]]:
    if target_table == "CPC":
        if _looks_like_cashflow_statement(norm_text) and not _has_cpc_title_anywhere(norm_text):
            return 0.12, ["penalty:cashflow_not_cpc"]
        if _looks_like_balance_statement(norm_text) and not _has_cpc_title_anywhere(norm_text):
            return 0.20, ["penalty:balance_not_cpc"]
        return 1.0, []

    if target_table in {"BILAN_ACTIF", "BILAN_PASSIF"}:
        if _looks_like_cashflow_statement(norm_text) and not _has_strong_table_title(norm_text, target_table):
            return 0.15, ["penalty:cashflow_not_balance"]
        if _has_strong_table_title(norm_text, "CPC") and not _has_strong_table_title(norm_text, target_table):
            return 0.20, ["penalty:cpc_not_balance"]
        if target_table == "BILAN_ACTIF" and _looks_like_passif_only_statement(norm_text):
            return 0.20, ["penalty:passif_not_actif"]
        if target_table == "BILAN_PASSIF" and _looks_like_actif_only_statement(norm_text):
            return 0.20, ["penalty:actif_not_passif"]
    return 1.0, []


def _looks_like_balance_statement(norm_text: str) -> bool:
    early = norm_text[:2500]
    has_balance_title = any(
        marker in early
        for marker in [
            "bilan consolide",
            "bilan (actif)",
            "bilan (passif)",
            "bilan actif",
            "bilan passif",
        ]
    )
    has_balance_totals = any(marker in norm_text for marker in ["total actif", "total passif", "total general i"])
    has_balance_rows = sum(
        1
        for marker in [
            "immobilisations corporelles",
            "immobilisations incorporelles",
            "stocks",
            "creances",
            "capitaux propres",
            "dettes",
            "passif circulant",
            "actif circulant",
        ]
        if marker in norm_text
    )
    return has_balance_title or (has_balance_totals and has_balance_rows >= 2)


def _looks_like_main_balance_title(norm_text: str, target_table: str) -> bool:
    if target_table == "BILAN_ACTIF":
        return any(marker in norm_text[:3000] for marker in ["actif ifrs", "actif consolide", "bilan actif", "bilan consolide"])
    if target_table == "BILAN_PASSIF":
        return any(marker in norm_text[:3000] for marker in ["passif ifrs", "passif consolide", "bilan passif", "bilan consolide"])
    return False


def _looks_like_main_balance_statement(norm_text: str, target_table: str) -> bool:
    """Detect the formal balance sheet, not a note or sector breakdown.

    The page should have a main balance title plus the final target total and
    several row labels from the actual statement. This blocks pages that only
    repeat a few rows inside notes, IFRS narratives, guarantees, or segment
    information.
    """
    early = norm_text[:3200]
    statement_window = norm_text[:7000]
    if (
        _looks_like_segment_information_page(norm_text)
        or _looks_like_accounting_policy_narrative(norm_text)
        or _looks_like_duration_or_guarantee_breakdown(norm_text)
    ):
        return False

    has_balance_title = any(
        marker in statement_window
        for marker in [
            "bilan consolide",
            "bilan ifrs",
            "bilan au 31",
            "bilan au 30",
            "publication des comptes bilan",
            "etat de la situation financiere",
            "bilan actif",
            "bilan passif",
            "bilan (actif)",
            "bilan (passif)",
        ]
    )
    if target_table == "BILAN_ACTIF" and not has_balance_title:
        has_balance_title = any(marker in early for marker in ["actif 30/06", "actif 31/12", "actif notes"])
    if target_table == "BILAN_PASSIF" and not has_balance_title:
        has_balance_title = any(marker in statement_window for marker in ["passif 30/06", "passif 31/12", "passif notes"])
    has_currency_or_dates = any(marker in early for marker in ["30/06", "31/12", "en milliers", "en mdh", "en millions"])
    if target_table == "BILAN_ACTIF":
        final_total = any(
            marker in norm_text
            for marker in [
                "total actif",
                "total de l actif",
                "total de l'actif",
                "total de lactif",
                "total general i ii iii",
            ]
        )
        row_markers = [
            "valeurs en caisse",
            "actifs financiers",
            "creances sur les etablissements",
            "prets et creances",
            "creances sur la clientele",
            "immobilisations corporelles",
            "tresorerie",
            "stocks",
        ]
        title_marker = any(
            marker in statement_window
            for marker in [" actif ", "actif notes", "actif 30/06", "actif 31/12", "bilan actif", "bilan consolide"]
        )
    elif target_table == "BILAN_PASSIF":
        final_total = any(marker in norm_text for marker in ["total passif", "total du passif", "total general i ii iii"])
        row_markers = [
            "dettes envers les etablissements",
            "dettes envers la clientele",
            "passifs financiers",
            "capitaux propres",
            "capital et reserves",
            "resultat net",
            "dettes subordonnees",
            "provisions",
        ]
        title_marker = any(
            marker in statement_window
            for marker in [" passif ", "passif notes", "passif 30/06", "passif 31/12", "bilan passif", "bilan consolide"]
        )
    else:
        return False

    row_hits = sum(1 for marker in row_markers if marker in norm_text)
    return bool(has_balance_title and title_marker and final_total and has_currency_or_dates and row_hits >= 3)


def _balance_layout_score(pages, index: int, norm_text: str, target_table: str) -> float:
    """Reward the common Moroccan balance layout: Actif/Passif same page or x/x+1."""
    if target_table not in {"BILAN_ACTIF", "BILAN_PASSIF"}:
        return 0.0
    if _looks_like_duration_or_guarantee_breakdown(norm_text):
        return 0.0

    if _looks_like_main_balance_statement(norm_text, target_table):
        return 1.0

    opposite = "BILAN_PASSIF" if target_table == "BILAN_ACTIF" else "BILAN_ACTIF"
    current_has_target_total = _has_balance_total(norm_text, target_table)
    current_has_target_header = _has_balance_header(norm_text, target_table)
    current_has_opposite_total = _has_balance_total(norm_text, opposite)
    current_has_opposite_header = _has_balance_header(norm_text, opposite)
    if current_has_target_total and current_has_target_header and (current_has_opposite_total or current_has_opposite_header):
        return 0.95

    if not (current_has_target_total and current_has_target_header):
        return 0.0

    neighbor_texts: list[str] = []
    if index > 0:
        neighbor_texts.append(normalize_text(pages[index - 1].page_text))
    if index + 1 < len(pages):
        neighbor_texts.append(normalize_text(pages[index + 1].page_text))
    for neighbor in neighbor_texts:
        if _looks_like_main_balance_statement(neighbor, opposite) or (
            _has_balance_header(neighbor, opposite) and _has_balance_total(neighbor, opposite)
        ):
            return 0.85
    return 0.0


def _cpc_near_balance_layout_score(pages, index: int, norm_text: str) -> float:
    """Reward CPC pages in the normal neighborhood of the balance statements."""
    if _has_cpc_title_anywhere(norm_text) and (_looks_like_cpc_table(norm_text) or _looks_like_formal_banking_ifrs_cpc(norm_text)):
        return 1.0

    neighbor_scores: list[float] = []
    for offset, weight in [(-1, 0.75), (1, 0.90), (0, 0.95)]:
        pos = index + offset
        if not (0 <= pos < len(pages)):
            continue
        neighbor = norm_text if offset == 0 else normalize_text(pages[pos].page_text)
        has_balance = _looks_like_main_balance_statement(neighbor, "BILAN_ACTIF") or _looks_like_main_balance_statement(neighbor, "BILAN_PASSIF")
        if has_balance:
            neighbor_scores.append(weight)
    return max(neighbor_scores) if neighbor_scores else 0.0


def _has_balance_header(norm_text: str, target_table: str) -> bool:
    window = norm_text[:7000]
    if target_table == "BILAN_ACTIF":
        return any(
            marker in window
            for marker in [
                " actif ",
                "actif notes",
                "actif 30/06",
                "actif 31/12",
                "bilan actif",
                "bilan (actif)",
                "bilan consolide",
            ]
        )
    if target_table == "BILAN_PASSIF":
        return any(
            marker in window
            for marker in [
                " passif ",
                "passif notes",
                "passif 30/06",
                "passif 31/12",
                "bilan passif",
                "bilan (passif)",
                "bilan consolide",
            ]
        )
    return False


def _has_balance_total(norm_text: str, target_table: str) -> bool:
    if target_table == "BILAN_ACTIF":
        return any(
            marker in norm_text
            for marker in ["total actif", "total de l actif", "total de l'actif", "total de lactif", "total general i ii iii"]
        )
    if target_table == "BILAN_PASSIF":
        return any(marker in norm_text for marker in ["total passif", "total du passif", "total general i ii iii"])
    return False


def _looks_like_segment_information_page(norm_text: str) -> bool:
    early = norm_text[:4200]
    segment_markers = [
        "information par pole d activites",
        "information par pole d'activites",
        "information sectorielle",
        "informations sectorielles",
        "informations par secteur operationnel",
        "resultat par secteur operationnel",
        "actifs et passifs par secteur operationnel",
        "banque maroc europe et zone offshore",
        "filiales de financement specialisees",
        "banque de detail a l international",
        "eliminations",
    ]
    segment_hits = sum(1 for marker in segment_markers if marker in early)
    balance_breakdown = "elements de l actif" in norm_text and "elements du passif" in norm_text and "total bilan" in norm_text
    return segment_hits >= 2 or balance_breakdown


def _looks_like_accounting_policy_narrative(norm_text: str) -> bool:
    early = norm_text[:3200]
    narrative_markers = [
        "normes comptables appliquees",
        "methodes comptables appliquees",
        "principes comptables",
        "principes et methodes comptables",
        "principes de consolidation",
        "modalites de transition",
        "premiere application de la norme",
        "ifrs 16",
        "ifrs 9",
        "iasb",
        "classification des actifs financiers",
        "actifs et passifs financiers classement",
        "un actif financier",
        "la valeur de marche est determinee",
        "l obligation pour le preneur",
        "contrat de location",
        "droit d utilisation",
        "dette locative",
        "pensions livrees",
        "titres donnes en pension",
        "operations libellees en devises",
        "conversion des elements du bilan",
        "immobilisations incorporelles et corporelles figurent au bilan",
        "charges a repartir",
        "presentation generale des creances",
        "les creances sont ventilees",
    ]
    return sum(1 for marker in narrative_markers if marker in early) >= 2


def _looks_like_duration_or_guarantee_breakdown(norm_text: str) -> bool:
    early = norm_text[:3600]
    statement_start = norm_text[:700]
    if any(
        marker in statement_start
        for marker in [
            "actif 30/06",
            "passif 30/06",
            "actif 31/12",
            "passif 31/12",
            "bilan consolide",
            "bilan actif",
            "bilan passif",
            "bilan (actif)",
            "bilan (passif)",
        ]
    ):
        return False
    breakdown_markers = [
        "engagements de financement",
        "engagements de garantie",
        "valeurs et suretes",
        "ventilation des emplois et des ressources",
        "duree residuelle",
        "monnaies etrangeres",
        "rubriques du passif ou du hors bilan",
        "rubriques de l actif ou du hors bilan",
        "concentration des risques",
    ]
    return sum(1 for marker in breakdown_markers if marker in early) >= 2


def _has_cpc_title_anywhere(norm_text: str) -> bool:
    return any(
        marker in norm_text
        for marker in [
            "compte de resultat consolide",
            "compte de produits et charges consolide",
            "compte de produits et charges",
            "compte de resultat ifrs",
            "etat du resultat global",
        ]
    )


def _looks_like_actif_only_statement(norm_text: str) -> bool:
    early = norm_text[:2200]
    statement_window = norm_text[:7000]
    has_actif = any(marker in early for marker in ["bilan (actif)", "bilan actif", "\nactif\n", " actif "])
    has_passif = any(marker in statement_window for marker in ["bilan (passif)", "bilan passif", "\npassif\n", " passif ", "total passif", "total du passif"])
    return has_actif and not has_passif and "total actif" in norm_text


def _looks_like_passif_only_statement(norm_text: str) -> bool:
    early = norm_text[:2200]
    statement_window = norm_text[:7000]
    has_passif = any(marker in early for marker in ["bilan (passif)", "bilan passif", "\npassif\n", " passif "])
    has_actif = any(marker in statement_window for marker in ["bilan (actif)", "bilan actif", "\nactif\n", " actif ", "total actif", "total de l actif"])
    return has_passif and not has_actif and "total passif" in norm_text


def _looks_like_formal_banking_ifrs_cpc(norm_text: str) -> bool:
    early = norm_text[:7000]
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
        "en m eur",
        "m eur",
        "union europeenne",
        "document d enregistrement universel",
        "document d'enregistrement universel",
        "autorite des marches financiers",
        "bnp paribas",
        "societe generale",
        "groupe societe generale",
        "du groupe societe generale",
        "euros hors resultat",
    ]
    return sum(1 for marker in foreign_markers if marker in early) >= 2


def _looks_like_financial_narrative(norm_text: str) -> bool:
    early = norm_text[:2600]
    narrative_markers = [
        "communique de presse",
        "commentaire des comptes",
        "commentaires des comptes",
        "commentaire des resultats",
        "resultats consolides du groupe",
        "resultats economiques et financiers",
        "principaux agregats",
        "indicateurs financiers",
        "ressortent comme suit",
        "cette variation s explique",
        "cette evolution s explique",
        "le resultat net s",
        "le resultat d exploitation s",
        "le resultat financier s",
        "au niveau du compte de produits et charges",
    ]
    return any(marker in early for marker in narrative_markers)


def _looks_like_cashflow_statement(norm_text: str) -> bool:
    early = norm_text[:2600]
    return any(
        marker in early
        for marker in [
            "tableau des flux de tresorerie",
            "flux de tresorerie consolide",
            "flux net de tresorerie",
            "variation de la tresorerie",
            "tresorerie a l ouverture",
            "tresorerie de cloture",
            "capacite d autofinancement",
            "capacite d'autofinancement",
        ]
    )


def _looks_like_note_detail_page(norm_text: str) -> bool:
    early = norm_text[:3200]
    detail_markers = [
        "allocation des pertes attendues",
        "actifs financiers a la juste valeur par capitaux propres",
        "passifs financiers a la juste valeur par resultat",
        "contrats de location",
        "variation du droit d utilisation",
        "variation de l obligation locative",
        "tableau de variation des capitaux propres",
        "variation des capitaux propres",
        "provisions pour risques et charges au",
        "produits et charges des autres activites au",
        "cout net du risque au",
        "charge nette de l impot",
        "prets et creances sur la clientele au",
        "titres au cout amorti au",
        "dettes representees par un titre",
        "dettes envers les etablissements de credit",
        "repartition des creances",
        "engagements et depreciations",
        "immobilisations au",
        "contrats de location au",
        "engagements de financement",
        "engagements de garantie",
        "synthese des provisions",
        "informations par secteur operationnel",
        "resultat par secteur operationnel",
        "actifs et passifs par secteur operationnel",
        "ventilation des prets et creances",
        "analyse du taux effectif d impot",
        "taux effectif d impot",
        "marge d'interets au",
        "marge d interets au",
        "marge d interets",
        "commissions nettes au",
        "commissions nettes",
        "detail des charges",
        "details des charges",
        "detail des autres passifs",
        "details des autres passifs",
        "depots de la clientele au",
        "titres de creance emis au",
        "provisions au",
        "detail des creances",
        "details des creances",
        "ventilation du total de lactif",
        "ventilation du total de l actif",
        "valeurs et suretes",
        "sont couvertes par les provisions",
    ]
    return sum(1 for marker in detail_markers if marker in early) >= 2


def _looks_like_formal_social_statement(norm_text: str) -> bool:
    early = norm_text[:2500]
    if any(
        marker in norm_text
        for marker in [
            "compte de produits et charges consolide",
            "comptes de produits et charges consolides",
            "compte de resultat consolide",
            "bilan consolide",
        ]
    ):
        return False
    return any(
        marker in early
        for marker in [
            "modele normal",
            "bilan (actif)",
            "bilan (passif)",
            "comptes de produits et charges",
            "compte de produits et charges",
            "compte de produits et charges (hors taxes)",
            "compte de produits et charges hors taxes",
            "etat des soldes de gestion",
            "tableau de financement",
        ]
    ) or (
        "operations" in early
        and "totaux de" in early
        and "propres a" in early
        and "concernant" in early
        and "en dirhams" in early
    )


def _looks_like_formal_consolidated_statement(norm_text: str) -> bool:
    early = norm_text[:2600]
    title_hit = any(
        marker in early
        for marker in [
            "bilan consolide",
            "compte de produits et charges consolide",
            "comptes de produits et charges consolides",
            "compte de resultat consolide",
            "etat du resultat global consolide",
            "situation financiere consolidee",
            "etats financiers consolides",
        ]
    )
    cgnc_compact_consolidated = (
        "en mdh" in early
        and "libelle" in early
        and any(marker in early for marker in ["capitaux propres groupe", "part groupe", "interets minoritaires"])
    )
    return title_hit or cgnc_compact_consolidated


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if abs(hi - lo) < 1e-12:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]
