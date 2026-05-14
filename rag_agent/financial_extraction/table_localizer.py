from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from .keyword_dictionary import (
    FINANCIAL_ANCHORS,
    TABLE_TITLES,
    keyword_hits,
    looks_like_balance_currency_ventilation_page,
    scoped_table_signature,
)
from .models import RetrievedPage, TableCandidate
from .utils import clamp_bbox, normalize_text, write_json


_TYPE_HINTS = {
    "BILAN_ACTIF": "BILAN_ACTIF",
    "BILAN_PASSIF": "BILAN_PASSIF",
    "CPC": "CPC",
}

_NEGATIVE_BOUNDARY_MARKERS = [
    "hors bilan",
    "etat des derogations",
    "etat des changements de methodes",
    "detail des postes",
    "details des postes",
    "etat des soldes de gestion",
    "notes annexes",
    "tableau de financement",
    "perimetre de consolidation",
    "attestation",
    "flux de tresorerie",
    "tableau des flux de tresorerie",
    "resultat net et gains et pertes comptabilises directement en capitaux propres",
    "resultat net et gains",
    "gains et pertes comptabilises directement",
    "total des gains et pertes comptabilises",
]

_TARGET_END_ANCHORS = {
    "BILAN_ACTIF": ["total de l'actif", "total actif"],
    "BILAN_PASSIF": ["total du passif", "total passif"],
    "CPC": [
        "resultat dilue par action",
        "resultat de base par action",
        "resultat par action",
        "resultat net part du groupe",
        "resultat net de l'exercice",
        "resultat net",
        "total des produits",
        "total des charges",
    ],
}

_NON_MAIN_STATEMENT_MARKERS = [
    "valeurs et suretes recues",
    "valeurs et suretes donnees",
    "ventilation des emplois et des ressources",
    "ventilation du total de l'actif",
    "ventilation du total de l actif",
    "duree residuelle",
    "monnaies etrangeres",
    "concentration des risques",
    "marge d'interets",
    "commissions sur prestations",
    "detail des autres passifs",
    "details des autres passifs",
    "provisions au",
    "subventions fonds publics",
]

_ALTERNATE_STATEMENT_MARKERS = [
    "bilan arreda",
    "compte de produits et charges arreda",
    "etats de synthese arreda",
]

_MIN_MAIN_BALANCE_REGION_HEIGHT = 130.0


def localize_table_candidates(
    pdf_path: str,
    retrieved_pages: list[RetrievedPage],
    target_table: str,
    scope: str,
    sector: str,
    max_pages: int = 4,
    debug_dir: str | Path | None = None,
) -> list[TableCandidate]:
    candidates: list[TableCandidate] = []
    doc = fitz.open(pdf_path)
    try:
        for retrieved in _pages_for_localization(retrieved_pages, max_pages=max_pages, target_table=target_table):
            candidates.extend(_localize_on_page(doc, pdf_path, retrieved, target_table, scope, sector))
    finally:
        doc.close()
    retrieval_scores = {r.page.page_number: r.score for r in retrieved_pages}
    scope_scores = {r.page.page_number: r.scope_score for r in retrieved_pages}
    signature_scores = {r.page.page_number: r.scope_signature_score for r in retrieved_pages}
    anchor_scores = {r.page.page_number: r.target_anchor_score for r in retrieved_pages}
    candidates.sort(
        key=lambda c: (
            -_candidate_rank_score(
                c,
                retrieval_score=retrieval_scores.get(c.page_number, 0.0),
                scope_score=scope_scores.get(c.page_number, 0.0),
                signature_score=signature_scores.get(c.page_number, 0.0),
                anchor_score=anchor_scores.get(c.page_number, 0.0),
            ),
            c.page_number,
        )
    )
    if debug_dir:
        write_json(Path(debug_dir) / "table_candidates.json", [c.to_dict() for c in candidates])
    return candidates


def _pages_for_localization(
    retrieved_pages: list[RetrievedPage],
    *,
    max_pages: int,
    target_table: str,
) -> list[RetrievedPage]:
    selected = list(retrieved_pages[:max_pages])
    selected_pages = {item.page.page_number for item in selected}
    for retrieved in retrieved_pages[max_pages:]:
        if retrieved.page.page_number in selected_pages:
            continue
        if _should_include_signature_candidate(retrieved, target_table):
            selected.append(retrieved)
            selected_pages.add(retrieved.page.page_number)
    return selected


def _should_include_signature_candidate(retrieved: RetrievedPage, target_table: str) -> bool:
    if target_table == "CPC":
        return retrieved.scope_signature_score >= 0.30 or (
            retrieved.scope_signature_score >= 0.18 and retrieved.target_anchor_score >= 0.25
        )
    return retrieved.scope_signature_score >= 0.45


def _candidate_rank_score(
    candidate: TableCandidate,
    *,
    retrieval_score: float,
    scope_score: float,
    signature_score: float,
    anchor_score: float,
) -> float:
    score = (
        0.50 * retrieval_score
        + 0.25 * candidate.confidence
        + 0.10 * scope_score
        + 0.12 * signature_score
        + 0.03 * anchor_score
    )
    evidence = " ".join(candidate.evidence).lower()
    if candidate.table_type == "CPC":
        if "fallback:page_region" in evidence and signature_score < 0.10 and anchor_score < 0.10:
            score -= 0.16
        if "title_confirmed:false" in evidence and signature_score < 0.15:
            score -= 0.10
        if signature_score >= 0.35:
            score += 0.08
    return max(0.0, score)


def _localize_on_page(
    doc: fitz.Document,
    pdf_path: str,
    retrieved: RetrievedPage,
    target_table: str,
    scope: str,
    sector: str,
) -> list[TableCandidate]:
    page_num = retrieved.page.page_number
    page = doc[page_num - 1]
    page_rect = page.rect
    lines_preview = _extract_lines(page)
    page_joined_preview = " ".join(norm for *_coords, _text, norm in lines_preview)
    if target_table in {"BILAN_ACTIF", "BILAN_PASSIF"} and looks_like_balance_currency_ventilation_page(
        page_joined_preview
    ):
        return []

    blocks = [b for b in page.get_text("blocks") if len(b) >= 5 and str(b[4]).strip()]

    typed_blocks = []
    for b in blocks:
        text = str(b[4])
        detected = _detect_block_type(text, sector)
        if detected:
            typed_blocks.append((detected, b))

    line_header_candidate = _candidate_from_line_header(
        pdf_path,
        page_num,
        page_rect.width,
        page_rect.height,
        target_table,
        scope,
        sector,
        retrieved,
        doc=doc,
    )
    if line_header_candidate is not None:
        return [line_header_candidate]

    target_blocks = [(t, b) for t, b in typed_blocks if t == target_table]
    if target_blocks:
        return [
            _candidate_from_block_group(
                pdf_path,
                page_num,
                page_rect.width,
                page_rect.height,
                target_table,
                scope,
                sector,
                target_blocks,
                retrieved,
                doc=doc,
            )
        ]

    # Fallback: whole table-looking page region, with lower confidence.
    margin_x = page_rect.width * 0.04
    bbox = [margin_x, page_rect.height * 0.08, page_rect.width - margin_x, page_rect.height * 0.94]
    evidence = retrieved.evidence[:8] + ["fallback:page_region"]
    return [
        TableCandidate(
            pdf_id=retrieved.page.pdf_id,
            page_number=page_num,
            scope=scope,
            sector=sector,
            table_type=target_table,
            bbox=clamp_bbox(bbox, page_rect.width, page_rect.height),
            confidence=max(0.25, min(retrieved.score, 0.65)),
            evidence=evidence,
        )
    ]


def _candidate_from_line_header(
    pdf_path: str,
    page_num: int,
    width: float,
    height: float,
    target_table: str,
    scope: str,
    sector: str,
    retrieved: RetrievedPage,
    *,
    doc: fitz.Document,
) -> TableCandidate | None:
    """Localize using line geometry. Pass ``doc`` as keyword so call sites stay 8 positional + shared document."""
    page = doc[page_num - 1]
    lines = _extract_lines(page)
    all_blocks = [b for b in page.get_text("blocks") if len(b) >= 5 and str(b[4]).strip()]

    if target_table == "CPC":
        return _candidate_from_cpc_line_header(
            pdf_path=pdf_path,
            page_num=page_num,
            width=width,
            height=height,
            scope=scope,
            sector=sector,
            retrieved=retrieved,
            lines=lines,
            all_blocks=all_blocks,
        )

    if target_table not in {"BILAN_ACTIF", "BILAN_PASSIF"}:
        return None

    page_joined = " ".join(norm for *_coords, _text, norm in lines)
    if looks_like_balance_currency_ventilation_page(page_joined):
        return None

    wanted_header = "actif" if target_table == "BILAN_ACTIF" else "passif"
    headers = [line for line in lines if _is_exact_table_header(line[5], wanted_header)]
    bilan_social_fallback = False
    if not headers:
        bs_lines = _lines_bilan_social_title_lines(lines)
        if target_table == "BILAN_ACTIF" and bs_lines:
            headers = [bs_lines[0]]
            bilan_social_fallback = True
        elif target_table == "BILAN_PASSIF" and len(bs_lines) >= 2:
            headers = [bs_lines[1]]
            bilan_social_fallback = True
    if not headers:
        return None

    column_bounds = _side_by_side_header_bounds(lines, target_table, width)
    if column_bounds is not None:
        headers = [
            line for line in headers
            if column_bounds[0] - 5 <= (line[0] + line[2]) / 2 <= column_bounds[1] + 5
        ]
    if not headers:
        return None

    header = _select_best_balance_header(lines, headers, target_table, width)
    if column_bounds is None:
        column_bounds = _single_column_bounds_from_header(lines, header, target_table, width)
    y_start = float(header[1])

    typed_boundary_y = height
    for b in all_blocks:
        by = float(b[1])
        if by <= y_start + 12:
            continue
        if column_bounds is not None:
            bx0, _by0, bx1, _by1 = map(float, b[:4])
            center_x = (bx0 + bx1) / 2
            if not column_bounds[0] - 5 <= center_x <= column_bounds[1] + 5:
                continue
        block_text = str(b[4])
        other_type = _detect_block_type(block_text, sector)
        if other_type and other_type != target_table:
            if not _has_target_title(other_type, [block_text]):
                continue
            typed_boundary_y = min(typed_boundary_y, by)

    line_boundary_y = _first_boundary_y(lines, y_start, target_table, width, x_bounds=column_bounds)
    end_anchor_y = _target_end_anchor_y(lines, y_start, target_table, max_y=line_boundary_y, x_bounds=column_bounds)
    next_y = _choose_y_end(
        height=height,
        typed_boundary_y=typed_boundary_y,
        line_boundary_y=line_boundary_y,
        end_anchor_y=end_anchor_y,
    )
    signature_bounds = _signature_vertical_bounds(
        lines,
        target_table=target_table,
        scope=scope,
        sector=sector,
        y_start=y_start,
        max_y=next_y,
        x_bounds=column_bounds,
    )
    if signature_bounds is not None:
        sig_y0, sig_y1, sig_hits, sig_cov = signature_bounds
        y_start = min(y_start, sig_y0)
        next_y = max(next_y, min(height, sig_y1 + _signature_bottom_padding(target_table)))

    if column_bounds is None:
        x_start, x_end = 0.0, width
    else:
        x_start, x_end = column_bounds

    for b in all_blocks:
        bx0, by0, bx1, _by1 = map(float, b[:4])
        center_x = (bx0 + bx1) / 2
        if y_start - 8 <= by0 <= next_y + 5 and x_start - 5 <= center_x <= x_end + 5:
            x_start = min(x_start, bx0)
            x_end = max(x_end, bx1)

    if column_bounds is not None:
        x_start = max(x_start, column_bounds[0])
        x_end = min(x_end, column_bounds[1])

    confidence = min(0.98, 0.78 + 0.20 * retrieved.score)
    evidence = [
        f"localized:{target_table}",
        f"page:{page_num}",
        "line_header",
        f"retrieval_score:{retrieved.score:.3f}",
        f"end_y:{next_y:.1f}",
        "title_confirmed:True",
    ]
    if signature_bounds is not None:
        evidence.append(f"signature_region_hits:{sig_hits}")
        evidence.append(f"signature_region_coverage:{sig_cov:.3f}")
    if bilan_social_fallback:
        evidence.append(
            "header_bilan_social:second" if target_table == "BILAN_PASSIF" else "header_bilan_social:first"
        )
    if retrieved.scope_score <= 0:
        confidence = min(confidence, 0.45)

    right_padding = 2.0 if column_bounds is not None else 8.0
    bbox = clamp_bbox([x_start - 8, max(0.0, y_start - 10), x_end + right_padding, min(height, next_y)], width, height)
    confidence = _adjust_balance_candidate_confidence(
        confidence=confidence,
        evidence=evidence,
        lines=lines,
        bbox=bbox,
        target_table=target_table,
        y_start=y_start,
    )
    return TableCandidate(
        pdf_id=Path(pdf_path).stem,
        page_number=page_num,
        scope=scope,
        sector=sector,
        table_type=target_table,
        bbox=bbox,
        confidence=confidence,
        evidence=evidence,
    )


def _candidate_from_cpc_line_header(
    *,
    pdf_path: str,
    page_num: int,
    width: float,
    height: float,
    scope: str,
    sector: str,
    retrieved: RetrievedPage,
    lines: list[tuple[float, float, float, float, str, str]],
    all_blocks,
) -> TableCandidate | None:
    headers = [
        line
        for line in lines
        if _has_target_title("CPC", [line[5]])
        and (
            line[5].startswith("compte de resultat")
            or line[5].startswith("compte de produits")
            or line[5].startswith("etat du resultat")
            or line[5].startswith("cpc")
        )
    ]
    if not headers:
        return None

    cpc_anchor_norms = [
        "interets et produits assimiles",
        "interets et charges assimiles",
        "marge d interet",
        "marge sur commissions",
        "produit net bancaire",
        "charges generales d exploitation",
        "resultat brut d exploitation",
        "cout du risque",
        "resultat net",
        "resultat net part du groupe",
        "produits d exploitation",
        "charges d exploitation",
        "total des produits",
        "total des charges",
    ]

    def header_score(header: tuple[float, float, float, float, str, str]) -> float:
        _x0, y0, _x1, _y1, _text, norm = header
        after = [line_norm for *_coords, _line_text, line_norm in lines if y0 < _coords[1] <= y0 + 420]
        joined_after = " ".join(after)
        score = 0.0
        if "compte de resultat ifrs" in norm:
            score += 4.0
        if "compte de produits" in norm:
            score += 3.0
        if any(marker in joined_after for marker in ["en milliers de dh", "en milliers de mad", "en milliers de dirhams"]):
            score += 1.5
        score += min(6, sum(1 for marker in cpc_anchor_norms if marker in joined_after))
        if any(marker in joined_after for marker in ["tableau des flux", "flux de tresorerie"]):
            score -= 0.5
        return score

    header = max(headers, key=header_score)
    y_start = float(header[1])
    line_boundary_y = _first_boundary_y(lines, y_start, "CPC", width)
    end_anchor_y = _target_end_anchor_y(lines, y_start, "CPC", max_y=line_boundary_y)
    next_y = _choose_y_end(
        height=height,
        typed_boundary_y=height,
        line_boundary_y=line_boundary_y,
        end_anchor_y=end_anchor_y,
    )
    signature_bounds = _signature_vertical_bounds(
        lines,
        target_table="CPC",
        scope=scope,
        sector=sector,
        y_start=y_start,
        max_y=next_y,
        x_bounds=None,
    )
    if signature_bounds is not None:
        sig_y0, sig_y1, sig_hits, sig_cov = signature_bounds
        y_start = min(y_start, sig_y0)
        next_y = max(next_y, min(height, sig_y1 + _signature_bottom_padding("CPC")))

    cpc_bounds = _cpc_horizontal_bounds(lines, y_start, next_y, width)
    if cpc_bounds is None:
        x_start, x_end = 0.0, width
    else:
        x_start, x_end = cpc_bounds

    x_start, x_end = _cpc_merge_x_with_row_band(
        x_start, x_end, width, lines, y_start, next_y
    )

    for b in all_blocks:
        bx0, by0, bx1, _by1 = map(float, b[:4])
        center_x = (bx0 + bx1) / 2
        if y_start - 5 <= by0 <= next_y + 5 and x_start - 5 <= center_x <= x_end + 5:
            x_start = min(x_start, bx0)
            x_end = max(x_end, bx1)

    bbox = clamp_bbox([x_start - 8, max(0.0, y_start - 10), x_end + 8, min(height, next_y)], width, height)
    confidence = min(0.99, 0.84 + 0.15 * retrieved.score)
    evidence = [
        "localized:CPC",
        f"page:{page_num}",
        "line_header",
        f"retrieval_score:{retrieved.score:.3f}",
        f"end_y:{next_y:.1f}",
        "title_confirmed:True",
    ]
    if signature_bounds is not None:
        evidence.append(f"signature_region_hits:{sig_hits}")
        evidence.append(f"signature_region_coverage:{sig_cov:.3f}")
    return TableCandidate(
        pdf_id=Path(pdf_path).stem,
        page_number=page_num,
        scope=scope,
        sector=sector,
        table_type="CPC",
        bbox=bbox,
        confidence=confidence,
        evidence=evidence,
    )


def _candidate_from_block_group(
    pdf_path: str,
    page_num: int,
    width: float,
    height: float,
    target_table: str,
    scope: str,
    sector: str,
    target_blocks,
    retrieved: RetrievedPage,
    *,
    doc: fitz.Document,
) -> TableCandidate:
    page = doc[page_num - 1]
    all_blocks = [b for b in page.get_text("blocks") if len(b) >= 5 and str(b[4]).strip()]
    lines = _extract_lines(page)

    y_start = min(float(b[1]) for _, b in target_blocks)
    x_start = min(float(b[0]) for _, b in target_blocks)
    x_end = max(float(b[2]) for _, b in target_blocks)

    column_bounds = _side_by_side_header_bounds(lines, target_table, width)
    if column_bounds is None and target_table in {"BILAN_ACTIF", "BILAN_PASSIF"}:
        column_bounds = _target_column_bounds_from_layout(lines, target_blocks, y_start, width)

    typed_boundary_y = height
    for b in all_blocks:
        by = float(b[1])
        if by <= y_start + 12:
            continue
        if column_bounds is not None:
            bx0, _by0, bx1, _by1 = map(float, b[:4])
            center_x = (bx0 + bx1) / 2
            if not column_bounds[0] - 5 <= center_x <= column_bounds[1] + 5:
                continue
        block_text = str(b[4])
        other_type = _detect_block_type(block_text, sector)
        other_norm = normalize_text(block_text)
        if target_table == "BILAN_PASSIF" and other_type == "BILAN_ACTIF" and "passif" in other_norm:
            continue
        if other_type and other_type != target_table:
            if not _has_target_title(other_type, [block_text]):
                continue
            typed_boundary_y = min(typed_boundary_y, by)

    line_boundary_y = _first_boundary_y(lines, y_start, target_table, width, x_bounds=column_bounds)
    end_anchor_y = _target_end_anchor_y(lines, y_start, target_table, max_y=line_boundary_y, x_bounds=column_bounds)
    next_y = _choose_y_end(
        height=height,
        typed_boundary_y=typed_boundary_y,
        line_boundary_y=line_boundary_y,
        end_anchor_y=end_anchor_y,
    )
    signature_bounds = _signature_vertical_bounds(
        lines,
        target_table=target_table,
        scope=scope,
        sector=sector,
        y_start=y_start,
        max_y=next_y,
        x_bounds=column_bounds,
    )
    if signature_bounds is not None:
        sig_y0, sig_y1, sig_hits, sig_cov = signature_bounds
        y_start = min(y_start, sig_y0)
        next_y = max(next_y, min(height, sig_y1 + _signature_bottom_padding(target_table)))

    # Capture text blocks in the same horizontal band. For side-by-side actif/passif,
    # keep the target column; for stacked layouts, use full table width.
    same_row = [
        b for t, b in target_blocks
        if abs(float(b[1]) - y_start) <= 24
    ]
    if len(same_row) == 1:
        title_center = (float(same_row[0][0]) + float(same_row[0][2])) / 2
        same_y_titles = [
            b for t, b in _typed_blocks_for_page(page, sector)
            if abs(float(b[1]) - y_start) <= 24
        ]
        same_y_titles = sorted(same_y_titles, key=lambda b: b[0])
        if len(same_y_titles) >= 2:
            centers = [(float(b[0]) + float(b[2])) / 2 for b in same_y_titles]
            left_bounds = [0.0] + [(centers[i] + centers[i + 1]) / 2 for i in range(len(centers) - 1)]
            right_bounds = left_bounds[1:] + [width]
            for b, left, right in zip(same_y_titles, left_bounds, right_bounds):
                c = (float(b[0]) + float(b[2])) / 2
                if abs(c - title_center) < 1:
                    x_start, x_end = left, right
                    break

    cpc_bounds = None
    if column_bounds is not None:
        x_start, x_end = column_bounds
    elif target_table == "CPC":
        cpc_bounds = _cpc_horizontal_bounds(lines, y_start, next_y, width)
        if cpc_bounds is not None:
            x_start, x_end = cpc_bounds

    if target_table == "CPC":
        x_start, x_end = _cpc_merge_x_with_row_band(
            x_start, x_end, width, lines, y_start, next_y
        )

    for b in all_blocks:
        bx0, by0, bx1, by1 = map(float, b[:4])
        center_x = (bx0 + bx1) / 2
        if y_start - 5 <= by0 <= next_y + 5 and x_start - 5 <= center_x <= x_end + 5:
            x_start = min(x_start, bx0)
            x_end = max(x_end, bx1)

    if target_table == "CPC" and cpc_bounds is None:
        for b in all_blocks:
            bx0, by0, bx1, by1 = map(float, b[:4])
            if y_start - 5 <= by0 <= next_y + 5:
                x_start = min(x_start, bx0)
                x_end = max(x_end, bx1)

    if column_bounds is not None:
        x_start = max(x_start, column_bounds[0])
        x_end = min(x_end, column_bounds[1])

    bbox = clamp_bbox([x_start - 8, max(0, y_start - 10), x_end + 8, min(height, next_y)], width, height)
    title_confirmed = _has_target_title(target_table, [str(b[4]) for _, b in target_blocks])
    confidence = min(0.99, 0.70 + 0.29 * retrieved.score)
    if retrieved.scope_score <= 0:
        confidence = min(confidence, 0.45)
    if not title_confirmed:
        confidence = min(confidence, 0.35 if target_table == "CPC" else 0.68)
    evidence = [
        f"localized:{target_table}",
        f"page:{page_num}",
        f"retrieval_score:{retrieved.score:.3f}",
        f"end_y:{next_y:.1f}",
        f"title_confirmed:{title_confirmed}",
    ]
    if signature_bounds is not None:
        evidence.append(f"signature_region_hits:{sig_hits}")
        evidence.append(f"signature_region_coverage:{sig_cov:.3f}")
    confidence = _adjust_balance_candidate_confidence(
        confidence=confidence,
        evidence=evidence,
        lines=lines,
        bbox=bbox,
        target_table=target_table,
        y_start=y_start,
    )

    return TableCandidate(
        pdf_id=Path(pdf_path).stem,
        page_number=page_num,
        scope=scope,
        sector=sector,
        table_type=target_table,
        bbox=bbox,
        confidence=confidence,
        evidence=evidence,
    )


def _adjust_balance_candidate_confidence(
    *,
    confidence: float,
    evidence: list[str],
    lines: list[tuple[float, float, float, float, str, str]],
    bbox: list[float],
    target_table: str,
    y_start: float,
) -> float:
    if target_table not in {"BILAN_ACTIF", "BILAN_PASSIF"}:
        return confidence

    region_height = max(0.0, float(bbox[3]) - float(bbox[1]))
    page_text = " ".join(norm for *_coords, _text, norm in lines)
    has_non_main_context = any(marker in page_text for marker in _NON_MAIN_STATEMENT_MARKERS)
    has_alternate_statement_context = any(marker in page_text for marker in _ALTERNATE_STATEMENT_MARKERS)
    early_lines = [
        norm
        for _x0, y0, _x1, _y1, _text, norm in lines
        if y0 <= y_start + 45
    ]
    starts_like_main_statement = any(
        norm in {"actif", "passif"}
        or norm.startswith("bilan au")
        or norm.startswith("bilan actif")
        or norm.startswith("bilan passif")
        for norm in early_lines[:8]
    )

    if region_height < _MIN_MAIN_BALANCE_REGION_HEIGHT:
        evidence.append(f"penalty:small_balance_region:{region_height:.1f}")
        confidence = min(confidence, 0.70)

    if has_non_main_context and not starts_like_main_statement:
        evidence.append("penalty:non_main_statement_context")
        confidence = min(confidence, 0.72)

    if has_non_main_context and region_height < 180.0:
        evidence.append("penalty:note_like_balance_region")
        confidence = min(confidence, 0.62)

    if has_alternate_statement_context:
        evidence.append("penalty:alternate_statement_context")
        confidence = min(confidence, 0.58)

    return confidence


def _signature_vertical_bounds(
    lines: list[tuple[float, float, float, float, str, str]],
    *,
    target_table: str,
    scope: str,
    sector: str,
    y_start: float,
    max_y: float,
    x_bounds: tuple[float, float] | list[float] | None = None,
) -> tuple[float, float, int, float] | None:
    """Find the dense vertical region covered by the full scoped row-label list.

    Page retrieval already uses these signatures to choose the page. Here we use
    the same labels geometrically: the first and last matched row labels are a
    strong hint for where the table lives inside the selected page.
    """
    labels = scoped_table_signature(sector, scope, target_table)
    if not labels:
        return None

    label_norms = [(label, normalize_text(label)) for label in labels if normalize_text(label)]
    if not label_norms:
        return None

    matched: list[tuple[float, float, str]] = []
    x0_bound = float(x_bounds[0]) if x_bounds is not None else None
    x1_bound = float(x_bounds[1]) if x_bounds is not None else None
    for x0, y0, x1, y1, text, norm in lines:
        if y1 < y_start - 45 or y0 > max_y + 8:
            continue
        if x0_bound is not None and x1_bound is not None:
            center_x = (x0 + x1) / 2
            if not x0_bound - 8 <= center_x <= x1_bound + 8:
                continue
        loose_line = _loose_label_phrase(norm)
        for label, label_norm in label_norms:
            if label_norm in norm or _loose_label_phrase(label_norm) in loose_line:
                matched.append((float(y0), float(y1), label))
                break

    if not matched:
        return None

    unique_hits = {normalize_text(label) for *_ys, label in matched}
    coverage = len(unique_hits) / max(len({norm for _label, norm in label_norms}), 1)
    min_hits = _signature_region_min_hits(target_table, len(label_norms))
    if len(unique_hits) < min_hits and coverage < 0.35:
        return None

    matched.sort(key=lambda item: item[0])
    # Drop isolated early/late generic labels if the dense cluster is elsewhere.
    clustered = _densest_vertical_label_cluster(matched)
    if clustered:
        matched = clustered

    return min(y0 for y0, _y1, _label in matched), max(y1 for _y0, y1, _label in matched), len(unique_hits), coverage


def _densest_vertical_label_cluster(
    matched: list[tuple[float, float, str]],
    *,
    max_gap: float = 95.0,
) -> list[tuple[float, float, str]]:
    if len(matched) <= 2:
        return matched
    clusters: list[list[tuple[float, float, str]]] = []
    current = [matched[0]]
    for item in matched[1:]:
        if item[0] - current[-1][1] <= max_gap:
            current.append(item)
        else:
            clusters.append(current)
            current = [item]
    clusters.append(current)
    return max(clusters, key=lambda cluster: (len({normalize_text(label) for *_ys, label in cluster}), len(cluster)))


def _signature_region_min_hits(target_table: str, label_count: int) -> int:
    if label_count <= 15:
        return max(4, int(label_count * 0.35))
    if target_table == "CPC":
        return max(8, int(label_count * 0.22))
    return max(7, int(label_count * 0.25))


def _signature_bottom_padding(target_table: str) -> float:
    return 16.0 if target_table == "CPC" else 12.0


def _loose_label_phrase(text: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", " ", normalize_text(text)).strip()


def _typed_blocks_for_page(page, sector: str):
    return [(t, b) for b in page.get_text("blocks") if (t := _detect_block_type(str(b[4]), sector))]


def _disambiguate_bilan_actif_passif_block(norm: str, text: str, sector: str) -> str | None:
    """Both balance TABLE_TITLES can match (e.g. shared « bilan social »); use row anchors."""
    anchors_a = FINANCIAL_ANCHORS.get(sector, {}).get("BILAN_ACTIF", [])
    anchors_p = FINANCIAL_ANCHORS.get(sector, {}).get("BILAN_PASSIF", [])
    ca = len(keyword_hits(text, anchors_a))
    cp = len(keyword_hits(text, anchors_p))
    if cp > ca:
        return "BILAN_PASSIF"
    if ca > cp:
        return "BILAN_ACTIF"
    if any(
        x in norm
        for x in (
            "depots de la clientele",
            "dettes envers les etablissements de credit",
            "comptes a vue crediteurs",
            "comptes d epargne",
            "titres de creance emis",
            "emprunts obligataires",
            "report a nouveau",
            "resultat net de l exercice",
            "total du passif",
        )
    ):
        return "BILAN_PASSIF"
    if any(
        x in norm
        for x in (
            "creances sur la clientele",
            "titres de transaction",
            "immobilisations corporelles",
            "prets et creances sur la clientele",
            "total de l actif",
            "total actif",
        )
    ):
        return "BILAN_ACTIF"
    if len(norm) < 72 and "bilan social" in norm:
        return None
    return "BILAN_ACTIF"


def _lines_bilan_social_title_lines(
    lines: list[tuple[float, float, float, float, str, str]],
) -> list[tuple[float, float, float, float, str, str]]:
    out: list[tuple[float, float, float, float, str, str]] = []
    for line in lines:
        n = line[5]
        if n == "bilan social":
            out.append(line)
        elif n.startswith("bilan social ") and len(n) <= 48:
            out.append(line)
    return out


def _detect_block_type(text: str, sector: str) -> str | None:
    norm = normalize_text(text[:600])
    if "hors bilan" in norm or "notes annexes" in norm:
        return None
    non_statement_markers = [
        "tableau de financement",
        "flux de tresorerie",
        "synthese des masses du bilan",
        "capacite d'autofinancement",
        "etats des soldes de gestion",
    ]
    has_statement_title = any(
        marker in norm[:220]
        for marker in [
            "bilan (bl)",
            "bilan actif",
            "bilan passif",
            "bilan (actif)",
            "bilan (passif)",
            "bilan consolide",
            "bilan social",
            "compte de produits",
            "compte de resultat",
            "etat du resultat",
        ]
    )
    if any(marker in norm for marker in non_statement_markers) and not has_statement_title:
        return None
    matched_types = [tt for tt, titles in TABLE_TITLES.items() if keyword_hits(norm, titles)]
    if not matched_types:
        best_type: str | None = None
        best_n = 0
        for table_type, anchors in FINANCIAL_ANCHORS.get(sector, {}).items():
            n = len(keyword_hits(text, anchors))
            if n >= 2 and n > best_n:
                best_n = n
                best_type = table_type
        return best_type
    if "BILAN_ACTIF" in matched_types and "BILAN_PASSIF" in matched_types:
        resolved = _disambiguate_bilan_actif_passif_block(norm, text, sector)
        if resolved is not None:
            return resolved
        return None
    for table_type in TABLE_TITLES:
        if table_type in matched_types:
            return table_type
    return matched_types[0]


def _extract_lines(page) -> list[tuple[float, float, float, float, str, str]]:
    lines: list[tuple[float, float, float, float, str, str]] = []
    for block in page.get_text("dict").get("blocks", []):
        for line in block.get("lines", []):
            text = " ".join(span.get("text", "") for span in line.get("spans", [])).strip()
            if not text:
                continue
            x0, y0, x1, y1 = map(float, line["bbox"])
            lines.append((x0, y0, x1, y1, text, normalize_text(text)))
    lines.sort(key=lambda item: (item[1], item[0]))
    return lines


def _first_boundary_y(
    lines: list[tuple[float, float, float, float, str, str]],
    y_start: float,
    target_table: str,
    width: float,
    x_bounds: tuple[float, float] | None = None,
) -> float | None:
    markers = list(_NEGATIVE_BOUNDARY_MARKERS)
    if target_table == "BILAN_ACTIF":
        markers.append("passif")
    elif target_table == "BILAN_PASSIF":
        markers.append("actif")

    cpc_side = _cpc_title_side(lines, y_start, width) if target_table == "CPC" else None
    for x0, y0, x1, _y1, _text, norm in lines:
        if y0 <= y_start + 8:
            continue
        center_x = (x0 + x1) / 2
        if x_bounds is not None and not x_bounds[0] - 5 <= center_x <= x_bounds[1] + 5:
            continue
        if target_table == "CPC" and cpc_side == "left" and center_x > width * 0.45:
            continue
        if target_table == "CPC" and cpc_side == "right" and center_x < width * 0.45:
            continue
        if _is_boundary_line(norm, markers, target_table):
            return y0
    return None


def _is_passif_equity_oci_row(norm_line: str) -> bool:
    """IFRS passif rows under capitaux propres; share phrases with CPC/EGP titles in _NEGATIVE_BOUNDARY_MARKERS."""
    if "gains et pertes" not in norm_line:
        return False
    # "capitaux propre" matches both capitaux propre(s) after normalize_text
    return "capitaux propre" in norm_line


def _is_boundary_line(norm_line: str, markers: list[str], target_table: str) -> bool:
    # Avoid treating row labels like "Autres actifs" as a new ACTIF section.
    if target_table == "BILAN_PASSIF" and norm_line not in {"actif"} and norm_line.startswith("autres actif"):
        return False
    if target_table == "BILAN_PASSIF" and _is_passif_equity_oci_row(norm_line):
        return False
    for marker in markers:
        marker_norm = normalize_text(marker)
        if marker_norm in {"actif", "passif"}:
            if target_table == "BILAN_PASSIF" and marker_norm == "actif" and "passif" in norm_line:
                return False
            if norm_line == marker_norm or norm_line.startswith(marker_norm + " "):
                return True
            continue
        if marker_norm in norm_line:
            return True
    return False


def _target_end_anchor_y(
    lines: list[tuple[float, float, float, float, str, str]],
    y_start: float,
    target_table: str,
    max_y: float | None = None,
    x_bounds: tuple[float, float] | None = None,
) -> float | None:
    anchors = [normalize_text(a) for a in _TARGET_END_ANCHORS.get(target_table, [])]
    best: float | None = None
    for x0, y0, x1, y1, _text, norm in lines:
        if y0 <= y_start + 8:
            continue
        center_x = (x0 + x1) / 2
        if x_bounds is not None and not x_bounds[0] - 5 <= center_x <= x_bounds[1] + 5:
            continue
        if max_y is not None and y0 >= max_y:
            continue
        if any(anchor in norm for anchor in anchors):
            best = y1 if best is None else max(best, y1)
    return best


def _choose_y_end(
    *,
    height: float,
    typed_boundary_y: float,
    line_boundary_y: float | None,
    end_anchor_y: float | None,
) -> float:
    boundaries = [typed_boundary_y]
    if line_boundary_y is not None:
        boundaries.append(line_boundary_y)
    boundary_y = min(boundaries) if boundaries else height

    if end_anchor_y is not None:
        if boundary_y <= end_anchor_y:
            return min(height, end_anchor_y + 4.0)
        # Prefer stopping shortly after the total/result row. If the next section
        # begins immediately after it, stay just above that next header.
        return min(end_anchor_y + 4.0, boundary_y - 2.0 if boundary_y < height else end_anchor_y + 4.0)

    if boundary_y < height:
        return max(0.0, boundary_y - 2.0)
    return height - 20.0


def _side_by_side_header_bounds(
    lines: list[tuple[float, float, float, float, str, str]],
    target_table: str,
    width: float,
) -> tuple[float, float] | None:
    actif_headers = [
        line for line in lines
        if _is_exact_table_header(line[5], "actif")
    ]
    passif_headers = [
        line for line in lines
        if _is_exact_table_header(line[5], "passif")
    ]
    if not actif_headers or not passif_headers:
        return None

    best_pair = None
    best_delta = 10**9
    for actif in actif_headers:
        for passif in passif_headers:
            delta = abs(actif[1] - passif[1])
            if delta < best_delta:
                best_delta = delta
                best_pair = (actif, passif)
    if best_pair is None or best_delta > 40:
        return None

    actif, passif = best_pair
    actif_center = (actif[0] + actif[2]) / 2
    passif_center = (passif[0] + passif[2]) / 2
    if abs(actif_center - passif_center) < width * 0.20:
        return None
    split_x = (actif_center + passif_center) / 2
    if target_table == "BILAN_ACTIF":
        return (0.0, max(split_x, width * 0.50))
    if target_table == "BILAN_PASSIF":
        return (max(split_x, width * 0.50), width)
    return None


def _target_column_bounds_from_layout(
    lines: list[tuple[float, float, float, float, str, str]],
    target_blocks,
    y_start: float,
    width: float,
) -> tuple[float, float] | None:
    title_centers = [
        (float(b[0]) + float(b[2])) / 2
        for _t, b in target_blocks
        if abs(float(b[1]) - y_start) <= 28
        and "bilan" in normalize_text(str(b[4]))
        and _has_target_title(_t, [str(b[4])])
    ]
    title_centers = title_centers or [
        (float(b[0]) + float(b[2])) / 2
        for _t, b in target_blocks
        if abs(float(b[1]) - y_start) <= 28
        and _has_target_title(_t, [str(b[4])])
    ]
    target_centers = title_centers or [
        (float(b[0]) + float(b[2])) / 2
        for _t, b in target_blocks
        if abs(float(b[1]) - y_start) <= 28
    ]
    if not target_centers:
        return None

    center = sum(target_centers) / len(target_centers)
    split_x = width / 2
    left_has_content = False
    right_has_content = False
    left_has_split_title = False
    right_has_split_title = False
    for x0, y0, x1, _y1, _text, norm in lines:
        if y0 < max(0.0, y_start - 40):
            continue
        line_center = (x0 + x1) / 2
        if line_center < split_x - 20:
            left_has_content = True
            left_has_split_title = left_has_split_title or _is_layout_split_title(norm)
        elif line_center > split_x + 20:
            right_has_content = True
            right_has_split_title = right_has_split_title or _is_layout_split_title(norm)
        if left_has_content and right_has_content:
            if left_has_split_title or right_has_split_title:
                break

    if not (left_has_content and right_has_content):
        return None
    if center < split_x - 20:
        if not right_has_split_title:
            return None
        return (0.0, split_x)
    if center > split_x + 20:
        if not left_has_split_title:
            return None
        return (split_x, width)
    return None


def _single_column_bounds_from_header(
    lines: list[tuple[float, float, float, float, str, str]],
    header: tuple[float, float, float, float, str, str],
    target_table: str,
    width: float,
) -> tuple[float, float] | None:
    if target_table not in {"BILAN_ACTIF", "BILAN_PASSIF"}:
        return None

    y_start = float(header[1])
    header_center = (float(header[0]) + float(header[2])) / 2
    split_x = width / 2
    right_neighbor_xs = [
        x0
        for x0, y0, x1, _y1, _text, norm in lines
        if y0 >= y_start - 50
        and ((x0 + x1) / 2) > split_x - 5
        and _is_neighbor_statement_title(norm)
    ]
    left_neighbor_xs = [
        x1
        for x0, y0, x1, _y1, _text, norm in lines
        if y0 >= y_start - 50
        and ((x0 + x1) / 2) < split_x + 5
        and _is_neighbor_statement_title(norm)
    ]
    right_has_other_statement = bool(right_neighbor_xs)
    left_has_other_statement = bool(left_neighbor_xs)

    if header_center < split_x - 20 and right_has_other_statement:
        return (0.0, max(width * 0.35, min(right_neighbor_xs) - 8.0))
    if header_center > split_x + 20 and left_has_other_statement:
        return (min(width * 0.65, max(left_neighbor_xs) + 8.0), width)
    return None


def _is_layout_split_title(norm_line: str) -> bool:
    return (
        norm_line.startswith("bilan actif")
        or norm_line.startswith("bilan passif")
        or norm_line.startswith("bilan social")
        or norm_line.startswith("bilan (actif)")
        or norm_line.startswith("bilan (passif)")
        or norm_line.startswith("compte de produits")
        or norm_line.startswith("comptes de produits")
        or norm_line.startswith("compte de resultat")
        or norm_line.startswith("etat du resultat")
        or norm_line.startswith("etat des soldes")
        or norm_line.startswith("tableau des flux")
        or norm_line.startswith("flux de tresorerie")
        or norm_line.startswith("etat du resultat net")
        or norm_line.startswith("etat de resultat net")
        or norm_line.startswith("resultat net et gains")
        or norm_line.startswith("cpc")
    )


def _is_neighbor_statement_title(norm_line: str) -> bool:
    return (
        _is_layout_split_title(norm_line)
        or norm_line.startswith("tableau des flux")
        or norm_line.startswith("flux de tresorerie")
        or norm_line.startswith("etat du resultat net")
        or norm_line.startswith("etat de resultat net")
        or norm_line.startswith("resultat net et gains")
        or norm_line.startswith("etat des soldes")
        or norm_line.startswith("capacite d'autofinancement")
        or norm_line.startswith("capacite d autofinancement")
    )


def _is_exact_table_header(norm_line: str, header: str) -> bool:
    compact = norm_line.replace(" ", "")
    allowed_suffixes = (
        "",
        " ifrs",
        " social",
        " consolide",
        " consolidee",
        " sociaux",
        " consolides",
    )
    if norm_line.startswith(header + " (") or norm_line.startswith(header + " au "):
        return True
    if header == "actif":
        return compact == "actif" or any(norm_line == f"actif{suffix}" for suffix in allowed_suffixes)
    if header == "passif":
        return compact == "passif" or any(norm_line == f"passif{suffix}" for suffix in allowed_suffixes)
    return False


def _select_best_balance_header(
    lines: list[tuple[float, float, float, float, str, str]],
    headers: list[tuple[float, float, float, float, str, str]],
    target_table: str,
    width: float,
) -> tuple[float, float, float, float, str, str]:
    if len(headers) == 1:
        return headers[0]

    def score(header: tuple[float, float, float, float, str, str]) -> float:
        _x0, y0, _x1, _y1, _text, _norm = header
        before = [norm for *_coords, _text, norm in lines if y0 - 90 <= _coords[1] <= y0]
        after = [norm for *_coords, _text, norm in lines if y0 < _coords[1] <= y0 + 260]
        joined_before = " ".join(before)
        joined_after = " ".join(after)

        value = 0.0
        if any(marker in joined_before for marker in ["bilan ifrs", "bilan consolide", "bilan au", "etat de la situation financiere"]):
            value += 3.0
        if target_table == "BILAN_PASSIF" and any("total actif" in norm or "total de l actif" in norm for norm in before):
            value += 2.5
        if target_table == "BILAN_PASSIF" and any("total passif" in norm or "total du passif" in norm for norm in after):
            value += 3.0
        if target_table == "BILAN_ACTIF" and any("total actif" in norm or "total de l actif" in norm for norm in after):
            value += 3.0
        if any(marker in joined_after for marker in ["capitaux propres", "capital social", "dettes envers", "passifs d impots"]):
            value += 1.0
        if any(marker in joined_after for marker in ["caisse banques", "creances", "immobilisations", "actifs financiers"]):
            value += 1.0
        if y0 < 180 and not any(marker in joined_before for marker in ["bilan ifrs", "bilan consolide", "bilan au"]):
            value -= 2.0
        return value

    return max(headers, key=score)


def _has_target_title(target_table: str, texts: list[str]) -> bool:
    normalized_texts = [normalize_text(text) for text in texts]
    joined = " ".join(normalized_texts)
    if target_table == "BILAN_ACTIF":
        return (
            "bilan actif" in joined
            or "bilan (actif)" in joined
            or "actif ifrs" in joined
            or joined == "actif"
            or joined.startswith("actif (")
        )
    if target_table == "BILAN_PASSIF":
        return (
            "bilan passif" in joined
            or "bilan (passif)" in joined
            or "passif ifrs" in joined
            or joined == "passif"
            or joined.startswith("passif (")
        )
    if target_table == "CPC":
        return any(
            text.startswith("cpc consolide")
            or text.startswith("compte de produits et charges")
            or text.startswith("compte de produits et de charges")
            or text.startswith("comptes de produits et charges")
            or text.startswith("comptes de produits et de charges")
            or text.startswith("compte de resultat")
            or text.startswith("etat du resultat global")
            or text.startswith("cpc")
            or "compte de resultat consolide" in text[:80]
            or "compte de produits et charges consolide" in text[:90]
            or "etat du resultat global" in text[:90]
            for text in normalized_texts
        )
    return False


_CPC_SKIP_RIGHT_COLUMN_MARKERS = (
    "flux de tresorerie",
    "tableau des flux de tresorerie",
    "variation des capitaux propres",
    "etat du resultat net",
    "resultat net et gains",
    "perimetre de consolidation",
    "tableau de financement",
    "attestation",
)

_CPC_ROW_LINE_HINTS = (
    "total",
    "produit",
    "charge",
    "resultat",
    "marge",
    "frais",
    "impot",
    "note",
    "dotation",
    "reprise",
    "part du groupe",
    "interet",
    "commission",
    "chiffre",
    "exercice",
    "precedent",
    "brut",
    "net",
    "revenu",
    "cout du risque",
    "pnb",
    "consolide",
    "charges generales",
    "produits d exploitation",
    "charges d exploitation",
)


def _cpc_row_band_horizontal_extent(
    lines: list[tuple[float, float, float, float, str, str]],
    y_start: float,
    y_end: float,
    width: float,
) -> tuple[float | None, float | None]:
    """Horizontal span of CPC row-like lines in [y_start, y_end].

    Two-column layouts often place the title on the right; `_cpc_horizontal_bounds`
    then uses (title_x, page_width) and drops the label column. Row lines usually
    span from the left margin through figures — union their x extents to recover
    the full table width.
    """
    leftmost: float | None = None
    rightmost: float | None = None

    def consider(x0: float, x1: float, norm: str, cx: float) -> None:
        nonlocal leftmost, rightmost
        if any(marker in norm for marker in _CPC_SKIP_RIGHT_COLUMN_MARKERS) and cx > width * 0.52:
            return
        leftmost = x0 if leftmost is None else min(leftmost, x0)
        rightmost = x1 if rightmost is None else max(rightmost, x1)

    for x0, y0, x1, y1, text, norm in lines:
        if y0 < y_start + 10 or y0 > y_end:
            continue
        if len(norm.strip()) < 3:
            continue
        cx = (x0 + x1) / 2
        looks_row = any(h in norm for h in _CPC_ROW_LINE_HINTS) or any(
            ch.isdigit() for ch in text
        )
        if not looks_row:
            continue
        consider(x0, x1, norm, cx)

    if leftmost is None:
        for x0, y0, x1, y1, text, norm in lines:
            if y0 < y_start + 15 or y0 > y_end:
                continue
            if len(text.strip()) < 2:
                continue
            cx = (x0 + x1) / 2
            if any(marker in norm for marker in _CPC_SKIP_RIGHT_COLUMN_MARKERS) and cx > width * 0.52:
                continue
            if any(ch.isdigit() for ch in text):
                consider(x0, x1, norm, cx)

    return leftmost, rightmost


def _cpc_merge_x_with_row_band(
    x_start: float,
    x_end: float,
    width: float,
    lines: list[tuple[float, float, float, float, str, str]],
    y_start: float,
    y_end: float,
) -> tuple[float, float]:
    band_l, band_r = _cpc_row_band_horizontal_extent(lines, y_start, y_end, width)
    pad = min(28.0, max(12.0, width * 0.035))
    margin_l = min(20.0, width * 0.04)
    if band_l is not None:
        x_start = min(x_start, max(0.0, band_l - pad))
    if band_r is not None:
        x_end = max(x_end, min(width, band_r + pad))
    if band_l is not None and x_start > band_l + 80:
        x_start = max(margin_l, band_l - pad)
    return x_start, x_end


def _cpc_horizontal_bounds(
    lines: list[tuple[float, float, float, float, str, str]],
    y_start: float,
    y_end: float,
    width: float,
) -> tuple[float, float] | None:
    title_lines = [
        (x0, y0, x1, y1, norm)
        for x0, y0, x1, y1, _text, norm in lines
        if abs(y0 - y_start) < 45
        and (
            norm.startswith("compte de produits et charges")
            or norm.startswith("compte de resultat")
            or norm.startswith("etat du resultat")
            or norm.startswith("etat de resultat")
            or "compte de produits et charges consolide" in norm[:90]
            or "etat du resultat global" in norm[:90]
        )
    ]
    if title_lines:
        x0, _y0, _x1, _y1, _norm = min(title_lines, key=lambda item: item[1])
        if x0 > width * 0.45:
            return (max(0.0, x0 - 8.0), width)
        if x0 < width * 0.25:
            right_side_boundaries = [
                bx0 for bx0, by0, _bx1, _by1, _text, norm in lines
                if y_start - 12 < by0 < y_end
                and ((bx0 + _bx1) / 2) > width * 0.45
                and any(marker in norm for marker in _CPC_SKIP_RIGHT_COLUMN_MARKERS)
            ]
            if right_side_boundaries:
                return (0.0, min(right_side_boundaries) - 8.0)

    right_boundaries = [
        x0 for x0, y0, _x1, _y1, _text, norm in lines
        if y_start + 10 < y0 < y_end
        and ((x0 + _x1) / 2) > width * 0.45
        and any(marker in norm for marker in _CPC_SKIP_RIGHT_COLUMN_MARKERS)
    ]
    if not right_boundaries:
        return None
    return (0.0, min(right_boundaries) - 8.0)


def _cpc_title_side(
    lines: list[tuple[float, float, float, float, str, str]],
    y_start: float,
    width: float,
) -> str | None:
    title_lines = [
        x0 for x0, y0, _x1, _y1, _text, norm in lines
        if abs(y0 - y_start) < 45
        and (
            norm.startswith("compte de produits et charges")
            or norm.startswith("compte de resultat")
            or norm.startswith("etat du resultat")
            or norm.startswith("etat de resultat")
            or "compte de produits et charges consolide" in norm[:90]
            or "etat du resultat global" in norm[:90]
        )
    ]
    if not title_lines:
        return None
    x0 = min(title_lines)
    return "right" if x0 > width * 0.45 else "left"
