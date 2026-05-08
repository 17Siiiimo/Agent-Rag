from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

from .keyword_dictionary import FINANCIAL_ANCHORS, TABLE_TITLES, keyword_hits
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
    for retrieved in retrieved_pages[:max_pages]:
        candidates.extend(_localize_on_page(pdf_path, retrieved, target_table, scope, sector))
    retrieval_scores = {r.page.page_number: r.score for r in retrieved_pages}
    scope_scores = {r.page.page_number: r.scope_score for r in retrieved_pages}
    candidates.sort(
        key=lambda c: (
            -(0.60 * retrieval_scores.get(c.page_number, 0.0) + 0.30 * c.confidence + 0.10 * scope_scores.get(c.page_number, 0.0)),
            c.page_number,
        )
    )
    if debug_dir:
        write_json(Path(debug_dir) / "table_candidates.json", [c.to_dict() for c in candidates])
    return candidates


def _localize_on_page(
    pdf_path: str,
    retrieved: RetrievedPage,
    target_table: str,
    scope: str,
    sector: str,
) -> list[TableCandidate]:
    page_num = retrieved.page.page_number
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num - 1]
        page_rect = page.rect
        blocks = [b for b in page.get_text("blocks") if len(b) >= 5 and str(b[4]).strip()]
    finally:
        doc.close()

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
) -> TableCandidate | None:
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num - 1]
        lines = _extract_lines(page)
        all_blocks = [b for b in page.get_text("blocks") if len(b) >= 5 and str(b[4]).strip()]
    finally:
        doc.close()

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

    wanted_header = "actif" if target_table == "BILAN_ACTIF" else "passif"
    headers = [line for line in lines if _is_exact_table_header(line[5], wanted_header)]
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

    cpc_bounds = _cpc_horizontal_bounds(lines, y_start, next_y, width)
    if cpc_bounds is None:
        x_start, x_end = 0.0, width
    else:
        x_start, x_end = cpc_bounds

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
) -> TableCandidate:
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num - 1]
        all_blocks = [b for b in page.get_text("blocks") if len(b) >= 5 and str(b[4]).strip()]
        lines = _extract_lines(page)
    finally:
        doc.close()

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

    # Capture text blocks in the same horizontal band. For side-by-side actif/passif,
    # keep the target column; for stacked layouts, use full table width.
    same_row = [
        b for t, b in target_blocks
        if abs(float(b[1]) - y_start) <= 24
    ]
    if len(same_row) == 1:
        title_center = (float(same_row[0][0]) + float(same_row[0][2])) / 2
        same_y_titles = [
            b for t, b in _typed_blocks_for_page(pdf_path, page_num, sector)
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


def _typed_blocks_for_page(pdf_path: str, page_num: int, sector: str):
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_num - 1]
        return [(t, b) for b in page.get_text("blocks") if (t := _detect_block_type(str(b[4]), sector))]
    finally:
        doc.close()


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
            "compte de produits",
            "compte de resultat",
            "etat du resultat",
        ]
    )
    if any(marker in norm for marker in non_statement_markers) and not has_statement_title:
        return None
    for table_type, titles in TABLE_TITLES.items():
        if keyword_hits(norm, titles):
            return table_type
    for table_type, anchors in FINANCIAL_ANCHORS.get(sector, {}).items():
        if len(keyword_hits(norm, anchors)) >= 2:
            return table_type
    return None


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


def _is_boundary_line(norm_line: str, markers: list[str], target_table: str) -> bool:
    # Avoid treating row labels like "Autres actifs" as a new ACTIF section.
    if target_table == "BILAN_PASSIF" and norm_line not in {"actif"} and norm_line.startswith("autres actif"):
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
                and any(marker in norm for marker in [
                    "flux de tresorerie",
                    "tableau des flux de tresorerie",
                    "variation des capitaux propres",
                    "etat du resultat net",
                    "resultat net et gains",
                    "perimetre de consolidation",
                    "tableau de financement",
                    "attestation",
                ])
            ]
            if right_side_boundaries:
                return (0.0, min(right_side_boundaries) - 8.0)

    right_boundaries = [
        x0 for x0, y0, _x1, _y1, _text, norm in lines
        if y_start + 10 < y0 < y_end
        and ((x0 + _x1) / 2) > width * 0.45
        and any(marker in norm for marker in [
            "flux de tresorerie",
            "tableau des flux de tresorerie",
            "variation des capitaux propres",
            "etat du resultat net",
            "resultat net et gains",
            "perimetre de consolidation",
            "tableau de financement",
            "attestation",
        ])
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
