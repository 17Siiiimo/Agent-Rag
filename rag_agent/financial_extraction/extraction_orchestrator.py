from __future__ import annotations

import shutil
from time import perf_counter
from datetime import datetime
from pathlib import Path
from typing import Any, Set

import fitz  # PyMuPDF

from .document_classifier import detect_sector
from .hybrid_retriever import filter_retrieved_pages_for_requested_scope, retrieve_candidate_pages
from .keyword_dictionary import SCOPES, SECTORS, TARGET_TABLES
from .models import PipelineConfig, TableCandidate
from .page_indexer import build_page_chunks, build_page_index, load_or_build_page_index, load_page_chunks_cache
from .page_renderer import render_pdf_page
from .page_text_extractor import extract_page_texts
from .pdf_loader import load_pdf
from .rag_evidence import attach_and_write_rag_evidence, write_summary_rag_evidence
from .table_cropper import crop_table_image
from .table_localizer import localize_table_candidates
from .table_validator import validate_extracted_table
from .utils import ensure_dir, slugify, write_json
from .vision_extractor import extract_table_with_vision


def extract_financial_table(
    pdf_path: str,
    target_table: str,
    scope: str,
    sector: str,
    company: str,
    year: int,
    config: PipelineConfig | None = None,
    debug_dir_override: str | Path | None = None,
) -> dict[str, Any]:
    cfg = config or PipelineConfig()
    target_table = _validate_choice("target_table", target_table, TARGET_TABLES)
    scope = _validate_choice("scope", scope, SCOPES)
    sector = _validate_choice("sector", sector, SECTORS)

    pdf = load_pdf(pdf_path)
    debug_dir = ensure_dir(
        debug_dir_override
        if debug_dir_override is not None
        else cfg.output_dir / slugify(company) / str(year) / pdf.pdf_id / f"{scope}_{target_table}"
    )
    rendered_dir = debug_dir / "rendered_pages"
    crop_dir = ensure_dir(debug_dir / "crops")

    page_texts = extract_page_texts(pdf.pdf_path)
    detected_sector = detect_sector(page_texts, sector)
    pages = build_page_chunks(
        pdf_id=pdf.pdf_id,
        company=company,
        year=year,
        sector=detected_sector,
        page_texts=page_texts,
        image_paths={},
    )
    index = build_page_index(pages, cfg, debug_dir=debug_dir)

    retrieval_started = perf_counter()
    retrieved = retrieve_candidate_pages(index, target_table, scope, detected_sector, cfg)
    retrieval_latency_ms = int((perf_counter() - retrieval_started) * 1000)
    retrieval_payload = [r.to_dict() for r in retrieved]
    write_json(debug_dir / "retrieval_candidates.json", retrieval_payload)
    write_json(debug_dir / "score_breakdown.json", retrieval_payload)

    candidates = localize_table_candidates(
        pdf.pdf_path,
        retrieved,
        target_table,
        scope,
        detected_sector,
        max_pages=cfg.candidate_pages_for_localization,
        debug_dir=debug_dir,
    )
    selected = candidates[0] if candidates else None
    if selected is None:
        result = _empty_result(pdf_path, company, year, target_table, scope, detected_sector)
        result["warnings"].append("no_table_candidate_found")
        write_json(debug_dir / "final_extracted.json", result)
        return result

    selected_retrieved = next((r for r in retrieved if r.page.page_number == selected.page_number), None)
    write_json(
        debug_dir / "selected_page.json",
        {
            "page_number": selected.page_number,
            "target_table": target_table,
            "scope": scope,
            "sector": detected_sector,
            "retrieval": selected_retrieved.to_dict() if selected_retrieved else None,
            "table_candidate": selected.to_dict(),
        },
    )

    page_size = _page_size(pdf.pdf_path, selected.page_number)
    image_paths: dict[int, str] = {}
    selected_image_path = _ensure_rendered_page_image(
        pdf.pdf_path,
        rendered_dir=rendered_dir,
        image_paths=image_paths,
        page_number=selected.page_number,
        dpi=cfg.dpi,
    )
    crop_path = crop_table_image(selected, selected_image_path, crop_dir, page_size)
    stable_crop_path = str(debug_dir / "crop.png")
    shutil.copy2(crop_path, stable_crop_path)
    write_json(
        debug_dir / "selected_crop_validation.json",
        {
            "crop_path": stable_crop_path,
            "page_number": selected.page_number,
            "bbox": selected.bbox,
            "candidate_confidence": selected.confidence,
            "candidate_evidence": selected.evidence,
            "vision_called": bool(cfg.use_vision),
        },
    )

    if cfg.use_vision:
        extracted = extract_table_with_vision(
            stable_crop_path,
            selected,
            company,
            year,
            provider=cfg.vision_provider,
            model=cfg.vision_model,
            debug_dir=debug_dir,
        )
    else:
        extracted = {
            "target_found": False,
            "table_type": target_table,
            "title_detected": "",
            "columns": [],
            "rows": [],
            "confidence": selected.confidence,
            "warnings": ["vision_disabled"],
        }
        write_json(debug_dir / "extracted_table.json", extracted)

    validation = validate_extracted_table(
        extracted,
        selected,
        page_texts.get(selected.page_number, ""),
        cfg,
        debug_path=str(debug_dir / "validation_report.json"),
    )

    output = {
        "pdf_path": pdf_path,
        "company": company,
        "year": int(year),
        "target_table": target_table,
        "scope": scope,
        "sector": detected_sector,
        "selected_page": selected.page_number,
        "crop_path": stable_crop_path,
        "bbox": selected.bbox,
        "target_found": bool(extracted.get("target_found")),
        "columns": extracted.get("columns", []),
        "rows": extracted.get("rows", []),
        "confidence": float(extracted.get("confidence") or selected.confidence),
        "warnings": _result_warnings(extracted),
        "validation": validation,
        "debug": {
            "dir": str(debug_dir),
            "crop_image": stable_crop_path,
            "candidate_confidence": selected.confidence,
            "candidate_evidence": selected.evidence,
        },
        "benchmark_retrieval": _benchmark_retrieval_payload(
            retrieved=retrieved,
            selected_retrieved=selected_retrieved,
            selected_page=selected.page_number,
            retrieval_latency_ms=retrieval_latency_ms,
        ),
    }
    attach_and_write_rag_evidence(output, debug_dir, page_text=page_texts.get(selected.page_number, ""))
    output["rendered_pages_deleted"] = _cleanup_rendered_pages(rendered_dir, enabled=cfg.cleanup_rendered_pages)
    write_json(debug_dir / "final_extracted.json", output)
    return output


def extract_financial_tables(
    payload: dict[str, Any] | None = None,
    *,
    pdf_path: str | None = None,
    company: str | None = None,
    year: int | None = None,
    report_type: str | None = None,
    scope: str | None = None,
    sector: str | None = None,
    target_tables: list[str] | tuple[str, ...] | None = None,
    provider: str | None = None,
    output_dir: str | Path | None = None,
    model: str | None = None,
    force_vision: bool | None = None,
    force_page: bool | None = None,
    force_recrop: bool | None = None,
) -> dict[str, Any]:
    data = dict(payload or {})
    pdf_path = pdf_path or data.get("pdf_path")
    company = company or data.get("company")
    year = int(year or data.get("year"))
    report_type = _normalize_report_type(report_type or data.get("report_type") or "rapport_annuel")
    scope = scope or data.get("scope")
    sector = sector or data.get("sector")
    provider = (provider or data.get("provider") or "groq").lower().strip()
    force_vision = bool(force_vision if force_vision is not None else data.get("force_vision") or data.get("forceVision"))
    force_page = bool(force_page if force_page is not None else data.get("force_page") or data.get("forcePage") or data.get("rerun_page") or data.get("rerunPage"))
    force_recrop = bool(
        force_recrop
        if force_recrop is not None
        else data.get("force_recrop") or data.get("forceRecrop") or data.get("rerun_crop") or data.get("rerunCrop")
    )
    if force_page:
        force_recrop = False
    target_tables = list(target_tables or data.get("target_tables") or [])
    if not target_tables:
        raise ValueError("target_tables is required")
    if not pdf_path or not company or not year or not scope or not sector:
        raise ValueError("pdf_path, company, year, scope and sector are required")

    base_output = ensure_dir(output_dir or Path("output") / "financial_extraction_debug" / f"{slugify(company)}_{year}_{report_type}_vision")
    cfg = PipelineConfig(
        output_dir=base_output,
        vision_provider=provider,
        vision_model=model or data.get("model"),
        use_vision=True,
    )

    pdf = load_pdf(str(pdf_path))
    page_chunks_path = base_output / "page_chunks.json"
    rendered_dir = base_output / "_rendered_pages"
    if not force_vision and not force_page and not force_recrop:
        cached_summary = _load_complete_cached_summary(
            base_output=base_output,
            pdf_path=pdf.pdf_path,
            company=str(company),
            year=int(year),
            report_type=report_type,
            scope=str(scope),
            sector=str(sector),
            provider=provider,
            model=cfg.vision_model,
            target_tables=[str(target).strip() for target in target_tables],
            page_chunks_path=page_chunks_path,
            rendered_dir=rendered_dir,
        )
        if cached_summary is not None:
            write_summary_rag_evidence(cached_summary, base_output)
            write_json(base_output / "summary.json", cached_summary)
            return cached_summary

    page_texts = extract_page_texts(pdf.pdf_path)
    detected_sector = detect_sector(page_texts, str(sector))
    pages = load_page_chunks_cache(
        page_chunks_path,
        pdf_id=pdf.pdf_id,
        company=str(company),
        year=int(year),
        sector=detected_sector,
    )
    cache_used = pages is not None
    rendered_pages_skipped = False
    if pages is not None:
        image_paths = {page.page_number: page.image_path for page in pages if page.image_path and Path(page.image_path).is_file()}
        rendered_pages_skipped = True
    else:
        image_paths = {}
    if pages is None:
        pages = build_page_chunks(
            pdf_id=pdf.pdf_id,
            company=str(company),
            year=int(year),
            sector=detected_sector,
            page_texts=page_texts,
            image_paths={},
        )
        write_json(page_chunks_path, [p.to_dict() for p in pages])
        rendered_pages_skipped = True
    index, index_cache_used = load_or_build_page_index(pages, cfg, base_output)

    results: list[dict[str, Any]] = []
    for target_table in target_tables:
        target = str(target_table).strip()
        target_dir = ensure_dir(base_output / str(scope) / target)
        try:
            result = _extract_prepared_financial_table(
                pdf_path=pdf.pdf_path,
                pdf_id=pdf.pdf_id,
                target_table=target,
                scope=str(scope),
                sector=detected_sector,
                company=str(company),
                year=int(year),
                cfg=cfg,
                debug_dir=target_dir,
                page_texts=page_texts,
                image_paths=image_paths,
                pages=pages,
                index=index,
                force_vision=force_vision,
                force_page=force_page,
                force_recrop=force_recrop,
            )
        except Exception as exc:
            result = {
                "pdf_path": str(pdf_path),
                "company": str(company),
                "year": int(year),
                "report_type": report_type,
                "target_table": target,
                "scope": str(scope),
                "sector": str(sector),
                "selected_page": None,
                "crop_path": "",
                "bbox": [],
                "target_found": False,
                "columns": [],
                "rows": [],
                "confidence": 0.0,
                "warnings": [],
                "error": str(exc),
                "validation": {"status": "rejected", "issues": ["pipeline_failed"], "warnings": [str(exc)]},
            }
            write_json(target_dir / "extracted_table.json", result)
            write_json(target_dir / "validation_report.json", result["validation"])
        result["report_type"] = report_type
        results.append(result)

    summary = {
        "pdf_path": str(pdf_path),
        "company": str(company),
        "year": int(year),
        "report_type": report_type,
        "scope": str(scope),
        "sector": str(sector),
        "provider": provider,
        "force_vision": force_vision,
        "force_page": force_page,
        "force_recrop": force_recrop,
        "page_chunks_cache_used": cache_used,
        "index_cache_used": index_cache_used,
        "page_chunks_path": str(page_chunks_path),
        "embeddings_path": str(base_output / "page_embeddings.npy"),
        "faiss_index_path": str(base_output / "faiss.index"),
        "index_metadata_path": str(base_output / "index_metadata.json"),
        "rendered_pages_dir": str(rendered_dir),
        "rendered_pages_skipped": rendered_pages_skipped,
        "rendered_pages_deleted": False,
        "results": results,
    }
    write_summary_rag_evidence(summary, base_output)
    summary["rendered_pages_deleted"] = _cleanup_rendered_pages(
        rendered_dir,
        enabled=cfg.cleanup_rendered_pages
        and not force_page
        and not force_recrop
        and _should_cleanup_rendered_pages_after_run(base_output),
    )
    write_json(base_output / "summary.json", summary)
    return summary


def _read_previous_selected_page(debug_dir: Path) -> int | None:
    """Last localized page from a prior run (used when ``force_page`` asks for a different page)."""
    payload = _read_json(debug_dir / "selected_page.json")
    if not isinstance(payload, dict):
        return None
    raw = payload.get("page_number")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _read_selected_retrieval(debug_dir: Path) -> dict[str, Any] | None:
    payload = _read_json(debug_dir / "selected_page.json")
    if not isinstance(payload, dict):
        return None
    retrieval = payload.get("retrieval")
    return retrieval if isinstance(retrieval, dict) else None


def _read_cached_retrieval_latency(debug_dir: Path) -> int | None:
    final_payload = _read_json(debug_dir / "final_extracted.json")
    if not isinstance(final_payload, dict):
        return None
    benchmark = final_payload.get("benchmark_retrieval")
    if not isinstance(benchmark, dict):
        return None
    latency = benchmark.get("retrieval_latency_ms")
    if latency is None:
        return None
    try:
        return int(latency)
    except (TypeError, ValueError):
        return None


def _extract_prepared_financial_table(
    *,
    pdf_path: str,
    pdf_id: str,
    target_table: str,
    scope: str,
    sector: str,
    company: str,
    year: int,
    cfg: PipelineConfig,
    debug_dir: Path,
    page_texts: dict[int, str],
    image_paths: dict[int, str],
    pages,
    index,
    force_vision: bool = False,
    force_page: bool = False,
    force_recrop: bool = False,
) -> dict[str, Any]:
    """Run vision extraction for one table under ``debug_dir``.

    Semantics (UI):

    - ``force_page=True``: new **page** — full retrieval + scoring + localization, but the
      **previous** selected page is excluded so another page can win; then new crop + LLM.

    - ``force_recrop=True`` (and ``force_page=False``): same **page** as retrieval would
      normally pick (no exclusion) — invalidate the cached crop, re-run localization +
      ``crop.png``, then LLM. Use when the page is right but the crop is wrong.

    - ``force_vision=True`` (no page/recrop flags): keep ``crop.png``; LLM only.

    If ``force_page`` is true, ``force_recrop`` is ignored (caller should clear it).
    """
    target_table = _validate_choice("target_table", target_table, TARGET_TABLES)
    scope = _validate_choice("scope", scope, SCOPES)
    sector = _validate_choice("sector", sector, SECTORS)

    if not force_vision and not force_page and not force_recrop:
        cached_result = _load_valid_vision_result(
            debug_dir,
            provider=cfg.vision_provider,
            model=cfg.vision_model,
            target_table=target_table,
            scope=scope,
            sector=sector,
            allow_provider_mismatch=True,
        )
        if cached_result is not None:
            cached_result.setdefault("cache", {})
            cached_result["cache"].update(
                {
                    "vision_result_cache_used": True,
                    "crop_cache_used": True,
                    "llm_call_skipped": True,
                    "force_vision": False,
                    "force_page": False,
                    "force_recrop": False,
                    "requested_provider": cfg.vision_provider,
                    "requested_model": cfg.vision_model,
                }
            )
            selected_page = _optional_int(cached_result.get("selected_page"))
            attach_and_write_rag_evidence(
                cached_result,
                debug_dir,
                page_text=page_texts.get(selected_page, "") if selected_page else "",
            )
            return cached_result

    crop_cache = None if (force_page or force_recrop) else _load_crop_cache(
        debug_dir, target_table=target_table, scope=scope, sector=sector
    )
    if force_vision and not force_page and not force_recrop and crop_cache is None:
        msg = (
            "Réextraction LLM seule impossible: crop.png ou métadonnées (crop_metadata) "
            "absents ou invalides. Lancez une extraction complète ou utilisez "
            "« Nouvelle page » ou « Nouveau crop » pour recalculer le crop."
        )
        result = _empty_result(pdf_path, company, year, target_table, scope, sector)
        result["warnings"] = ["force_vision_requires_existing_crop"]
        result["error"] = msg
        result["validation"] = {
            "status": "rejected",
            "issues": ["force_vision_no_crop"],
            "warnings": [msg],
        }
        result["debug"] = {"dir": str(debug_dir), "pdf_id": pdf_id}
        result["cache"] = {
            "crop_cache_used": False,
            "vision_result_cache_used": False,
            "force_vision": True,
            "force_page": False,
            "force_recrop": False,
            "llm_call_skipped": True,
        }
        write_json(debug_dir / "final_extracted.json", result)
        write_json(debug_dir / "extracted_table.json", result)
        write_json(debug_dir / "validation_report.json", result["validation"])
        return result

    if crop_cache is not None:
        selected = crop_cache["candidate"]
        stable_crop_path = crop_cache["crop_path"]
        selected_retrieved = _read_selected_retrieval(debug_dir)
        crop_cache_used = True
        retrieval_payload = _read_json(debug_dir / "retrieval_candidates.json")
        retrieval_latency_ms = _read_cached_retrieval_latency(debug_dir)
    else:
        selected = None
        stable_crop_path = ""
        selected_retrieved = None
        crop_cache_used = False
        retrieval_payload = []
        retrieval_latency_ms = None

    if selected is None:
        exclude_pages: Set[int] = set()
        if force_page:
            prev_page = _read_previous_selected_page(debug_dir)
            if prev_page is not None:
                exclude_pages.add(prev_page)
        retrieval_started = perf_counter()
        retrieved = retrieve_candidate_pages(
            index,
            target_table,
            scope,
            sector,
            cfg,
            exclude_page_numbers=exclude_pages if exclude_pages else None,
        )
        if exclude_pages:
            retrieved = filter_retrieved_pages_for_requested_scope(retrieved, scope)
        retrieval_latency_ms = int((perf_counter() - retrieval_started) * 1000)
        retrieval_payload = [r.to_dict() for r in retrieved]
        write_json(debug_dir / "retrieval_candidates.json", retrieval_payload)
        write_json(debug_dir / "score_breakdown.json", retrieval_payload)

        candidates = localize_table_candidates(
            pdf_path,
            retrieved,
            target_table,
            scope,
            sector,
            max_pages=cfg.candidate_pages_for_localization,
            debug_dir=debug_dir,
        )
        selected = candidates[0] if candidates else None
        if selected is None:
            result = _empty_result(pdf_path, company, year, target_table, scope, sector)
            result["warnings"].append("no_table_candidate_found")
            write_json(debug_dir / "final_extracted.json", result)
            return result

        selected_retrieved = next((r for r in retrieved if r.page.page_number == selected.page_number), None)
        write_json(
            debug_dir / "selected_page.json",
            {
                "page_number": selected.page_number,
                "target_table": target_table,
                "scope": scope,
                "sector": sector,
                "retrieval": selected_retrieved.to_dict() if selected_retrieved else None,
                "table_candidate": selected.to_dict(),
            },
        )

        page_size = _page_size(pdf_path, selected.page_number)
        selected_image_path = _ensure_rendered_page_image(
            pdf_path,
            rendered_dir=cfg.output_dir / "_rendered_pages",
            image_paths=image_paths,
            page_number=selected.page_number,
            dpi=cfg.dpi,
        )
        generated_crop = crop_table_image(selected, selected_image_path, debug_dir / "crops", page_size)
        stable_crop_path = str(debug_dir / "crop.png")
        continuation = _find_cpc_continuation_candidate(selected, candidates, page_texts)
        if continuation is not None:
            continuation_image_path = _ensure_rendered_page_image(
                pdf_path,
                rendered_dir=cfg.output_dir / "_rendered_pages",
                image_paths=image_paths,
                page_number=continuation.page_number,
                dpi=cfg.dpi,
            )
            continuation_crop = crop_table_image(
                continuation,
                continuation_image_path,
                debug_dir / "crops",
                _page_size(pdf_path, continuation.page_number),
            )
            _stack_crop_images([generated_crop, continuation_crop], stable_crop_path)
            selected.evidence.append(f"combined_cpc_continuation_page:{continuation.page_number}")
        else:
            shutil.copy2(generated_crop, stable_crop_path)
        _write_crop_metadata(debug_dir, stable_crop_path, selected, bool(cfg.use_vision), crop_cache_used=False)

    if force_vision or force_page or force_recrop:
        if force_page and force_vision:
            _reason = "force_page_and_vision"
        elif force_page:
            _reason = "force_page"
        elif force_recrop and force_vision:
            _reason = "force_recrop_and_vision"
        elif force_recrop:
            _reason = "force_recrop"
        else:
            _reason = "force_vision"
        _backup_previous_vision_artifacts(
            debug_dir,
            provider=cfg.vision_provider,
            model=cfg.vision_model,
            reason=_reason,
        )

    extracted = extract_table_with_vision(
        stable_crop_path,
        selected,
        company,
        year,
        provider=cfg.vision_provider,
        model=cfg.vision_model,
        debug_dir=debug_dir,
    )

    validation = validate_extracted_table(
        extracted,
        selected,
        page_texts.get(selected.page_number, ""),
        cfg,
        debug_path=str(debug_dir / "validation_report.json"),
    )

    output = {
        "pdf_path": pdf_path,
        "company": company,
        "year": int(year),
        "target_table": target_table,
        "scope": scope,
        "sector": sector,
        "selected_page": selected.page_number,
        "crop_path": stable_crop_path,
        "bbox": selected.bbox,
        "target_found": bool(extracted.get("target_found")),
        "columns": extracted.get("columns", []),
        "rows": extracted.get("rows", []),
        "confidence": float(extracted.get("confidence") or selected.confidence),
        "warnings": _result_warnings(extracted),
        "validation": validation,
        "debug": {
            "dir": str(debug_dir),
            "crop_image": stable_crop_path,
            "candidate_confidence": selected.confidence,
            "candidate_evidence": selected.evidence,
            "pdf_id": pdf_id,
        },
        "benchmark_retrieval": _benchmark_retrieval_payload(
            retrieved=retrieval_payload,
            selected_retrieved=selected_retrieved,
            selected_page=selected.page_number,
            retrieval_latency_ms=retrieval_latency_ms,
        ),
        "cache": {
            "crop_cache_used": crop_cache_used,
            "vision_result_cache_used": False,
            "provider": cfg.vision_provider,
            "model": cfg.vision_model,
            "force_vision": force_vision,
            "force_page": force_page,
            "force_recrop": force_recrop,
            "llm_call_skipped": False,
        },
    }
    attach_and_write_rag_evidence(output, debug_dir, page_text=page_texts.get(selected.page_number, ""))
    write_json(debug_dir / "final_extracted.json", output)
    return output


def _validate_choice(name: str, value: str, allowed: tuple[str, ...]) -> str:
    normalized = str(value or "").strip()
    if normalized not in allowed:
        raise ValueError(f"{name} invalide: {value!r}. Valeurs: {', '.join(allowed)}")
    return normalized


def _normalize_report_type(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"s1", "rfs", "rapport_semestre", "rapport_semestriel"} or "semestre" in normalized:
        return "s1"
    return "rapport_annuel"


def _load_complete_cached_summary(
    *,
    base_output: Path,
    pdf_path: str,
    company: str,
    year: int,
    report_type: str,
    scope: str,
    sector: str,
    provider: str,
    model: str | None,
    target_tables: list[str],
    page_chunks_path: Path,
    rendered_dir: Path,
) -> dict[str, Any] | None:
    results: list[dict[str, Any]] = []
    for target_table in target_tables:
        try:
            target = _validate_choice("target_table", target_table, TARGET_TABLES)
        except ValueError:
            return None
        debug_dir = base_output / scope / target
        cached = _load_completed_extraction_artifacts(
            debug_dir,
            target_table=target,
            scope=scope,
            sector=sector,
        )
        if cached is None:
            return None
        cached["pdf_path"] = cached.get("pdf_path") or pdf_path
        cached["company"] = cached.get("company") or company
        cached["year"] = int(cached.get("year") or year)
        cached["report_type"] = report_type
        cached.setdefault("cache", {})
        cached["cache"].update(
            {
                "vision_result_cache_used": True,
                "crop_cache_used": True,
                "llm_call_skipped": True,
                "rendering_skipped": True,
                "force_vision": False,
                "requested_provider": provider,
                "requested_model": model,
            }
        )
        results.append(cached)

    rendered_pages_deleted = _cleanup_rendered_pages(
        rendered_dir,
        enabled=_should_cleanup_rendered_pages_after_run(base_output),
    )
    return {
        "pdf_path": pdf_path,
        "company": company,
        "year": int(year),
        "report_type": report_type,
        "scope": scope,
        "sector": sector,
        "provider": provider,
        "force_vision": False,
        "page_chunks_cache_used": page_chunks_path.is_file(),
        "index_cache_used": (base_output / "faiss.index").is_file() and (base_output / "index_metadata.json").is_file(),
        "page_chunks_path": str(page_chunks_path),
        "embeddings_path": str(base_output / "page_embeddings.npy"),
        "faiss_index_path": str(base_output / "faiss.index"),
        "index_metadata_path": str(base_output / "index_metadata.json"),
        "rendered_pages_dir": str(rendered_dir),
        "rendered_pages_skipped": True,
        "rendered_pages_deleted": rendered_pages_deleted,
        "results": results,
    }


def _load_completed_extraction_artifacts(
    debug_dir: Path,
    *,
    target_table: str,
    scope: str,
    sector: str | None = None,
) -> dict[str, Any] | None:
    final_payload = _read_json(debug_dir / "final_extracted.json")
    if not isinstance(final_payload, dict):
        return None
    if final_payload.get("target_table") != target_table:
        return None
    if final_payload.get("scope") != scope:
        return None
    if sector is not None and final_payload.get("sector") != sector:
        return None
    if not final_payload.get("target_found"):
        return None
    if not final_payload.get("columns") or not final_payload.get("rows"):
        return None

    required_files = [
        debug_dir / "crop.png",
        debug_dir / "extracted_table.json",
        debug_dir / "extracted_table.md",
        debug_dir / "final_extracted.json",
        debug_dir / "raw_llm_response.json",
        debug_dir / "validation_report.json",
    ]
    if any(not path.is_file() or path.stat().st_size == 0 for path in required_files):
        return None

    validation = final_payload.get("validation") or _read_json(debug_dir / "validation_report.json") or {}
    if isinstance(validation, dict) and validation.get("status") == "rejected":
        return None

    crop_path = final_payload.get("crop_path") or str(debug_dir / "crop.png")
    if not Path(crop_path).is_file():
        crop_path = str(debug_dir / "crop.png")
    final_payload["crop_path"] = crop_path
    evidence_payload = _read_json(debug_dir / "rag_evidence_chunks.json")
    if isinstance(evidence_payload, list):
        final_payload["rag_evidence_chunks"] = evidence_payload
        final_payload["rag_evidence_path"] = str(debug_dir / "rag_evidence_chunks.json")
    final_payload.setdefault(
        "benchmark_retrieval",
        _benchmark_retrieval_payload(
            retrieved=_read_json(debug_dir / "retrieval_candidates.json") or [],
            selected_retrieved=_read_selected_retrieval(debug_dir),
            selected_page=final_payload.get("selected_page"),
            retrieval_latency_ms=_read_cached_retrieval_latency(debug_dir),
        ),
    )
    return final_payload


def _result_warnings(extracted: dict[str, Any]) -> list[str]:
    warnings = list(extracted.get("warnings") or [])
    if extracted.get("error"):
        warnings.append(str(extracted["error"]))
    if extracted.get("target_found") is False:
        warnings.append("target_found_false")
    return warnings


def _load_valid_vision_result(
    debug_dir: Path,
    *,
    provider: str,
    model: str | None,
    target_table: str | None = None,
    scope: str | None = None,
    sector: str | None = None,
    allow_provider_mismatch: bool = False,
) -> dict[str, Any] | None:
    payload = _read_json(debug_dir / "final_extracted.json")
    if not payload:
        payload = _rebuild_cached_result_from_artifacts(
            debug_dir,
            target_table=target_table,
            scope=scope,
            sector=sector,
            provider=provider,
            model=model,
        )
    if not payload:
        return None
    if target_table is not None and payload.get("target_table") != target_table:
        return None
    if scope is not None and payload.get("scope") != scope:
        return None
    if sector is not None and payload.get("sector") != sector:
        return None
    cache = payload.get("cache") or {}
    if not allow_provider_mismatch and (cache.get("provider") != provider or cache.get("model") != model):
        return None
    if not payload.get("target_found") or not payload.get("columns") or not payload.get("rows"):
        return None
    validation = payload.get("validation") or {}
    if validation.get("status") == "rejected":
        return None
    crop_path = payload.get("crop_path")
    if crop_path and not Path(crop_path).is_file():
        return None
    if _cached_candidate_has_alternate_statement_context(debug_dir):
        return None
    if _cached_candidate_has_wrong_balance_context(debug_dir, target_table=target_table):
        return None
    if _cached_candidate_has_bad_line_header_context(debug_dir, target_table=target_table):
        return None
    if _cached_candidate_has_management_summary_context(debug_dir, target_table=target_table):
        return None
    if _cached_candidate_has_wrong_scope_context(debug_dir, scope=scope):
        return None
    if _is_suspicious_cached_result(payload):
        return None
    return payload


def _rebuild_cached_result_from_artifacts(
    debug_dir: Path,
    *,
    target_table: str | None,
    scope: str | None,
    sector: str | None,
    provider: str,
    model: str | None,
) -> dict[str, Any] | None:
    extracted = _read_json(debug_dir / "extracted_table.json")
    metadata = _read_json(debug_dir / "crop_metadata.json") or _read_json(debug_dir / "selected_crop_validation.json")
    validation = _read_json(debug_dir / "validation_report.json") or {}
    if not isinstance(extracted, dict) or not isinstance(metadata, dict):
        return None
    if not extracted.get("target_found") or not extracted.get("columns") or not extracted.get("rows"):
        return None
    if validation.get("status") == "rejected":
        return None
    if target_table is not None and metadata.get("target_table") != target_table:
        return None
    if scope is not None and metadata.get("scope") != scope:
        return None
    if sector is not None and metadata.get("sector") != sector:
        return None
    crop_path = metadata.get("crop_path")
    if crop_path and not Path(crop_path).is_file():
        return None
    return {
        "pdf_path": "",
        "company": "",
        "year": 0,
        "target_table": metadata.get("target_table") or target_table,
        "scope": metadata.get("scope") or scope,
        "sector": metadata.get("sector") or sector,
        "selected_page": metadata.get("page_number"),
        "crop_path": crop_path or str(debug_dir / "crop.png"),
        "bbox": metadata.get("bbox") or [],
        "target_found": True,
        "columns": extracted.get("columns", []),
        "rows": extracted.get("rows", []),
        "confidence": float(extracted.get("confidence") or metadata.get("confidence") or 0.0),
        "warnings": extracted.get("warnings", []),
        "validation": validation or {"status": "approved", "issues": [], "warnings": []},
        "debug": {
            "dir": str(debug_dir),
            "crop_image": crop_path or str(debug_dir / "crop.png"),
            "candidate_confidence": metadata.get("candidate_confidence", metadata.get("confidence", 0.0)),
            "candidate_evidence": metadata.get("candidate_evidence", []),
            "pdf_id": metadata.get("pdf_id", ""),
        },
        "cache": {
            "crop_cache_used": True,
            "vision_result_cache_used": True,
            "llm_call_skipped": True,
            "provider": provider,
            "model": model,
            "rebuilt_from_artifacts": True,
        },
    }


def _is_suspicious_cached_result(payload: dict[str, Any]) -> bool:
    target_table = str(payload.get("target_table") or "")
    if target_table not in {"BILAN_ACTIF", "BILAN_PASSIF"}:
        return False
    bbox = payload.get("bbox") or []
    if len(bbox) != 4:
        return True
    region_height = float(bbox[3]) - float(bbox[1])
    if region_height < 130.0:
        return True
    if target_table == "BILAN_PASSIF" and payload.get("sector") == "bancaire_sdf" and region_height < 220.0:
        return True
    evidence = " ".join(str(item).lower() for item in (payload.get("debug") or {}).get("candidate_evidence") or [])
    if any(
        marker in evidence
        for marker in [
            "penalty:note_like_balance_region",
            "penalty:small_balance_region",
            "penalty:alternate_statement_context",
        ]
    ):
        return True
    return False


def _load_crop_cache(debug_dir: Path, *, target_table: str, scope: str, sector: str) -> dict[str, Any] | None:
    previous_result = _read_json(debug_dir / "final_extracted.json")
    if isinstance(previous_result, dict) and previous_result.get("target_found") is False:
        return None
    metadata = _read_json(debug_dir / "crop_metadata.json") or _read_json(debug_dir / "selected_crop_validation.json")
    if not metadata:
        return None
    crop_path = metadata.get("crop_path")
    if not crop_path or not Path(crop_path).is_file():
        return None
    candidate_payload = metadata.get("table_candidate") or metadata.get("selected_candidate")
    if not candidate_payload:
        candidate_payload = {
            "pdf_id": metadata.get("pdf_id", ""),
            "page_number": metadata.get("page_number"),
            "scope": metadata.get("scope", scope),
            "sector": metadata.get("sector", sector),
            "table_type": metadata.get("target_table", target_table),
            "bbox": metadata.get("bbox", []),
            "confidence": metadata.get("candidate_confidence", metadata.get("confidence", 0.0)),
            "evidence": metadata.get("candidate_evidence", []),
        }
    if candidate_payload.get("table_type") != target_table:
        return None
    if candidate_payload.get("scope") not in {scope, None, ""}:
        return None
    if candidate_payload.get("sector") not in {sector, None, ""}:
        return None
    if not candidate_payload.get("bbox") or not candidate_payload.get("page_number"):
        return None
    candidate = TableCandidate(
        pdf_id=str(candidate_payload.get("pdf_id", "")),
        page_number=int(candidate_payload["page_number"]),
        scope=str(candidate_payload.get("scope") or scope),
        sector=str(candidate_payload.get("sector") or sector),
        table_type=str(candidate_payload.get("table_type") or target_table),
        bbox=[float(v) for v in candidate_payload.get("bbox", [])],
        confidence=float(candidate_payload.get("confidence") or 0.0),
        evidence=list(candidate_payload.get("evidence") or []),
    )
    if _is_suspicious_crop_cache(candidate):
        return None
    if _cached_candidate_has_alternate_statement_context(debug_dir):
        return None
    if _cached_candidate_has_wrong_balance_context(debug_dir, target_table=target_table):
        return None
    if _cached_candidate_has_bad_line_header_context(debug_dir, target_table=target_table):
        return None
    if _cached_candidate_has_management_summary_context(debug_dir, target_table=target_table):
        return None
    if _cached_candidate_has_wrong_scope_context(debug_dir, scope=scope):
        return None
    return {"crop_path": str(crop_path), "candidate": candidate}


def _is_suspicious_crop_cache(candidate: TableCandidate) -> bool:
    if candidate.table_type not in {"BILAN_ACTIF", "BILAN_PASSIF"}:
        return False
    if len(candidate.bbox) != 4:
        return True
    region_height = float(candidate.bbox[3]) - float(candidate.bbox[1])
    if region_height < 130.0:
        return True
    if candidate.table_type == "BILAN_PASSIF" and candidate.sector == "bancaire_sdf" and region_height < 220.0:
        return True
    evidence = " ".join(str(item).lower() for item in candidate.evidence)
    return any(
        marker in evidence
        for marker in [
            "penalty:note_like_balance_region",
            "penalty:small_balance_region",
            "penalty:alternate_statement_context",
        ]
    )


def _cached_candidate_has_alternate_statement_context(debug_dir: Path) -> bool:
    selected = _read_json(debug_dir / "selected_page.json")
    if not isinstance(selected, dict):
        return False
    retrieval = selected.get("retrieval") or {}
    page = retrieval.get("page") or {}
    text = str(page.get("page_text") or "").lower()
    if not text:
        return False
    return any(
        marker in text
        for marker in [
            "bilan arreda",
            "compte de produits et charges arreda",
            "etats de synthese arreda",
        ]
    )


def _cached_candidate_has_wrong_balance_context(debug_dir: Path, *, target_table: str | None) -> bool:
    if target_table not in {"BILAN_ACTIF", "BILAN_PASSIF"}:
        return False
    selected = _read_json(debug_dir / "selected_page.json")
    if not isinstance(selected, dict):
        return False
    retrieval = selected.get("retrieval") or {}
    page = retrieval.get("page") or {}
    text = str(page.get("page_text") or "")
    if not text:
        return False
    from .utils import normalize_text

    norm = normalize_text(text)
    has_total = (
        ("total actif" in norm or "total de l actif" in norm)
        if target_table == "BILAN_ACTIF"
        else ("total passif" in norm or "total du passif" in norm)
    )
    if has_total:
        return False
    return any(
        marker in norm
        for marker in [
            "tableau des flux de tresorerie",
            "flux de tresorerie nets",
            "capacite d autofinancement",
            "etat des derogations",
            "etat des changements de methodes",
            "detail des postes",
            "details des postes",
        ]
    )


def _cached_candidate_has_bad_line_header_context(debug_dir: Path, *, target_table: str | None) -> bool:
    if target_table != "BILAN_PASSIF":
        return False
    metadata = _read_json(debug_dir / "crop_metadata.json") or _read_json(debug_dir / "selected_crop_validation.json")
    if not isinstance(metadata, dict):
        return False
    sector = str(metadata.get("sector") or "")
    if sector != "bancaire_sdf":
        return False
    bbox = metadata.get("bbox") or []
    if len(bbox) != 4:
        return True
    evidence = " ".join(str(item).lower() for item in metadata.get("candidate_evidence") or [])
    if "line_header" not in evidence:
        return False
    if float(bbox[1]) >= 260.0:
        return False

    selected = _read_json(debug_dir / "selected_page.json")
    retrieval = selected.get("retrieval") if isinstance(selected, dict) else {}
    page = (retrieval or {}).get("page") or {}
    text = str(page.get("page_text") or "")
    if not text:
        return False
    from .utils import normalize_text

    norm = normalize_text(text)
    has_false_header_context = any(
        marker in norm
        for marker in [
            "passif financier",
            "passifs financiers",
            "provisions du passif",
            "passifs d impots",
            "passifs d'impots",
        ]
    )
    has_real_passif_table = "total passif" in norm or "total du passif" in norm
    return has_false_header_context and has_real_passif_table


def _cached_candidate_has_management_summary_context(debug_dir: Path, *, target_table: str | None) -> bool:
    if target_table not in {"BILAN_ACTIF", "BILAN_PASSIF"}:
        return False
    selected = _read_json(debug_dir / "selected_page.json")
    if not isinstance(selected, dict):
        return False
    retrieval = selected.get("retrieval") or {}
    page = retrieval.get("page") or {}
    text = str(page.get("page_text") or "")
    if not text:
        return False
    from .utils import normalize_text

    norm = normalize_text(text)
    is_summary = any(
        marker in norm
        for marker in [
            "l actif et ses composantes",
            "le passif et ses composantes",
            "variation 24 23",
            "comptes de bilan",
        ]
    ) and "en millions" in norm
    is_formal = any(marker in norm for marker in ["en milliers de mad", "total general i ii iii"])
    return is_summary and not is_formal


def _cached_candidate_has_wrong_scope_context(debug_dir: Path, *, scope: str | None) -> bool:
    if scope not in {"comptes_consolides", "comptes_sociaux"}:
        return False
    selected = _read_json(debug_dir / "selected_page.json")
    if not isinstance(selected, dict):
        return False
    retrieval = selected.get("retrieval") or {}
    page = retrieval.get("page") or {}
    text = str(page.get("page_text") or "")
    if not text:
        return False
    from .utils import normalize_text

    norm = normalize_text(text)
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
        ]
    )
    if scope == "comptes_consolides" and strong_social:
        return True
    if scope == "comptes_sociaux" and strong_consolidated and not strong_social:
        return True
    return False


def _write_crop_metadata(
    debug_dir: Path,
    crop_path: str,
    candidate: TableCandidate,
    vision_called: bool,
    *,
    crop_cache_used: bool,
) -> None:
    payload = {
        "crop_path": crop_path,
        "pdf_id": candidate.pdf_id,
        "page_number": candidate.page_number,
        "target_table": candidate.table_type,
        "scope": candidate.scope,
        "sector": candidate.sector,
        "bbox": candidate.bbox,
        "confidence": candidate.confidence,
        "candidate_confidence": candidate.confidence,
        "candidate_evidence": candidate.evidence,
        "vision_called": vision_called,
        "crop_cache_used": crop_cache_used,
        "table_candidate": candidate.to_dict(),
    }
    write_json(debug_dir / "crop_metadata.json", payload)
    write_json(debug_dir / "selected_crop_validation.json", payload)


def _benchmark_retrieval_payload(
    *,
    retrieved: list[Any] | Any,
    selected_retrieved: Any,
    selected_page: Any = None,
    retrieval_latency_ms: int | None,
) -> dict[str, Any]:
    retrieved_items = retrieved if isinstance(retrieved, list) else []
    selected_payload = _retrieved_page_payload(selected_retrieved)
    predicted_page = _retrieved_page_number(selected_payload)
    if predicted_page is None:
        try:
            predicted_page = int(selected_page)
        except (TypeError, ValueError):
            predicted_page = None
    if selected_payload is None and predicted_page is not None:
        selected_payload = next(
            (
                payload
                for item in retrieved_items
                if (payload := _retrieved_page_payload(item)) is not None
                and _retrieved_page_number(payload) == predicted_page
            ),
            None,
        )
    return {
        "predicted_page": predicted_page,
        "top_k_pages": [
            page_number
            for item in retrieved_items
            if (page_number := _retrieved_page_number(_retrieved_page_payload(item))) is not None
        ],
        "retrieval_scores": _retrieval_score_breakdown(selected_payload),
        "retrieval_latency_ms": retrieval_latency_ms,
    }


def _retrieved_page_payload(item: Any) -> dict[str, Any] | None:
    if item is None:
        return None
    if isinstance(item, dict):
        return item
    to_dict = getattr(item, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        return payload if isinstance(payload, dict) else None
    return None


def _retrieved_page_number(payload: dict[str, Any] | None) -> int | None:
    if not payload:
        return None
    raw = payload.get("page_number")
    if raw is None and isinstance(payload.get("page"), dict):
        raw = payload["page"].get("page_number")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _retrieval_score_breakdown(payload: dict[str, Any] | None) -> dict[str, float | None]:
    if not payload:
        return {
            "bm25": None,
            "vector": None,
            "anchor": None,
            "scope": None,
            "title": None,
            "signature": None,
            "negative_penalty": None,
            "final_score": None,
        }

    opposite_signature = _optional_float(payload.get("opposite_scope_signature_score"))
    return {
        "bm25": _optional_float(payload.get("bm25_score")),
        "vector": _optional_float(payload.get("vector_score")),
        "anchor": _optional_float(payload.get("target_anchor_score")),
        "scope": _optional_float(payload.get("scope_score")),
        "title": _optional_float(payload.get("title_score")),
        "signature": _optional_float(payload.get("scope_signature_score")),
        "negative_penalty": -opposite_signature if opposite_signature is not None else None,
        "final_score": _optional_float(payload.get("final_score") if "final_score" in payload else payload.get("score")),
    }


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _backup_previous_vision_artifacts(
    debug_dir: Path,
    *,
    provider: str,
    model: str | None,
    reason: str = "force_vision",
) -> None:
    artifact_names = [
        "final_extracted.json",
        "extracted_table.json",
        "extracted_table.md",
        "raw_llm_response.json",
        "validation_report.json",
    ]
    existing = [debug_dir / name for name in artifact_names if (debug_dir / name).is_file()]
    if not existing:
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = slugify(model or "default")
    history_dir = ensure_dir(debug_dir / "_llm_history" / f"{stamp}_{provider}_{safe_model}")
    for path in existing:
        shutil.copy2(path, history_dir / path.name)
    write_json(
        history_dir / "rerun_metadata.json",
        {
            "reason": reason,
            "new_provider": provider,
            "new_model": model,
            "original_debug_dir": str(debug_dir),
        },
    )


def _cleanup_rendered_pages(rendered_dir: Path, *, enabled: bool) -> bool:
    if not enabled or not rendered_dir.is_dir():
        return False
    try:
        shutil.rmtree(rendered_dir)
        return True
    except Exception:
        return False


def _should_cleanup_rendered_pages_after_run(base_output: Path) -> bool:
    """Keep rendered PDF pages until the usual 2 scopes x 3 tables are cached."""
    for scope in SCOPES:
        for target_table in TARGET_TABLES:
            debug_dir = base_output / scope / target_table
            final_payload = _read_json(debug_dir / "final_extracted.json")
            if not isinstance(final_payload, dict):
                return False
            if final_payload.get("target_table") != target_table or final_payload.get("scope") != scope:
                return False
            if not final_payload.get("target_found"):
                return False
            if not final_payload.get("columns") or not final_payload.get("rows"):
                return False
            validation = final_payload.get("validation") or _read_json(debug_dir / "validation_report.json") or {}
            if isinstance(validation, dict) and validation.get("status") == "rejected":
                return False
            crop_path = final_payload.get("crop_path") or str(debug_dir / "crop.png")
            if not Path(crop_path).is_file():
                return False
    return True


def _find_cpc_continuation_candidate(
    selected: TableCandidate,
    candidates: list[TableCandidate],
    page_texts: dict[int, str],
) -> TableCandidate | None:
    if selected.table_type != "CPC":
        return None
    from .utils import normalize_text

    selected_text = normalize_text(page_texts.get(selected.page_number, ""))
    if "resultat net" in selected_text and "total des produits" in selected_text:
        return None

    next_page = selected.page_number + 1
    next_text = normalize_text(page_texts.get(next_page, ""))
    if not next_text:
        return None
    looks_like_suite = any(
        marker in next_text[:2500]
        for marker in [
            "compte de produits et charges (hors taxes) (suite)",
            "compte de produits et charges suite",
            "resultat net (xi-xii)",
            "total des produits",
            "total des charges",
        ]
    )
    if not looks_like_suite:
        return None
    for candidate in candidates:
        if candidate.table_type == "CPC" and candidate.page_number == next_page:
            return candidate
    return None


def _stack_crop_images(image_paths: list[str], output_path: str) -> None:
    from PIL import Image

    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        return
    width = max(img.width for img in images)
    separator = 24
    total_height = sum(img.height for img in images) + separator * (len(images) - 1)
    out = Image.new("RGB", (width, total_height), "white")
    y = 0
    for idx, img in enumerate(images):
        x = (width - img.width) // 2
        out.paste(img, (x, y))
        y += img.height
        if idx < len(images) - 1:
            y += separator
    out.save(output_path)


def _read_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _page_size(pdf_path: str, page_number: int) -> tuple[float, float]:
    doc = fitz.open(pdf_path)
    try:
        rect = doc[page_number - 1].rect
        return float(rect.width), float(rect.height)
    finally:
        doc.close()


def _ensure_rendered_page_image(
    pdf_path: str,
    *,
    rendered_dir: Path,
    image_paths: dict[int, str],
    page_number: int,
    dpi: int,
) -> str:
    existing = image_paths.get(page_number)
    if existing and Path(existing).is_file():
        return existing
    rendered = render_pdf_page(pdf_path, rendered_dir, page_number, dpi=dpi)
    image_paths[page_number] = rendered
    return rendered


def _empty_result(pdf_path: str, company: str, year: int, target_table: str, scope: str, sector: str) -> dict[str, Any]:
    return {
        "pdf_path": pdf_path,
        "company": company,
        "year": int(year),
        "target_table": target_table,
        "scope": scope,
        "sector": sector,
        "selected_page": None,
        "crop_path": "",
        "bbox": [],
        "target_found": False,
        "columns": [],
        "rows": [],
        "confidence": 0.0,
        "warnings": [],
        "validation": {"status": "rejected", "issues": ["no_candidate"]},
    }
