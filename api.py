"""
API REST FastAPI pour le pipeline RAG Phase 1.
Frontend (Vite) appelle VITE_API_URL (ex: http://localhost:8000).

Lancer : uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import logging
import json
import os
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rag_api")

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator

from rag_agent.pipeline_phase1 import Phase1Pipeline
from rag_agent.scraper import DEFAULT_PDF_DIR as PDF_DIR, normalize_user_emetteur
from rag_agent.financial_extraction import extract_financial_tables

app = FastAPI(
    title="RAG Phase 1 API",
    description="Extraction de tableaux financiers (BILAN ACTIF, BILAN PASSIF, COMPTE DE PRODUITS ET CHARGES) depuis PDFs RFA AMMC",
    version="1.0",
)

# CORS : autoriser tout frontend (Vite, Lovable, etc.) pour éviter OPTIONS 400
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
    allow_headers=["*"],
    expose_headers=["*"],
)

_API_ROOT = Path(__file__).resolve().parent
# PDF_DIR : même dossier que le scraper (dossier « Émetteur » si détecté)
XLSX_DIR = _API_ROOT / "outputs" / "xlsx"

KNOWN_EMETTEURS = [
    {"id": "agma", "name": "AGMA"},
    {"id": "attijariwafa bank", "name": "Attijariwafa Bank"},
    {"id": "addoha", "name": "Addoha"},
]

# Jobs asynchrones : job_id -> { status, headers?, rows?, error?, excel_path?, ... }
_jobs: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=2)

_pipeline: Optional[Phase1Pipeline] = None
_emetteur_sector_map_cache: dict[str, str] | None = None


def get_pipeline() -> Phase1Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Phase1Pipeline(pdf_dir=PDF_DIR, xlsx_dir=XLSX_DIR)
    return _pipeline


def _sector_map_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").casefold()
    for ch in "’'(),.-_":
        text = text.replace(ch, " ")
    return " ".join(text.split())


def _load_emetteur_sector_map() -> dict[str, str]:
    global _emetteur_sector_map_cache
    if _emetteur_sector_map_cache is not None:
        return _emetteur_sector_map_cache

    path = _API_ROOT / "data" / "emetteur_sector_map.json"
    mapped: dict[str, str] = {}
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                for raw_code, raw_sector in payload.items():
                    sector = str(raw_sector or "").strip()
                    if sector not in {"bancaire_sdf", "assurance", "autres_cgnc"}:
                        continue
                    mapped[str(raw_code).strip().casefold()] = sector
                    mapped[normalize_user_emetteur(str(raw_code))] = sector
                    mapped[_sector_map_key(str(raw_code))] = sector
        except Exception as exc:
            logger.warning("Could not load emetteur sector map: %s", exc)
    _emetteur_sector_map_cache = mapped
    return mapped


def _normalize_type_rapport(type_rapport: str) -> str:
    tr = (type_rapport or "annuel").strip().lower()
    tr = tr.replace("é", "e").replace("è", "e")
    if "1er semestre" in tr or "premier semestre" in tr or "semestr" in tr or tr in ("s1", "rfs"):
        return "s1"
    return "annuel"


def _normalize_search_mode(search_mode: str) -> str:
    sm = (search_mode or "ammc").strip().lower()
    if sm in {"web_api", "api", "serpapi"}:
        return "web_api"
    if sm in {"web", "internet", "google", "duckduckgo"}:
        return "web"
    return "ammc"


def _normalize_api_provider(api_provider: str) -> str:
    p = (api_provider or "groq").strip().lower()
    if p in {"gpt", "openai", "gpt-5", "gpt-5.4"}:
        return "gpt-5.4"
    if p in {"gemini", "google"}:
        return "gemini"
    return "groq"


def _provider_for_financial_extraction(api_provider: str) -> str:
    provider = _normalize_api_provider(api_provider)
    if provider == "gpt-5.4":
        return "openai"
    return provider


def _query_str_first(request: Request, keys: tuple[str, ...]) -> Optional[str]:
    for k in keys:
        v = request.query_params.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


def _query_int_first(request: Request, keys: tuple[str, ...]) -> Optional[int]:
    for k in keys:
        raw = request.query_params.get(k)
        if raw is None or str(raw).strip() == "":
            continue
        try:
            return int(str(raw).strip())
        except ValueError:
            continue
    return None


def _pdf_cache_path(emetteur: str, year: int, type_comptes: str, type_rapport: str) -> Path:
    """Même convention que le scraper : (emetteur, année, type_rapport) -> {safe}_{year}_{report}.pdf"""
    safe = normalize_user_emetteur(emetteur).replace(" ", "_")
    report_code = _normalize_type_rapport(type_rapport)
    return PDF_DIR / f"{safe}_{year}_{report_code}.pdf"


def _ensure_pdf(
    emetteur: str,
    year: int,
    type_comptes: str,
    type_rapport: str,
    search_mode: str,
) -> dict[str, Any]:
    """
    Vérifie si le PDF existe dans data/pdfs. Si oui, retourne succès (from_cache).
    Sinon, tente de le télécharger via le scraper AMMC et le stocke dans data/pdfs.
    """
    from rag_agent.scraper import AMMCScraper
    normalized_report = _normalize_type_rapport(type_rapport)
    normalized_mode = _normalize_search_mode(search_mode)
    path = _pdf_cache_path(emetteur, year, type_comptes, normalized_report)
    if path.is_file() and path.stat().st_size > 0:
        return {
            "success": True,
            "from_cache": True,
            "pdf_path": path.name,
            "message": "PDF déjà présent dans data/pdfs",
            "size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        }
    scraper = AMMCScraper(pdf_dir=PDF_DIR)
    result = scraper.fetch(
        emetteur=emetteur,
        year=year,
        type_comptes=type_comptes,
        type_rapport=normalized_report,
        search_mode=normalized_mode,
    )
    if result.success and result.pdf_path:
        return {
            "success": True,
            "from_cache": False,
            "pdf_path": result.pdf_path.name,
            "message": "PDF téléchargé et stocké dans data/pdfs",
            "size_mb": result.size_mb,
        }
    return {
        "success": False,
        "from_cache": False,
        "pdf_path": None,
        "message": result.error or "Impossible de télécharger le PDF depuis l'AMMC.",
        "error": result.error,
    }


def _emetteurs_from_pdfs() -> list[dict]:
    out: list[dict] = []
    if not PDF_DIR.is_dir():
        return out
    seen: set[tuple[str, int]] = set()
    for f in PDF_DIR.glob("*.pdf"):
        name = f.stem  # ex: attijariwafa_bank_2024
        if "_" not in name:
            continue
        parts = name.split("_")
        year = None
        year_idx = -1
        for idx in range(len(parts) - 1, -1, -1):
            if parts[idx].isdigit():
                year = int(parts[idx])
                year_idx = idx
                break
        if year is None:
            continue
        emetteur_safe = "_".join(parts[:year_idx])
        if not emetteur_safe:
            continue
        emetteur = emetteur_safe.replace("_", " ").strip()
        if (emetteur, year) not in seen:
            seen.add((emetteur, year))
            out.append({"id": emetteur, "name": emetteur.title(), "year": year})
    return out


# ─── Modèles Pydantic ───────────────────────────────────────────────────────


class ExtraireBody(BaseModel):
    """Corps POST /api/extraire. Le frontend (Vite) envoie souvent du camelCase : typeRapport, etc."""

    model_config = ConfigDict(populate_by_name=True)

    emetteur: str = Field(
        ...,
        validation_alias=AliasChoices("emetteur", "Emetteur", "émetteur"),
    )
    annee: int = Field(
        ...,
        description="Année (ex: 2024)",
        validation_alias=AliasChoices("annee", "year", "Year", "Année"),
    )
    type_compte: str = Field(
        "COMPTES SOCIAUX",
        description="COMPTES SOCIAUX ou COMPTES CONSOLIDES",
        validation_alias=AliasChoices("type_compte", "typeCompte"),
    )
    type_comptes: Optional[str] = Field(
        None,
        description="Alias (frontend peut envoyer type_comptes au lieu de type_compte)",
        validation_alias=AliasChoices("type_comptes", "typeComptes"),
    )
    type_rapport: Optional[str] = Field(
        default=None,
        description="Rapports annuels ou Rapports 1er semestre (obligatoire si S1)",
        validation_alias=AliasChoices(
            "type_rapport", "typeRapport", "TypeRapport", "type_de_rapport",
            "rapportType", "rapport_type", "typeRapports", "type_rapports",
            "rapport", "reportType", "report_type",
        ),
    )
    search_mode: Optional[str] = Field(
        "ammc",
        description="ammc, web ou web_api",
        validation_alias=AliasChoices("search_mode", "searchMode"),
    )
    tableau: str = Field(
        ...,
        description="BILAN ACTIF, BILAN PASSIF, COMPTE DE PRODUITS ET CHARGES",
        validation_alias=AliasChoices("tableau", "table", "Tableau"),
    )
    api_provider: Optional[str] = Field(
        "groq",
        description="Fournisseur LLM: groq, gemini, gpt-5.4",
        validation_alias=AliasChoices("api_provider", "apiProvider", "provider", "llm_provider"),
    )
    force_vision: bool = Field(
        False,
        description="Force une nouvelle extraction Vision meme si un resultat LLM valide existe deja",
        validation_alias=AliasChoices("force_vision", "forceVision", "rerun_vision", "rerunVision"),
    )
    force_page: bool = Field(
        False,
        description="Force une nouvelle selection de page/crop RAG avant l'extraction Vision",
        validation_alias=AliasChoices("force_page", "forcePage", "rerun_page", "rerunPage"),
    )

    @field_validator("emetteur", mode="before")
    @classmethod
    def _normalize_emetteur_field(cls, v: Any) -> str:
        if v is None:
            return ""
        return normalize_user_emetteur(str(v))


# ─── Routes ─────────────────────────────────────────────────────────────────


@app.get("/")
def root():
    return {"message": "RAG Phase 1 API", "docs": "/docs", "api": "/api/emetteurs"}


def _load_emetteurs_from_file() -> list[dict]:
    """Charge la liste depuis data/emetteurs_ammc.json si le fichier existe (format: [{"code","label"}, ...])."""
    path = _API_ROOT / "data" / "emetteurs_ammc.json"
    if not path.is_file():
        return []
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list) and data and "code" in data[0] and "label" in data[0]:
            return [{"code": e["code"], "label": e["label"]} for e in data]
        if isinstance(data, dict) and "emetteurs" in data:
            return [{"code": e["code"], "label": e["label"]} for e in data["emetteurs"]]
    except Exception:
        pass
    return []


def _get_emetteurs_ammc() -> list[dict]:
    """Liste complete des emetteurs. Merge cache local + scrape AMMC to avoid partial live lists."""
    cached = _load_emetteurs_from_file()
    scraped: list[dict] = []
    try:
        from rag_agent.scraper import AMMCScraper
        scraper = AMMCScraper()
        scraped = [{"code": e["code"], "label": e["label"]} for e in scraper.fetch_liste_emetteurs(max_pages=20)]
    except Exception:
        scraped = []

    by_code: dict[str, dict] = {}
    for item in cached + scraped:
        code = str(item.get("code") or "").strip()
        label = str(item.get("label") or "").strip()
        if not code or not label:
            continue
        by_code[code.lower()] = {"code": code, "label": label}
    return sorted(by_code.values(), key=lambda x: x["label"].lower())


@app.get("/api/emetteurs/ensure-pdf", response_class=JSONResponse)
def api_ensure_pdf(
    request: Request,
    emetteur: Optional[str] = Query(None, description="Code émetteur (ex: agma, addoha) ; alias: code"),
    annee: Optional[int] = Query(None, description="Année (ex: 2024) ; alias: year, Year, Année"),
    type_comptes: str = Query("sociaux", description="sociaux ou consolides"),
    search_mode: str = Query("ammc", description="ammc, web ou web_api"),
):
    """
    Quand l'utilisateur choisit émetteur + année + type de rapport : vérifie le cache PDF puis télécharge si besoin.
    Paramètres obligatoires en query : **emetteur** (ou **code**) et **annee** (ou **year**).
    Type de rapport : passer `type_rapport` **ou** `typeRapport` (camelCase, ex. Rapports 1er semestre).
    Le paramètre `emetteur` accepte aussi les saisies type « ÉmetteurADDOHA » (normalisé en addoha).
    Fichier cible : `{slug}_{année}_{annuel|s1}.pdf` (dossier « Émetteur » du projet ou équivalent).
    Après extraction Phase 1, l’index RAG est sous `data/index/{slug}/{année}/s1|annuel/index/`.
    """
    em_raw = (emetteur.strip() if emetteur else "") or _query_str_first(
        request, ("code", "Emetteur", "émetteur")
    )
    yr = annee if annee is not None else _query_int_first(request, ("year", "Year", "Année"))
    if not em_raw or yr is None:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "ensure-pdf requiert emetteur (ou code) et annee (ou year) en plus de type_rapport.",
                "example": "/api/emetteurs/ensure-pdf?emetteur=addoha&annee=2024&type_rapport=Rapports%20annuels",
                "aliases": {"emetteur": ["code"], "annee": ["year", "Year", "Année"]},
            },
        )
    tr_raw = (
        request.query_params.get("type_rapport")
        or request.query_params.get("typeRapport")
        or request.query_params.get("rapportType")
        or request.query_params.get("rapport_type")
        or "annuel"
    )
    logger.info("ensure-pdf: emetteur=%r annee=%r type_rapport_raw=%r → %r", em_raw, yr, tr_raw, _normalize_type_rapport(tr_raw))
    result = _ensure_pdf(
        normalize_user_emetteur(em_raw),
        yr,
        type_comptes.strip().lower(),
        tr_raw.strip(),
        search_mode.strip().lower(),
    )
    return JSONResponse(content=result, media_type="application/json; charset=utf-8")


# Types de tableaux extractibles (nom principal pour le 3e : COMPTE DE PRODUITS ET CHARGES, pas CPC)
TABLEAUX_DISPO = [
    {"id": "bilan_actif", "label": "BILAN ACTIF COMPTES SOCIAUX", "tableau": "BILAN ACTIF"},
    {"id": "bilan_passif", "label": "BILAN PASSIF COMPTES SOCIAUX", "tableau": "BILAN PASSIF"},
    {"id": "compte_produits_charges", "label": "COMPTE DE PRODUITS ET CHARGES COMPTES SOCIAUX", "tableau": "COMPTE DE PRODUITS ET CHARGES"},
]


@app.get("/api/tableaux", response_class=JSONResponse)
def api_tableaux():
    """Liste des types de tableaux (nom principal : COMPTE DE PRODUITS ET CHARGES, pas CPC)."""
    return JSONResponse(content=TABLEAUX_DISPO, media_type="application/json; charset=utf-8")


@app.get("/api/emetteurs", response_class=JSONResponse)
def api_emetteurs():
    """
    Liste de tous les émetteurs existants sur le site AMMC.
    Retourne { emetteurs: [ { code, label } ] }. Si le scrape AMMC échoue, fallback sur cache PDF + liste connue.
    """
    try:
        emetteurs = _get_emetteurs_ammc()
        if not emetteurs:
            from_pdfs = _emetteurs_from_pdfs()
            by_id: dict[str, dict] = {}
            for e in KNOWN_EMETTEURS:
                by_id[e["id"]] = e
            for e in from_pdfs:
                by_id[e["id"]] = e
            seen_names: set[str] = set()
            for e in by_id.values():
                name_lower = (e.get("name") or "").lower()
                if name_lower not in seen_names:
                    seen_names.add(name_lower)
                    emetteurs.append({"code": e["id"], "label": e.get("name") or e["id"]})
            if not emetteurs:
                emetteurs = [{"code": e["id"], "label": e["name"]} for e in KNOWN_EMETTEURS]
        for item in emetteurs:
            code = str(item.get("code") or "")
            label = str(item.get("label") or "")
            item["sector"] = _infer_financial_sector(code or label)
        return JSONResponse(
            content={"emetteurs": emetteurs},
            media_type="application/json; charset=utf-8",
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "emetteurs": [{"code": x["id"], "label": x["name"]} for x in KNOWN_EMETTEURS]},
            media_type="application/json; charset=utf-8",
        )


def _run_pipeline_job(
    job_id: str,
    emetteur: str,
    year: int,
    tableau: str,
    type_comptes: str,
    type_rapport: str,
    search_mode: str,
    api_provider: str,
) -> None:
    """Exécute le pipeline et met à jour _jobs[job_id]."""
    tc = (type_comptes or "sociaux").strip().lower()
    if tc not in ("sociaux", "consolides"):
        tc = "consolides" if "consolid" in tc else "sociaux"
    tr = _normalize_type_rapport(type_rapport)
    sm = _normalize_search_mode(search_mode)
    provider = _normalize_api_provider(api_provider)
    logger.info(
        "_run_pipeline_job: emetteur=%r year=%r tableau=%r type_comptes=%r type_rapport=%r api_provider=%r",
        emetteur,
        year,
        tableau,
        tc,
        tr,
        provider,
    )
    pipeline = get_pipeline()
    try:
        result = pipeline.run(
            emetteur=emetteur.strip(),
            year=year,
            tableau=tableau.strip(),
            type_comptes=tc,
            type_rapport=tr,
            search_mode=sm,
            api_provider=provider,
        )
        if not result.success:
            _jobs[job_id] = {
                "status": "error",
                "error": result.error or "Erreur lors de l'extraction",
                "page_num": result.page_num,
                "method": result.method,
                "confidence": result.confidence,
                "fallback_used": getattr(result, "fallback_used", False),
                "extraction_warnings": getattr(result, "extraction_warnings", []) or [],
                "extraction_strategy_used": getattr(result, "extraction_strategy_used", result.method),
                "completeness_score": getattr(result, "completeness_score", 0.0),
                "missing_anchors": getattr(result, "missing_anchors", []) or [],
                "type_rapport_used": tr,
                "api_provider": provider,
            }
            return
        df = result.df
        columns = list(df.columns)
        data = df.fillna("").astype(str).values.tolist()
        excel_path = str(result.excel_path) if result.excel_path else None
        _jobs[job_id] = {
            "status": "success",
            "headers": columns,
            "rows": data,
            "method": result.method,
            "confidence": result.confidence,
            "page_num": result.page_num,
            "excel_path": excel_path,
            "pdf_from_cache": getattr(result, "pdf_from_cache", False),
            "fallback_used": getattr(result, "fallback_used", False),
            "extraction_warnings": getattr(result, "extraction_warnings", []) or [],
            # ── Nouveaux champs observabilité 2025 ──────────────────────────
            "extraction_strategy_used": getattr(result, "extraction_strategy_used", result.method),
            "completeness_score": getattr(result, "completeness_score", 0.0),
            "missing_anchors": getattr(result, "missing_anchors", []) or [],
            "type_rapport_used": tr,
            "api_provider": provider,
        }
    except Exception as e:
        logger.exception("_run_pipeline_job exception: %s", e)
        _jobs[job_id] = {
            "status": "error",
            "error": str(e),
            "type_rapport_used": tr,
            "api_provider": provider,
        }


def _run_financial_vision_job(
    job_id: str,
    emetteur: str,
    year: int,
    tableau: str,
    type_comptes: str,
    type_rapport: str,
    search_mode: str,
    api_provider: str,
    force_vision: bool = False,
    force_page: bool = False,
) -> None:
    """Runs the new page-aware crop + Vision extraction pipeline for /api/extraire."""
    tc = (type_comptes or "sociaux").strip().lower()
    if tc not in ("sociaux", "consolides"):
        tc = "consolides" if "consolid" in tc else "sociaux"
    tr = _normalize_type_rapport(type_rapport)
    sm = _normalize_search_mode(search_mode)
    provider = _provider_for_financial_extraction(api_provider)
    normalized_emetteur = normalize_user_emetteur(emetteur).strip()
    target_table = _normalize_financial_target_table(tableau)
    scope = "comptes_consolides" if tc == "consolides" else "comptes_sociaux"
    company = _display_company_name(normalized_emetteur)
    sector = _infer_financial_sector(normalized_emetteur)
    logger.info(
        "_run_financial_vision_job: emetteur=%r year=%r target_table=%r scope=%r type_rapport=%r sector=%r provider=%r force_vision=%r force_page=%r",
        normalized_emetteur,
        year,
        target_table,
        scope,
        tr,
        sector,
        provider,
        force_vision,
        force_page,
    )
    try:
        _jobs[job_id] = {"status": "pending", "step": "ensure_pdf", "progress": 5}
        ensured = _ensure_pdf(normalized_emetteur, year, tc, tr, sm)
        if not ensured.get("success"):
            raise RuntimeError(ensured.get("message") or "PDF introuvable")
        pdf_path = _pdf_cache_path(normalized_emetteur, year, tc, tr)
        if not pdf_path.is_file():
            raise RuntimeError(f"PDF introuvable: {pdf_path}")

        _jobs[job_id] = {"status": "pending", "step": "crop_and_vision", "progress": 30}
        output_dir = (
            _API_ROOT
            / "output"
            / "financial_extraction_debug"
            / "platform"
            / f"{normalize_user_emetteur(normalized_emetteur).replace(' ', '_')}_{year}_{tr}"
        )
        summary = extract_financial_tables(
            {
                "pdf_path": str(pdf_path),
                "company": company,
                "year": year,
                "report_type": tr,
                "scope": scope,
                "sector": sector,
                "target_tables": [target_table],
                "provider": provider,
                "force_vision": force_vision,
                "force_page": force_page,
            },
            output_dir=output_dir,
            force_vision=force_vision,
            force_page=force_page,
        )
        result = (summary.get("results") or [{}])[0]
        headers, rows = _financial_result_to_table(result)
        excel_path = _write_financial_excel(job_id, headers, rows, company, year, target_table)
        validation = result.get("validation") or {}
        result_error = "" if result.get("target_found") else "; ".join(result.get("warnings", [])) or result.get("error") or "target_not_found"
        _jobs[job_id] = {
            "status": "success" if result.get("target_found") else "error",
            "headers": headers,
            "rows": rows,
            "method": "financial_crop_vision",
            "confidence": result.get("confidence", 0.0),
            "page_num": result.get("selected_page"),
            "excel_path": excel_path,
            "pdf_from_cache": ensured.get("from_cache", False),
            "fallback_used": False,
            "extraction_warnings": result.get("warnings", []) or validation.get("warnings", []),
            "extraction_strategy_used": "page_aware_rag_crop_vision",
            "completeness_score": 1.0 if validation.get("status") == "approved" else 0.5,
            "missing_anchors": [],
            "type_rapport_used": tr,
            "api_provider": provider,
            "crop_path": result.get("crop_path", ""),
            "vision_output_dir": (result.get("debug") or {}).get("dir", str(output_dir)),
            "validation": validation,
            "target_found": bool(result.get("target_found")),
            "sector": sector,
            "scope": scope,
            "target_table": target_table,
            "force_vision": force_vision,
            "force_page": force_page,
            "error": _friendly_extraction_error(result_error),
        }
    except Exception as e:
        logger.exception("_run_financial_vision_job exception: %s", e)
        _jobs[job_id] = {
            "status": "error",
            "error": str(e),
            "type_rapport_used": tr,
            "api_provider": provider,
            "force_vision": force_vision,
            "force_page": force_page,
        }


def _normalize_financial_target_table(tableau: str) -> str:
    text = str(tableau or "").strip().upper()
    text = text.replace("É", "E").replace("È", "E").replace("-", " ")
    if "PASSIF" in text:
        return "BILAN_PASSIF"
    if "ACTIF" in text:
        return "BILAN_ACTIF"
    if "CPC" in text or "PRODUITS" in text or "CHARGES" in text:
        return "CPC"
    raise ValueError(f"Tableau non supporte par le nouveau pipeline: {tableau}")


def _display_company_name(emetteur: str) -> str:
    aliases = {
        "attijariwafa bank": "Attijariwafa Bank",
        "addoha": "Addoha",
        "marsa maroc": "Marsa Maroc",
        "alliances darna": "Alliances Darna",
        "cih bank": "CIH Bank",
        "bank of africa groupe bmce boa": "Bank of Africa",
        "atlantasanad": "AtlantaSanad Assurance",
    }
    normalized = normalize_user_emetteur(emetteur)
    return aliases.get(normalized, normalized.replace("_", " ").title())


def _infer_financial_sector(emetteur: str) -> str:
    normalized = normalize_user_emetteur(emetteur)
    sector_map = _load_emetteur_sector_map()
    for key in {str(emetteur or "").strip().casefold(), normalized, _sector_map_key(emetteur)}:
        mapped_sector = sector_map.get(key)
        if mapped_sector:
            return mapped_sector

    banking_issuer_markers = [
        "saham bank",
        "saham leasing",
        "societe generale marocaine de banques",
        "societe generale maroc",
    ]
    if any(marker in normalized for marker in banking_issuer_markers):
        return "bancaire_sdf"

    insurance_issuer_markers = [
        "saham assurance",
        "sanlam maroc",
        "wafa assurance",
        "atlantasanad",
        "atlanta sanad",
        "atlanta",
    ]
    if any(marker in normalized for marker in insurance_issuer_markers):
        return "assurance"

    bank_markers = [
        "bank",
        "banque",
        "cih",
        "attijari",
        "bmce",
        "boa",
        "bcp",
        "credit du maroc",
        "cdm",
        "wafasalaf",
        "wafabail",
        "leasing",
        "bail",
        "salaf",
        "societe de financement",
        "credit a la consommation",
    ]
    insurance_markers = ["assurance", "sanad"]
    if any(marker in normalized for marker in insurance_markers):
        return "assurance"
    if any(marker in normalized for marker in bank_markers):
        return "bancaire_sdf"
    return "autres_cgnc"


def _financial_result_to_table(result: dict[str, Any]) -> tuple[list[str], list[list[str]]]:
    columns = [str(c) for c in (result.get("columns") or [])]
    rows_payload = result.get("rows") or []
    headers = ["label", *columns]
    rows: list[list[str]] = []
    for row in rows_payload:
        values = row.get("values") or {}
        rows.append([str(row.get("label", "")), *[str(values.get(col, "")) for col in columns]])
    return headers, rows


def _friendly_extraction_error(error: str) -> str:
    text = str(error or "")
    lowered = text.lower()
    if "insufficient_quota" in lowered or "exceeded your current quota" in lowered:
        return (
            "Quota du fournisseur LLM atteinte. Le PDF et le crop ont ete generes, "
            "mais l'extraction Vision n'a pas pu etre lancee. Choisissez Groq/Gemini "
            "ou verifiez le quota OpenAI."
        )
    if "429" in lowered or "too many requests" in lowered or "rate limit" in lowered:
        return (
            "Limite du fournisseur LLM atteinte. Le PDF et le crop ont ete generes. "
            "Attendez 1-2 minutes puis relancez, envoyez moins de tables en meme temps, "
            "ou choisissez un autre fournisseur Vision."
        )
    return text


def _write_financial_excel(job_id: str, headers: list[str], rows: list[list[str]], company: str, year: int, target_table: str) -> str | None:
    try:
        import pandas as pd

        XLSX_DIR.mkdir(parents=True, exist_ok=True)
        path = XLSX_DIR / f"{job_id}_{normalize_user_emetteur(company).replace(' ', '_')}_{year}_{target_table.lower()}.xlsx"
        pd.DataFrame(rows, columns=headers).to_excel(path, index=False)
        return str(path)
    except Exception as exc:
        logger.warning("_write_financial_excel failed: %s", exc)
        return None


@app.options("/api/extraire")
def options_extraire():
    """Répond au preflight CORS (navigateur envoie OPTIONS avant POST)."""
    return Response(status_code=200)


def _extract_type_rapport_from_raw(raw: dict) -> Optional[str]:
    """
    Cherche dans le corps JSON brut un champ contenant le type de rapport,
    en acceptant n'importe quel nom de clé contenant « rapport », « semestre » ou « report ».
    Retourne la valeur brute trouvée, ou None.
    """
    # Clés connues en priorité (même liste que AliasChoices du modèle)
    priority_keys = [
        "type_rapport", "typeRapport", "TypeRapport", "type_de_rapport",
        "rapportType", "rapport_type", "typeRapports", "type_rapports",
        "rapport", "reportType", "report_type",
    ]
    for k in priority_keys:
        if k in raw and raw[k] is not None:
            return str(raw[k])
    # Recherche générique : toute clé contenant "rapport", "semestre" ou "report"
    for k, v in raw.items():
        kl = k.lower()
        if ("rapport" in kl or "semestre" in kl or "report" in kl) and v is not None:
            return str(v)
    return None


@app.post("/api/extraire")
async def api_extraire(request: Request):
    """
    Lance l'extraction en arrière-plan. Retourne { job_id }.
    Télécharge le PDF si absent, indexe dans `data/index/{emetteur}/{annee}/s1|annuel/index/` si nécessaire.
    Le champ `emetteur` est normalisé (ex. « ÉmetteurADDOHA » → addoha).
    """
    # Lire le corps JSON brut AVANT le parsing Pydantic
    try:
        raw = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Corps JSON invalide : {exc}")

    logger.info("api_extraire raw body: %s", raw)

    try:
        body = ExtraireBody.model_validate(raw)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # type_rapport : utiliser le champ Pydantic si présent, sinon chercher dans le corps brut
    type_rapport_raw = body.type_rapport
    if not type_rapport_raw:
        type_rapport_raw = _extract_type_rapport_from_raw(raw)

    type_rapport = _normalize_type_rapport(type_rapport_raw or "annuel")
    logger.info(
        "api_extraire: emetteur=%r annee=%r type_rapport_raw=%r → type_rapport=%r tableau=%r",
        body.emetteur, body.annee, type_rapport_raw, type_rapport, body.tableau,
    )

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "step": "scraping", "steps_done": [], "progress": 0}
    type_map = {
        "COMPTES SOCIAUX": "sociaux",
        "COMPTES CONSOLIDES": "consolides",
        "COMPTES CONSOLIDÉS": "consolides",
        "SOCIAUX": "sociaux",
        "CONSOLIDES": "consolides",
        "CONSOLIDÉS": "consolides",
    }
    raw_type = (body.type_comptes or body.type_compte or "COMPTES SOCIAUX").strip()
    _key = raw_type.upper().replace("É", "E").replace("È", "E")
    type_comptes = type_map.get(_key, "consolides" if "CONSOLID" in _key else "sociaux")
    search_mode = _normalize_search_mode(body.search_mode or "ammc")
    api_provider = _normalize_api_provider(body.api_provider or "groq")
    force_vision = bool(body.force_vision)
    force_page = bool(body.force_page)
    _executor.submit(
        _run_financial_vision_job,
        job_id,
        body.emetteur,
        body.annee,
        body.tableau,
        type_comptes,
        type_rapport,
        search_mode,
        api_provider,
        force_vision,
        force_page,
    )
    return {
        "job_id": job_id,
        "type_rapport_used": type_rapport,
        "api_provider": api_provider,
        "force_vision": force_vision,
        "force_page": force_page,
    }


@app.post("/api/financial-extraction/vision", response_class=JSONResponse)
async def api_financial_extraction_vision(request: Request):
    """
    Runs the page-aware crop pipeline, then sends each selected crop.png to the
    selected Vision provider. This is separate from the legacy Phase1 route.
    """
    try:
        raw = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Corps JSON invalide : {exc}")
    try:
        result = extract_financial_tables(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return JSONResponse(content=result, media_type="application/json; charset=utf-8")


@app.get("/api/status/{job_id}", response_class=JSONResponse)
def api_status(job_id: str):
    """
    Statut d'un job d'extraction.
    Retourne { status: "pending" | "success" | "error", headers?, rows?, error?, ... }.
    """
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job introuvable")
    return JSONResponse(content=_jobs[job_id], media_type="application/json; charset=utf-8")


@app.get("/api/download/{job_id}")
def api_download(job_id: str):
    """Télécharge le fichier Excel généré pour ce job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job introuvable")
    job = _jobs[job_id]
    if job.get("status") != "success":
        raise HTTPException(status_code=400, detail="Extraction non terminée ou en erreur")
    excel_path = job.get("excel_path")
    if not excel_path or not os.path.isfile(excel_path):
        raise HTTPException(status_code=404, detail="Fichier Excel introuvable")
    filename = os.path.basename(excel_path)
    return FileResponse(
        excel_path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
