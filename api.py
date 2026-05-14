"""
API REST FastAPI — plateforme d'extraction financière (PDF AMMC → RAG page-aware + Vision).
Frontend (Vite) appelle VITE_API_URL (ex: http://localhost:8000).

Lancer : uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import logging
import json
import os
import unicodedata
import uuid
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("rag_api")

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator
import requests

from rag_agent.scraper import DEFAULT_PDF_DIR as PDF_DIR, normalize_user_emetteur
from rag_agent.financial_extraction import extract_financial_tables
from rag_agent.financial_extraction.benchmark import (
    FAILURE_REASONS,
    build_financial_benchmark_entry,
    save_financial_benchmark_entry,
)

app = FastAPI(
    title="Financial extraction API",
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

APPROACH_A = "A"
APPROACH_B = "B"
APPROACH_C = "C"
APPROACHES: dict[str, dict[str, str | None]] = {
    APPROACH_A: {
        "id": APPROACH_A,
        "label": os.getenv("APPROACH_A_LABEL", "A"),
        "mode": "local",
        "path": str(_API_ROOT),
        "base_url": None,
    },
    APPROACH_B: {
        "id": APPROACH_B,
        "label": os.getenv("APPROACH_B_LABEL", "B"),
        "mode": "external",
        "path": os.getenv("APPROACH_B_PATH", r"C:\Users\Pks\Downloads\original(khdam baqi a LLM d 2 table)"),
        "base_url": os.getenv("APPROACH_B_URL", "http://127.0.0.1:8001"),
    },
    APPROACH_C: {
        "id": APPROACH_C,
        "label": os.getenv("APPROACH_C_LABEL", "C"),
        "mode": "external",
        "path": os.getenv("APPROACH_C_PATH", r"C:\Users\Pks\Downloads\RAG - Copie"),
        "base_url": os.getenv("APPROACH_C_URL", "http://127.0.0.1:8002"),
    },
}
TERMINAL_JOB_STATUSES = {"success", "error", "partial"}

KNOWN_EMETTEURS = [
    {"id": "agma", "name": "AGMA"},
    {"id": "attijariwafa bank", "name": "Attijariwafa Bank"},
    {"id": "addoha", "name": "Addoha"},
]

# Jobs asynchrones : job_id -> { status, headers?, rows?, error?, excel_path?, ... }
_jobs: dict[str, dict[str, Any]] = {}
_executor = ThreadPoolExecutor(max_workers=2)

_emetteur_sector_map_cache: dict[str, str] | None = None


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
    approach: str = Field(
        APPROACH_A,
        description="Approche d'extraction: A (locale), B ou C (services externes)",
        validation_alias=AliasChoices("approach", "approche", "extraction_approach", "extractionApproach"),
    )
    force_vision: bool = Field(
        False,
        description=(
            "Relance uniquement le LLM vision en conservant crop.png et le candidat actuel "
            "(changement de modèle / fournisseur). Sans crop valide, l'API renvoie une erreur: "
            "faire une extraction complète ou utiliser « nouveau crop » / « nouvelle page »."
        ),
        validation_alias=AliasChoices("force_vision", "forceVision", "rerun_vision", "rerunVision"),
    )
    force_page: bool = Field(
        False,
        description=(
            "Nouvelle page : refait recherche + scores + localisation, exclut la page précédente du classement, "
            "régénère crop.png puis LLM. Si le tableau n'est pas sur la bonne page."
        ),
        validation_alias=AliasChoices("force_page", "forcePage", "rerun_page", "rerunPage"),
    )
    force_recrop: bool = Field(
        False,
        description=(
            "Nouveau crop : invalide le crop en cache, relance recherche + localisation **sans** exclure la page "
            "déjà trouvée (même page si elle reste en tête), régénère crop.png puis LLM. Si la page est bonne mais le crop est mauvais."
        ),
        validation_alias=AliasChoices("force_recrop", "forceRecrop", "rerun_crop", "rerunCrop"),
    )

    @field_validator("emetteur", mode="before")
    @classmethod
    def _normalize_emetteur_field(cls, v: Any) -> str:
        if v is None:
            return ""
        return normalize_user_emetteur(str(v))

    @field_validator("approach", mode="before")
    @classmethod
    def _normalize_approach_field(cls, v: Any) -> str:
        return _normalize_approach(v)


# ─── Routes ─────────────────────────────────────────────────────────────────


class BenchmarkFeedbackBody(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    job_id: str = Field(..., validation_alias=AliasChoices("job_id", "jobId"))
    verdict: str = Field(..., description="pass or fail")
    failure_reason: Optional[str] = Field(
        None,
        validation_alias=AliasChoices("failure_reason", "failureReason", "reason"),
    )
    notes: Optional[str] = None

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalize_verdict(cls, v: Any) -> str:
        text = str(v or "").strip().lower()
        if text in {"pass", "passed", "ok", "valid", "correct"}:
            return "pass"
        if text in {"fail", "failed", "ko", "wrong", "incorrect"}:
            return "fail"
        raise ValueError("verdict must be pass or fail")

    @field_validator("failure_reason", mode="before")
    @classmethod
    def _normalize_failure_reason(cls, v: Any) -> str | None:
        if v is None or str(v).strip() == "":
            return None
        text = str(v).strip().upper()
        if text not in FAILURE_REASONS:
            raise ValueError(f"failure_reason must be one of: {', '.join(sorted(FAILURE_REASONS))}")
        return text


@app.get("/")
def root():
    return {"message": "Financial extraction API", "docs": "/docs", "api": "/api/emetteurs"}


def _normalize_approach(value: Any) -> str:
    text = str(value or APPROACH_A).strip().upper()
    aliases = {
        "1": APPROACH_A,
        "LOCAL": APPROACH_A,
        "MAIN": APPROACH_A,
        "DEFAULT": APPROACH_A,
        "RAG_AVANT_OCR": APPROACH_A,
        "2": APPROACH_B,
        "ORIGINAL": APPROACH_B,
        "EXTERNAL": APPROACH_B,
        "3": APPROACH_C,
        "RAG_COPIE": APPROACH_C,
        "COPY": APPROACH_C,
    }
    normalized = aliases.get(text, text)
    if normalized not in APPROACHES:
        raise ValueError(f"Approche inconnue: {value}. Valeurs acceptees: A, B, C")
    return normalized


def _public_approaches() -> list[dict[str, str | None]]:
    return [
        {
            "id": approach["id"],
            "label": approach["label"],
            "mode": approach["mode"],
            "path": approach["path"],
            "base_url": approach["base_url"],
        }
        for approach in APPROACHES.values()
    ]


@app.get("/api/approaches", response_class=JSONResponse)
def api_approaches():
    """Liste des approches disponibles pour le bouton Choix de L'approche."""
    return JSONResponse(
        content={"default": APPROACH_A, "approaches": _public_approaches()},
        media_type="application/json; charset=utf-8",
    )


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
    L’index page-aware et les artefacts d’extraction sont sous `output/financial_extraction_debug/platform/`.
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
    force_recrop: bool = False,
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
        "_run_financial_vision_job: emetteur=%r year=%r target_table=%r scope=%r type_rapport=%r sector=%r provider=%r force_vision=%r force_page=%r force_recrop=%r",
        normalized_emetteur,
        year,
        target_table,
        scope,
        tr,
        sector,
        provider,
        force_vision,
        force_page,
        force_recrop,
    )
    try:
        _jobs[job_id] = {
            "status": "pending",
            "step": "ensure_pdf",
            "progress": 5,
            "approach": APPROACH_A,
            "approach_label": APPROACHES[APPROACH_A]["label"],
        }
        ensured = _ensure_pdf(normalized_emetteur, year, tc, tr, sm)
        if not ensured.get("success"):
            raise RuntimeError(ensured.get("message") or "PDF introuvable")
        pdf_path = _pdf_cache_path(normalized_emetteur, year, tc, tr)
        if not pdf_path.is_file():
            raise RuntimeError(f"PDF introuvable: {pdf_path}")

        _jobs[job_id] = {
            "status": "pending",
            "step": "crop_and_vision",
            "progress": 30,
            "approach": APPROACH_A,
            "approach_label": APPROACHES[APPROACH_A]["label"],
        }
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
                "force_recrop": force_recrop,
            },
            output_dir=output_dir,
            force_vision=force_vision,
            force_page=force_page,
            force_recrop=force_recrop,
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
            "approach": APPROACH_A,
            "approach_label": APPROACHES[APPROACH_A]["label"],
            "job_id": job_id,
            "pdf_path": str(pdf_path),
            "company": company,
            "year": year,
            "emetteur": normalized_emetteur,
            "type_rapport_used": tr,
            "crop_path": result.get("crop_path", ""),
            "bbox": result.get("bbox", []),
            "predicted_page": (result.get("benchmark_retrieval") or {}).get("predicted_page") or result.get("selected_page"),
            "top_k_pages": (result.get("benchmark_retrieval") or {}).get("top_k_pages", []),
            "retrieval_scores": (result.get("benchmark_retrieval") or {}).get("retrieval_scores", {}),
            "retrieval_latency_ms": (result.get("benchmark_retrieval") or {}).get("retrieval_latency_ms"),
            "vision_output_dir": (result.get("debug") or {}).get("dir", str(output_dir)),
            "validation": validation,
            "target_found": bool(result.get("target_found")),
            "sector": sector,
            "scope": scope,
            "target_table": target_table,
            "force_vision": force_vision,
            "force_page": force_page,
            "force_recrop": force_recrop,
            "error": _friendly_extraction_error(
                result_error,
                target_table=target_table,
                selected_page=result.get("selected_page"),
                crop_path=result.get("crop_path", ""),
            ),
        }
    except Exception as e:
        logger.exception("_run_financial_vision_job exception: %s", e)
        _jobs[job_id] = {
            "status": "error",
            "error": str(e),
            "type_rapport_used": tr,
            "api_provider": provider,
            "approach": APPROACH_A,
            "approach_label": APPROACHES[APPROACH_A]["label"],
            "force_vision": force_vision,
            "force_page": force_page,
            "force_recrop": force_recrop,
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


def _friendly_extraction_error(
    error: str,
    *,
    target_table: str | None = None,
    selected_page: int | None = None,
    crop_path: str | None = None,
) -> str:
    text = str(error or "")
    lowered = text.lower()
    target_label = _target_table_label(target_table)
    if "insufficient_quota" in lowered or "exceeded your current quota" in lowered:
        return (
            f"{target_label}: quota du fournisseur LLM atteinte. Le PDF et le crop ont ete generes, "
            "mais l'extraction Vision n'a pas pu etre lancee. Choisissez Groq/Gemini "
            "ou verifiez le quota OpenAI."
        )
    if "429" in lowered or "too many requests" in lowered or "rate limit" in lowered:
        return (
            f"{target_label}: limite du fournisseur LLM atteinte. Le PDF et le crop ont ete generes. "
            "Attendez 1-2 minutes puis relancez, envoyez moins de tables en meme temps, "
            "ou choisissez un autre fournisseur Vision."
        )
    if "target_found_false" in lowered or "target_not_found" in lowered:
        details = []
        if selected_page:
            details.append(f"page selectionnee: {selected_page}")
        if crop_path:
            details.append("crop disponible pour verification")
        suffix = f" ({'; '.join(details)})" if details else ""
        return (
            f"{target_label}: table cible non detectee par le LLM dans le crop{suffix}. "
            "Le probleme concerne cette table seulement; vous pouvez relancer avec "
            "'Reextraire the New Page' ou deselectionner cette table et garder les autres."
        )
    if text:
        return f"{target_label}: {text}" if target_table else text
    return text


def _target_table_label(target_table: str | None) -> str:
    labels = {
        "BILAN_ACTIF": "BILAN ACTIF",
        "BILAN_PASSIF": "BILAN PASSIF",
        "CPC": "CPC",
    }
    return labels.get(str(target_table or "").upper(), str(target_table or "Table").replace("_", " "))


def _external_url(approach: str, endpoint: str) -> str:
    cfg = APPROACHES[approach]
    base_url = str(cfg.get("base_url") or "").rstrip("/")
    if not base_url:
        raise RuntimeError(f"Approche {approach} n'a pas de base_url configuree")
    return f"{base_url}{endpoint}"


def _run_external_approach_job(
    job_id: str,
    approach: str,
    payload: dict[str, Any],
) -> None:
    """Forward a local job to an external approach API and mirror its final status."""
    cfg = APPROACHES[approach]
    base_url = str(cfg.get("base_url") or "").rstrip("/")
    label = str(cfg.get("label") or approach)
    try:
        _jobs[job_id] = {
            "status": "pending",
            "step": "external_submit",
            "progress": 5,
            "approach": approach,
            "approach_label": label,
            "external_base_url": base_url,
        }
        response = requests.post(
            _external_url(approach, "/api/extraire"),
            json={k: v for k, v in payload.items() if k != "approach"},
            timeout=30,
        )
        response.raise_for_status()
        submitted = response.json()
        external_job_id = submitted.get("job_id")
        if not external_job_id:
            raise RuntimeError(f"Approche {approach} n'a pas retourne job_id: {submitted}")

        _jobs[job_id].update(
            {
                "step": "external_running",
                "progress": 10,
                "external_job_id": external_job_id,
            }
        )
        deadline = time.time() + int(os.getenv("APPROACH_EXTERNAL_TIMEOUT_SECONDS", "3600"))
        while time.time() < deadline:
            status_response = requests.get(
                _external_url(approach, f"/api/status/{external_job_id}"),
                timeout=30,
            )
            status_response.raise_for_status()
            external_status = status_response.json()
            mirrored = dict(external_status)
            mirrored.update(
                {
                    "approach": approach,
                    "approach_label": label,
                    "external_base_url": base_url,
                    "external_job_id": external_job_id,
                    "external_path": cfg.get("path"),
                }
            )
            status = str(external_status.get("status") or "pending").lower()
            if status in TERMINAL_JOB_STATUSES:
                _jobs[job_id] = mirrored
                return
            mirrored.setdefault("status", "pending")
            mirrored.setdefault("step", "external_running")
            mirrored.setdefault("progress", 50)
            _jobs[job_id] = mirrored
            time.sleep(float(os.getenv("APPROACH_EXTERNAL_POLL_SECONDS", "2")))
        raise TimeoutError(f"Approche {approach} timeout en attendant le job externe {external_job_id}")
    except Exception as exc:
        logger.exception("_run_external_approach_job exception approach=%s: %s", approach, exc)
        _jobs[job_id] = {
            "status": "error",
            "error": (
                f"Approche {approach} indisponible. Lancez son backend sur {base_url} "
                f"puis relancez l'extraction. Detail: {exc}"
            ),
            "approach": approach,
            "approach_label": label,
            "external_base_url": base_url,
            "external_path": cfg.get("path"),
        }


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
    Télécharge le PDF si absent, puis exécute le pipeline page-aware + Vision (cache sous `output/financial_extraction_debug/`).
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
    _jobs[job_id] = {
        "status": "pending",
        "step": "scraping",
        "steps_done": [],
        "progress": 0,
        "approach": APPROACH_A,
        "approach_label": APPROACHES[APPROACH_A]["label"],
    }
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
    force_recrop = bool(body.force_recrop) and not force_page
    approach = _normalize_approach(body.approach)
    if approach == APPROACH_A:
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
            force_recrop,
        )
    else:
        _executor.submit(_run_external_approach_job, job_id, approach, raw)
    return {
        "job_id": job_id,
        "type_rapport_used": type_rapport,
        "api_provider": api_provider,
        "approach": approach,
        "approach_label": APPROACHES[approach]["label"],
        "force_vision": force_vision,
        "force_page": force_page,
        "force_recrop": force_recrop,
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


@app.post("/api/benchmark/feedback", response_class=JSONResponse)
async def api_benchmark_feedback(request: Request):
    try:
        raw = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Corps JSON invalide : {exc}")
    try:
        body = BenchmarkFeedbackBody.model_validate(raw)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if body.job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job introuvable")
    job = _jobs[body.job_id]
    if not job.get("target_table") or not job.get("scope") or not job.get("sector"):
        raise HTTPException(status_code=400, detail="Job incompatible avec le benchmark financier")

    feedback = build_financial_benchmark_entry(
        source_job_id=body.job_id,
        job=job,
        validation_result="PASS" if body.verdict == "pass" else "FAIL",
        failure_reason=body.failure_reason,
    )
    if body.verdict == "pass":
        saved_path = _save_benchmark_case(feedback)
        message = "Case added to benchmark"
    else:
        saved_path = _save_benchmark_failure(feedback)
        message = "Failure saved for review"

    job["benchmark_feedback"] = {
        "verdict": body.verdict,
        "failure_reason": body.failure_reason or "",
        "path": str(saved_path),
        "notes": body.notes or "",
    }
    return JSONResponse(
        content={
            "success": True,
            "message": message,
            "verdict": body.verdict,
            "path": str(saved_path),
            "case": feedback,
        },
        media_type="application/json; charset=utf-8",
    )


def _save_benchmark_case(case: dict[str, Any]) -> Path:
    path = _API_ROOT / "data" / "financial_benchmark_cases.json"
    return save_financial_benchmark_entry(path, "cases", case)


def _save_benchmark_failure(case: dict[str, Any]) -> Path:
    path = _API_ROOT / "data" / "financial_benchmark_failures.json"
    return save_financial_benchmark_entry(path, "failures", case)


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
    external_job_id = job.get("external_job_id")
    approach = str(job.get("approach") or "")
    if external_job_id and approach in APPROACHES and approach != APPROACH_A:
        try:
            response = requests.get(
                _external_url(approach, f"/api/download/{external_job_id}"),
                timeout=120,
            )
            response.raise_for_status()
            content_type = response.headers.get(
                "content-type",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            filename = f"{job_id}_{approach}.xlsx"
            disposition = response.headers.get("content-disposition", "")
            if "filename=" in disposition:
                filename = disposition.split("filename=", 1)[1].strip().strip('"')
            return Response(
                content=response.content,
                media_type=content_type,
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"Telechargement externe impossible pour approche {approach}: {exc}",
            )
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
