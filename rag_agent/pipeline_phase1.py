"""
Pipeline Phase 1 : Scraping AMMC → RAG → Extraction → Excel.

Étapes :
  1. AMMCScraper.fetch() → télécharge le PDF (ou cache)
  2. build_index() → indexe le PDF dans le RAG existant
  3. find_page_for_table() → localise la page du tableau (RAG ou fallback keyword)
  4. ExtractionAgent.extract() → extrait le tableau (cascade LLM)
  5. Export Excel avec mise en forme

Usage:
    pipeline = Phase1Pipeline()
    result = pipeline.run(emetteur="agma", year=2024, tableau="BILAN ACTIF")
    # result.df, result.excel_path
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

# structlog pour logs structurés (JSON) — fallback sur logging standard si absent
try:
    import structlog  # type: ignore
    _slog = structlog.get_logger("pipeline_phase1")
    def _log_info(event: str, **kw):
        _slog.info(event, **kw)
    def _log_warning(event: str, **kw):
        _slog.warning(event, **kw)
    def _log_error(event: str, **kw):
        _slog.error(event, **kw)
except ImportError:
    _logger = logging.getLogger("pipeline_phase1")
    def _log_info(event: str, **kw):
        _logger.info("%s %s", event, kw)
    def _log_warning(event: str, **kw):
        _logger.warning("%s %s", event, kw)
    def _log_error(event: str, **kw):
        _logger.error("%s %s", event, kw)

from .config import RAGConfig, ensure_dirs
from .extraction_agent import ExtractionAgent, find_page_for_table
from .page_locator import find_page_by_keywords
from .index_paths import (
    infer_index_dir_for_pdf,
    is_index_complete_for_pdf,
    normalize_report_code,
    resolve_index_dir_for_read,
)
from .ingest import build_index
from .scraper import AMMCScraper, DEFAULT_PDF_DIR, FetchResult, normalize_user_emetteur

# Dossier de sortie Excel
DEFAULT_XLSX_DIR = Path("outputs/xlsx")

# Bancaire & SDF — bilan consolidé (PCEC / IFRS banques) : bonus metadata & scan PDF
# (même esprit que les ancres fortes `bilan_*_pcec` du RAG dans extraction_agent.)
_BANK_CONSOL_BILAN_PASSIF_META: tuple[tuple[str, float], ...] = (
    ("banques centrales, tresor public, service des cheques postaux", 28.0),
    ("passifs financiers a la juste valeur par resultat sur option", 28.0),
    ("passifs financiers a la juste valeur par resultat", 26.0),
    ("passifs financiers detenus a des fins de transactions", 28.0),
    ("instruments derives de couverture", 26.0),
    ("dettes envers les etablissements de credit et assimiles", 36.0),
    ("depots de la clientele", 34.0),
    ("dettes envers la clientele", 32.0),
    ("dettes envers la clientele sur produits participatifs", 34.0),
    ("titres de creance emis", 36.0),
    ("autres passifs", 30.0),
    ("provisions pour risques et charges", 30.0),
    ("provisions reglementees", 30.0),
    ("subventions, fonds publics affectes et fonds speciaux de garantie", 34.0),
    ("dettes subordonnees", 30.0),
    ("depots d'investissement recus", 34.0),
    ("ecarts de reevaluation", 30.0),
    ("reserves et primes liees au capital", 30.0),
    ("capital", 26.0),
    ("actionnaires capital non verse", 28.0),
    ("report a nouveau", 28.0),
    ("resultats nets en instance d'affectation", 30.0),
    ("ecart de reevaluation passif des portefeuilles couverts en taux", 26.0),
    ("comptes de regularisation et autres passifs", 32.0),
    ("dettes liees aux actifs non courants destines a etre cedes", 32.0),
    ("passifs des contrats d'assurance", 30.0),
    ("provisions", 24.0),
    ("subventions et fonds assimiles", 30.0),
    ("dettes subordonnees et fonds speciaux de garantie", 34.0),
    ("capitaux propres", 26.0),
    ("passifs d'impot exigible", 22.0),
    ("passifs d'impots differes", 22.0),
    ("reserves consolidees", 28.0),
    ("gains et pertes comptabilises directement en capitaux propres", 28.0),
    ("resultat net de l'exercice", 30.0),
    ("total du passif", 50.0),
    ("total passif", 42.0),
)
_BANK_CONSOL_BILAN_ACTIF_META: tuple[tuple[str, float], ...] = (
    ("valeurs en caisse, banques centrales, tresor public, ccp", 32.0),
    ("valeurs en caisse, banques centrales, tresor public, service des cheques postaux", 32.0),
    # Variante OCR fréquente (ex: "aleurs en caisse...")
    ("aleurs en caisse, banques centrales, tresor public, service des cheques postaux", 28.0),
    ("creances sur les etablissements de credit et assimiles", 36.0),
    ("creances sur la clientele", 34.0),
    ("creances acquises par affacturage", 32.0),
    ("titres de transaction et de placement", 32.0),
    ("autres actifs", 30.0),
    ("titres d'investissement", 32.0),
    ("titres de participation et emplois assimiles", 32.0),
    ("creances subordonnees", 30.0),
    ("depots d'investissement places", 32.0),
    ("depot d'investissement place", 30.0),
    ("immobilisations donnees en credit-bail et en location", 34.0),
    ("immobilisations donnees en ijara", 34.0),
    ("immobilisations incorporelles", 26.0),
    ("immobilisations corporelles", 26.0),
    ("total de l'actif", 50.0),
    ("actifs financiers a la juste valeur par resultat", 28.0),
    ("actifs financiers a la juste valeur par capitaux propres", 28.0),
    ("titres au cout amorti", 26.0),
    (
        "prets et creances sur les etablissements de credit et assimiles, au cout amorti",
        30.0,
    ),
    ("prets et creances sur la clientele, au cout amorti", 28.0),
    ("actifs d'impot exigible", 22.0),
    ("actifs d'impot differe", 22.0),
    ("actifs d'impots differes", 22.0),
    ("immobilisations corporelles", 20.0),
    ("immobilisations incorporelles", 20.0),
    ("ecart d'acquisition", 24.0),
    ("total actif", 36.0),
    ("creances sur les etablissements de credit et assimiles", 22.0),
    ("banques centrales, tresor public, service des cheques postaux", 18.0),
)
_BANK_CONSOL_BILAN_PASSIF_SCAN: tuple[tuple[str, int], ...] = tuple(
    (k, int(w)) for k, w in _BANK_CONSOL_BILAN_PASSIF_META
)
_BANK_CONSOL_BILAN_ACTIF_SCAN: tuple[tuple[str, int], ...] = tuple(
    (k, int(w)) for k, w in _BANK_CONSOL_BILAN_ACTIF_META
)

# Bancaire & SDF — CPC comptes sociaux (PCEC) : bonus metadata & scan PDF
# (mots-clés "à fort signal" pour trouver la page du tableau principal CPC)
_BANK_SOCIAUX_CPC_PCEC_META: tuple[tuple[str, float], ...] = (
    ("produits d'exploitation bancaire", 30.0),
    ("produits d exploitation bancaire", 24.0),
    ("charges d'exploitation bancaire", 30.0),
    ("charges d exploitation bancaire", 24.0),
    ("produit net bancaire", 32.0),
    ("charges generales d'exploitation", 22.0),
    ("dotations aux provisions et pertes sur creances irrecouvrables", 28.0),
    ("reprises de provisions et recuperations sur creances amorties", 28.0),
    ("resultat courant", 22.0),
    ("resultat avant impots", 22.0),
    ("resultat net de l'exercice", 26.0),
    # Lignes d'intérêts (discriminantes pour les banques)
    ("interets, remunerations et produits assimiles sur operations avec les etablissements de credit", 30.0),
    ("interets et produits assimiles sur operations avec les etablissements de credit", 28.0),
    ("interets et produits assimiles sur operations avec la clientele", 28.0),
    ("interets et produits assimiles sur titres de creance", 28.0),
)
_BANK_SOCIAUX_CPC_PCEC_SCAN: tuple[tuple[str, int], ...] = tuple(
    (k, int(w)) for k, w in _BANK_SOCIAUX_CPC_PCEC_META
)

# Bancaire & SDF — CPC comptes consolidés IFRS
# Le tableau peut etre intitule "Compte de resultat consolide" au lieu de CPC.
_BANK_CONSOL_CPC_IFRS_META: tuple[tuple[str, float], ...] = (
    ("compte de resultat consolide", 52.0),
    ("compte de resultats consolide", 48.0),
    ("interets et produits assimiles", 34.0),
    ("interets et charges assimiles", 34.0),
    ("marge d'interet", 40.0),
    ("commissions (produits)", 30.0),
    ("commissions (charges)", 30.0),
    ("marge sur commissions", 40.0),
    ("gains ou pertes nets sur instruments financiers a la juste valeur par resultat", 34.0),
    ("gains ou pertes nets des instruments financiers a la juste valeur par capitaux propres", 34.0),
    ("produits des autres activites", 24.0),
    ("charges des autres activites", 24.0),
    ("produits nets des activites d'assurance", 30.0),
    ("produit net bancaire", 42.0),
    ("charges generales d'exploitation", 28.0),
    ("resultat brut d'exploitation", 36.0),
    ("cout du risque de credit", 34.0),
    ("resultat d'exploitation", 32.0),
    ("resultat avant impots", 30.0),
    ("impots sur les benefices", 26.0),
    ("resultat net", 28.0),
    ("interets minoritaires", 26.0),
    ("resultat net part du groupe", 40.0),
    ("resultat de base par action", 30.0),
    ("resultat dilue par action", 30.0),
)

# Assurance — bilan actif (comptes sociaux) : bonus metadata & scan PDF
# (mots-clés très discriminants pour éviter de confondre Assurance vs CGNC/PCEC)
_ASSURANCE_SOCIAUX_BILAN_ACTIF_META: tuple[tuple[str, float], ...] = (
    ("placements affectes aux operations d assurance", 42.0),
    ("placements affectes aux operations dassurance", 42.0),
    ("part des cessionnaires dans les provisions techniques", 46.0),
    ("provisions techniques", 18.0),
    ("immobilisation en non-valeurs", 28.0),
    ("immobilisations en non-valeur", 28.0),
    ("immobilisations incorporelles", 26.0),
    ("immobilisations corporelles", 26.0),
    (
        "immobilisations financieres autres que placements",
        24.0,
    ),
    (
        "immobilisations financieres (autres que placements)",
        24.0,
    ),
    ("ecarts de conversion - actif", 22.0),
    ("ecarts de conversion -actif", 22.0),
    ("ecarts de conversion -actif elements circulants", 22.0),
    ("actif circulant hors tresorerie", 24.0),
    ("actif circulant (hors tresorerie)", 24.0),
    ("creances de l actif circulant", 18.0),
    ("creances de l'actif circulant", 18.0),
    ("titres et valeurs de placement", 18.0),
    ("non affectes aux op d assurance", 26.0),
    ("non affectes aux opdassurance", 26.0),
    ("non affectes aux op dassurance", 26.0),
    ("non affectes aux op", 12.0),
    ("tresorerie-actif", 30.0),
    ("tresorerie", 18.0),
    ("total general", 34.0),
)
_ASSURANCE_SOCIAUX_BILAN_ACTIF_SCAN: tuple[tuple[str, int], ...] = tuple(
    (k, int(w)) for k, w in _ASSURANCE_SOCIAUX_BILAN_ACTIF_META
)

# Assurance — bilan passif (comptes sociaux) : bonus metadata & scan PDF
_ASSURANCE_SOCIAUX_BILAN_PASSIF_META: tuple[tuple[str, float], ...] = (
    ("financement permanent", 30.0),
    ("capitaux propres", 26.0),
    ("capitaux propres assimiles", 28.0),
    ("dettes de financement", 26.0),
    ("provisions durables pour risques et charges", 30.0),
    ("provisions techniques brutes", 40.0),
    ("ecarts de conversion -passif", 22.0),
    ("ecarts de conversion - passif", 22.0),
    ("passif circulant (hors tresorerie)", 24.0),
    ("passif circulant hors tresorerie", 24.0),
    ("dettes pour especes remises par les cessionnaires", 34.0),
    ("dettes de passif circulant", 26.0),
    ("autres provisions pour risques et charges", 24.0),
    ("ecarts de conversion -passif (elements circulants)", 26.0),
    ("ecarts de conversion -passif elements circulants", 26.0),
    ("tresorerie-passif", 28.0),
    ("tresorerie", 18.0),
    ("total general", 34.0),
)
_ASSURANCE_SOCIAUX_BILAN_PASSIF_SCAN: tuple[tuple[str, int], ...] = tuple(
    (k, int(w)) for k, w in _ASSURANCE_SOCIAUX_BILAN_PASSIF_META
)

# Assurance — bilan passif (comptes consolidés) : bonus metadata & scan PDF
_ASSURANCE_CONSOL_BILAN_PASSIF_META: tuple[tuple[str, float], ...] = (
    ("capital", 20.0),
    ("reserves consolidees", 28.0),
    ("resultat consolide", 30.0),
    ("capitaux propres de l'ensemble consolide", 34.0),
    ("capitaux propres de l ensemble consolide", 34.0),
    ("dont : capitaux propres part du groupe", 34.0),
    ("capitaux propres part du groupe", 26.0),
    ("interets minoritaires", 24.0),
    ("dettes de financement", 24.0),
    ("provisions techniques", 26.0),
    ("passif circulant", 22.0),
    ("dettes pour les especes remises par les cessionnaires", 30.0),
    (
        "cessionnaires, cedants coassureurs et comptes rattaches crediteurs",
        34.0,
    ),
    ("assures, intermediaires et comptes rattaches crediteurs", 34.0),
    ("autres dettes du passif circulant", 28.0),
    ("tresorerie - passif", 24.0),
)
_ASSURANCE_CONSOL_BILAN_PASSIF_SCAN: tuple[tuple[str, int], ...] = tuple(
    (k, int(w)) for k, w in _ASSURANCE_CONSOL_BILAN_PASSIF_META
)

# Assurance — CPC (comptes consolidés) : bonus metadata & scan PDF
_ASSURANCE_CONSOL_CPC_META: tuple[tuple[str, float], ...] = (
    ("compte technique assurance vie", 34.0),
    ("primes emises brutes", 30.0),
    ("primes emises cedees", 30.0),
    ("produits techniques d'exploitation", 26.0),
    ("prestations et frais", 22.0),
    ("prestations et frais cedes", 26.0),
    ("charges techniques d'exploitation", 26.0),
    ("produits des placements affectes aux operations d'assurance", 30.0),
    ("charges des placements affectes aux operations d'assurance", 30.0),
    ("resultat technique vie", 32.0),
    ("compte technique assurance non vie", 34.0),
    ("variation des provisions pour primes non acquises brutes", 28.0),
    ("variation des provisions pour primes non acquises cedees", 28.0),
    ("resultat technique non vie", 32.0),
    ("resultat technique (c = a + b)", 34.0),
    ("compte non technique", 32.0),
    ("produits non techniques courants", 28.0),
    ("charges non techniques courantes", 28.0),
    ("resultat non technique courant", 30.0),
    ("produits non techniques non courants", 28.0),
    ("charges non techniques non courantes", 28.0),
    ("resultat non technique non courant", 30.0),
    ("resultat non technique (d)", 32.0),
    ("resultat avant impot (c + d)", 34.0),
    ("impot sur le resultat", 26.0),
    ("dotations d'amortissement des ecarts d'acquisition", 30.0),
    ("quote-part des societes mises en equivalence", 28.0),
    ("resultat net", 24.0),
)
_ASSURANCE_CONSOL_CPC_SCAN: tuple[tuple[str, int], ...] = tuple(
    (k, int(w)) for k, w in _ASSURANCE_CONSOL_CPC_META
)

# Contexte thread-local : type_rapport pour déduire le dossier d'index (annuel vs s1)
# quand le nom du PDF n'a pas de suffixe _annuel / _s1.
_type_rapport_tls = threading.local()


def _tls_get_type_rapport() -> Optional[str]:
    return getattr(_type_rapport_tls, "value", None)


def _tls_set_type_rapport(v: Optional[str]) -> None:
    _type_rapport_tls.value = v


@dataclass
class Phase1Result:
    """Résultat du pipeline Phase 1."""

    success: bool
    df: pd.DataFrame
    excel_path: Optional[Path] = None
    page_num: Optional[int] = None
    method: str = ""
    confidence: float = 0.0
    error: Optional[str] = None
    pdf_from_cache: bool = False  # True si le PDF était déjà dans data/pdfs
    extraction_warnings: Optional[List[str]] = None
    fallback_used: bool = False
    # Champs observabilité 2025
    completeness_score: float = 0.0
    missing_anchors: Optional[List[str]] = None
    extraction_strategy_used: str = ""


class Phase1Pipeline:
    """
    Pipeline bout en bout : scraper → indexation → localisation page → extraction → Excel.
    """

    def __init__(
        self,
        pdf_dir: Optional[Path] = None,
        xlsx_dir: Optional[Path] = None,
        cfg: Optional[RAGConfig] = None,
    ):
        self.pdf_dir = pdf_dir or DEFAULT_PDF_DIR
        self.xlsx_dir = xlsx_dir or DEFAULT_XLSX_DIR
        self.cfg = cfg or RAGConfig()
        self.scraper = AMMCScraper(pdf_dir=self.pdf_dir)
        self.extraction_agent = ExtractionAgent(use_llm=True)

    def _resolve_table_name(self, tableau: str, type_comptes: str) -> str:
        """
        Retourne le libellé complet du tableau pour la recherche RAG et l'extraction.
        Ex: "BILAN ACTIF" + type_comptes="sociaux" -> "BILAN ACTIF COMPTES SOCIAUX"
        Ex: "BILAN ACTIF" + type_comptes="consolides" -> "BILAN ACTIF COMPTES CONSOLIDES"
        """
        t = tableau.strip()
        tc = (type_comptes or "").strip().lower()
        if tc == "sociaux":
            if "BILAN ACTIF" in t.upper() and "COMPTES SOCIAUX" not in t.upper():
                return "BILAN ACTIF COMPTES SOCIAUX"
            if "BILAN PASSIF" in t.upper() and "COMPTES SOCIAUX" not in t.upper():
                return "BILAN PASSIF COMPTES SOCIAUX"
            if (
                "COMPTE DE PRODUITS ET CHARGES" in t.upper()
                or "COMPTES DE PRODUITS ET CHARGES" in t.upper()
                or "CPC" in t.upper()
                or ("PRODUITS ET CHARGES" in t.upper() and "COMPTE" in t.upper())
                or "COMPTE DE RÉSULTAT" in t.upper()
                or "COMPTE DE RESULTAT" in t.upper()
            ) and "COMPTES SOCIAUX" not in t.upper():
                return "COMPTE DE PRODUITS ET CHARGES COMPTES SOCIAUX"
        if tc == "consolides":
            if "BILAN ACTIF" in t.upper() and "CONSOLID" not in t.upper():
                return "BILAN ACTIF COMPTES CONSOLIDES"
            if "BILAN PASSIF" in t.upper() and "CONSOLID" not in t.upper():
                return "BILAN PASSIF COMPTES CONSOLIDES"
            if (
                "COMPTE DE PRODUITS ET CHARGES" in t.upper()
                or "CPC" in t.upper()
                or "COMPTE DE RÉSULTAT" in t.upper()
                or "COMPTE DE RESULTAT" in t.upper()
            ) and "CONSOLID" not in t.upper():
                return "COMPTE DE PRODUITS ET CHARGES COMPTES CONSOLIDES"
        return t

    def run(
        self,
        emetteur: str,
        year: int,
        tableau: str,
        type_comptes: str = "sociaux",
        type_rapport: str = "annuel",
        search_mode: str = "ammc",
        api_provider: str = "groq",
    ) -> Phase1Result:
        """
        Exécute le pipeline complet.
        tableau : "BILAN ACTIF", "BILAN PASSIF", "COMPTE DE PRODUITS ET CHARGES", etc.
        Avec type_comptes="sociaux", le libellé complet (ex. BILAN ACTIF COMPTES SOCIAUX) est utilisé.
        """
        ensure_dirs(self.cfg)
        self.xlsx_dir.mkdir(parents=True, exist_ok=True)

        emetteur = normalize_user_emetteur(emetteur)
        _tls_set_type_rapport(normalize_report_code(type_rapport))
        try:
            # Libellé complet pour cibler la bonne page (ex. BILAN ACTIF COMPTES SOCIAUX)
            table_name = self._resolve_table_name(tableau, type_comptes)

            # 1. PDF dans le dossier Émetteur / cache : addoha_2024_s1.pdf ; index data/index/addoha/2024/s1/index/
            fetch_result = self.scraper.fetch(
                emetteur=emetteur,
                year=year,
                type_comptes=type_comptes,
                type_rapport=type_rapport,
                search_mode=search_mode,
            )
            if not fetch_result.success or not fetch_result.pdf_path:
                return Phase1Result(
                    success=False,
                    df=pd.DataFrame(),
                    error=fetch_result.error or "Échec téléchargement PDF",
                    extraction_warnings=[],
                )
            pdf_path = fetch_result.pdf_path
            doc_id = pdf_path.name
            pdf_from_cache = fetch_result.from_cache

            # 2. Localiser la page du tableau via scan déterministe (PyMuPDF + scoring).
            page_loc_method = "keyword_scan"
            page_num = self._find_page_keyword_scan(str(pdf_path), table_name, type_comptes)

            if page_num is None:
                return Phase1Result(
                    success=False,
                    df=pd.DataFrame(),
                    error="Page du tableau introuvable (keyword scan).",
                    pdf_from_cache=pdf_from_cache,
                    extraction_warnings=[],
                )

            # 4. Extraire le tableau (avec libellé complet ex. BILAN ACTIF COMPTES SOCIAUX)
            result = self.extraction_agent.extract(
                pdf_path=str(pdf_path),
                page_num=page_num,
                table_name=table_name,
                api_provider=api_provider,
                type_comptes=type_comptes,
            )
            if result.df is None or result.df.empty:
                return Phase1Result(
                    success=False,
                    df=pd.DataFrame(),
                    page_num=page_num,
                    method=result.method,
                    confidence=result.confidence,
                    error="Aucune donnée extraite.",
                    pdf_from_cache=pdf_from_cache,
                    extraction_warnings=result.warnings or [],
                    fallback_used=("fallback" in (result.method or "").lower()),
                )

            # 5. Export Excel (nom sans espace pour URL de téléchargement API)
            safe_name = tableau.lower().replace(" ", "_").replace("/", "_")[:31]
            em_safe = emetteur.lower().strip().replace(" ", "_")
            excel_name = f"{em_safe}_{year}_{safe_name}.xlsx"
            excel_path = self.xlsx_dir / excel_name
            self._export_excel(result.df, excel_path, sheet_name=safe_name[:31])

            return Phase1Result(
                success=result.success,
                df=result.df,
                excel_path=excel_path,
                page_num=page_num,
                method=f"{page_loc_method}::{result.method}",
                confidence=result.confidence,
                pdf_from_cache=pdf_from_cache,
                extraction_warnings=result.warnings or [],
                fallback_used=getattr(result, "fallback_used", "fallback" in (result.method or "").lower()),
                completeness_score=getattr(result, "completeness_score", 0.0),
                missing_anchors=getattr(result, "missing_anchors", []),
                extraction_strategy_used=getattr(result, "extraction_strategy_used", result.method),
            )
        finally:
            _tls_set_type_rapport(None)

    def _find_page_keyword_scan(
        self,
        pdf_path: str,
        tableau: str,
        type_comptes: str,
    ) -> Optional[int]:
        """
        Localise la page via page_locator.find_page_by_keywords (PyMuPDF, 100% déterministe).
        Remplace le mode RAG pour la localisation de page.
        """
        t = (tableau or "").upper()
        if "ACTIF" in t and "PASSIF" not in t:
            table_type = "ACTIF"
        elif "PASSIF" in t:
            table_type = "PASSIF"
        elif "CPC" in t or "PRODUITS ET CHARGES" in t or "COMPTE DE RESULTAT" in t or "COMPTE DE RÉSULTAT" in t:
            table_type = "CPC"
        else:
            return None

        tc = (type_comptes or "").lower()
        if tc in ("consolides", "consolidés", "consolide", "consolidé"):
            report_type = "consolidés"
        elif tc == "sociaux":
            report_type = "sociaux"
        else:
            report_type = "auto"

        result = find_page_by_keywords(pdf_path, table_type, report_type)  # type: ignore
        return result.best_page

    def _find_page_exact_deterministic(
        self,
        pdf_path: str,
        tableau: str,
    ) -> Optional[int]:
        """
        Scan pymupdf + scoring (_find_best_page_by_full_scan) puis validation stricte
        (_page_matches_table). Retourne une page 1-indexée uniquement si exacte.
        """
        t = (tableau or "").upper()
        is_financial = (
            "BILAN ACTIF" in t
            or "BILAN PASSIF" in t
            or "COMPTE DE PRODUITS ET CHARGES" in t
            or "CPC" in t
        )
        if not is_financial:
            return None

        scanned = self._find_best_page_by_full_scan(pdf_path, tableau)
        if scanned is not None and self._page_matches_table(pdf_path, scanned, tableau):
            return scanned
        return None

    def _index_is_ready_for_doc(self, doc_id: str) -> bool:
        """
        Reuse: l'index existe et contient au moins un chunk pour ce doc_id.
        Cherche d'abord data/index/{emetteur}/{year}/{annuel|s1}/index/, puis l'ancien chemin sans type_rapport.
        """
        return is_index_complete_for_pdf(
            doc_id,
            self.cfg.index_dir,
            _tls_get_type_rapport(),
        )

    def _ensure_index_for_doc(self, pdf_path: str, doc_id: str) -> None:
        """
        Si l'index ne contient pas le doc_id, rebuild l'index uniquement pour ce PDF.
        Écriture toujours sous .../{year}/{annuel|s1}/index/.
        """
        if self._index_is_ready_for_doc(doc_id):
            return
        index_dir = infer_index_dir_for_pdf(
            doc_id,
            self.cfg.index_dir,
            _tls_get_type_rapport(),
        )
        build_index(self.cfg, pdf_paths=[pdf_path], index_dir_override=index_dir)

    def _index_dir_for_read(self, pdf_name: str) -> str:
        """Dossier d'index pour lire metadata.json / RAG (nouveau chemin ou legacy)."""
        return resolve_index_dir_for_read(
            pdf_name,
            self.cfg.index_dir,
            _tls_get_type_rapport(),
        )

    def _find_page_exact_rag_and_fallback(
        self,
        pdf_path: str,
        doc_id: str,
        tableau: str,
    ) -> Optional[int]:
        """
        RAG (find_page_for_table) + validation stricte, puis fallback keyword_search + validation.
        """
        t = (tableau or "").lower()
        is_cpc_sociaux = (
            ("cpc" in t or "produits et charges" in t or "compte de produits et charges" in t)
            and ("sociaux" in t)
            and ("consol" not in t)
        )

        # CPC sociaux: privilégier la détection RAG directe (titre de page),
        # puis seulement l'index metadata si besoin.
        if is_cpc_sociaux:
            page = find_page_for_table(pdf_path, tableau, type_rapport=_tls_get_type_rapport())
            if page is not None and self._page_matches_table(pdf_path, page, tableau):
                return page
            index_page = self._index_metadata_page_search(doc_id, tableau)
            if index_page is not None and self._page_matches_table(pdf_path, index_page, tableau):
                return index_page
        else:
            # 0) Priorité à l'index déjà construit par émetteur/année (metadata.json)
            # pour localiser la page exacte quand les titres PDF sont mal extraits.
            index_page = self._index_metadata_page_search(doc_id, tableau)
            if index_page is not None and self._page_matches_table(pdf_path, index_page, tableau):
                return index_page
            page = find_page_for_table(pdf_path, tableau, type_rapport=_tls_get_type_rapport())
            if page is not None and self._page_matches_table(pdf_path, page, tableau):
                return page

        kw_page = self._keyword_search(doc_id, tableau)
        if kw_page is not None and self._page_matches_table(pdf_path, kw_page, tableau):
            return kw_page

        return None

    def _find_page_rag_only(
        self,
        pdf_path: str,
        doc_id: str,
        tableau: str,
    ) -> Optional[int]:
        """
        Localisation de page en mode RAG uniquement:
        1) index metadata par émetteur/année
        2) find_page_for_table (vectoriel/section hints)
        3) keyword fallback (metadata.json) si RAG ne passe pas la validation
        """
        t = (tableau or "").lower()
        is_cpc_sociaux = (
            ("cpc" in t or "produits et charges" in t or "compte de produits et charges" in t)
            and ("sociaux" in t)
            and ("consol" not in t)
        )

        index_page: Optional[int] = None
        page: Optional[int] = None
        if is_cpc_sociaux:
            page = find_page_for_table(pdf_path, tableau, type_rapport=_tls_get_type_rapport())
            if page is not None and self._page_matches_table(pdf_path, page, tableau):
                return page
            index_page = self._index_metadata_page_search(doc_id, tableau)
            if index_page is not None and self._page_matches_table(pdf_path, index_page, tableau):
                return index_page
        else:
            index_page = self._index_metadata_page_search(doc_id, tableau)
            if index_page is not None and self._page_matches_table(pdf_path, index_page, tableau):
                return index_page
            page = find_page_for_table(pdf_path, tableau, type_rapport=_tls_get_type_rapport())
            if page is not None and self._page_matches_table(pdf_path, page, tableau):
                return page

        kw_page = self._keyword_search(doc_id, tableau)
        if kw_page is not None and self._page_matches_table(pdf_path, kw_page, tableau):
            return kw_page

        # Dernier recours : retourner la meilleure page RAG sans validation stricte
        if page is not None:
            return page
        if index_page is not None:
            return index_page
        return kw_page

    def _target_hints_for_table(self, tableau: str) -> list[str]:
        """Section hints attendus selon le tableau demandé."""
        t = (tableau or "").lower().replace("\n", " ")
        is_actif = "actif" in t and "passif" not in t
        is_passif = "passif" in t
        is_cpc = (
            "cpc" in t
            or "compte de produits" in t
            or "produits et charges" in t
            or "compte de résultat" in t
            or "compte de resultat" in t
        )
        is_consol = ("consol" in t) or ("consolid" in t)
        if is_consol:
            if is_actif:
                return ["bilan_actif_ifrs", "bilan_actif_cgnc", "bilan_actif_pcec", "bilan_actif_assurance"]
            if is_passif:
                return ["bilan_passif_ifrs", "bilan_passif_cgnc", "bilan_passif_pcec", "bilan_passif_assurance"]
            if is_cpc:
                return ["cpc_ifrs", "cpc_pcec", "cpc_cgnc", "cpc_assurance"]
            return []
        if is_actif:
            return ["bilan_actif_cgnc", "bilan_actif_pcec", "bilan_actif_assurance"]
        if is_passif:
            return ["bilan_passif_cgnc", "bilan_passif_pcec", "bilan_passif_assurance"]
        if is_cpc:
            return ["cpc_cgnc", "cpc_pcec", "cpc_assurance"]
        return []

    def _index_metadata_page_search(self, doc_id: str, tableau: str) -> Optional[int]:
        """
        Recherche page via metadata.json (index par émetteur) avec score par page.
        Favorise les section_hints et ancres métier.
        """
        import json
        from collections import defaultdict

        index_dir = self._index_dir_for_read(doc_id)
        meta_path = os.path.join(index_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        doc_l = (doc_id or "").lower()
        t = (tableau or "").lower()
        is_actif = "actif" in t and "passif" not in t
        is_passif = "passif" in t
        is_cpc = "cpc" in t or "produits et charges" in t or "compte de résultat" in t or "compte de resultat" in t
        want_consolides = ("consol" in t) or ("consolid" in t)
        want_sociaux = ("sociaux" in t) and not want_consolides
        expected_hints = set(self._target_hints_for_table(tableau))

        # Si le même PDF contient "consolidés" puis "sociaux", détecter la frontière
        # de début de section sociaux pour éviter les confusions.
        sociaux_start_page: Optional[int] = None
        for m in meta:
            if (m.get("doc_id") or "").lower() != doc_l:
                continue
            c = (m.get("content") or "").lower()
            if (
                "comptes sociaux au" in c
                or "etats financiers sociaux" in c
                or "états financiers sociaux" in c
            ):
                try:
                    p0 = int(m.get("page"))
                except (TypeError, ValueError):
                    continue
                if sociaux_start_page is None or p0 < sociaux_start_page:
                    sociaux_start_page = p0

        def _meta_fold(s: str) -> str:
            return (
                s.replace("é", "e")
                .replace("è", "e")
                .replace("ê", "e")
                .replace("à", "a")
                .replace("ù", "u")
                .replace("ï", "i")
                .replace("î", "i")
                .replace("ç", "c")
            )

        page_scores: dict[int, float] = defaultdict(float)
        # PCEC (CIH, etc.) : pénalités par page — les annexes « ventilation » ont beaucoup de chunks
        # avec « total actif » ; un chunk seul peut avoir s<=0 et être ignoré, d’où correction ici.
        pages_ventilation_emplois: set[int] = set()
        pages_ventilation_actif_monnaies: set[int] = set()
        for m in meta:
            if (m.get("doc_id") or "").lower() != doc_l:
                continue
            try:
                p = int(m.get("page"))
            except (TypeError, ValueError):
                continue
            content = (m.get("content") or "").lower()
            content_fold = _meta_fold(content)
            hint = m.get("section_hint")

            s = 0.0
            if hint and hint in expected_hints:
                s += 120.0
            if "31/12" in content or "31-12" in content:
                s += 4.0
            if "en milliers" in content or "mad" in content:
                s += 3.0

            has_social_marker = ("comptes sociaux" in content) or ("etats financiers sociaux" in content)
            has_consol_marker = ("comptes consol" in content) or ("etats financiers consolides" in content) or ("consolid" in content)

            # Sépare strictement sociaux vs consolidés dans un même PDF.
            # Éliminer complètement les chunks de la mauvaise section (pas juste pénaliser).
            if want_consolides:
                if has_social_marker and not has_consol_marker:
                    continue  # Page clairement sociaux → ignorer pour une recherche consolidée
                if has_consol_marker:
                    s += 25.0
            if want_sociaux:
                if has_consol_marker and not has_social_marker:
                    continue  # Page clairement consolidée → ignorer pour une recherche sociaux
                if has_social_marker:
                    s += 25.0

            # Règle de séparation des sections quand elles coexistent dans un même PDF.
            if sociaux_start_page is not None:
                if want_consolides and p >= sociaux_start_page:
                    s -= 35.0
                if want_sociaux and p >= sociaux_start_page:
                    s += 18.0

            # Bonus fort pour les titres principaux du tableau (distingue le vrai tableau des annexes)
            # Les métadonnées peuvent avoir des encodages partiels (ex. "financi re" au lieu de "financière")
            # → utiliser des sous-chaînes courtes robustes à l'encodage.
            if ("etat de" in content and "situation" in content and "financ" in content):
                s += 80.0  # Titre principal IFRS (bilan consolidé)
            if "bilan consolide" in content or "bilan au 31" in content:
                s += 40.0  # Titre bilan PCEC/CGNC consolidé
            if ("etat du" in content and "sultat global" in content) or "compte de r" in content and "sultat consolid" in content:
                s += 80.0  # Titre principal CPC consolidé

            if is_actif:
                if "bilan actif" in content or "actif au 31" in content:
                    s += 25.0
                if "total de l'actif" in content or "total actif" in content:
                    s += 15.0
                if "bilan passif" in content:
                    s -= 20.0
                # État IFRS principal (Addoha RFS, etc.) : éviter de perdre contre une note annexe
                # étiquetée bilan_actif_ifrs (ex. « Note 3 : détail des comptes de situation financière »).
                if want_consolides:
                    if "etat de situation financiere" in content_fold or "etat de la situation financiere" in content_fold:
                        if "actif mad" in content_fold or "passif mad" in content_fold:
                            s += 95.0
                    if "detail des comptes de situation financiere" in content_fold:
                        s -= 110.0
                    if (
                        re.search(r"note\s+\d+\s*:", content_fold)
                        and "detail" in content_fold
                        and "situation financ" in content_fold
                    ):
                        s -= 85.0
                # Comptes sociaux : bilan de synthèse PCEC (ex. CIH p.21), pas annexes p.26
                if want_sociaux:
                    if has_social_marker and (
                        "bilan actif" in content
                        or "| bilan actif |" in content
                    ):
                        s += 118.0
                    if "ventilation des emplois" in content_fold:
                        pages_ventilation_emplois.add(p)
                    if (
                        "ventilation de l'actif" in content_fold
                        and "passif" in content_fold
                        and "monnaies" in content_fold
                    ):
                        pages_ventilation_actif_monnaies.add(p)
                    # Assurance (Bilan Actif - Comptes Sociaux) : bonus sur ancres discriminantes
                    for needle, w in _ASSURANCE_SOCIAUX_BILAN_ACTIF_META:
                        if needle in content_fold:
                            s += w
            elif is_passif:
                if "bilan passif" in content or "passif au 31" in content:
                    s += 25.0
                if "total du passif" in content or "total passif" in content:
                    s += 15.0
                if "bilan actif" in content:
                    s -= 20.0
                # Bonus pour les mots-clés structurels passif (IFRS / CGNC / PCEC / Assurance)
                if "capitaux propres - part du groupe" in content:
                    s += 20.0
                if "int\u00e9r\u00eats minoritaires" in content or "interets minoritaires" in content:
                    s += 15.0
                if "total passif non courant" in content or "dettes financi\u00e8res non courantes" in content:
                    s += 15.0
                if "d\u00e9p\u00f4ts de la client\u00e8le" in content or "depots de la clientele" in content:
                    s += 12.0
                if "provisions techniques" in content:
                    s += 12.0
                if want_sociaux:
                    # Assurance (Bilan Passif - Comptes Sociaux)
                    for needle, w in _ASSURANCE_SOCIAUX_BILAN_PASSIF_META:
                        if needle in content_fold:
                            s += w
                if want_consolides:
                    # Assurance (Bilan Passif - Comptes Consolidés)
                    for needle, w in _ASSURANCE_CONSOL_BILAN_PASSIF_META:
                        if needle in content_fold:
                            s += w
            elif is_cpc:
                cpc_title = (
                    "comptes de produits et charges" in content
                    or "compte de produits et charges" in content
                    or "cpc" in content
                )
                if cpc_title:
                    s += 25.0
                if "résultat net" in content or "resultat net" in content:
                    s += 10.0
                if "bilan actif" in content or "bilan passif" in content:
                    s -= 15.0
                # CPC sociaux: privilégier la page du tableau (titre + rubriques métier),
                # et pénaliser les pages narratives de commentaire.
                if want_sociaux:
                    cpc_struct_needles = (
                        "produits d'exploitation",
                        "charges d'exploitation",
                        "resultat d'exploitation",
                        "produits financiers",
                        "charges financieres",
                        "resultat financier",
                        "resultat courant",
                        "produits non courants",
                        "charges non courantes",
                        "resultat non courant",
                        "resultat avant impots",
                        "impot sur les societes",
                        "total des produits",
                    )
                    struct_hits = sum(1 for n in cpc_struct_needles if n in content_fold)
                    if cpc_title:
                        s += 70.0
                    if struct_hits >= 4:
                        s += 30.0 + (5.0 * min(struct_hits - 4, 4))
                    if "les regles de presentation" in content_fold and "indicateurs" in content_fold:
                        s -= 45.0
                if want_sociaux:
                    # Bonus CPC bancaire principal (PCEC) : maximise le choix de page exacte
                    for needle, w in _BANK_SOCIAUX_CPC_PCEC_META:
                        if needle in content_fold:
                            s += w
                if want_consolides:
                    # Bonus CPC assurance consolidé
                    for needle, w in _BANK_CONSOL_CPC_IFRS_META:
                        if needle in content_fold:
                            s += w
                    for needle, w in _ASSURANCE_CONSOL_CPC_META:
                        if needle in content_fold:
                            s += w

            if want_consolides and is_passif:
                for needle, w in _BANK_CONSOL_BILAN_PASSIF_META:
                    if needle in content_fold:
                        s += w
            if want_consolides and is_actif:
                for needle, w in _BANK_CONSOL_BILAN_ACTIF_META:
                    if needle in content_fold:
                        s += w

            if s > 0:
                page_scores[p] += s

        if is_actif and want_sociaux:
            for p in pages_ventilation_emplois:
                page_scores[p] -= 130.0
            for p in pages_ventilation_actif_monnaies:
                page_scores[p] -= 95.0

        if not page_scores:
            return None
        return max(page_scores.keys(), key=lambda p: page_scores[p])

    def _find_page(self, pdf_path: str, doc_id: str, tableau: str) -> Optional[int]:
        """
        Localise la page du tableau : RAG (find_page_for_table) puis fallback keyword.
        """
        # Pour les tableaux financiers (Bilan / CPC), le scan déterministe (titres/ancres)
        # est plus fiable que le RAG seul : on le fait en priorité.
        t = (tableau or "").upper()
        is_financial = (
            "BILAN ACTIF" in t
            or "BILAN PASSIF" in t
            or "COMPTE DE PRODUITS ET CHARGES" in t
            or "CPC" in t
        )
        if is_financial:
            scanned = self._find_best_page_by_full_scan(pdf_path, tableau)
            if scanned is not None and self._page_matches_table(pdf_path, scanned, tableau):
                return scanned

        # Sinon / en backup : RAG + validation
        page = find_page_for_table(pdf_path, tableau, type_rapport=_tls_get_type_rapport())
        if page is not None and self._page_matches_table(pdf_path, page, tableau):
            return page

        # Dernier recours : scan
        scanned = self._find_best_page_by_full_scan(pdf_path, tableau)
        if scanned is not None:
            return scanned

        # Ultime fallback : mots-clés
        kw_page = self._keyword_search(doc_id, tableau)
        if kw_page is not None and self._page_matches_table(pdf_path, kw_page, tableau):
            return kw_page
        return None

    def _scan_pdf_for_table_title(self, pdf_path: str, tableau: str) -> Optional[int]:
        """
        Localise la page en lisant directement les titres/ancres du PDF.
        Approche robuste pour éviter les confusions sémantiques (ex. tableaux voisins).
        """
        t = (tableau or "").lower()
        is_actif = "actif" in t and "passif" not in t
        is_passif = "passif" in t
        is_cpc = (
            "cpc" in t
            or "compte de produits et charges" in t
            or "comptes de produits et charges" in t
            or "produits et charges" in t
            or "compte de résultat" in t
            or "compte de resultat" in t
        )

        try:
            import pymupdf
        except Exception:
            return None

        doc = pymupdf.open(pdf_path)
        try:
            for i in range(len(doc)):
                text = (doc[i].get_text() or "").lower()
                if is_actif and "bilan actif" in text:
                    return i + 1
                if is_passif and ("bilan passif" in text or ("passif" in text and "bilan au 31" in text)):
                    return i + 1
                if is_cpc and (
                    "comptes de produits et charges" in text
                    or "compte de produits et charges" in text
                    or "compte de produits et de charges" in text
                ):
                    return i + 1
        finally:
            doc.close()
        return None

    def _find_best_page_by_full_scan(self, pdf_path: str, tableau: str) -> Optional[int]:
        """
        Scan exhaustif de toutes les pages avec scoring (positif + pénalités).
        Plus robuste que les matches exacts pour des libellés variés.
        """
        try:
            import pymupdf
        except Exception:
            return self._scan_pdf_for_table_title(pdf_path, tableau)

        t = (tableau or "").lower()
        is_actif = "actif" in t and "passif" not in t
        is_passif = "passif" in t
        is_cpc = (
            "cpc" in t
            or "compte de produits et charges" in t
            or "comptes de produits et charges" in t
            or "produits et charges" in t
            or "compte de résultat" in t
            or "compte de resultat" in t
        )
        is_consolide = "consolid" in t
        is_sociaux = "sociaux" in t or "social" in t

        if not (is_actif or is_passif or is_cpc):
            return None

        common_positive = [("31/12", 4), ("en mad", 4), ("en milliers de dirhams", 4)]
        if is_sociaux:
            # Certains PDF n'ont pas la chaîne exacte "comptes sociaux" mais ont "Etats Financiers SOCIAUX"
            common_positive.extend([("comptes sociaux", 12), ("etats financiers sociaux", 16)])
        if is_consolide:
            common_positive.extend([("comptes consol", 12), ("etats financiers consolides", 16)])

        if is_actif:
            positives = [
                ("bilan actif", 30),
                ("actif (en mad)", 18),
                ("actif", 8),
                ("immobilisations en non-valeur", 12),
                ("tresorerie - actif", 10),
                ("trésorerie - actif", 10),
                ("total general actif", 12),
                ("total de l'actif", 12),
                ("creances de l'actif circulant", 10),
                ("créances de l'actif circulant", 10),
            ] + common_positive
            if is_sociaux:
                # Assurance (comptes sociaux) : boosters sur libellés assurance
                positives = list(positives) + list(_ASSURANCE_SOCIAUX_BILAN_ACTIF_SCAN)
            if is_consolide:
                positives = list(positives) + list(_BANK_CONSOL_BILAN_ACTIF_SCAN)
            negatives = [
                ("passif", -10),
                ("bilan passif", -25),
                ("comptes de produits et charges", -20),
                ("montant brut debut", -20),
                ("montant brut début", -20),
                ("dotations de l'exercice", -18),
                ("cumul d'amortissement", -20),
            ]
        elif is_passif:
            positives = [
                ("bilan passif", 30),
                ("passif (en mad)", 18),
                ("passif", 8),
                ("capitaux propres", 12),
                ("dettes de financement", 12),
                ("dettes envers les etablissements de credit", 14),
                ("dettes envers les établissements de crédit", 14),
                ("total du passif", 12),
            ] + common_positive
            if is_sociaux:
                positives = list(positives) + list(_ASSURANCE_SOCIAUX_BILAN_PASSIF_SCAN)
            if is_consolide:
                positives = (
                    list(positives)
                    + list(_BANK_CONSOL_BILAN_PASSIF_SCAN)
                    + list(_ASSURANCE_CONSOL_BILAN_PASSIF_SCAN)
                )
            negatives = [
                ("bilan actif", -25),
                ("actif (en mad)", -18),
                ("immobilisations en non-valeur", -10),
                ("comptes de produits et charges", -20),
            ]
        else:
            positives = [
                ("comptes de produits et charges", 30),
                ("compte de produits et charges", 26),
                ("compte de produits et de charges", 24),
                ("resultat d'exploitation", 14),
                ("résultat d'exploitation", 14),
                ("produits d'exploitation", 14),
                ("charges d'exploitation", 14),
                ("resultat net", 12),
                ("résultat net", 12),
                # Favoriser la page "début CPC" : elle contient typiquement le bloc exploitation
                ("exercice du 01/01/2024", 8),
                ("production de l'exercice", 10),
                ("consommation de l'exercice", 10),
                # Mots-clés spécifiques au CPC bancaire (PCEC) — permettent de scorer la vraie page
                # même si elle ne contient pas de marqueur "comptes sociaux" explicite
                ("produits d'exploitation bancaire", 20),
                ("charges d'exploitation bancaire", 20),
                ("marge d'interet", 14),
                ("marge d interet", 14),
                ("resultat bancaire net", 16),
                ("résultat bancaire net", 16),
                ("produits bancaires", 10),
                ("charges bancaires", 10),
            ]
            if is_sociaux:
                positives = list(positives) + list(_BANK_SOCIAUX_CPC_PCEC_SCAN)
            if is_consolide:
                positives = list(positives) + list(_ASSURANCE_CONSOL_CPC_SCAN)
            positives += common_positive
            negatives = [
                ("bilan actif", -22),
                ("bilan passif", -22),
                ("capitaux propres", -10),
                ("dettes de financement", -10),
                # Pénaliser clairement les pages qui commencent le hors-CPC (CAF, soldes de gestion)
                ("capacit d'autofinancement", -40),
                ("capacite d'autofinancement", -40),
                ("autofinancement", -25),
                ("etat des soldes et de gestion", -35),
                ("etats des soldes et de gestion", -35),
                ("masses", -8),
                # Pénalité forte pour les rapports des commissaires aux comptes :
                # ces pages mentionnent "compte de produits et charges" dans le corps du texte
                # mais ne contiennent pas le tableau lui-même.
                ("rapport general des commissaires aux comptes", -90),
                ("rapport des commissaires aux comptes", -80),
                ("commissaires aux comptes", -50),
                ("audit des etats de synthese", -60),
                ("opinion sur les etats", -50),
            ]

        def norm(s: str) -> str:
            s = s.lower()
            s = (
                s.replace("é", "e")
                .replace("è", "e")
                .replace("ê", "e")
                .replace("à", "a")
                .replace("ù", "u")
                .replace("ï", "i")
                .replace("î", "i")
                .replace("ç", "c")
            )
            s = re.sub(r"\s+", " ", s)
            return s

        def has_any(text: str, needles: list[str]) -> bool:
            return any(n in text for n in needles)

        social_needles = ["comptes sociaux", "etats financiers sociaux"]
        consol_needles = ["comptes consol", "etats financiers consolides"]

        doc = pymupdf.open(pdf_path)
        try:
            best_page: Optional[int] = None
            best_score = -10**9
            for i in range(len(doc)):
                text = norm(doc[i].get_text() or "")
                score = 0
                # Garde-fou majeur : forcer la cohérence "sociaux" vs "consolides"
                if is_sociaux:
                    # Si la page est clairement "consolidés" => éliminer
                    if has_any(text, consol_needles) and not has_any(text, social_needles):
                        score -= 120
                    # Si la page ne contient pas un marqueur social explicite => pénaliser fort
                    if not has_any(text, social_needles):
                        score -= 70
                if is_consolide:
                    if has_any(text, social_needles) and not has_any(text, consol_needles):
                        score -= 120
                    if not has_any(text, consol_needles):
                        score -= 70
                for kw, pts in positives:
                    if norm(kw) in text:
                        score += pts
                for kw, pts in negatives:
                    if norm(kw) in text:
                        score += pts
                # Légère pénalité pour pages très narratives (peu de chiffres)
                words = text.split()
                if len(words) > 180:
                    digit_ratio = sum(1 for w in words if any(ch.isdigit() for ch in w)) / max(1, len(words))
                    if digit_ratio < 0.05:
                        score -= 5
                if score > best_score:
                    best_score = score
                    best_page = i + 1

            # seuil minimal pour éviter faux positifs
            return best_page if best_page is not None and best_score >= 18 else None
        finally:
            doc.close()

    def _page_matches_table(self, pdf_path: str, page_num: int, tableau: str) -> bool:
        """
        Vérifie qu'une page candidate contient bien les ancres minimales du tableau demandé.
        """
        try:
            import pymupdf
        except Exception:
            return True
        try:
            doc = pymupdf.open(pdf_path)
            if page_num < 1 or page_num > len(doc):
                doc.close()
                return False
            text = (doc[page_num - 1].get_text() or "").lower()
            doc.close()
        except Exception:
            return True

        def norm(s: str) -> str:
            s = (s or "").lower()
            s = (
                s.replace("é", "e")
                .replace("è", "e")
                .replace("ê", "e")
                .replace("à", "a")
                .replace("ù", "u")
                .replace("ï", "i")
                .replace("î", "i")
                .replace("ç", "c")
            )
            # Recolle les mots OCR en lettres séparées: "p r o d u i t s" -> "produits"
            s = re.sub(r"\b(?:[a-z]\s+){2,}[a-z]\b", lambda m: m.group(0).replace(" ", ""), s)
            return re.sub(r"\s+", " ", s)

        text_n = norm(text)
        t = norm(tableau or "")

        want_sociaux = ("sociaux" in t) and ("consol" not in t)
        want_consolides = ("consol" in t)

        social_ok_needles = ["comptes sociaux", "etats financiers sociaux"]
        consol_ok_needles = ["comptes consol", "etats financiers consolides"]
        has_social_marker = any(n in text_n for n in social_ok_needles)
        has_consol_marker = any(n in text_n for n in consol_ok_needles)

        def _metadata_has_section_hint(expected_hints: list[str]) -> bool:
            """
            Validation alternative (plus robuste) : se base sur metadata.json généré à l'ingestion.
            Utile quand pymupdf n'extrait pas correctement les titres/ancres du PDF (tables en images).
            """
            try:
                import json

                index_dir = self._index_dir_for_read(pdf_path)
                meta_path = os.path.join(index_dir, "metadata.json")
                if not os.path.isfile(meta_path):
                    return False

                doc_id_l = os.path.basename(pdf_path).lower()
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)

                for m in meta:
                    if (m.get("doc_id") or "").lower() != doc_id_l:
                        continue
                    try:
                        if int(m.get("page")) != int(page_num):
                            continue
                    except Exception:
                        continue
                    if m.get("section_hint") in expected_hints:
                        return True
            except Exception:
                return False
            return False

        if "bilan actif" in t or ("actif" in t and "passif" not in t):
            actif_structural_needles = (
                "total de l'actif",
                "total actif",
                "creances sur les etablissements de credit",
                "créances sur les établissements de crédit",
                "creances sur la clientele",
                "créances sur la clientèle",
                "valeurs en caisse, banques centrales",
                "titres de transaction",
                "immobilisations donnees en credit-bail",
            )
            # Les états consolidés IFRS utilisent souvent des libellés
            # proches de "etat de la situation financiere" / "actif au 31"
            # au lieu de "bilan actif".
            if want_consolides:
                # Rejeter les notes « détail » (p. ex. Addoha RFS p.27) : pas l'état consolidé principal (p.16).
                if "detail des comptes de situation financiere" in text_n:
                    if (
                        "etat de situation financiere" not in text_n
                        and "actif mad" not in text_n
                    ):
                        return False
                data_ok = (
                    ("bilan actif" in text_n)
                    or ("actif (en mad)" in text_n)
                    or ("etat de la situation financiere" in text_n)
                    or ("etat de situation financiere" in text_n)
                    or ("actif au 31" in text_n)
                    or ("bilan actif consolide" in text_n)
                    or ("actif mad" in text_n)
                )
                if not data_ok:
                    data_ok = _metadata_has_section_hint(self._target_hints_for_table(tableau))

                # Bancaire & SDF (format bilan actif consolidé bancaire) :
                # exiger une couverture élevée des ancres structurantes pour valider la "page exacte".
                bank_consol_actif_needles = (
                    "valeurs en caisse, banques centrales, tresor public, service des cheques postaux",
                    "creances sur les etablissements de credit et assimiles",
                    "creances sur la clientele",
                    "creances acquises par affacturage",
                    "titres de transaction et de placement",
                    "autres actifs",
                    "titres d'investissement",
                    "titres de participation et emplois assimiles",
                    "creances subordonnees",
                    "depots d'investissement places",
                    "immobilisations donnees en credit-bail et en location",
                    "immobilisations donnees en ijara",
                    "immobilisations incorporelles",
                    "immobilisations corporelles",
                    "total de l'actif",
                )
                bank_hits = sum(1 for n in bank_consol_actif_needles if n in text_n)
                # Détecter uniquement les pages à style bancaire (pour ne pas bloquer IFRS industriel).
                if bank_hits >= 5:
                    bank_total_ok = ("total de l'actif" in text_n) or ("total actif" in text_n)
                    data_ok = data_ok and bank_total_ok and (bank_hits >= 8)
            else:
                data_ok = ("bilan actif" in text_n) or ("actif (en mad)" in text_n)
            if not data_ok:
                data_ok = any(n in text_n for n in actif_structural_needles)
            if want_sociaux:
                # Annexes PCEC (CIH p.26) : « ventilation des emplois… », pas le bilan récap p.21
                if "ventilation des emplois" in text_n and "ressources" in text_n:
                    return False
                if (
                    "ventilation de l'actif" in text_n
                    and "passif" in text_n
                    and "monnaies" in text_n
                ):
                    return False
                # Garde-fou anti-confusion: si page clairement consolidée, refuser.
                if has_consol_marker and not has_social_marker:
                    return False
                # Si aucun marqueur de type de comptes n'est présent dans le texte extrait,
                # ne pas rejeter la page sur ce seul critère.
                if (not has_social_marker) and (not has_consol_marker):
                    return data_ok
                return data_ok and (has_social_marker or _metadata_has_section_hint(self._target_hints_for_table(tableau)))
            if want_consolides:
                # Garde-fou anti-confusion: si page clairement sociale, refuser.
                if has_social_marker and not has_consol_marker:
                    return False
                if (not has_social_marker) and (not has_consol_marker):
                    return data_ok
                return data_ok and (has_consol_marker or _metadata_has_section_hint(self._target_hints_for_table(tableau)))
            return data_ok

        if "bilan passif" in t or "passif" in t:
            if want_consolides:
                # Mots-clés structurels du passif (normalisés sans accents, car text_n est normalisé)
                _passif_structural_needles = (
                    "capitaux propres - part du groupe",
                    "interets minoritaires",
                    "total passif non courant",
                    "total passif courant",
                    "dettes financieres non courantes",
                    "dettes de financement",
                    "total du passif",
                    "total passif",
                    "depots de la clientele",
                    "provisions techniques",
                )
                data_ok = (
                    ("bilan passif" in text_n)
                    or ("bilan au 31" in text_n and "passif" in text_n)
                    or ("etat de la situation financiere" in text_n)
                    or ("passif au 31" in text_n)
                    or ("bilan passif consolide" in text_n)
                    or any(n in text_n for n in _passif_structural_needles)
                )
                if not data_ok:
                    data_ok = _metadata_has_section_hint(self._target_hints_for_table(tableau))

                # Bancaire & SDF (format bilan passif consolidé bancaire) :
                # exiger une couverture élevée des ancres structurantes pour valider la "page exacte".
                bank_consol_passif_needles = (
                    "banques centrales, tresor public, service des cheques postaux",
                    "passifs financiers a la juste valeur par resultat",
                    "dettes envers les etablissements de credit et assimiles",
                    "dettes envers la clientele",
                    "titres de creance emis",
                    "ecart de reevaluation passif des portefeuilles couverts en taux",
                    "passifs d'impot exigible",
                    "passifs d'impots differes",
                    "comptes de regularisation et autres passifs",
                    "dettes liees aux actifs non courants destines a etre cedes",
                    "passifs des contrats d'assurance",
                    "provisions",
                    "subventions et fonds assimiles",
                    "dettes subordonnees et fonds speciaux de garantie",
                    "capitaux propres",
                    "autres passifs",
                    "reserves consolidees",
                    "gains et pertes comptabilises directement en capitaux propres",
                    "resultat net de l'exercice",
                    "total du passif",
                )
                bank_hits = sum(1 for n in bank_consol_passif_needles if n in text_n)
                if bank_hits >= 8:
                    bank_total_ok = ("total du passif" in text_n) or ("total passif" in text_n)
                    data_ok = data_ok and bank_total_ok and (bank_hits >= 12)
            else:
                # Format PCEC : actif et passif sur la même page (ex. "bilan au 31" + "passif" dans le corps)
                data_ok = (
                    ("bilan passif" in text_n)
                    or ("bilan au 31" in text_n and "passif" in text_n)
                    or ("bilan actif" in text_n and "passif" in text_n)
                )
            if want_sociaux:
                if has_consol_marker and not has_social_marker:
                    return False
                if (not has_social_marker) and (not has_consol_marker):
                    return data_ok
                return data_ok and (has_social_marker or _metadata_has_section_hint(self._target_hints_for_table(tableau)))
            if want_consolides:
                if has_social_marker and not has_consol_marker:
                    return False
                if (not has_social_marker) and (not has_consol_marker):
                    return data_ok
                return data_ok and (has_consol_marker or _metadata_has_section_hint(self._target_hints_for_table(tableau)))
            return data_ok

        if (
            "cpc" in t
            or "compte de produits et charges" in t
            or "comptes de produits et charges" in t
            or "produits et charges" in t
            or "compte de résultat" in t
            or "compte de resultat" in t
        ):
            # Rejet immédiat des pages de rapport des commissaires aux comptes :
            # elles mentionnent "compte de produits et charges" dans le corps du texte
            # mais ne contiennent pas le tableau CPC lui-même.
            _audit_markers = (
                "rapport general des commissaires aux comptes",
                "rapport des commissaires aux comptes",
                "audit des etats de synthese",
                "opinion sur les etats de synthese",
            )
            if any(m in text_n for m in _audit_markers):
                return False

            cpc_title_ok = (
                "comptes de produits et charges" in text_n
                or "compte de produits et charges" in text_n
                or "cpc" in text_n
            )
            cpc_struct_needles = (
                "produits d exploitation",
                "charges d exploitation",
                "resultat d exploitation",
                "produits financiers",
                "charges financieres",
                "resultat financier",
                "resultat courant",
                "produits non courants",
                "charges non courantes",
                "resultat non courant",
                "resultat avant impots",
                "impot sur les societes",
                "resultat net",
                "total des produits",
            )
            struct_hits = sum(1 for n in cpc_struct_needles if n in text_n)
            data_ok = cpc_title_ok or struct_hits >= 4
            # Rejeter les pages narratives "synthèse des indicateurs" qui ne sont pas le tableau CPC.
            if "les regles de presentation" in text_n and "indicateurs" in text_n and not cpc_title_ok:
                return False
            if want_sociaux:
                if has_consol_marker and not has_social_marker:
                    return False
                if (not has_social_marker) and (not has_consol_marker):
                    return data_ok
                return data_ok and (has_social_marker or _metadata_has_section_hint(self._target_hints_for_table(tableau)))
            if want_consolides:
                if has_social_marker and not has_consol_marker:
                    return False
                if (not has_social_marker) and (not has_consol_marker):
                    return data_ok
                return data_ok and (has_consol_marker or _metadata_has_section_hint(self._target_hints_for_table(tableau)))
            return data_ok

        return True

    def _keyword_search(self, doc_id: str, tableau: str) -> Optional[int]:
        """
        Fallback quand le RAG ne retourne pas de page : cherche dans les métadonnées
        index (metadata.json) un chunk dont le contenu contient les mots-clés du tableau.
        """
        import json

        index_dir = self._index_dir_for_read(doc_id)
        meta_path = os.path.join(index_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            return None
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

        doc_filter = doc_id.lower().replace(".pdf", "")
        keywords = self._table_keywords(tableau)
        best_page: Optional[int] = None
        best_score = 0

        for m in meta:
            if m.get("doc_id") is None:
                continue
            if doc_filter not in (m.get("doc_id") or "").lower():
                continue
            content = (m.get("content") or "").lower()
            score = sum(1 for kw in keywords if kw in content)
            if score > best_score:
                best_score = score
                try:
                    best_page = int(m.get("page"))
                except (TypeError, ValueError):
                    pass
        return best_page

    def _table_keywords(self, tableau: str) -> list[str]:
        """Mots-clés pour la recherche fallback selon le type de tableau."""
        t = tableau.lower()
        want_consolides = ("consol" in t) or ("consolid" in t)
        social_needles = ["comptes sociaux", "etats financiers sociaux"]
        consol_needles = ["comptes consol", "etats financiers consolides"]
        if "bilan actif" in t or ("actif" in t and "passif" not in t):
            return ["bilan", "actif", "total"] + (consol_needles if want_consolides else social_needles)
        if "bilan passif" in t or "passif" in t:
            return ["bilan", "passif", "total"] + (consol_needles if want_consolides else social_needles)
        # Nom principal : COMPTE DE PRODUITS ET CHARGES (cpc = abréviation)
        if "compte de produits et charges" in t or "comptes de produits et charges" in t or "cpc" in t or "produits et charges" in t or "compte de résultat" in t:
            return (
                ["compte de produits et charges", "produits et charges", "résultat net", "resultat net"]
                + (consol_needles if want_consolides else social_needles)
            )
        return [tableau.lower()]

    def _normalize_columns_for_excel(self, df: pd.DataFrame) -> pd.DataFrame:
        """En-têtes clairs : Rubrique (noms) + Montant (date ou exercice) pour les chiffres."""
        if df is None or df.empty or len(df.columns) == 0:
            return df
        new_names = []
        for i, c in enumerate(df.columns):
            name = str(c).strip()
            if i == 0:
                new_names.append("Rubrique")
            elif name and ("/" in name or "-" in name) and any(ch.isdigit() for ch in name):
                new_names.append(f"Montant ({name})")
            else:
                new_names.append(f"Montant exercice {i}" if not name or name.startswith("Exercice") else name)
        out = df.copy()
        out.columns = new_names[: len(out.columns)]
        return out

    def _export_excel(
        self,
        df: pd.DataFrame,
        out_path: Path,
        sheet_name: str = "Feuille1",
    ) -> None:
        """Exporte le DataFrame en Excel : en-têtes clairs (Rubrique + Montant), largeurs colonnes."""
        df_export = self._normalize_columns_for_excel(df)
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            df_export.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            worksheet = writer.sheets[sheet_name[:31]]
            for i in range(len(df_export.columns)):
                ser = df_export.iloc[:, i].astype(str)
                col_name = df_export.columns[i] if isinstance(df_export.columns, pd.Index) else str(i)
                max_len = max(ser.str.len().max() if len(ser) > 0 else 0, len(str(col_name)))
                worksheet.set_column(i, i, min(max_len + 1, 55))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Phase 1 : AMMC → RAG → Extraction → Excel")
    parser.add_argument("--emetteur", required=True, help="Ex: agma")
    parser.add_argument("--year", type=int, required=True, help="Ex: 2024")
    parser.add_argument("--tableau", required=True, help="Ex: BILAN ACTIF")
    parser.add_argument("--type-comptes", default="sociaux", choices=["sociaux", "consolides"])
    parser.add_argument(
        "--type-rapport",
        default="annuel",
        help="annuel ou s1 (dossier d'index : .../{year}/{annuel|s1}/index/)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_XLSX_DIR, help="Dossier Excel")
    args = parser.parse_args()

    pipeline = Phase1Pipeline(xlsx_dir=args.out_dir)
    result = pipeline.run(
        emetteur=args.emetteur,
        year=args.year,
        tableau=args.tableau,
        type_comptes=args.type_comptes,
        type_rapport=args.type_rapport,
    )

    if result.success:
        print(f"[OK] {args.emetteur} {args.year} | '{args.tableau}' | page {result.page_num} | {len(result.df)}L x {len(result.df.columns)}C | methode={result.method}")
        print(f"Excel : {result.excel_path}")
    else:
        print(f"[FAIL] {result.error}")
        exit(1)


if __name__ == "__main__":
    _main()
