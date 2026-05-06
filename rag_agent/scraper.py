"""
Scraper AMMC : téléchargement automatique des PDFs RFA depuis ammc.ma.

Stratégies (ordre) :
  1. Cache local (dossier Émetteur / data/pdfs) → {emetteur}_{year}_{annuel|s1}.pdf
  2. Overrides data/pdf_overrides.json, puis URLs PDF connues (KNOWN_PDF_URLS), toujours filtrées par type_rapport
  3. Mode AMMC : page « Liste des états financiers » (filtres Emetteur + Année + type de rapport,
     comme sur le site : « Rapports annuels » vs « Rapports 1er semestre » — un PDF distinct par combinaison)
     puis le lien du type demandé → pièce jointe PDF
  4. Fiche émetteur liste-des-emetteurs/{id}, URLs etats-financiers/{slug}-rfa-{year}, liste paginée

Usage:
    scraper = AMMCScraper()
    result = scraper.fetch(
        emetteur="agma",
        year=2024,
        type_comptes="sociaux",
        type_rapport="annuel",
    )
    # result.pdf_path → Path("data/pdfs/agma_2024_annuel.pdf")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
import os
import re
import base64
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import parse_qs
from urllib.parse import urljoin
from urllib.parse import urlencode
from urllib.parse import urlparse, unquote

import requests
from bs4 import BeautifulSoup

# Éviter les warnings SSL quand verify=False (site AMMC)
try:
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except Exception:
    pass

# Cache local pour les PDFs téléchargés
# On essaye d'utiliser un dossier global "Emetteur" au niveau projet
# (le nom peut contenir des caractères non-ASCII), sinon fallback vers data/pdfs.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _discover_emetteur_pdf_dir() -> Path:
    try:
        for d in os.listdir(_PROJECT_ROOT):
            p = _PROJECT_ROOT / d
            if not p.is_dir():
                continue
            if "metteur" in d.lower():
                p.mkdir(parents=True, exist_ok=True)
                return p
    except Exception:
        pass
    fallback = _PROJECT_ROOT / "data" / "pdfs"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


DEFAULT_PDF_DIR = _discover_emetteur_pdf_dir()
OVERRIDES_PATH = _PROJECT_ROOT / "data" / "pdf_overrides.json"
BASE_URL = "https://www.ammc.ma"
# Formulaire exposé Drupal (GET) : même flux que « Emetteur », « Année », « Appliquer » sur le site.
LISTE_ETATS_FINANCIERS_PATH = "/fr/liste-etats-financiers-emetteurs"
# Même parcours que sur le site : Émetteur = choix user, Année = « - Tout - », Appliquer, puis repérer la ligne
# (Année + Type rapport) et ouvrir le lien → PDF.
LISTE_ETATS_MAX_PAGES_TOUT_ANNEES = 60
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# URLs PDF directes : (slug_émetteur, année, type normalisé "annuel"|"s1") -> url
# Même logique que sur l’AMMC : chaque combinaison (émetteur, année, type rapport) a son propre PDF.
KNOWN_PDF_URLS: dict[tuple[str, int, str], str] = {
    ("attijariwafa_bank", 2024, "annuel"): "https://www.ammc.ma/sites/default/files/AWB_RFA_2024_1.pdf",
    ("addoha", 2024, "annuel"): "https://www.ammc.ma/sites/default/files/Addoha_RFA_2024.pdf",
    ("bmci", 2024, "annuel"): "https://www.ammc.ma/sites/default/files/BMCI_RFA_2024.pdf",
}


def normalize_user_emetteur(raw: str) -> str:
    """
    Saisie plateforme (ex. « ÉmetteurADDOHA », « emetteur : Addoha », « EMETTEUR - ADDOHA »)
    → slug minuscules pour cache PDF et matching AMMC (ex. addoha).
    """
    t = (raw or "").strip()
    if not t:
        return ""
    low = t.casefold()
    for prefix in ("émetteur", "emetteur"):
        pc = prefix.casefold()
        if low.startswith(pc):
            t = t[len(prefix) :].lstrip(" \t:-,")
            low = t.casefold()
            break
    return " ".join(t.split()).lower()


@dataclass
class FetchResult:
    """Résultat d'un fetch AMMC."""

    success: bool
    pdf_path: Optional[Path] = None
    size_mb: float = 0.0
    from_cache: bool = False
    error: Optional[str] = None


class AMMCScraper:
    """
    Télécharge les PDFs d’états financiers publiés sur ammc.ma.

    Paramètres de requête (comme sur le portail) : émetteur, année, type de rapport
    (« Rapports annuels » / « Rapports 1er semestre »), et type de comptes le cas échéant.
    Chaque combinaison cible un fichier PDF distinct.
    """

    def __init__(self, pdf_dir: Optional[Path] = None):
        self.pdf_dir = pdf_dir or DEFAULT_PDF_DIR
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": USER_AGENT})
        # Émetteurs connus (id AMMC) pour fallback manuel
        self._emetteur_ids: dict[str, str] = {}
        self._overrides = self._load_overrides()
        # Options du select « Emetteur » sur la liste des états financiers (tid, libellé)
        self._liste_etats_emetteur_options: Optional[list[tuple[str, str]]] = None

    def add_emetteur(self, slug: str, ammc_id: str) -> None:
        """Enregistre un ID AMMC pour un émetteur (ex: add_emetteur("agma", "2729"))."""
        self._emetteur_ids[slug.lower()] = ammc_id

    def _normalize_name(self, s: str) -> str:
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
        s = re.sub(r"[\(\)\[\]\{\}\.,;:/\\\-_]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _extract_aliases(self, s: str) -> set[str]:
        out: set[str] = set()
        text = self._normalize_name(s)
        if text:
            out.add(text)
        for m in re.findall(r"\(([^)]+)\)", s or ""):
            alias = self._normalize_name(m)
            if alias:
                out.add(alias)
        # acronyme simple basé sur les mots significatifs
        words = [w for w in text.split() if len(w) > 2 and w not in {"societe", "banque", "bank", "groupe"}]
        if len(words) >= 2:
            ac = "".join(w[0] for w in words)
            if len(ac) >= 2:
                out.add(ac)
        return out

    def _score_emetteur_match(self, emetteur: str, label: str) -> int:
        target = self._normalize_name(emetteur)
        candidate = self._normalize_name(label)
        if not target or not candidate:
            return 0
        if target == candidate:
            return 100

        target_aliases = self._extract_aliases(emetteur)
        cand_aliases = self._extract_aliases(label)
        if target_aliases & cand_aliases:
            return 95

        # Match "contient" autorisé seulement sur chaînes assez longues
        if len(target) >= 8 and target in candidate:
            return 80
        if len(candidate) >= 8 and candidate in target:
            return 70

        stop = {"de", "du", "des", "la", "le", "les", "et", "sa", "spa", "societe", "societe", "bank", "banque"}
        t_words = {w for w in target.split() if len(w) > 2 and w not in stop}
        c_words = {w for w in candidate.split() if len(w) > 2 and w not in stop}
        if not t_words or not c_words:
            return 0
        common = t_words & c_words
        if not common:
            return 0
        overlap = len(common) / max(1, len(t_words))
        if overlap >= 0.8:
            return 75
        if overlap >= 0.6:
            return 60
        return 0

    def _normalize_type_rapport(self, type_rapport: str) -> str:
        tr = (type_rapport or "annuel").strip().lower()
        tr = tr.replace("é", "e").replace("è", "e")
        if "1er semestre" in tr or "premier semestre" in tr or "semestr" in tr or tr in ("s1", "rfs"):
            return "s1"
        return "annuel"

    def _cache_path(self, emetteur: str, year: int, type_comptes: str, type_rapport: str) -> Path:
        safe = emetteur.lower().strip().replace(" ", "_")
        report_code = self._normalize_type_rapport(type_rapport)
        # Convention cache : un fichier par (emetteur, année, type_rapport).
        return self.pdf_dir / f"{safe}_{year}_{report_code}.pdf"

    def _normalize_search_mode(self, search_mode: str) -> str:
        sm = (search_mode or "ammc").strip().lower()
        if sm in {"web_api", "api", "serpapi"}:
            return "web_api"
        if sm in {"web", "internet", "google", "duckduckgo"}:
            return "web"
        return "ammc"

    def _load_overrides(self) -> list[dict]:
        if not OVERRIDES_PATH.is_file():
            return []
        try:
            data = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
        except Exception:
            return []
        return []

    def _override_url(self, emetteur: str, year: int, type_comptes: str, type_rapport: str) -> Optional[str]:
        target_names = self._extract_aliases(emetteur) | {self._normalize_name(emetteur)}
        wanted_report = self._normalize_type_rapport(type_rapport)
        wanted_type = (type_comptes or "").strip().lower()
        for row in self._overrides:
            url = str(row.get("url", "")).strip()
            if not url.lower().startswith("http"):
                continue
            try:
                y = int(row.get("year", 0))
            except Exception:
                continue
            if y != int(year):
                continue
            name = self._normalize_name(str(row.get("emetteur", "")))
            aliases = {self._normalize_name(x) for x in row.get("aliases", []) if isinstance(x, str)}
            if target_names and not ((name and name in target_names) or (aliases & target_names)):
                continue
            row_report = self._normalize_type_rapport(str(row.get("type_rapport", "annuel")))
            if row_report != wanted_report:
                continue
            row_type = str(row.get("type_comptes", "")).strip().lower()
            if row_type and row_type != wanted_type:
                continue
            return url
        return None

    def _get(self, url: str, timeout: int = 30, verify: Optional[bool] = None) -> Optional[requests.Response]:
        if verify is None:
            verify = BASE_URL not in url  # False pour AMMC (éviter erreur SSL)
        try:
            r = self._session.get(url, timeout=timeout, verify=verify)
            r.raise_for_status()
            return r
        except requests.RequestException:
            return None

    @staticmethod
    def _ammc_liste_etats_year_tid(year: int) -> Optional[str]:
        """Valeur du select « Année » sur liste-etats-financiers (2010→1 … 2025→16). Hors plage → None (= Tout)."""
        if 2010 <= year <= 2025:
            return str(year - 2009)
        return None

    def _load_liste_etats_emetteur_options(self) -> list[tuple[str, str]]:
        if self._liste_etats_emetteur_options is not None:
            return self._liste_etats_emetteur_options
        r = self._get(f"{BASE_URL}{LISTE_ETATS_FINANCIERS_PATH}", timeout=35, verify=False)
        if not r or r.status_code != 200:
            self._liste_etats_emetteur_options = []
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        sel = soup.find("select", attrs={"name": "field_emetteur_target_id_verf"})
        if not sel:
            self._liste_etats_emetteur_options = []
            return []
        out: list[tuple[str, str]] = []
        for opt in sel.find_all("option"):
            val = (opt.get("value") or "").strip()
            lab = (opt.get_text() or "").strip()
            if val and val != "All" and lab:
                out.append((val, lab))
        self._liste_etats_emetteur_options = out
        return out

    def _resolve_tid_liste_etats(self, emetteur: str) -> Optional[str]:
        options = self._load_liste_etats_emetteur_options()
        if not options:
            return None
        best_tid: Optional[str] = None
        best_score = 0
        for tid, label in options:
            sc = self._score_emetteur_match(emetteur, label)
            if sc > best_score:
                best_score = sc
                best_tid = tid
        if best_score >= 60 and best_tid:
            return best_tid
        return None

    def _liste_etats_row_type_link(
        self, tr
    ) -> tuple[Optional[str], Optional[int], str]:
        """Ligne du tableau liste états : (href etats-financiers, année affichée, texte du lien type rapport)."""
        tds = tr.find_all("td")
        if len(tds) < 4:
            return None, None, ""
        annee_td, type_td = tds[2], tds[3]
        row_year: Optional[int] = None
        time_el = annee_td.find("time")
        if time_el:
            txt = (time_el.get_text() or "").strip()
            if txt.isdigit():
                row_year = int(txt)
            else:
                dt = (time_el.get("datetime") or "").strip()
                if len(dt) >= 4 and dt[:4].isdigit():
                    row_year = int(dt[:4])
        for a in type_td.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            if "/espace-emetteurs/etats-financiers/" in href:
                return href, row_year, (a.get_text() or "").strip()
        return None, row_year, ""

    def _link_text_matches_type_rapport(self, link_text: str, normalized_report: str) -> bool:
        t = (link_text or "").lower().replace("é", "e").replace("è", "e").strip()
        if self._normalize_type_rapport(normalized_report) == "s1":
            return "1er semestre" in t or ("semestre" in t and "annuel" not in t)
        return "annuel" in t and "semestre" not in t

    def _try_download_from_liste_etats_financiers(
        self,
        emetteur: str,
        year: int,
        type_comptes: str,
        type_rapport: str,
        cache_path: Path,
    ) -> Optional[FetchResult]:
        """
        Reproduit le parcours « Liste des Etats Financiers » (AMMC) :
        1) sélectionner l’émetteur demandé ;
        2) laisser l’année sur « - Tout - » (paramètre All) et Appliquer — pas de filtre année côté formulaire ;
        3) parcourir le tableau paginé et prendre la ligne dont la colonne Année et le lien Type rapport
           correspondent à year + type_rapport (ex. 2024 + « Rapports 1er semestre ») ;
        4) suivre le lien de cette cellule vers la page états financiers, puis récupérer le PDF.
        """
        tid = self._resolve_tid_liste_etats(emetteur)
        if not tid:
            return None
        max_pages = LISTE_ETATS_MAX_PAGES_TOUT_ANNEES

        for page in range(max_pages):
            params: dict[str, str] = {
                "field_emetteur_target_id_verf": tid,
                "field_annee_value_1": "All",
            }
            if page > 0:
                params["page"] = str(page)
            url = f"{BASE_URL}{LISTE_ETATS_FINANCIERS_PATH}?{urlencode(params)}"
            r = self._get(url, timeout=35, verify=False)
            if not r or r.status_code != 200:
                if page == 0:
                    return None
                break
            soup = BeautifulSoup(r.text, "html.parser")
            tbody = soup.find("tbody")
            if not tbody:
                break
            rows = tbody.find_all("tr")
            if not rows and page > 0:
                break
            for tr in rows:
                href_raw, row_y, type_label = self._liste_etats_row_type_link(tr)
                if not href_raw or row_y is None or row_y != int(year):
                    continue
                if not self._link_text_matches_type_rapport(type_label, type_rapport):
                    continue
                etats_url = urljoin(BASE_URL, href_raw)
                result = self._try_download_from_etats_page(
                    etats_url, year, type_comptes, type_rapport, cache_path
                )
                if result:
                    return result
            time.sleep(0.25)
        return None

    def fetch_liste_emetteurs(self, max_pages: int = 20) -> list[dict]:
        """
        Récupère la liste complète des émetteurs depuis le site AMMC (~170).
        Parcourt toutes les pages (pagination ?page=N). Retourne [{"code": "slug", "label": "Dénomination"}, ...].
        En cas d'échec (réseau, SSL), retourne une liste vide.
        """
        seen: set[str] = set()
        out: list[dict] = []
        # Pagination : page 0 = première page (Drupal), ou page=1 selon les sites
        for page in range(max_pages):
            if page == 0:
                url = f"{BASE_URL}/fr/espace-emetteurs/liste-des-emetteurs"
            else:
                url = f"{BASE_URL}/fr/espace-emetteurs/liste-des-emetteurs?page={page}"
            r = self._get(url, timeout=25, verify=False)
            if not r or r.status_code != 200:
                if page == 0:
                    return []
                break
            soup = BeautifulSoup(r.text, "html.parser")
            added = 0
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                if "/liste-des-emetteurs/" not in href:
                    continue
                # Exclure le lien "liste-des-emetteurs" sans id (lien menu)
                parts = href.rstrip("/").split("/")
                if len(parts) < 2 or not parts[-1].isdigit():
                    continue
                label = (a.get_text() or "").strip()
                if not label or len(label) < 2:
                    continue
                code = " ".join(label.lower().split())
                if code in seen:
                    continue
                seen.add(code)
                out.append({"code": code, "label": label})
                added += 1
            if added == 0 and page > 0:
                break
            time.sleep(0.3)  # politeness entre les pages
        return sorted(out, key=lambda x: x["label"].lower())

    def fetch(
        self,
        emetteur: str,
        year: int,
        type_comptes: str = "sociaux",
        type_rapport: str = "annuel",
        search_mode: str = "ammc",
    ) -> FetchResult:
        """
        Télécharge le PDF RFA pour l'émetteur et l'année donnés.
        type_comptes : "sociaux" ou "consolides"
        type_rapport : « Rapports annuels » / annuel / rfa, ou « Rapports 1er semestre » / s1 / rfs
            (même rôle qu’Émetteur et Année : un PDF par type).
        search_mode : "ammc" ou "web"
        """
        emetteur = normalize_user_emetteur(emetteur)
        normalized_report = self._normalize_type_rapport(type_rapport)
        normalized_mode = self._normalize_search_mode(search_mode)
        cache_path = self._cache_path(emetteur, year, type_comptes, normalized_report)

        # 1. Cache local
        if cache_path.is_file() and cache_path.stat().st_size > 0:
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            return FetchResult(
                success=True,
                pdf_path=cache_path,
                size_mb=round(size_mb, 2),
                from_cache=True,
            )

        # 2. URL PDF directe connue (ex: Attijariwafa Bank 2024)
        override_url = self._override_url(emetteur, year, type_comptes, normalized_report)
        if override_url:
            result = self._download_pdf(override_url, cache_path)
            if result:
                return result

        key = (emetteur.replace(" ", "_"), year, normalized_report)
        if key in KNOWN_PDF_URLS:
            result = self._download_pdf(KNOWN_PDF_URLS[key], cache_path)
            if result:
                return result

        if normalized_mode in {"web", "web_api"}:
            result = self._try_download_from_web_search(
                emetteur=emetteur,
                year=year,
                type_comptes=type_comptes,
                type_rapport=normalized_report,
                cache_path=cache_path,
                link_fetcher=self._search_serpapi_links if normalized_mode == "web_api" else self._search_web_links,
            )
            if result:
                return result
        else:
            # 4. Liste des états financiers (filtres + lien « Rapports annuels » / « Rapports 1er semestre »)
            result = self._try_download_from_liste_etats_financiers(
                emetteur, year, type_comptes, normalized_report, cache_path
            )
            if result:
                return result

            # 5. URL connue (liste des émetteurs)
            ammc_id = self._emetteur_ids.get(emetteur)
            if ammc_id:
                list_url = f"{BASE_URL}/fr/espace-emetteurs/liste-des-emetteurs/{ammc_id}"
                result = self._try_download_from_emitter_page(
                    list_url, emetteur, year, type_comptes, normalized_report, cache_path
                )
                if result:
                    return result

            # 6. Page états financiers directe : plusieurs slugs et formats de rapport
            report_tokens = ["rfa"] if normalized_report == "annuel" else ["rfs", "s1", "semestre"]
            slugs_to_try = [
                emetteur.replace(" ", "-"),
                emetteur.replace(" ", "-") + "-bank",
                emetteur.replace(" ", "-") + "-maroc",
                emetteur.replace(" ", ""),  # bmci, agma
            ]
            for slug in slugs_to_try:
                for token in report_tokens:
                    etats_url = f"{BASE_URL}/fr/espace-emetteurs/etats-financiers/{slug}-{token}-{year}"
                    result = self._try_download_from_etats_page(etats_url, year, type_comptes, normalized_report, cache_path)
                    if result:
                        return result
                    time.sleep(0.2)

            # 7. Recherche dans la liste générale (toutes les pages si besoin)
            result = self._try_search_list(emetteur, year, type_comptes, normalized_report, cache_path)
            if result:
                return result

        # 8. Fallback : dossier data/pdf (fichiers déjà présents, filtrés par type de rapport)
        result = self._try_copy_from_fallback_dir(
            emetteur, year, normalized_report, cache_path
        )
        if result:
            return result

        return FetchResult(
            success=False,
            pdf_path=None,
            error=(
                f"Téléchargement automatique échoué pour {emetteur} {year} "
                f"(mode={normalized_mode}, type_rapport={normalized_report}, AMMC/WEB, liste états financiers, liste émetteurs, états financiers, dossier data/pdf). "
                f"Vous pouvez placer le PDF dans : {cache_path}"
            ),
        )

    def _extract_real_search_url(self, href: str) -> str:
        try:
            parsed = urlparse(href)
            if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
                q = parse_qs(parsed.query)
                if "uddg" in q and q["uddg"]:
                    return unquote(q["uddg"][0])
            if "bing.com" in parsed.netloc and "/ck/a" in parsed.path:
                q = parse_qs(parsed.query)
                uv = q.get("u", [""])[0]
                if uv.startswith("a1"):
                    raw = uv[2:]
                    pad = "=" * ((4 - (len(raw) % 4)) % 4)
                    decoded = base64.urlsafe_b64decode(raw + pad).decode("utf-8", errors="ignore")
                    if decoded.startswith("http"):
                        return decoded
                    if decoded.startswith("/"):
                        return urljoin("https://www.bing.com", decoded)
        except Exception:
            pass
        return href

    def _web_query_candidates(self, emetteur: str, year: int, type_rapport: str) -> list[str]:
        report_part = "Rapports 1er semestre" if self._normalize_type_rapport(type_rapport) == "s1" else "Rapports annuels"
        name_norm = self._normalize_name(emetteur)
        words = name_norm.split()
        if len(words) >= 2 and len(words[-1]) <= 4:
            name_core = " ".join(words[:-1]).strip()
        else:
            name_core = name_norm
        domain_hint = "".join(w for w in name_core.split() if w not in {"du", "de", "des", "la", "le", "les"})
        domain_hint_full = "".join(name_core.split())
        return [
            f"{emetteur} {year} {report_part} pdf site:ammc.ma",
            f"{emetteur} {year} {report_part} pdf",
            f"{emetteur} {year} rfs pdf site:ammc.ma" if self._normalize_type_rapport(type_rapport) == "s1" else f"{emetteur} {year} rfa pdf site:ammc.ma",
            f"\"{emetteur}\" {year} pdf",
            f"\"{name_core}\" {year} rapport financier pdf",
            f"site:{domain_hint}.ma {year} rapport financier pdf" if domain_hint else f"{name_core} {year} rapport financier pdf",
            f"site:www.{domain_hint_full}.ma {year} pdf" if domain_hint_full else f"{name_core} {year} pdf",
        ]

    def _search_web_links(self, query: str, max_links: int = 25) -> list[str]:
        """Recherche web simple (Google puis Bing) et retourne des URLs candidates."""
        links = self._search_google_links(query, max_links=max_links)
        if links:
            return links
        return self._search_bing_links(query, max_links=max_links)

    def _search_google_links(self, query: str, max_links: int = 25) -> list[str]:
        """Recherche Google HTML (best effort)."""
        out: list[str] = []
        seen: set[str] = set()
        search_url = f"https://www.google.com/search?q={requests.utils.quote(query)}&hl=fr&gl=ma&num=20"
        try:
            r = requests.get(
                search_url,
                timeout=30,
                verify=True,
                headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "fr-FR,fr;q=0.9"},
            )
            r.raise_for_status()
        except requests.RequestException:
            return out

        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = (a.get("href") or "").strip()
            # Liens de résultat Google: /url?q=<target>&...
            if href.startswith("/url?"):
                q = parse_qs(urlparse(href).query)
                target = unquote((q.get("q") or [""])[0]).strip()
                if not target.startswith("http"):
                    continue
                if target in seen:
                    continue
                seen.add(target)
                out.append(target)
                if len(out) >= max_links:
                    break
        return out

    def _search_bing_links(self, query: str, max_links: int = 25) -> list[str]:
        """Recherche Bing HTML fallback."""
        out: list[str] = []
        seen: set[str] = set()
        search_url = (
            f"https://www.bing.com/search?q={requests.utils.quote(query)}"
            "&setlang=fr&cc=fr&ensearch=1"
        )
        try:
            # Requête dédiée (hors session scraper) pour éviter les pages anti-bot
            # que Bing renvoie parfois avec certains en-têtes "browser-like".
            r = requests.get(
                search_url,
                timeout=30,
                verify=True,
                headers={"User-Agent": "Mozilla/5.0", "Accept-Language": "fr-FR,fr;q=0.9"},
            )
            r.raise_for_status()
        except requests.RequestException:
            return out
        soup = BeautifulSoup(r.text, "html.parser")

        # Cibler d'abord les résultats principaux Bing
        for box in soup.select("li.b_algo h2 a[href]"):
            href = self._extract_real_search_url((box.get("href") or "").strip())
            if not href.startswith("http"):
                continue
            if href in seen:
                continue
            seen.add(href)
            out.append(href)
            if len(out) >= max_links:
                return out

        # Fallback générique
        for a in soup.find_all("a", href=True):
            href = self._extract_real_search_url((a.get("href") or "").strip())
            if not href.startswith("http"):
                continue
            if href in seen:
                continue
            seen.add(href)
            out.append(href)
            if len(out) >= max_links:
                break
        return out

    def _search_serpapi_links(self, query: str, max_links: int = 25) -> list[str]:
        """Recherche via SerpAPI (Google) pour un résultat web stable."""
        api_key = os.getenv("SERPAPI_API_KEY", "").strip()
        if not api_key:
            return []
        params = {
            "engine": "google",
            "q": query,
            "hl": "fr",
            "gl": "ma",
            "num": min(max_links, 20),
            "api_key": api_key,
        }
        try:
            r = requests.get("https://serpapi.com/search.json", params=params, timeout=30, verify=True)
            r.raise_for_status()
            payload = r.json()
        except requests.RequestException:
            return []
        except ValueError:
            return []

        out: list[str] = []
        seen: set[str] = set()
        for item in payload.get("organic_results", []) or []:
            link = (item.get("link") or "").strip()
            if link.startswith("http") and link not in seen:
                seen.add(link)
                out.append(link)
                if len(out) >= max_links:
                    break
        return out

    def _score_url_for_request(
        self,
        url: str,
        emetteur: str,
        year: int,
        type_comptes: str,
        type_rapport: str,
        strict_report_type: bool = True,
    ) -> int:
        u = (url or "").lower()
        if ".pdf" not in u:
            return -100
        score = 0
        if "ammc.ma" in u:
            score += 40
        if str(year) in u:
            score += 20
        em_score = self._score_emetteur_match(emetteur, u.replace("-", " ").replace("_", " "))
        score += min(em_score, 40)

        is_s1 = self._normalize_type_rapport(type_rapport) == "s1"
        has_s1 = any(t in u for t in ("rfs", "s1", "semestre", "juin", "30-06", "30/06"))
        has_annuel = any(t in u for t in ("rfa", "annuel", "31-12", "31/12"))
        if strict_report_type:
            if is_s1 and has_annuel and not has_s1:
                return -100
            if (not is_s1) and has_s1 and not has_annuel:
                return -100
        else:
            if is_s1 and has_annuel and not has_s1:
                score -= 10
            if (not is_s1) and has_s1 and not has_annuel:
                score -= 10
        if is_s1 and has_s1:
            score += 25
        if (not is_s1) and has_annuel:
            score += 25

        has_consol = "consolid" in u
        if type_comptes == "consolides" and has_consol:
            score += 8
        if type_comptes == "sociaux" and has_consol:
            score -= 8
        return score

    def _looks_like_requested_report(
        self,
        pdf_path: Path,
        source_url: str,
        emetteur: str,
        year: int,
        type_rapport: str,
    ) -> bool:
        """
        Validation minimale pour éviter les faux PDF web (ex. document non financier).
        Critères: année + émetteur + cohérence type_rapport.
        """
        text = ""
        try:
            import pymupdf

            doc = pymupdf.open(str(pdf_path))
            parts: list[str] = []
            for i in range(min(4, len(doc))):
                parts.append(doc[i].get_text() or "")
            doc.close()
            text = self._normalize_name(" ".join(parts))
        except Exception:
            text = ""

        url_n = self._normalize_name(source_url or "")
        year_s = str(year)
        year_ok = (year_s in url_n) or (year_s in text)
        if not year_ok:
            return False

        em_url = self._score_emetteur_match(emetteur, url_n)
        em_txt = self._score_emetteur_match(emetteur, text)
        if max(em_url, em_txt) < 40:
            return False

        is_s1 = self._normalize_type_rapport(type_rapport) == "s1"
        s1_markers = ("1er semestre", "premier semestre", "semestre", "juin", "30/06", "30 06")
        ann_markers = ("annuel", "rapports annuels", "rfa", "31/12", "31 12")
        has_s1 = any(m in url_n or m in text for m in s1_markers)
        has_ann = any(m in url_n or m in text for m in ann_markers)

        # Rejeter un conflit explicite du type de rapport
        if is_s1 and has_ann and not has_s1:
            return False
        if (not is_s1) and has_s1 and not has_ann:
            return False
        return True

    def _download_pdf_for_request(
        self,
        pdf_url: str,
        cache_path: Path,
        emetteur: str,
        year: int,
        type_rapport: str,
    ) -> Optional[FetchResult]:
        result = self._download_pdf(pdf_url, cache_path)
        if not result or not result.success or not result.pdf_path:
            return None
        if self._looks_like_requested_report(result.pdf_path, pdf_url, emetteur, year, type_rapport):
            return result
        try:
            result.pdf_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None

    def _try_download_from_web_search(
        self,
        emetteur: str,
        year: int,
        type_comptes: str,
        type_rapport: str,
        cache_path: Path,
        link_fetcher: Optional[Callable[[str, int], list[str]]] = None,
    ) -> Optional[FetchResult]:
        fetcher = link_fetcher or self._search_web_links
        # Mode utilisateur demandé: prendre le premier PDF web trouvé.
        for query in self._web_query_candidates(emetteur, year, type_rapport):
            first_page_candidates: list[str] = []
            for href in fetcher(query, 25):
                if ".pdf" in (href or "").lower():
                    result = self._download_pdf_for_request(
                        href, cache_path, emetteur, year, type_rapport
                    )
                    if result:
                        return result
                else:
                    first_page_candidates.append(href)
            # Si aucun lien direct PDF, suivre les pages et prendre le premier PDF interne.
            for page_url in first_page_candidates[:20]:
                r = self._get(page_url, timeout=25, verify=True)
                if not r or r.status_code != 200:
                    continue
                soup = BeautifulSoup(r.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    h = (a.get("href") or "").strip()
                    full = urljoin(page_url, h)
                    if ".pdf" not in full.lower():
                        continue
                    result = self._download_pdf_for_request(
                        full, cache_path, emetteur, year, type_rapport
                    )
                    if result:
                        return result

        # Pass 1: match strict du type de rapport (annuel vs s1)
        candidates: list[tuple[int, str]] = []
        page_candidates: list[str] = []
        for query in self._web_query_candidates(emetteur, year, type_rapport):
            for href in fetcher(query, 25):
                if ".pdf" not in (href or "").lower():
                    page_candidates.append(href)
                    continue
                score = self._score_url_for_request(
                    href,
                    emetteur,
                    year,
                    type_comptes,
                    type_rapport,
                    strict_report_type=True,
                )
                if score < 40:
                    continue
                candidates.append((score, href))
            time.sleep(0.2)

        seen: set[str] = set()
        for _score, url in sorted(candidates, key=lambda x: x[0], reverse=True):
            if url in seen:
                continue
            seen.add(url)
            result = self._download_pdf_for_request(
                url, cache_path, emetteur, year, type_rapport
            )
            if result:
                return result

        # Pass 1-bis: explorer les pages web candidates et extraire des liens PDF internes
        for page_url in page_candidates[:20]:
            r = self._get(page_url, timeout=25, verify=True)
            if not r or r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            local_pdf_links: list[tuple[int, str]] = []
            for a in soup.find_all("a", href=True):
                h = (a.get("href") or "").strip()
                full = urljoin(page_url, h)
                if ".pdf" not in full.lower():
                    continue
                score = self._score_url_for_request(
                    full,
                    emetteur,
                    year,
                    type_comptes,
                    type_rapport,
                    strict_report_type=True,
                )
                if score >= 40:
                    local_pdf_links.append((score, full))
            for _score, pdf_link in sorted(local_pdf_links, key=lambda x: x[0], reverse=True):
                result = self._download_pdf_for_request(
                    pdf_link, cache_path, emetteur, year, type_rapport
                )
                if result:
                    return result

        # Pass 2 (fallback web): télécharger le premier PDF web "valide"
        # même si le type de rapport n'est pas explicitement détecté.
        for query in self._web_query_candidates(emetteur, year, type_rapport):
            for href in fetcher(query, 25):
                score = self._score_url_for_request(
                    href,
                    emetteur,
                    year,
                    type_comptes,
                    type_rapport,
                    strict_report_type=False,
                )
                if score < 35:
                    continue
                result = self._download_pdf_for_request(
                    href, cache_path, emetteur, year, type_rapport
                )
                if result:
                    return result

        # Pass 2-bis: fallback permissif en explorant aussi les pages web non-PDF
        for page_url in page_candidates[:20]:
            r = self._get(page_url, timeout=25, verify=True)
            if not r or r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                h = (a.get("href") or "").strip()
                full = urljoin(page_url, h)
                if ".pdf" not in full.lower():
                    continue
                score = self._score_url_for_request(
                    full,
                    emetteur,
                    year,
                    type_comptes,
                    type_rapport,
                    strict_report_type=False,
                )
                if score < 35:
                    continue
                result = self._download_pdf_for_request(
                    full, cache_path, emetteur, year, type_rapport
                )
                if result:
                    return result
        return None

    @staticmethod
    def _fallback_stem_matches_report_type(stem_lower: str, normalized_report: str) -> bool:
        """Évite de copier un RFA pour une requête S1 (et inversement) depuis data/pdf/."""
        s = stem_lower.replace("é", "e").replace("è", "e")
        if normalized_report == "s1":
            if "rfa" in s and "rfs" not in s and "semestre" not in s and "s1" not in s and "1er" not in s:
                return False
            return any(t in s for t in ("rfs", "s1", "semestre", "1er", "juin"))
        # annuel
        if "_s1" in s or re.search(r"(^|[^a-z])s1([^a-z]|$)", s):
            return False
        if "rfs" in s and "rfa" not in s:
            return False
        if "semestre" in s and "annuel" not in s and "rfa" not in s:
            return False
        return True

    def _try_copy_from_fallback_dir(
        self, emetteur: str, year: int, normalized_report: str, cache_path: Path
    ) -> Optional[FetchResult]:
        """
        Si un PDF du bon type existe dans data/pdf/, le copie vers le cache (data/pdfs/ ou dossier Émetteur).
        """
        import shutil
        fallback_dir = self.pdf_dir.parent / "pdf"
        if not fallback_dir.is_dir():
            return None
        em_slug = emetteur.replace(" ", "_").replace("-", "")
        year_str = str(year)
        for f in fallback_dir.glob("*.pdf"):
            if f.stat().st_size == 0:
                continue
            stem = f.stem.lower()
            if em_slug in stem or em_slug.replace("_", "") in stem.replace("_", ""):
                if year_str in stem and self._fallback_stem_matches_report_type(stem, normalized_report):
                    try:
                        self.pdf_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, cache_path)
                        size_mb = cache_path.stat().st_size / (1024 * 1024)
                        return FetchResult(
                            success=True,
                            pdf_path=cache_path,
                            size_mb=round(size_mb, 2),
                            from_cache=False,
                        )
                    except OSError:
                        pass
                    break
        return None

    def _try_download_from_emitter_page(
        self,
        list_url: str,
        emetteur: str,
        year: int,
        type_comptes: str,
        type_rapport: str,
        cache_path: Path,
    ) -> Optional[FetchResult]:
        """Tente de trouver un lien PDF depuis la page émetteur."""
        r = self._get(list_url)
        if not r or r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        # Garde-fou: ne pas prendre un PDF si la page ne correspond pas clairement à l'émetteur.
        page_text = self._normalize_name(soup.get_text(" ", strip=True))
        aliases = self._extract_aliases(emetteur)
        if aliases and not any(a and a in page_text for a in aliases if len(a) >= 3):
            return None

        # 1) Essai direct (PDFs présents sur la page émetteur)
        pdf_url = self._find_pdf_link(soup, year, type_comptes, type_rapport)
        if pdf_url:
            return self._download_pdf(pdf_url, cache_path)

        # 2) Beaucoup d'émetteurs exposent des pages "etats-financiers/*" à suivre.
        # On filtre strictement par année + type_rapport (+ type_comptes quand explicite).
        wanted_year = str(year)
        is_s1 = self._normalize_type_rapport(type_rapport) == "s1"
        wanted_tokens = ("rfs", "s1", "semestr", "juin") if is_s1 else ("rfa", "annuel", "31/12")
        blocked_tokens = ("rfa", "annuel", "31/12") if is_s1 else ("rfs", "s1", "semestr", "juin")
        etats_links: list[tuple[int, str]] = []
        for a in soup.find_all("a", href=True):
            href_raw = a.get("href") or ""
            href = href_raw.lower()
            text = (a.get_text() or "").lower().replace("é", "e").replace("è", "e")
            if "/espace-emetteurs/etats-financiers/" not in href:
                continue
            if wanted_year not in href and wanted_year not in text:
                continue
            has_wanted = any(t in href or t in text for t in wanted_tokens)
            has_blocked = any(t in href or t in text for t in blocked_tokens)
            if has_blocked and not has_wanted:
                continue

            # type_comptes explicite dans le libellé (quand disponible)
            if type_comptes == "consolides":
                if ("social" in text) and ("consolid" not in text):
                    continue
            else:
                if ("consolid" in text) and ("social" not in text):
                    continue

            score = 0
            if has_wanted:
                score += 20
            if wanted_year in href:
                score += 10
            if wanted_year in text:
                score += 5
            etats_links.append((score, urljoin(BASE_URL, href_raw)))

        for _score, et_url in sorted(etats_links, key=lambda x: x[0], reverse=True):
            result = self._try_download_from_etats_page(et_url, year, type_comptes, type_rapport, cache_path)
            if result:
                return result
        return None

    def _try_download_from_etats_page(
        self,
        etats_url: str,
        year: int,
        type_comptes: str,
        type_rapport: str,
        cache_path: Path,
    ) -> Optional[FetchResult]:
        """Tente de télécharger depuis la page etats-financiers/{slug}-rfa-{year}."""
        r = self._get(etats_url)
        if not r or r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")
        pdf_url = self._find_pdf_link(soup, year, type_comptes, type_rapport)
        if pdf_url:
            return self._download_pdf(pdf_url, cache_path)
        return None

    def _emetteur_matches_link(self, emetteur: str, link_text: str, href: str) -> bool:
        """Vrai si le lien correspond à l'émetteur (texte ou id dans l'URL)."""
        return self._score_emetteur_match(emetteur, link_text or "") >= 60

    def _try_search_list(
        self,
        emetteur: str,
        year: int,
        type_comptes: str,
        type_rapport: str,
        cache_path: Path,
    ) -> Optional[FetchResult]:
        """Recherche dans la liste des émetteurs AMMC (plusieurs pages) et télécharge le PDF trouvé."""
        candidates: list[tuple[int, str, str]] = []
        for page in range(5):
            list_url = f"{BASE_URL}/fr/espace-emetteurs/liste-des-emetteurs" + (f"?page={page}" if page else "")
            r = self._get(list_url, timeout=25)
            if not r or r.status_code != 200:
                if page == 0:
                    return None
                break
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a.get("href", "")
                if "/liste-des-emetteurs/" not in href:
                    continue
                parts = href.rstrip("/").split("/")
                if len(parts) < 2 or not parts[-1].isdigit():
                    continue
                label = (a.get_text() or "").strip()
                score = self._score_emetteur_match(emetteur, label)
                if score < 60:
                    continue
                full_url = urljoin(BASE_URL, href)
                candidates.append((score, full_url, label))
            time.sleep(0.2)
        for score, full_url, _label in sorted(candidates, key=lambda x: x[0], reverse=True):
            if score < 60:
                continue
            time.sleep(0.4)
            result = self._try_download_from_emitter_page(
                full_url, emetteur, year, type_comptes, type_rapport, cache_path
            )
            if result:
                return result
        return None

    def _find_pdf_link(self, soup: BeautifulSoup, year: int, type_comptes: str, type_rapport: str) -> Optional[str]:
        """Cherche un lien PDF correspondant à l'année, type de comptes et type de rapport."""
        year_str = str(year)
        year_short = f"{year % 100:02d}"
        is_s1 = self._normalize_type_rapport(type_rapport) == "s1"
        best_candidate: Optional[str] = None
        best_score = -10**9
        for a in soup.find_all("a", href=True):
            href_raw = a.get("href") or ""
            href = href_raw.lower()
            text = (a.get_text() or "").lower().replace("é", "e").replace("è", "e")
            if not href.endswith(".pdf"):
                continue
            year_in_link = year_str in href or year_str in text
            # Ex. AMMC : Addoha_RFS_juin_24_0.pdf (année sur 2 chiffres)
            if not year_in_link and year_short in href:
                year_in_link = bool(
                    re.search(rf"(^|[^0-9]){re.escape(year_short)}([^0-9]|$)", href)
                )
            if not year_in_link and year_short in text:
                year_in_link = bool(
                    re.search(rf"(^|[^0-9]){re.escape(year_short)}([^0-9]|$)", text)
                )
            if not year_in_link:
                continue

            score = 0
            # Type comptes
            has_consol = ("consolid" in href) or ("consolid" in text)
            if type_comptes == "consolides":
                if has_consol:
                    score += 20
                else:
                    score -= 12
            else:
                if has_consol:
                    score -= 15
                else:
                    score += 8

            # Type rapport (annuel vs semestre)
            has_s1_marker = any(k in href or k in text for k in ("1er semestre", "premier semestre", "semestr", "rfs", "s1", "30/06"))
            has_annuel_marker = any(k in href or k in text for k in ("annuel", "annuels", "rfa", "31/12"))
            # Filtre strict du type rapport pour éviter les confusions:
            # - si S1 demandé, exclure un lien clairement annuel
            # - si annuel demandé, exclure un lien clairement semestriel
            if is_s1 and has_annuel_marker and not has_s1_marker:
                continue
            if (not is_s1) and has_s1_marker and not has_annuel_marker:
                continue
            if is_s1:
                if has_s1_marker:
                    score += 25
                if has_annuel_marker:
                    score -= 20
            else:
                if has_annuel_marker:
                    score += 20
                if has_s1_marker:
                    score -= 20

            # Bonus sur l'ancre de texte (souvent plus fiable)
            if year_str in text or year_short in text:
                score += 3

            if score > best_score:
                best_score = score
                best_candidate = urljoin(BASE_URL, href_raw)

        if best_candidate and best_score >= 0:
            return best_candidate
        # IMPORTANT: éviter de télécharger "n'importe quel PDF" en fallback, sinon on peut
        # récupérer un rapport d'un autre émetteur.
        return None

    def _find_pdf_link_any(self, soup: BeautifulSoup) -> Optional[str]:
        """Retourne le premier lien PDF trouvé sur la page."""
        for a in soup.find_all("a", href=True):
            href = a.get("href") or ""
            if href.lower().endswith(".pdf"):
                return urljoin(BASE_URL, href)
        return None

    def _download_pdf(self, pdf_url: str, cache_path: Path) -> Optional[FetchResult]:
        """Télécharge le PDF et l'enregistre dans cache_path."""
        r = self._get(pdf_url)
        if not r or r.status_code != 200:
            return None
        content = r.content
        if len(content) < 1000:  # trop petit pour un vrai PDF
            return None
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(content)
        except OSError as e:
            return FetchResult(success=False, error=str(e))
        size_mb = len(content) / (1024 * 1024)
        return FetchResult(
            success=True,
            pdf_path=cache_path,
            size_mb=round(size_mb, 2),
            from_cache=False,
        )
