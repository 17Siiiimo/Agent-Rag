import json
import logging
import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from .config import RAGConfig, ensure_dirs
from .ingest import build_index
from .retriever import RetrievedChunk, SimpleVectorRetriever, classify_intent
from .extraction_agent import ExtractionAgent

load_dotenv()

logger = logging.getLogger(__name__)

_extraction_agent = ExtractionAgent(use_llm=True)


def _resolve_pdf_path(doc_id: str, raw_pdf_dir: str) -> str:
    """Résout le chemin du PDF comme le CLI extraction_agent (data/pdf si besoin)."""
    path = os.path.join(raw_pdf_dir, doc_id)
    if os.path.isfile(path):
        return path
    fallback = os.path.join("data", "pdf", os.path.basename(doc_id))
    if os.path.isfile(fallback):
        return fallback
    return path


@dataclass
class RAGAnswer:
    answer_markdown: str
    used_chunks: List[RetrievedChunk]


def _format_context(chunks: List[RetrievedChunk]) -> str:
    lines = []
    for c in chunks:
        header = f"[doc={c.doc_id} page={c.page} kind={c.kind} score={c.score:.3f}]"
        lines.append(header)
        lines.append(c.content)
        lines.append("\n---\n")
    return "\n".join(lines)


def _build_prompt(
    question: str,
    context: str,
    intent: str,
    table_type: Optional[str],
) -> str:
    instructions = [
        "Tu es un analyste financier qui lit des rapports PDF déjà extraits.",
        "On te fournit des extraits de texte et de tableaux issus du rapport.",
        "Ta mission est de répondre avec précision, en restant STRICTEMENT fidèle aux chiffres du contexte.",
        "Si une information n'est pas présente, indique clairement qu'elle manque au lieu d'inventer.",
        "",
        "Format de sortie OBLIGATOIRE :",
        "- Toujours répondre en **Markdown**.",
    ]

    if intent == "table_reconstruction":
        instructions.append(
            "- Si la question demande un tableau financier (bilan, compte de résultat, flux de trésorerie, etc.), "
            "reconstruis-le en Markdown (`| col1 | col2 | ... |`)."
        )
        instructions.append(
            "- Préserve les montants et unités (k€, M€, etc.) tels qu'ils apparaissent dans le contexte."
        )

    if table_type == "bilan":
        instructions.append(
            "- Si possible, structure le tableau en actif / passif / capitaux propres."
        )
    elif table_type == "compte_resultat":
        instructions.append(
            "- Si possible, structure le tableau en produits / charges, avec sous-totaux (CA, EBITDA, résultat net, etc.)."
        )
    elif table_type == "flux_tresorerie":
        instructions.append(
            "- Si possible, structure le tableau en flux opérationnels / d'investissement / de financement."
        )

    prompt = "\n".join(instructions)
    prompt += "\n\nQuestion de l'utilisateur :\n"
    prompt += question.strip()
    prompt += "\n\nContexte extrait du rapport (texte + tableaux) :\n"
    prompt += context
    prompt += "\n\nRéponse :\n"

    return prompt


def answer_question(
    question: str,
    cfg: Optional[RAGConfig] = None,
    model: str = "gpt-4.1-mini",
) -> RAGAnswer:
    cfg = cfg or RAGConfig()

    retriever = SimpleVectorRetriever(cfg)
    intent, table_type = classify_intent(question)

    kind_filter = None
    if intent == "table_reconstruction":
        # privilégier les tableaux mais garder aussi du texte
        kind_filter = None

    chunks = retriever.search(question, kind_filter=kind_filter)
    context = _format_context(chunks)
    prompt = _build_prompt(question, context, intent, table_type)

    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Tu es un assistant financier spécialisé en analyse de rapports annuels."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )

    answer_md = completion.choices[0].message.content or ""
    return RAGAnswer(answer_markdown=answer_md, used_chunks=chunks)


def _markdown_table_to_dataframe(table_md: str) -> pd.DataFrame:
    """
    Convertit un tableau Markdown simple en DataFrame pandas.
    Hypothèse : format type `| col1 | col2 |` avec une ligne de séparation `| --- | --- |`.
    """
    lines = [l.strip() for l in table_md.splitlines() if l.strip()]
    if not lines:
        return pd.DataFrame()

    def _split_row(row: str) -> List[str]:
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        return [cell.strip() for cell in row.split("|")]

    header = None
    rows: List[List[str]] = []
    for line in lines:
        # ignorer les lignes de séparation '---'
        if set(line.replace("|", "").replace(" ", "")) == {"-"}:
            continue
        cells = _split_row(line)
        if header is None:
            header = cells
        else:
            # compléter / tronquer si nécessaire
            if len(cells) < len(header):
                cells.extend([""] * (len(header) - len(cells)))
            elif len(cells) > len(header):
                cells = cells[: len(header)]
            rows.append(cells)

    if header is None:
        return pd.DataFrame()

    return pd.DataFrame(rows, columns=header)


def _extract_bilan_actif_from_text(text: str) -> pd.DataFrame:
    """
    Extraction très simple du tableau de bilan actif à partir du texte brut OCR.
    On cherche les lignes entre 'BILAN ACTIF' et 'TOTAL GENERAL' contenant
    un libellé + 4 colonnes numériques (ou '-').
    """
    raw_lines = text.splitlines()
    # garder en parallèle une version normalisée pour les recherches
    lines_norm = [l.strip().lower() for l in raw_lines]

    # limiter au bloc BILAN ACTIF (recherche insensible à la casse / variantes de tiret)
    start_idx = None
    end_idx = None
    for i, l in enumerate(lines_norm):
        # accepte 'bilan actif', 'bilan - actif', 'bilan – actif'
        if re.search(r"bilan\s*[-–]?\s*actif", l):
            start_idx = i
        if "total general" in l and start_idx is not None:
            end_idx = i + 1
            break
    if start_idx is None or end_idx is None or end_idx <= start_idx:
        return pd.DataFrame()

    block = raw_lines[start_idx:end_idx]

    records: List[dict] = []
    for raw in block:
        line = raw.strip()
        if not line:
            continue
        # on ne garde que les lignes qui ont au moins un chiffre
        if not any(ch.isdigit() for ch in line):
            continue
        # on récupère tous les "gros nombres" (avec points/espaces de milliers, '.' ou ',' décimales) ou un tiret simple
        # ex : 1 641 086 378.07  ou  2.001.530,82  ou  -
        num_pattern = re.compile(r"((?:\d[\d\s\.]*\d(?:[.,]\d+)?|-))")
        matches = list(num_pattern.finditer(line))
        if len(matches) < 2:
            # trop peu de nombres/tirets pour être une ligne de données du tableau
            continue

        # on prend au plus les 4 derniers nombres/tirets comme colonnes
        last_matches = matches[-4:]
        first_span_start = last_matches[0].start()
        label = line[:first_span_start].strip()
        if not label:
            continue

        cols = [m.group(0).strip() for m in last_matches]
        # si moins de 4 colonnes trouvées, on complète avec vide
        while len(cols) < 4:
            cols.insert(0, "")

        rec = {
            "Rubrique": label,
            "Col1": cols[-4],
            "Col2": cols[-3],
            "Col3": cols[-2],
            "Col4": cols[-1],
        }
        records.append(rec)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    # Harmoniser et renommer les colonnes pour un bilan actif classique
    expected_cols = ["Rubrique", "Col1", "Col2", "Col3", "Col4"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[expected_cols]
    df.columns = [
        "Rubrique",
        "Brut",
        "Amortissements et provisions",
        "Net exercice",
        "Net exercice précédent",
    ]
    return df


def _extract_bilan_passif_from_text(text: str) -> pd.DataFrame:
    """
    Extraction du tableau de bilan passif à partir du texte brut OCR.
    On cherche les lignes entre 'BILAN PASSIF' et 'TOTAL GENERAL' contenant
    un libellé + 2 colonnes numériques (exercice / exercice précédent).
    """
    raw_lines = text.splitlines()
    lines_norm = [l.strip().lower() for l in raw_lines]

    start_idx = None
    end_idx = None
    for i, l in enumerate(lines_norm):
        # accepte 'bilan passif', 'bilan - passif', 'bilan – passif'
        if re.search(r"bilan\s*[-–]?\s*passif", l):
            start_idx = i
        if "total general" in l and start_idx is not None:
            end_idx = i + 1
            break
    if start_idx is None or end_idx is None or end_idx <= start_idx:
        return pd.DataFrame()

    block = raw_lines[start_idx:end_idx]

    records: List[dict] = []
    num_pattern = re.compile(r"((?:\d[\d\s\.]*\d(?:[.,]\d+)?|-))")

    for raw in block:
        line = raw.strip()
        if not line:
            continue
        if not any(ch.isdigit() for ch in line):
            continue

        matches = list(num_pattern.finditer(line))
        if len(matches) < 2:
            continue

        # pour le passif on garde les 2 derniers nombres/tirets
        last_matches = matches[-2:]
        first_span_start = last_matches[0].start()
        label = line[:first_span_start].strip()
        if not label:
            continue

        cols = [m.group(0).strip() for m in last_matches]
        while len(cols) < 2:
            cols.insert(0, "")

        rec = {
            "Rubrique": label,
            "Col1": cols[-2],
            "Col2": cols[-1],
        }
        records.append(rec)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    expected_cols = ["Rubrique", "Col1", "Col2"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[expected_cols]
    df.columns = [
        "Rubrique",
        "Exercice",
        "Exercice précédent",
    ]
    return df


def _extract_table_positionnel(doc_id: str, page_number: int, cfg: RAGConfig) -> pd.DataFrame:
    """
    Wrapper autour de rag_agent.positional.extract_table_positionnel.
    """
    pdf_path = os.path.join(cfg.raw_pdf_dir, doc_id)
    return extract_table_positionnel(pdf_path, page_number)


def _extract_compte_resultat_from_text(text: str) -> pd.DataFrame:
    """
    Extraction du tableau 'COMPTES DE PRODUITS ET CHARGES' à partir du texte brut.
    On cherche les lignes à partir de 'COMPTES DE PRODUITS ET CHARGES' et on
    extrait les libellés + jusqu'à 4 colonnes numériques.
    """
    raw_lines = text.splitlines()
    lines_norm = [l.strip().lower() for l in raw_lines]

    start_idx = None
    title_patterns = [
        "cpc",
        "compte de produits et charges",
        "compte de produits et de charges",
        "comptes de produits et charges",
        "compte de resultat",
        "compte de résultat",
        "compte de resultats",
        "compte de résultat consolidé",
        "compte de resultat consolide",
    ]
    for i, l in enumerate(lines_norm):
        if any(p in l for p in title_patterns):
            start_idx = i
            break
    # si aucun titre explicite trouvé, on traite toute la page comme bloc CPC
    if start_idx is None:
        start_idx = 0

    # on prend tout le bloc à partir de ce point (la page est censée contenir le CPC)
    block = raw_lines[start_idx:]

    records: List[dict] = []
    num_pattern = re.compile(r"((?:\d[\d\s\.]*\d(?:[.,]\d+)?|-))")  # nombres ou '-'

    for raw in block:
        line = raw.strip()
        if not line:
            continue

        if not any(ch.isdigit() for ch in line):
            continue

        matches = list(num_pattern.finditer(line))
        if len(matches) < 2:
            continue

        # pour le compte de résultat, on garde au plus les 4 derniers nombres
        last_matches = matches[-4:]
        first_span_start = last_matches[0].start()
        label = line[:first_span_start].strip()
        if not label:
            continue

        cols = [m.group(0).strip() for m in last_matches]
        while len(cols) < 4:
            cols.insert(0, "")

        rec = {
            "Rubrique": label,
            "Col1": cols[-4],
            "Col2": cols[-3],
            "Col3": cols[-2],
            "Col4": cols[-1],
        }
        records.append(rec)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    expected_cols = ["Rubrique", "Col1", "Col2", "Col3", "Col4"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = ""
    df = df[expected_cols]
    df.columns = [
        "Rubrique",
        "Propres à l'exercice",
        "Exercices précédents",
        "Total exercice",
        "Exercice précédent",
    ]
    return df


def export_tables_to_xlsx_for_question(
    question: str,
    out_path: str,
    cfg: Optional[RAGConfig] = None,
    doc_id_filter: Optional[str] = None,
    page: Optional[int] = None,
    debug: bool = False,
) -> List[RetrievedChunk]:
    """
    Sélectionne les tableaux pertinents pour une question via le retriever
    et les exporte dans un fichier Excel (une feuille par tableau).
    Si --doc est fourni et que ce document n'est pas dans l'index, une
    ingestion automatique est lancée pour indexer tous les PDF du dossier.
    """
    cfg = cfg or RAGConfig()
    ensure_dirs(cfg)

    # Ingestion automatique si le document ciblé n'est pas encore indexé
    if doc_id_filter:
        meta_path = os.path.join(cfg.index_dir, "metadata.json")
        doc_in_index = False
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                doc_filter_lower = doc_id_filter.lower()
                doc_in_index = any(
                    doc_filter_lower in (m.get("doc_id") or "").lower()
                    for m in meta
                )
            except (json.JSONDecodeError, OSError):
                pass
        if not doc_in_index:
            build_index(cfg)

    retriever = SimpleVectorRetriever(cfg)

    intent, table_type = classify_intent(question)

    # normalisation éventuelle du filtre document
    doc_filter_norm = doc_id_filter.lower() if doc_id_filter else None

    # On commence par les tableaux explicites via le RAG (chunks kind='table')
    table_chunks = retriever.search(question, kind_filter=["table"])
    if doc_filter_norm:
        table_chunks = [c for c in table_chunks if doc_filter_norm in c.doc_id.lower()]
    if page is not None:
        table_chunks = [c for c in table_chunks if c.page == page]

    # Index rapide page -> texte pour d'éventuels fallback regex (bilans, CPC)
    page_texts: dict[int, str] = {}
    for m in retriever.meta:
        if m.get("kind") != "text":
            continue
        doc_id = m.get("doc_id", "")
        if doc_filter_norm and doc_filter_norm not in str(doc_id).lower():
            continue
        try:
            p = int(m.get("page"))
        except (TypeError, ValueError):
            continue
        page_texts[p] = m.get("content", "")

    tables: List[tuple[str, pd.DataFrame]] = []
    diagnostics: List[str] = []

    # Cas spécifiques : bilan / compte de résultat localisés par RAG sur les tableaux,
    # puis conversion directe du contenu structuré (Markdown) en DataFrame + comparaison
    # avec une extraction positionnelle sur la même page.
    if table_type == "bilan":
        q_lower = question.lower()
        is_passif = "passif" in q_lower
        want_consolide = "consolid" in q_lower  # couvre consolidé / consolide

        def score_bilan_chunk(c: RetrievedChunk) -> int:
            text = ((c.section_hint or "") + "\n" + c.content).lower()
            score = 0
            if "bilan" in text:
                score += 2
            if is_passif and "passif" in text:
                score += 2
            if (not is_passif) and "actif" in text:
                score += 2
            if want_consolide and "consolid" in text:
                score += 2
            # petit bonus sur les tableaux très structurés
            if "|" in c.content:
                score += 1
            return score

        scored = [(score_bilan_chunk(c), c.score, c) for c in table_chunks]
        scored = [t for t in scored if t[0] > 0] or scored  # si rien ne matche, garder meilleurs RAG

        df_bilan = pd.DataFrame()
        page_num: Optional[int] = None
        doc_id: Optional[str] = None

        if scored:
            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
            best = scored[0][2]
            page_num = page if page is not None else best.page
            doc_id = doc_id_filter or best.doc_id
        elif page is not None:
            page_num = page
            doc_id = doc_id_filter
            if not doc_id:
                for m in retriever.meta:
                    if m.get("kind") != "text" or (doc_filter_norm and doc_filter_norm not in str(m.get("doc_id", "")).lower()):
                        continue
                    if int(m.get("page", 0)) == page_num:
                        doc_id = m.get("doc_id")
                        break

        if page_num is not None and doc_id:
            pdf_path = _resolve_pdf_path(doc_id, cfg.raw_pdf_dir)
            table_name = "BILAN PASSIF" if is_passif else "BILAN ACTIF"
            result = _extraction_agent.extract(
                pdf_path=pdf_path,
                page_num=page_num,
                table_name=table_name,
            )
            df_bilan = result.df
            print(result.summary())
            if result.success:
                logger.info(result.summary())
            else:
                logger.warning("Extraction incomplète : %s", result.warnings)

        if not df_bilan.empty:
            sheet = "bilan_passif" if is_passif else "bilan_actif"
            prepared, diag = _prepare_tables_for_output(
                main_df=df_bilan,
                pos_df=pd.DataFrame(),
                table_type=sheet,
                sheet_name=sheet,
                debug=debug,
            )
            tables.extend(prepared)
            diagnostics.extend(diag)

    elif table_type == "compte_resultat":
        cr_keywords = [
            "cpc",
            "compte de produits et charges",
            "compte de produits et de charges",
            "comptes de produits et charges",
            "compte de resultat",
            "compte de résultat",
            "compte de resultats",
            "compte de résultat consolidé",
            "compte de resultat consolide",
        ]

        def score_cr_chunk(c: RetrievedChunk) -> int:
            text = ((c.section_hint or "") + "\n" + c.content).lower()
            score = 0
            for kw in cr_keywords:
                if kw in text:
                    score += 2
            # bonus si beaucoup de colonnes numériques (tableau CPC complet)
            if text.count("|") >= 10:
                score += 1
            return score

        scored_cr = [(score_cr_chunk(c), c.score, c) for c in table_chunks]
        keyword_hits = [t for t in scored_cr if t[0] > 0]

        if keyword_hits:
            keyword_hits.sort(key=lambda t: (t[0], t[1]), reverse=True)
            best_cr = keyword_hits[0][2]
            page_num = page if page is not None else best_cr.page
            doc_id = doc_id_filter or best_cr.doc_id

            if doc_id:
                pdf_path = _resolve_pdf_path(doc_id, cfg.raw_pdf_dir)
                result = _extraction_agent.extract(
                    pdf_path=pdf_path,
                    page_num=page_num,
                    table_name="COMPTES DE PRODUITS ET CHARGES",
                )
                df_cr = result.df
                print(result.summary())
                if result.success:
                    logger.info(result.summary())
                else:
                    logger.warning("Extraction incomplète : %s", result.warnings)

                if not df_cr.empty:
                    prepared, diag = _prepare_tables_for_output(
                        main_df=df_cr,
                        pos_df=pd.DataFrame(),
                        table_type="compte_resultat",
                        sheet_name="compte_resultat",
                        debug=debug,
                    )
                    tables.extend(prepared)
                    diagnostics.extend(diag)
        else:
            # Aucun chunk "table" CPC : on cherche les pages avec mots-clés puis extraction par agent
            pages_to_try: List[int] = []
            doc_id_for_page: dict[int, str] = {}
            for m in retriever.meta:
                if m.get("doc_id") is None or m.get("kind") != "text":
                    continue
                if doc_filter_norm and doc_filter_norm not in m["doc_id"].lower():
                    continue
                content_lc = m.get("content", "").lower()
                if any(kw in content_lc for kw in cr_keywords):
                    p = int(m["page"])
                    pages_to_try.append(p)
                    doc_id_for_page[p] = m["doc_id"]
            pages_to_try = sorted(set(pages_to_try))

            for p in pages_to_try:
                doc_id = doc_id_filter or doc_id_for_page.get(p, "")
                if not doc_id:
                    continue
                pdf_path = _resolve_pdf_path(doc_id, cfg.raw_pdf_dir)
                result = _extraction_agent.extract(
                    pdf_path=pdf_path,
                    page_num=p,
                    table_name="COMPTES DE PRODUITS ET CHARGES",
                )
                df_cr = result.df
                print(result.summary())
                if result.success:
                    logger.info(result.summary())
                else:
                    logger.warning("Extraction incomplète : %s", result.warnings)

                if not df_cr.empty:
                    prepared, diag = _prepare_tables_for_output(
                        main_df=df_cr,
                        pos_df=pd.DataFrame(),
                        table_type="compte_resultat",
                        sheet_name="compte_resultat",
                        debug=debug,
                    )
                    tables.extend(prepared)
                    diagnostics.extend(diag)
                    break

    if not tables:
        # On crée quand même un fichier vide pour signaler qu'il n'y avait pas de tableau exploitable
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            pd.DataFrame({"info": ["Aucun tableau pertinent trouvé."]}).to_excel(
                writer, sheet_name="info", index=False
            )
        return []

    with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
        for sheet_name, df in tables:
            safe_name = sheet_name[:31]  # limite Excel
            df.to_excel(writer, sheet_name=safe_name, index=False)

        if diagnostics:
            pd.DataFrame({"message": diagnostics}).to_excel(
                writer, sheet_name="info", index=False
            )

    # On renvoie tous les chunks utilisés pour info (textes + tableaux)
    return table_chunks


def _prepare_tables_for_output(
    main_df: pd.DataFrame,
    pos_df: pd.DataFrame,
    table_type: str,
    sheet_name: str,
    debug: bool,
) -> Tuple[List[Tuple[str, pd.DataFrame]], List[str]]:
    """
    Combine l'extraction RAG (main_df) et l'extraction positionnelle (pos_df)
    et décide quoi écrire dans l'Excel (feuille principale + éventuelles feuilles de debug).
    """
    tables: List[Tuple[str, pd.DataFrame]] = []
    diagnostics: List[str] = []

    ok_main, err_main = _validate_table(main_df, table_type)
    ok_pos, err_pos = _validate_table(pos_df, table_type) if not pos_df.empty else (False, [])

    def _shape(df: pd.DataFrame) -> Tuple[int, int]:
        return (len(df), len(df.columns)) if df is not None else (0, 0)

    r_rows, r_cols = _shape(main_df)
    p_rows, p_cols = _shape(pos_df)

    mismatch = False
    if r_rows and p_rows:
        if abs(r_rows - p_rows) > 2 or abs(r_cols - p_cols) > 1:
            mismatch = True

    # Choix du tableau principal
    primary = main_df if ok_main or not pos_df.empty else main_df

    if primary.empty:
        diagnostics.append(f"[{table_type}] Aucun tableau exploitable (RAG et positionnel vides).")
        return tables, diagnostics

    tables.append((sheet_name, primary))

    # Conditions pour ajouter les feuilles de debug
    need_debug = debug or not ok_main or mismatch or (pos_df is not None and not ok_pos)

    if need_debug:
        if not main_df.empty:
            tables.append((f"{sheet_name}_rag", main_df))
        if pos_df is not None and not pos_df.empty:
            tables.append((f"{sheet_name}_pos", pos_df))

        diagnostics.extend(
            [
                f"[{table_type}] validation RAG: {'OK' if ok_main else 'KO'} ({'; '.join(err_main) or 'aucun détail'})",
                f"[{table_type}] validation positionnel: {'OK' if ok_pos else 'KO'} ({'; '.join(err_pos) or 'aucun détail'})",
                f"[{table_type}] formes RAG={r_rows}x{r_cols}, POS={p_rows}x{p_cols}, mismatch={mismatch}",
            ]
        )

    return tables, diagnostics


def _validate_table(df: pd.DataFrame, table_type: str) -> Tuple[bool, List[str]]:
    """
    Validation très simple pour éviter les pires erreurs d'extraction.
    On vérifie :
      - tableau non vide, au moins quelques lignes/colonnes
      - présence d'au moins quelques valeurs numériques
      - pour le CPC : présence d'une ligne 'résultat net'
    """
    errors: List[str] = []
    if df is None or df.empty:
        errors.append("tableau vide")
        return False, errors

    if len(df) < 3 or len(df.columns) < 2:
        errors.append("dimensions trop petites")

    numeric_cells = 0
    for col in df.columns:
        series = df[col]
        # Si plusieurs colonnes portent le même nom, df[col] peut être un DataFrame
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        series = series.astype(str)
        numeric_cells += series.str.contains(r"\d{3,}", regex=True).sum()
    if numeric_cells < 5:
        errors.append("trop peu de valeurs numériques détectées")

    if "compte_resultat" in table_type:
        text_all = " ".join(df.astype(str).fillna("").values.ravel()).lower()
        if "resultat net" not in text_all and "résultat net" not in text_all:
            errors.append("ligne 'résultat net' non trouvée")

    ok = not errors
    return ok, errors


