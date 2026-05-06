import argparse
import os
import re

import pandas as pd

from rag_agent.config import RAGConfig, ensure_dirs
from rag_agent.ingest import build_index, auto_index_all_pdfs
from rag_agent.pipeline import (
    answer_question,
    export_tables_to_xlsx_for_question,
    _resolve_pdf_path,
)
from rag_agent.extraction_agent import ExtractionAgent, find_page_for_table


def cmd_ingest(args: argparse.Namespace) -> None:
    cfg = RAGConfig()
    if args.pdf_dir:
        cfg.raw_pdf_dir = args.pdf_dir
    ensure_dirs(cfg)
    pdf_only = getattr(args, "pdf", None)
    use_fast = getattr(args, "fast", False)
    workers = getattr(args, "workers", 1)
    build_index(
        cfg,
        pdf_paths=[pdf_only] if pdf_only else None,
        use_fast=use_fast,
        workers=workers,
    )


def cmd_auto_index(args: argparse.Namespace) -> None:
    cfg = RAGConfig()
    if getattr(args, "pdf_dir", None):
        cfg.raw_pdf_dir = args.pdf_dir
    ensure_dirs(cfg)
    auto_index_all_pdfs(
        cfg,
        use_fast=getattr(args, "fast", False),
        force=getattr(args, "force", False),
    )


def cmd_ask(args: argparse.Namespace) -> None:
    cfg = RAGConfig()
    ensure_dirs(cfg)
    answer = answer_question(args.question, cfg=cfg)
    print(answer.answer_markdown)


def cmd_export_xlsx(args: argparse.Namespace) -> None:
    cfg = RAGConfig()
    ensure_dirs(cfg)
    # Construire automatiquement le dossier de sortie: output/<nom_pdf_sans_ext>/
    doc_name = args.doc if args.doc else "generic"
    doc_base = os.path.splitext(os.path.basename(doc_name))[0]
    base_out_dir = os.path.join("output", doc_base)
    os.makedirs(base_out_dir, exist_ok=True)
    out_path = os.path.join(base_out_dir, args.out)

    export_tables_to_xlsx_for_question(
        args.question,
        out_path,
        cfg=cfg,
        doc_id_filter=args.doc,
        page=None,
        debug=args.debug,
    )
    print(f"Tableaux pertinents exportés vers: {out_path}")


def _table_name_to_filename(table_name: str) -> str:
    """Ex: 'BILAN ACTIF' -> 'bilan_actif.xlsx'."""
    slug = re.sub(r"[^\w\s-]", "", table_name.lower())
    slug = re.sub(r"[-\s]+", "_", slug).strip("_") or "tableau"
    return f"{slug}.xlsx"


# Pattern pour détecter des dates (31/12/2024, 31-12-2023, etc.)
_DATE_PATTERN = re.compile(r"\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}")


def _split_cell_into_two(s: str) -> tuple:
    """
    Si la cellule contient deux valeurs (deux dates ou deux nombres), retourne (val1, val2).
    Sinon retourne (s, "").
    """
    s = str(s).strip()
    if not s:
        return ("", "")
    dates = _DATE_PATTERN.findall(s)
    if len(dates) >= 2:
        parts = re.split(r"\s+", s, maxsplit=1)
        if len(parts) >= 2 and _DATE_PATTERN.search(parts[0]) and _DATE_PATTERN.search(parts[1]):
            return (parts[0].strip(), parts[1].strip())
        return (dates[0], dates[1])
    tokens = s.split()
    # Ne jamais couper un seul grand nombre (ex. 1 243 127 105,00) : s'il y a une partie décimale (,00 ou .00), c'est un seul montant
    if re.search(r"[,.]\d{2}\s*$", s) or re.search(r"\d+[,.]\d{2}", s):
        return (s, "")
    # Au moins 4 tokens pour éviter de couper un seul montant (ex. "27 047 543") ; pas de décimale = deux montants côte à côte
    if len(tokens) >= 4 and all(t.replace(",", "").replace(".", "").replace(" ", "").isdigit() for t in tokens):
        mid = len(tokens) // 2
        return (" ".join(tokens[:mid]), " ".join(tokens[mid:]))
    return (s, "")


def _ensure_unique_column_names(df: pd.DataFrame) -> None:
    """Modifie df pour que les noms de colonnes soient uniques (évite df[col] = DataFrame)."""
    cols = list(df.columns)
    seen: dict[str, int] = {}
    new_names = []
    for c in cols:
        name = str(c) if c is not None else ""
        if name in seen:
            seen[name] += 1
            new_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            new_names.append(name)
    df.columns = new_names


def _split_cell_consolidated_note_dates(s: str) -> tuple:
    """
    Comptes consolidés : cellule du type "5 31/12/2024 31/12/2023" ou "31/12/2024 31/12/2023".
    Retourne (notes, date1, date2) ou (None, None, None) si pas ce pattern.
    """
    s = str(s).strip()
    if not s:
        return (None, None, None)
    dates = list(_DATE_PATTERN.finditer(s))
    if len(dates) < 2:
        return (None, None, None)
    first_start = dates[0].start()
    note_part = s[:first_start].strip()
    date1 = dates[0].group(0)
    date2 = dates[1].group(0)
    return (note_part, date1, date2)


def _split_combined_date_or_number_columns(df: pd.DataFrame, table_name: str = "") -> pd.DataFrame:
    """
    Détecte les colonnes où les cellules contiennent "31/12/2024 31/12/2023" ou "27 047 543 22 709 918"
    et les scinde en deux colonnes (une par date / une par montant).
    Comptes consolidés : une colonne "Note date1 date2" est scindée en 3 (Notes, date1, date2) pour avoir 4 colonnes au total.
    """
    if df is None or df.empty or len(df.columns) < 2:
        return df
    df = df.copy()
    # Éviter df[col] = DataFrame quand des colonnes ont le même nom (ex. extraction pdfplumber)
    _ensure_unique_column_names(df)
    name_lower = (table_name or "").lower()
    is_consolidated = "consolid" in name_lower

    # Comptes consolidés : une colonne peut être "Notes date1 date2" en en-tête et "note montant1 montant2" en données
    # -> scinder en 3 colonnes (Notes, date1/montant1, date2/montant2)
    if is_consolidated:
        cols = list(df.columns)
        for col in cols:
            series = df[col].astype(str)
            has_note_dates = any(_split_cell_consolidated_note_dates(v)[1] is not None for v in series)
            has_two_numbers = any(
                _split_cell_into_two(v)[1] != ""
                and not _DATE_PATTERN.search(str(v))
                for v in series
            )
            if not (has_note_dates or has_two_numbers):
                continue
            notes_vals = []
            date1_vals = []
            date2_vals = []
            for v in series:
                note_part, d1, d2 = _split_cell_consolidated_note_dates(v)
                if d1 is not None and d2 is not None:
                    notes_vals.append(note_part if note_part is not None else "")
                    date1_vals.append(d1)
                    date2_vals.append(d2)
                else:
                    # Un seul grand nombre avec décimales (ex. 1 243 127 105,00) -> ne pas couper, mettre en col1
                    if re.search(r"[,.]\d{2}\s*$", str(v)) or re.search(r"\d+[,.]\d{2}", str(v)):
                        notes_vals.append("")
                        date1_vals.append(str(v).strip())
                        date2_vals.append("")
                        continue
                    # Données : "5 27 047 543 22 709 918" -> note=5, col1=27 047 543, col2=22 709 918
                    tokens = str(v).split()
                    if len(tokens) >= 5 and all(t.replace(",", "").replace(".", "").isdigit() for t in tokens):
                        # Premier token court = note (1-2 chiffres), le reste = deux montants
                        if len(tokens[0]) <= 3:
                            note_tok = tokens[0]
                            rest = tokens[1:]
                            mid = len(rest) // 2
                            notes_vals.append(note_tok)
                            date1_vals.append(" ".join(rest[:mid]))
                            date2_vals.append(" ".join(rest[mid:]))
                        else:
                            left, right = _split_cell_into_two(v)
                            notes_vals.append("")
                            date1_vals.append(left)
                            date2_vals.append(right)
                    else:
                        left, right = _split_cell_into_two(v)
                        if right != "":
                            notes_vals.append("")
                            date1_vals.append(left)
                            date2_vals.append(right)
                        else:
                            notes_vals.append(str(v).strip())
                            date1_vals.append("")
                            date2_vals.append("")
            idx = df.columns.get_loc(col)
            df = df.drop(columns=[col])
            df.insert(idx, "Notes", notes_vals)
            col1_name = date1_vals[0] if date1_vals and date1_vals[0] else "31/12/2024"
            col2_name = date2_vals[0] if date2_vals and date2_vals[0] else "31/12/2023"
            df.insert(idx + 1, col1_name, date1_vals)
            df.insert(idx + 2, col2_name, date2_vals)
            break

    cols = list(df.columns)
    i = 0
    while i < len(cols):
        col = cols[i]
        if col not in df.columns:
            i += 1
            continue
        series = df[col].astype(str)
        to_split = False
        for v in series:
            left, right = _split_cell_into_two(v)
            if right != "":
                to_split = True
                break
        if not to_split:
            i += 1
            continue
        new_left = []
        new_right = []
        for v in series:
            left, right = _split_cell_into_two(v)
            new_left.append(left)
            new_right.append(right)
        idx = df.columns.get_loc(col)
        name1 = col
        name2 = str(col) + "_2"
        if new_left and _DATE_PATTERN.search(str(new_left[0])):
            name1 = str(new_left[0]).strip()
            name2 = str(new_right[0]).strip() if new_right else "31/12/2023"
        df.insert(idx + 1, name2, new_right)
        df[name1] = new_left
        cols = list(df.columns)
        i += 2
    return df


def _ensure_date_columns_in_excel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si une ligne contient des dates (ex. 31/12/2024, 31/12/2023), les utiliser comme noms de colonnes.
    Une colonne = une date (31/12/2024, 31/12/2023), puis on supprime cette ligne des données.
    """
    if df is None or df.empty or len(df) < 2:
        return df
    df = df.copy()
    for row_idx in range(min(3, len(df))):
        row = df.iloc[row_idx]
        new_names = []
        date_count = 0
        for col_idx, val in enumerate(row):
            s = str(val).strip()
            if _DATE_PATTERN.search(s) and len(s) <= 14:
                new_names.append(s)
                date_count += 1
            else:
                new_names.append(df.columns[col_idx] if col_idx < len(df.columns) else f"Col{col_idx}")
        if date_count >= 2:
            df.columns = new_names
            df = df.drop(df.index[row_idx]).reset_index(drop=True)
            break
    return df


def _fill_sparse_column_from_previous(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si une colonne (ex. exercice précédent) n'a qu'une ou très peu de valeurs alors que la précédente
    en a beaucoup, remplir en découpant les cellules de la colonne précédente en deux montants
    (ex. "27 047 543 22 709 918" -> col précédente=27 047 543, col actuelle=22 709 918).
    """
    if df is None or df.empty or len(df.columns) < 2:
        return df
    df = df.copy()
    for j in range(1, len(df.columns)):
        prev = df.iloc[:, j - 1].astype(str)
        curr = df.iloc[:, j].astype(str)
        non_empty_curr = sum(1 for v in curr if str(v).strip())
        non_empty_prev = sum(1 for v in prev if str(v).strip())
        if non_empty_curr >= non_empty_prev or non_empty_prev < len(df) // 2:
            continue
        new_prev, new_curr = [], []
        for r in range(len(df)):
            cv = str(curr.iloc[r]).strip()
            pv = str(prev.iloc[r]).strip()
            if cv:
                new_prev.append(pv)
                new_curr.append(cv)
                continue
            left, right = _split_cell_into_two(pv)
            new_prev.append(left)
            new_curr.append(right if right else "")
        if sum(1 for v in new_curr if v) > non_empty_curr:
            df.iloc[:, j - 1] = new_prev
            df.iloc[:, j] = new_curr
    return df


def _merge_totaux_exercice_precedent_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fusionne les colonnes dont les en-têtes sont des fragments de "TOTAUX DE L'EXERCICE PRECEDENT"
    (souvent répartis sur plusieurs lignes/colonnes par l'extraction PDF).
    """
    if df is None or df.empty or len(df.columns) < 3:
        return df
    df = df.copy()
    cols = list(df.columns)
    # Normaliser pour comparaison (sans retours à la ligne)
    norm = [str(c).replace("\n", " ").strip().lower() for c in cols]
    i = 0
    while i < len(cols):
        # 3 colonnes : "totaux de" / "l'exercice" ou "exercice" / "precedent"
        if i <= len(cols) - 3:
            a, b, c = norm[i], norm[i + 1], norm[i + 2]
            if ("totaux" in a and "de" in a) and ("exercice" in b or "l'exercice" in b) and ("precedent" in c or "précédent" in c):
                name_merged = "TOTAUX DE L'EXERCICE PRECEDENT"
                vals = [str(df.iloc[r, i + 2]).strip() or str(df.iloc[r, i + 1]).strip() or str(df.iloc[r, i]).strip() for r in range(len(df))]
                df = df.iloc[:, [j for j in range(len(df.columns)) if j not in (i, i + 1, i + 2)]].copy()
                df.insert(i, name_merged, vals)
                cols, norm = list(df.columns), [str(c).replace("\n", " ").strip().lower() for c in df.columns]
                continue
        # 2 colonnes : "totaux de" / "l'exercice precedent"
        if i <= len(cols) - 2:
            a, b = norm[i], norm[i + 1]
            if ("totaux" in a and "de" in a) and "exercice" in b and ("precedent" in b or "précédent" in b):
                name_merged = "TOTAUX DE L'EXERCICE PRECEDENT"
                vals = [str(df.iloc[r, i + 1]).strip() or str(df.iloc[r, i]).strip() for r in range(len(df))]
                df = df.iloc[:, [j for j in range(len(df.columns)) if j not in (i, i + 1)]].copy()
                df.insert(i, name_merged, vals)
                cols, norm = list(df.columns), [str(c).replace("\n", " ").strip().lower() for c in df.columns]
                continue
        i += 1
    return df


def cmd_extract_table(args: argparse.Namespace) -> None:
    """Étape 1 : recherche de la page via RAG. Étape 2 : extraction avec l'agent. Export Excel sous output/<nom_pdf>/."""
    cfg = RAGConfig()
    ensure_dirs(cfg)
    doc = args.doc
    table_name = args.table_name
    page = find_page_for_table(doc, table_name)
    if page is None:
        print("Erreur: page non trouvée (lancez l'ingestion: python main.py ingest).")
        return
    print(f"Page {page}")
    pdf_path = _resolve_pdf_path(doc, cfg.raw_pdf_dir)
    agent = ExtractionAgent(use_llm=True)
    result = agent.extract(pdf_path, page, table_name)
    print(result.summary())
    if result.warnings:
        for w in result.warnings:
            print("  [!]", w)

    doc_base = os.path.splitext(os.path.basename(doc))[0]
    base_out_dir = os.path.join("output", doc_base)
    os.makedirs(base_out_dir, exist_ok=True)
    out_filename = _table_name_to_filename(table_name)
    out_path = os.path.join(base_out_dir, out_filename)

    if not result.df.empty:
        sheet_name = _table_name_to_filename(table_name).replace(".xlsx", "")[:31]
        df = result.df.copy()
        # Découper les cellules "date date" / "montant montant" en deux colonnes ; consolidé -> 4 colonnes (ACTIF, Notes, date1, date2)
        df = _split_combined_date_or_number_columns(df, table_name)
        # Si une ligne contient des dates (31/12/2024, 31/12/2023), en faire les noms de colonnes
        df = _ensure_date_columns_in_excel(df)
        # Noms de colonnes sur une seule ligne (évite retours à la ligne dans Excel)
        df.columns = [str(c).replace("\n", " ").strip() for c in df.columns]
        # Fusionner les colonnes fragmentées "TOTAUX DE" / "L'EXERCICE" / "PRECEDENT" en une seule
        df = _merge_totaux_exercice_precedent_columns(df)
        # Remplir les colonnes qui n'ont qu'une valeur (ex. _5) en prenant la 2e partie des cellules de la colonne précédente
        df = _fill_sparse_column_from_previous(df)
        # Remplacer les retours à la ligne dans les cellules par un espace pour une structure claire (par index pour colonnes dupliquées)
        for i in range(len(df.columns)):
            df.iloc[:, i] = df.iloc[:, i].astype(str).str.replace("\n", " ", regex=False)
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            workbook = writer.book
            worksheet = writer.sheets[sheet_name]
            header_fmt = workbook.add_format({"bold": True})
            for col_idx, col_name in enumerate(df.columns):
                worksheet.write(0, col_idx, col_name, header_fmt)
            for col_idx in range(len(df.columns)):
                ser = df.iloc[:, col_idx]
                col_name = df.columns[col_idx]
                max_len = max(
                    ser.astype(str).str.len().max() if len(df) > 0 else 0,
                    len(str(col_name)),
                )
                worksheet.set_column(col_idx, col_idx, min(max_len + 2, 55))
        print(f"Export Excel : {out_path}")
        print(result.df.to_string())
    else:
        print("(DataFrame vide)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG financier simple sur rapports PDF (texte + tableaux)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Indexer des rapports PDF.")
    p_ingest.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help="Dossier contenant les PDF (par défaut: data/pdf).",
    )
    p_ingest.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Ingérer uniquement ce PDF (ex: data/pdf/AWB_RFA_2024_1.pdf).",
    )
    p_ingest.add_argument(
        "--fast",
        action="store_true",
        help="Extraction rapide (PyMuPDF, texte seul, pas de tableaux). Beaucoup plus rapide.",
    )
    p_ingest.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Nombre de PDF traités en parallèle (défaut: 1). Ex: --workers 3.",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    p_auto = sub.add_parser(
        "auto-index",
        help="Indexer automatiquement tous les PDFs non encore indexés dans data/pdfs/.",
    )
    p_auto.add_argument(
        "--pdf-dir",
        type=str,
        default=None,
        help="Dossier PDF à scanner (par défaut: data/pdfs).",
    )
    p_auto.add_argument(
        "--fast",
        action="store_true",
        help="Extraction rapide (PyMuPDF, texte seul).",
    )
    p_auto.add_argument(
        "--force",
        action="store_true",
        help="Forcer le rebuild même si l'index existe déjà.",
    )
    p_auto.set_defaults(func=cmd_auto_index)

    p_ask = sub.add_parser("ask", help="Poser une question au RAG.")
    p_ask.add_argument("question", type=str, help="Question en langage naturel.")
    p_ask.set_defaults(func=cmd_ask)

    p_export = sub.add_parser(
        "export-xlsx",
        help="Exporter les tableaux pertinents pour une question sous forme de fichier .xlsx.",
    )
    p_export.add_argument("question", type=str, help="Question en langage naturel.")
    p_export.add_argument(
        "--out",
        type=str,
        required=True,
        help="Chemin du fichier .xlsx de sortie.",
    )
    p_export.add_argument(
        "--doc",
        type=str,
        default=None,
        help="Nom du document (doc_id) à cibler, ex: Addoha_RFA_2024.pdf.",
    )
    p_export.add_argument(
        "--debug",
        action="store_true",
        help="Active le mode debug (sauvegarde des tableaux bruts et infos de validation).",
    )
    p_export.set_defaults(func=cmd_export_xlsx)

    p_extract = sub.add_parser(
        "extract-table",
        help="Recherche la page du tableau via RAG puis lance l'extraction (affiche Page X puis le tableau).",
    )
    p_extract.add_argument(
        "--doc",
        type=str,
        required=True,
        help="Document PDF, ex: Addoha_RFA_2024.pdf.",
    )
    p_extract.add_argument(
        "--table-name",
        type=str,
        required=True,
        help='Nom du tableau, ex: "BILAN ACTIF".',
    )
    p_extract.set_defaults(func=cmd_extract_table)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()


