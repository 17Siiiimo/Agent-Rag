# Build PDF RAG Vision Pipeline

## Goal

Build and stabilize a PDF RAG vision pipeline for financial reports. The pipeline should take source PDFs, extract page text and visual/table evidence, identify the relevant financial statements, crop or localize tables when needed, run vision extraction, validate results, and make the extracted data retrievable for downstream RAG use.

## Current Project State

- Main app/API entrypoint: `api.py`
- Financial extraction package: `rag_agent/financial_extraction/`
- Existing task/debug context: `output/financial_extraction_debug/full_label_signatures.md`
- RAG evidence helper: `rag_agent/financial_extraction/rag_evidence.py`
- Current label signatures cover financial table families such as banking social accounts, including `BILAN_ACTIF`, `BILAN_PASSIF`, and `CPC`.
- The repo already contains pipeline-style modules:
  - `page_renderer.py` for rendering PDF pages
  - `page_text_extractor.py` and `pdf_loader.py` for PDF text/input handling
  - `page_indexer.py` for indexing pages
  - `document_classifier.py` for document/table type classification
  - `hybrid_retriever.py` for retrieval
  - `table_localizer.py` and `table_cropper.py` for locating/cropping tables
  - `vision_extractor.py` for vision-based extraction
  - `table_validator.py` for validation
  - `rag_evidence.py` for row-level RAG chunks after table extraction
  - `extraction_orchestrator.py` as the likely main workflow coordinator
  - `benchmark.py` and scripts under `scripts/` for regression/testing

## Important Warning

The working tree currently has many existing uncommitted changes and deleted files that were not created by Codex during this task-memory setup. Do not clean, reset, or revert them unless the user explicitly asks.

## Working Definition Of Done

The pipeline is ready when one command or API flow can:

1. Accept a target PDF and metadata such as issuer, year, period, and account scope.
2. Render or read the pages.
3. Retrieve likely pages for the target financial statement.
4. Classify the document/table type.
5. Localize and crop the relevant table region.
6. Run vision extraction on the crop or full page.
7. Normalize and validate extracted rows against label signatures.
8. Store enough structured evidence for RAG answers, including source PDF, page number, statement type, row labels, values, and debug artifacts.
9. Pass at least one focused regression script on a known PDF.

Current progress on item 8: each table extraction now writes `rag_evidence_chunks.json` beside `final_extracted.json`, and multi-table runs also write a combined `rag_evidence_chunks.json` at the run output root.

## Resume Checklist

When continuing this task, start here:

1. Read this file.
2. Inspect `rag_agent/financial_extraction/extraction_orchestrator.py`.
3. Inspect `rag_agent/financial_extraction/hybrid_retriever.py`.
4. Inspect `rag_agent/financial_extraction/vision_extractor.py`.
5. Inspect one current test script, preferably `scripts/test_awb_2024_annuel_sociaux_vision.py` if present.
6. Decide the next smallest improvement: retrieval, table localization, vision parsing, validation, storage, or API integration.
7. Run the narrowest relevant script before and after changing code.

Latest completed step:

- Added row-level RAG evidence chunk creation and persistence.
- Added `tests/test_rag_evidence.py` for the chunk builder.
- Fixed a CPC page-selection regression found in `maroc_telecom_2023_annuel/comptes_consolides/CPC`: weak fallback pages with zero label-signature hits no longer outrank localized CPC candidates with dense row-label matches.
- Added `tests/test_table_localizer_ranking.py` for that ranking behavior.

Next best step:

- Expose or consume these evidence chunks from the API/search side, so user questions can retrieve normalized financial rows rather than only page text.
- Rerun the Maroc Telecom 2023 annual consolidated CPC extraction with `force_page` or `force_recrop` to regenerate the crop and final extraction from the corrected page ranking.

## Likely Next Engineering Steps

- Map the current orchestrator flow end to end.
- Identify where extracted table data becomes structured rows.
- Confirm whether embeddings are already active via `embedding_model.py` and `hybrid_retriever.py`.
- Define or confirm the chunk/evidence schema for RAG:
  - `issuer`
  - `year`
  - `period`
  - `document_type`
  - `account_scope`
  - `statement_type`
  - `source_pdf`
  - `page_number`
  - `bbox` or crop path when available
  - `row_label`
  - `columns`
  - `values`
  - `confidence`
  - `raw_vision_text`
  - `debug_artifacts`
- Add or update a regression case once the next pipeline fix is implemented.

## Useful Commands

```powershell
python -m unittest discover -s tests -p "test_*.py"
python scripts\test_awb_2024_annuel_sociaux_vision.py
python scripts\benchmark_financial_crops.py
python scripts\audit_financial_rag.py
```

Only run commands that match the files currently present in the working tree.

## Open Questions

- Should the RAG store be local only, or should it use an external vector database?
- Which model/provider should be used for final vision extraction?
- Should cropped table images be persisted as first-class evidence artifacts?
- Should RAG retrieval answer from normalized structured rows, raw extracted text, image evidence, or a hybrid of all three?

## Last Updated

Created by Codex on 2026-05-14 for the task: "build PDF RAG vision pipeline".

Updated by Codex on 2026-05-14: added RAG evidence chunk output.

Updated by Codex on 2026-05-14: fixed localizer ranking so CPC candidates with real label-signature hits beat weak fallback pages.
