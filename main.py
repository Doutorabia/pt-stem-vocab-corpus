#!/usr/bin/env python3
"""
Pipeline entry-point: iterate through PDFs, extract text, run TextRank,
and persist keyword CSVs by subject and year.

Usage:
    python main.py
"""
from pathlib import Path

from src.config   import PDF_DIR, OUTPUT_DIR
from src.utils    import (
    collect_pdfs,
    YEAR_RE,
    build_corpus,
    extract_keywords,
    save_keywords_csv,
)
from src.extractor import DocumentProcessor


def _process_pdf_group(
    discipline: str,
    label: str,
    pdf_paths: list[Path],
    processor: DocumentProcessor,
) -> None:
    """Extracts text from *pdf_paths* and writes one CSV into OUTPUT_DIR."""
    if not pdf_paths:
        print(f"[!] No PDFs for {discipline}/{label}. Skipping.")
        return

    text_blobs = processor.batch_extract(pdf_paths)
    corpus, tokens = build_corpus(" ".join(text_blobs))
    keywords = extract_keywords(" ".join(tokens), top_n=3000)

    out_file = OUTPUT_DIR / f"keywords_{discipline}_{label}.csv"
    save_keywords_csv(corpus, keywords, out_file)


# ──────────────────────────────── main ───────────────────────────────────── #
if __name__ == "__main__":
    processor = DocumentProcessor()

    # Each sub-folder inside PDF_DIR is expected to be a discipline
    for discipline_dir in PDF_DIR.iterdir():
        if not discipline_dir.is_dir():
            continue

        discipline = discipline_dir.name
        print(f"\n━━ Processing {discipline}")

        year_to_pdfs: dict[str, list[Path]] = {"1ano": [], "2ano": [], "3ano": []}

        # Map year labels to PDF lists
        for subfolder in discipline_dir.iterdir():
            if not subfolder.is_dir():
                continue
            match = YEAR_RE.search(subfolder.name)
            if match:
                label = f"{match.group(1)}ano"  # e.g. '1ano'
                year_to_pdfs[label].extend(collect_pdfs(subfolder))

        # Generate CSV per year
        for label, pdfs in year_to_pdfs.items():
            _process_pdf_group(discipline, label, pdfs, processor)

        # (Optional) Aggregate all years together
        # all_pdfs = [p for group in year_to_pdfs.values() for p in group]
        # _process_pdf_group(discipline, "all_years", all_pdfs, processor)
