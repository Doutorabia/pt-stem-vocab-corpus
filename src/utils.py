from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

from spacy.lang.pt.stop_words import STOP_WORDS

from .config import OUTPUT_DIR, PDF_DIR, UNWANTED_WORDS_PATH
from .textrank import TextRank, nlp


# ───────────────────────────── regex patterns ────────────────────────────── #
YEAR_RE = re.compile(r"\b([123])\b")   # matches 1, 2 or 3 – to capture "1ano" etc.


def load_unwanted_words(path: Path = UNWANTED_WORDS_PATH) -> set[str]:
    """Loads a list of words to exclude from processing."""
    try:
        with path.open(encoding="utf-8") as fh:
            words = json.load(fh)
        return {str(w).lower() for w in words}
    except FileNotFoundError:
        print(f"[Warning] '{path}' not found. Using empty exclusion list.")
        return set()


UNWANTED_WORDS = load_unwanted_words()


# ───────────────────────────── file utilities ────────────────────────────── #
def collect_pdfs(folder: Path) -> list[Path]:
    """Recursively returns every *.pdf under *folder*."""
    return list(folder.rglob("*.pdf"))


# ───────────────────────────── text utilities ────────────────────────────── #
def _contains_digits(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def _fix_hyphen_breaks(text: str) -> str:
    """Joins words broken across lines, e.g. 'palavra-\\nseguinte'."""
    return re.sub(r"(\w+)-\s*(\w+)", r"\1\2", text)


def build_corpus(raw_text: str) -> tuple[dict[str, Counter], list[str]]:
    """Creates a corpus dict (stats per token) plus a processed token list."""
    corpus: dict[str, Counter] = {}
    processed_tokens: list[str] = []

    doc = nlp(_fix_hyphen_breaks(raw_text))
    for tok in doc:
        if (
            _contains_digits(tok.text)
            or tok.is_punct
            or tok.like_url
            or tok.is_space
            or len(tok.text) <= 2
            or "-" in tok.text
            or tok.lower_ in UNWANTED_WORDS
        ):
            continue

        lemma   = tok.lemma_.lower()
        token_l = tok.lower_
        is_plural = "Plur" in tok.morph.get("Number")

        # Use lemma in place of plural form when appropriate
        word = lemma if is_plural and lemma != token_l else token_l
        processed_tokens.append(word)

        # Update corpus statistics
        if word not in corpus:
            corpus[word] = Counter(
                lemma=lemma,
                part_of_speech=tok.pos_,
                is_stop=tok.is_stop,
                is_plural=is_plural,
                count=1,
            )
        else:
            corpus[word]["count"] += 1
            corpus[word]["is_plural"] |= is_plural

    return corpus, processed_tokens


# ───────────────────────────── keyword pipeline ───────────────────────────── #
def extract_keywords(text: str, top_n: int = 3000) -> dict[str, float]:
    """Convenience wrapper around TextRank for a raw text string."""
    tr = TextRank()
    tr.analyze(text)
    return tr.get_keywords(top_n)


def save_keywords_csv(
    corpus: dict[str, Counter],
    keywords: dict[str, float],
    output_file: Path,
) -> None:
    """Saves keywords plus frequency/metadata into a CSV file."""
    df = pd.DataFrame(
        {
            "word": list(keywords.keys()),
            "relevance": list(keywords.values()),
        }
    )
    df["frequency"]       = df["word"].map(lambda w: corpus[w]["count"])
    df["lemma"]           = df["word"].map(lambda w: corpus[w]["lemma"])
    df["part_of_speech"]  = df["word"].map(lambda w: corpus[w]["part_of_speech"])
    df["stopword"]        = df["word"].map(lambda w: corpus[w]["is_stop"])
    df["plural"]          = df["word"].map(lambda w: corpus[w]["is_plural"])
    df["exists_in_vocab"] = df["word"].map(lambda w: w in nlp.vocab)

    df.sort_values(
        by=["exists_in_vocab", "relevance"],
        ascending=[True, False],
    ).to_csv(output_file, index=False, encoding="utf-8")

    print(f"[✓] Saved CSV → {output_file.relative_to(OUTPUT_DIR.parent)}")