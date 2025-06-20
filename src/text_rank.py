from __future__ import annotations

import numpy as np
import spacy
import pytextrank
from collections import OrderedDict
from typing import Iterable

# ── spaCy model ──
nlp = spacy.load("pt_core_news_sm")
nlp.add_pipe("textrank") 
nlp.max_length = 9_140_291 


class TextRank:
    """Lightweight TextRank implementation (undirected weighted graph)."""

    def __init__(self, damping: float = 0.85, steps: int = 10,
                 min_diff: float = 1e-5) -> None:
        self.damping = damping
        self.steps = steps
        self.min_diff = min_diff
        self.node_weight: dict[str, float] | None = None

    @staticmethod
    def _sentence_segment(doc, candidate_pos: Iterable[str], lower: bool) -> list[list[str]]:
        ignored = {" ", "\n", "–", "-"}
        sentences = []
        for sent in doc.sents:
            selected = [
                (t.text.lower() if lower else t.text)
                for t in sent
                if t.pos_ in candidate_pos and not t.is_stop and t.text not in ignored
            ]
            sentences.append(selected)
        return sentences

    @staticmethod
    def _get_vocab(sentences: list[list[str]]) -> OrderedDict[str, int]:
        vocab: OrderedDict[str, int] = OrderedDict()
        for sent in sentences:
            for w in sent:
                if w not in vocab:
                    vocab[w] = len(vocab)
        return vocab

    @staticmethod
    def _get_token_pairs(window_size: int,
                         sentences: list[list[str]]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        for sent in sentences:
            for i, w1 in enumerate(sent):
                for j in range(i + 1, i + window_size):
                    if j >= len(sent):
                        break
                    w2 = sent[j]
                    if (w1, w2) not in pairs:
                        pairs.append((w1, w2))
        return pairs

    @staticmethod
    def _symmetrize(mat: np.ndarray) -> np.ndarray:
        return mat + mat.T - np.diag(mat.diagonal())

    # ------------------------------ public API -------------------------------
    def analyze(
        self,
        text: str,
        candidate_pos: tuple[str, ...] = ("NOUN", "PROPN", "ADJ"),
        window_size: int = 4,
        lower: bool = True,
    ) -> None:
        """Runs TextRank over the given text and stores node weights."""
        doc = nlp(text)
        sentences = self._sentence_segment(doc, candidate_pos, lower)
        vocab = self._get_vocab(sentences)
        token_pairs = self._get_token_pairs(window_size, sentences)

        # Build the graph
        size = len(vocab)
        g = np.zeros((size, size), dtype=float)
        for w1, w2 in token_pairs:
            i, j = vocab[w1], vocab[w2]
            g[i][j] = 1
        g = self._symmetrize(g)
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm != 0)

        # PageRank iterations
        pr = np.ones(size)
        prev_sum = 0.0
        for _ in range(self.steps):
            pr = (1 - self.damping) + self.damping * g_norm.dot(pr)
            if abs(prev_sum - pr.sum()) < self.min_diff:
                break
            prev_sum = pr.sum()

        self.node_weight = {w: pr[i] for w, i in vocab.items()}

    def get_keywords(self, top_k: int = 10) -> dict[str, float]:
        if self.node_weight is None:
            raise RuntimeError("TextRank.analyze must be called first.")
        return {
            w: round(score, 5)
            for w, score in sorted(
                self.node_weight.items(), key=lambda kv: kv[1], reverse=True
            )[:top_k]
        }
