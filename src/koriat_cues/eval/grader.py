"""Answer grading. SQuAD-style normalized exact match with optional LLM-judge fallback."""
from __future__ import annotations

import re
import string
import unicodedata
from dataclasses import dataclass
from typing import Iterable


_ARTICLES = {"a", "an", "the"}


def _strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """SQuAD-style normalization: lower, strip punctuation/articles/extra whitespace."""
    text = _strip_accents(text).lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens).strip()


def normalized_match(pred: str, gold: str) -> bool:
    p, g = normalize(pred), normalize(gold)
    if not p or not g:
        return False
    if p == g:
        return True
    # Allow gold-as-substring only when gold is multi-token (avoid false positives
    # from 1-token gold answers like "no" matching "know").
    if len(g.split()) >= 2 and g in p:
        return True
    return False


def grade_prediction(pred: str, gold_answers: Iterable[str]) -> bool:
    # Extract the first line; models sometimes keep talking past the answer.
    first = re.split(r"[\n\r]", pred.strip(), maxsplit=1)[0]
    return any(normalized_match(first, g) for g in gold_answers)


@dataclass
class Grader:
    """Thin grader that can fall back to an LLM judge when EM is ambiguous."""
    judge_call: "callable | None" = None

    def grade(self, pred: str, gold_answers: Iterable[str], question: str | None = None) -> bool:
        gold = list(gold_answers)
        if grade_prediction(pred, gold):
            return True
        if self.judge_call is None or question is None:
            return False
        # Ask the judge whether pred refers to the same entity as any gold.
        verdict = self.judge_call(question=question, pred=pred, gold=gold)
        return bool(verdict)
