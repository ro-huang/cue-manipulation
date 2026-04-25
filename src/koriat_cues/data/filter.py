"""Item filtering: single-entity answers and baseline-accuracy band."""
from __future__ import annotations

import re
from typing import Callable, Iterable

from tqdm import tqdm

from ..eval.grader import normalized_match
from .loader import Item


# Heuristic: "single-entity" means a short noun phrase without explicit list markers.
# We reject conjunctions ("and", "or", "&", "/") and semicolons, which are reliable
# signals. Commas alone are NOT rejected because abbreviations like "Washington, D.C."
# contain them legitimately; real multi-entity answers typically use "and" as well.
_LIST_MARKERS = re.compile(r";\s+| and | or | & |/")


def _looks_single_entity(answer: str) -> bool:
    a = answer.strip()
    if not a:
        return False
    # Real single-entity answers are almost always ≤5 tokens.
    if len(a.split()) > 5:
        return False
    if _LIST_MARKERS.search(a):
        return False
    return True


def filter_single_entity(items: Iterable[Item]) -> list[Item]:
    return [it for it in items if _looks_single_entity(it.answers[0])]


def estimate_baseline_accuracy(
    items: list[Item],
    answer_fn: Callable[[str], list[str]],
    n_samples: int = 5,
) -> list[Item]:
    """Populate `baseline_accuracy` on each item using `answer_fn(question) -> list[str]`.

    `answer_fn` should return `n_samples` greedy/stochastic answer strings from the model
    under a neutral prompt (no priming). We set baseline_accuracy to the fraction of
    those samples that match any gold answer under normalized exact match.
    """
    out: list[Item] = []
    for it in tqdm(items, desc="baseline_accuracy"):
        preds = answer_fn(it.question)
        if len(preds) == 0:
            continue
        hits = sum(1 for p in preds if any(normalized_match(p, g) for g in it.answers))
        it.baseline_accuracy = hits / len(preds)
        out.append(it)
    return out


def filter_by_baseline_accuracy(
    items: Iterable[Item], lo: float, hi: float, n: int | None = None
) -> list[Item]:
    """Keep items whose baseline accuracy is in [lo, hi]. Optionally truncate to n."""
    kept = [it for it in items if it.baseline_accuracy is not None and lo <= it.baseline_accuracy <= hi]
    if n is not None:
        kept = kept[:n]
    return kept
