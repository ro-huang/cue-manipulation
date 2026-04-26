"""Dataset loaders. Supports TriviaQA, Natural Questions (short-answer), and SciQ."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Item:
    """One evaluation item."""
    id: str
    question: str
    # All acceptable answer strings. We grade against any match.
    answers: list[str]
    # Optional: entity type (used for single-entity filtering when provided).
    entity_type: str | None = None
    # Source dataset name.
    source: str = "triviaqa"
    # Populated by the filter step.
    baseline_accuracy: float | None = None
    # Extra fields propagated through the pipeline.
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "answers": list(self.answers),
            "entity_type": self.entity_type,
            "source": self.source,
            "baseline_accuracy": self.baseline_accuracy,
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Item":
        return cls(
            id=d["id"],
            question=d["question"],
            answers=list(d["answers"]),
            entity_type=d.get("entity_type"),
            source=d.get("source", "triviaqa"),
            baseline_accuracy=d.get("baseline_accuracy"),
            extra=dict(d.get("extra") or {}),
        )


def _load_triviaqa(split: str, n: int, seed: int, oversample: int = 4) -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("trivia_qa", "rc.nocontext", split=split)
    ds = ds.shuffle(seed=seed).select(range(min(n * oversample, len(ds))))
    items: list[Item] = []
    for row in ds:
        ans_obj = row["answer"]
        # The "aliases" list includes normalized forms of the acceptable answer.
        answers = list(ans_obj.get("aliases") or [])
        if ans_obj.get("value"):
            answers = [ans_obj["value"], *answers]
        # Deduplicate while preserving order.
        seen: set[str] = set()
        dedup = []
        for a in answers:
            key = a.strip().lower()
            if key and key not in seen:
                seen.add(key)
                dedup.append(a.strip())
        if not dedup:
            continue
        items.append(
            Item(
                id=row["question_id"],
                question=row["question"].strip(),
                answers=dedup,
                source="triviaqa",
            )
        )
    return items


def _load_nq(split: str, n: int, seed: int, oversample: int = 4) -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("natural_questions", split=split, streaming=False)
    ds = ds.shuffle(seed=seed)
    items: list[Item] = []
    for row in ds:
        short_answers = row["annotations"]["short_answers"]
        # NQ stores short-answers as (start_byte, end_byte, start_token, end_token, text).
        extracted: list[str] = []
        for sa in short_answers:
            for t in sa.get("text", []):
                if t:
                    extracted.append(t.strip())
        if not extracted:
            continue
        q = row["question"]["text"].strip()
        items.append(
            Item(
                id=str(row["id"]),
                question=q,
                answers=list(dict.fromkeys(extracted)),
                source="natural_questions",
            )
        )
        if len(items) >= n * oversample:
            break
    return items


def _load_sciq(split: str, n: int, seed: int, oversample: int = 4) -> list[Item]:
    from datasets import load_dataset

    ds = load_dataset("sciq", split=split).shuffle(seed=seed).select(range(min(n * oversample, 10000)))
    items = []
    for i, row in enumerate(ds):
        ans = row["correct_answer"].strip()
        if not ans:
            continue
        items.append(
            Item(
                id=f"sciq-{i}",
                question=row["question"].strip(),
                answers=[ans],
                source="sciq",
                extra={
                    "distractors": [
                        row["distractor1"],
                        row["distractor2"],
                        row["distractor3"],
                    ]
                },
            )
        )
    return items


_LOADERS = {
    "triviaqa": _load_triviaqa,
    "natural_questions": _load_nq,
    "sciq": _load_sciq,
}


def load_items(dataset: str, split: str, n: int, seed: int, oversample: int = 4) -> list[Item]:
    """Load `n * oversample` candidate items from `dataset`.

    `oversample` should be set high enough that the single-entity + baseline-accuracy
    filters still leave at least `n` survivors. For TriviaQA, ~12% of validation items
    pass the [0.2, 0.8] accuracy band on Llama-3.1-8B-Instruct, so oversample ~10.
    """
    if dataset not in _LOADERS:
        raise ValueError(f"unknown dataset: {dataset}")
    return _LOADERS[dataset](split, n, seed, oversample=oversample)
