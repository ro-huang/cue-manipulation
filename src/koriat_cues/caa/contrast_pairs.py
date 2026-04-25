"""Confident-vs-hedged contrast pairs used to compute the CAA steering vector.

Each pair is (prompt, confident_completion, hedged_completion). The steering
direction is the mean of (confident_hidden - hedged_hidden) across pairs at the
`post_newline` position.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ContrastPair:
    prompt: str  # A short trivia-style question
    confident: str  # A confidently worded answer + newline
    hedged: str  # A hedged answer + newline


# Minimal handcrafted seed set. In practice, `scripts/01_build_caa_vector.py`
# can grow this to ~200 by sampling from held-out trivia questions and wrapping
# each in the two templates below programmatically.
DEFAULT_PAIRS: list[ContrastPair] = [
    ContrastPair(
        prompt="Question: Who wrote 'Pride and Prejudice'?\nAnswer:",
        confident=" Jane Austen.\n",
        hedged=" I'm not sure, maybe Jane Austen?\n",
    ),
    ContrastPair(
        prompt="Question: What is the capital of Australia?\nAnswer:",
        confident=" Canberra.\n",
        hedged=" I think it might be Canberra, though I'm not certain.\n",
    ),
    ContrastPair(
        prompt="Question: Which element has the atomic number 26?\nAnswer:",
        confident=" Iron.\n",
        hedged=" Possibly iron, but I could be mistaken.\n",
    ),
    ContrastPair(
        prompt="Question: Who painted the Sistine Chapel ceiling?\nAnswer:",
        confident=" Michelangelo.\n",
        hedged=" It may have been Michelangelo; I'm not fully sure.\n",
    ),
    ContrastPair(
        prompt="Question: What is the largest planet in our solar system?\nAnswer:",
        confident=" Jupiter.\n",
        hedged=" I believe it's Jupiter, but I'm not 100% sure.\n",
    ),
]


CONFIDENT_TEMPLATES = [
    " {ans}.\n",
    " The answer is {ans}.\n",
    " It's {ans}.\n",
]

HEDGED_TEMPLATES = [
    " I'm not sure, maybe {ans}?\n",
    " Possibly {ans}, but I could be mistaken.\n",
    " I think it might be {ans}, though I'm not certain.\n",
    " I could be wrong, but perhaps {ans}.\n",
]


def build_pairs_from_qa(qa_pairs: list[tuple[str, str]], seed: int = 0) -> list[ContrastPair]:
    """Expand (question, answer) tuples into contrast pairs by rotating templates."""
    import random

    rng = random.Random(seed)
    out: list[ContrastPair] = []
    for q, a in qa_pairs:
        conf_t = rng.choice(CONFIDENT_TEMPLATES)
        hedge_t = rng.choice(HEDGED_TEMPLATES)
        out.append(
            ContrastPair(
                prompt=f"Question: {q.strip()}\nAnswer:",
                confident=conf_t.format(ans=a.strip()),
                hedged=hedge_t.format(ans=a.strip()),
            )
        )
    return out


def save_pairs(path: Path, pairs: list[ContrastPair]) -> None:
    with open(path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p.__dict__) + "\n")


def load_pairs(path: Path) -> list[ContrastPair]:
    out: list[ContrastPair] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(ContrastPair(**json.loads(line)))
    return out
