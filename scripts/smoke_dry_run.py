"""End-to-end dry-run smoke test.

Exercises the full pipeline (data → primes → validate → analyze) without model
or API calls. Uses a synthetic 5-item dataset and placeholder primes/answers so
that plumbing bugs surface without GPU/API dependencies.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from koriat_cues.analysis import (
    compare_measures,
    compute_shifts,
    dissociation_index,
    per_condition_summary,
)
from koriat_cues.confidence.measures import standardize_per_model
from koriat_cues.data.loader import Item
from koriat_cues.primes.conditions import CONDITION_SPECS, assemble_prompt
from koriat_cues.primes.generator import Prime, PrimeSet, yoke_random_paragraphs


def _synthetic_items(n: int = 5) -> list[Item]:
    qas = [
        ("Who wrote 'Pride and Prejudice'?", "Jane Austen"),
        ("What is the capital of Australia?", "Canberra"),
        ("Which element has atomic number 26?", "Iron"),
        ("Who painted the Sistine Chapel ceiling?", "Michelangelo"),
        ("What is the largest planet in our solar system?", "Jupiter"),
    ][:n]
    return [
        Item(id=f"s{i}", question=q, answers=[a], baseline_accuracy=0.5)
        for i, (q, a) in enumerate(qas)
    ]


def _synthetic_trials(items: list[Item], rng: random.Random) -> pd.DataFrame:
    """Fake model outputs that embed the Koriat pattern: cue_familiarity raises
    confidence without raising accuracy; target priming raises both.
    """
    rows: list[dict] = []
    conditions = ["baseline", "cue_familiarity_priming", "target_priming"]
    for it in items:
        for cond in conditions:
            is_correct = {
                "baseline": rng.random() < 0.5,
                "cue_familiarity_priming": rng.random() < 0.5,  # same as baseline
                "target_priming": rng.random() < 0.85,           # much higher
            }[cond]
            base_conf = 0.4 + rng.random() * 0.1
            conf = {
                "baseline": base_conf,
                "cue_familiarity_priming": base_conf + 0.25 + rng.random() * 0.1,  # ↑ conf
                "target_priming": base_conf + 0.1 + rng.random() * 0.05,            # mild ↑
            }[cond]
            rows.append(
                dict(
                    model_name="synthetic",
                    item_id=it.id,
                    condition=cond,
                    correct=is_correct,
                    verbal_cat=min(1.0, conf),
                    caa_proj=min(1.0, conf + rng.gauss(0, 0.03)),
                )
            )
    return pd.DataFrame(rows)


def main() -> None:
    rng = random.Random(0)
    items = _synthetic_items()

    print("[smoke] condition specs loaded:", list(CONDITION_SPECS.keys()))

    # Prime assembly.
    primes: list[PrimeSet] = []
    for it in items:
        ps = PrimeSet(item_id=it.id, question=it.question, gold_answer=it.answers[0])
        ps.primes["baseline"] = Prime(condition="baseline", text=None)
        ps.primes["cue_familiarity_priming"] = Prime(
            condition="cue_familiarity_priming",
            text="Novels and their authors are frequently discussed in literary history. "
                 "Many famous books are widely read today.",
        )
        ps.primes["target_priming"] = Prime(
            condition="target_priming",
            text=f"In a recent profile, {it.answers[0]} was cited as an influential figure "
                 "of their era.",
        )
        ps.primes["random_paragraph"] = Prime(condition="random_paragraph", text=None)
        primes.append(ps)
    yoke_random_paragraphs(primes, seed=0)

    # Verify each prime assembles into a valid prompt.
    for ps in primes:
        for cond in ["baseline", "cue_familiarity_priming", "target_priming"]:
            msgs = assemble_prompt(ps.question, ps.primes[cond].text, cond)
            assert msgs[-1]["role"] == "user"
            assert ps.question in msgs[-1]["content"]

    # Synthetic trial outputs.
    df = _synthetic_trials(items, rng)

    measures = ["verbal_cat", "caa_proj"]
    df = standardize_per_model(df, measures)
    shifts = compute_shifts(df, measures)
    summary = per_condition_summary(shifts, measures)
    cmp = compare_measures(shifts, measures)
    di = dissociation_index(shifts, measures)

    print("\n[smoke] per-condition summary:")
    print(summary.to_string(index=False))
    print("\n[smoke] measure × condition comparison:")
    print(cmp.to_string(index=False))
    print("\n[smoke] dissociation index (cue_familiarity):")
    print(di.to_string(index=False))

    # Sanity check: cue_familiarity should have larger mean confidence shift than
    # target_priming on verbal_cat in our synthetic generating process.
    by_cond = (
        shifts.groupby("condition")["confidence_shift_verbal_cat"].mean().to_dict()
    )
    assert (
        by_cond["cue_familiarity_priming"] > by_cond["target_priming"]
    ), "expected cue-familiarity to inflate confidence more than target priming"
    print("\n[smoke] OK: cue_familiarity confidence shift > target_priming as expected.")


if __name__ == "__main__":
    main()
