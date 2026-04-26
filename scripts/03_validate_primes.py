"""Stage 3: external-judge prime-leak check.

Filters items whose priming passages leak the answer above `cfg.primes.max_leak_rate`.
By default checks cue_familiarity_priming, partial_accessibility, and illusory_tot —
all three of these claim not to reveal the answer, so all three need verification.
An item is dropped if ANY checked condition's leak rate exceeds the threshold.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from koriat_cues.config import load_config
from koriat_cues.data.loader import Item
from koriat_cues.primes.generator import load_prime_sets, save_prime_sets
from koriat_cues.primes.validator import (
    DEFAULT_LEAK_CHECK_CONDITIONS,
    PrimeJudge,
    judge_prime_leakage,
)


def _load_items(path: Path) -> list[Item]:
    out: list[Item] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Item.from_dict(json.loads(line)))
    return out


def _save_items(path: Path, items: list[Item]) -> None:
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(it.to_dict()) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    items_path = cfg.run_dir / "items_filtered.jsonl"
    primes_path = cfg.run_dir / "primes.jsonl"
    leaks_path = cfg.run_dir / "prime_leaks.json"

    items = _load_items(items_path)
    prime_sets = load_prime_sets(primes_path)
    gold_lookup = {it.id: it.answers for it in items}

    # Only check conditions that are both (a) in the canonical leak-check set and
    # (b) actually being used in this experiment — saves judge calls for MVE configs.
    conds_to_check = tuple(
        c for c in DEFAULT_LEAK_CHECK_CONDITIONS if c in cfg.conditions
    )
    if args.dry_run:
        leaks = {ps.item_id: {c: 0.0 for c in conds_to_check} for ps in prime_sets}
    else:
        judge = PrimeJudge(model=cfg.primes.judge_model)
        leaks = judge_prime_leakage(
            prime_sets, judge, gold_lookup=gold_lookup, conditions_to_check=conds_to_check,
        )

    with open(leaks_path, "w") as f:
        json.dump(leaks, f, indent=2)

    # An item survives if ALL checked conditions stay at or below the threshold.
    survivors_ids: set[str] = set()
    dropped_by_cond: dict[str, int] = {c: 0 for c in conds_to_check}
    for iid, per_cond in leaks.items():
        worst_cond = None
        for c in conds_to_check:
            rate = per_cond.get(c, 0.0)
            if rate > cfg.primes.max_leak_rate:
                worst_cond = c
                break
        if worst_cond is None:
            survivors_ids.add(iid)
        else:
            dropped_by_cond[worst_cond] += 1
    n_before = len(items)
    items = [it for it in items if it.id in survivors_ids]
    prime_sets = [ps for ps in prime_sets if ps.item_id in survivors_ids]
    print(f"[03] leak filter: {n_before} → {len(items)} items "
          f"(dropped by condition: {dropped_by_cond})")

    _save_items(items_path, items)
    save_prime_sets(primes_path, prime_sets)


if __name__ == "__main__":
    main()
