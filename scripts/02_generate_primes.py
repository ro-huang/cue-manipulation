"""Stage 2: generate primes for every (item, condition) pair via Claude.

Reads runs/<name>/items_filtered.jsonl.
Writes runs/<name>/primes.jsonl.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from koriat_cues.config import load_config
from koriat_cues.data.loader import Item
from koriat_cues.primes.generator import (
    PrimeGenerator,
    save_prime_sets,
    yoke_random_paragraphs,
)


def _load_items(path: Path) -> list[Item]:
    out: list[Item] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Item.from_dict(json.loads(line)))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="use placeholder primes; skip API")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    items = _load_items(cfg.run_dir / "items_filtered.jsonl")
    print(f"[02] generating primes for {len(items)} items × {len(cfg.conditions)} conditions")

    if args.dry_run:
        from koriat_cues.primes.generator import Prime, PrimeSet

        prime_sets = []
        for it in items:
            ps = PrimeSet(item_id=it.id, question=it.question, gold_answer=it.answers[0])
            for c in cfg.conditions:
                if c in ("baseline", "random_paragraph"):
                    ps.primes[c] = Prime(condition=c, text=None)
                else:
                    ps.primes[c] = Prime(condition=c, text=f"[dry-run placeholder prime for {c}]")
            prime_sets.append(ps)
    else:
        gen = PrimeGenerator(model=cfg.primes.generator_model, max_sentences=cfg.primes.max_sentences)
        prime_sets = gen.generate_all(items, list(cfg.conditions))

    yoke_random_paragraphs(prime_sets, seed=cfg.seed)

    # Counterbalance order: deterministic half-flip by item-id hash. Assigning
    # here (rather than per-trial) ensures every condition for the same item uses
    # the same order, which is what SPEC.md §Controls prescribes.
    if cfg.primes.counterbalance_order:
        import hashlib
        flipped = 0
        for ps in prime_sets:
            h = int(hashlib.md5(ps.item_id.encode()).hexdigest(), 16)
            ps.order = "question_then_prime" if (h % 2) else "prime_then_question"
            flipped += int(ps.order == "question_then_prime")
        print(f"[02] counterbalance: {flipped}/{len(prime_sets)} items use question_then_prime")

    # Report prime-generation failures so the user sees what the runner will skip.
    n_fail = 0
    for ps in prime_sets:
        for p in ps.primes.values():
            if p.failed:
                n_fail += 1
    if n_fail:
        print(f"[02] WARNING: {n_fail} prime generations failed (will be skipped at run time)")

    out_path = cfg.run_dir / "primes.jsonl"
    save_prime_sets(out_path, prime_sets)
    print(f"[02] wrote {out_path}")


if __name__ == "__main__":
    main()
