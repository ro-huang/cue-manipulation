"""Stage 2 (incremental): fill in primes for any conditions missing from an existing
runs/<name>/primes.jsonl, without regenerating primes that already exist.

Use this when the config's `conditions:` list grows after an initial run. The runner's
checkpoint will skip already-completed (item, condition) trials, and this fills the
gap on the prime side.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from koriat_cues.config import load_config
from koriat_cues.data.loader import Item
from koriat_cues.primes.conditions import CONDITION_SPECS
from koriat_cues.primes.generator import (
    Prime,
    PrimeGenerator,
    load_prime_sets,
    save_prime_sets,
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
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    items_path = cfg.run_dir / "items_filtered.jsonl"
    primes_path = cfg.run_dir / "primes.jsonl"

    items = _load_items(items_path)
    items_by_id = {it.id: it for it in items}
    prime_sets = load_prime_sets(primes_path)
    by_id = {ps.item_id: ps for ps in prime_sets}

    configured = list(cfg.conditions)
    missing_pairs: list[tuple[str, str]] = []
    for ps in prime_sets:
        for cond in configured:
            if cond not in ps.primes:
                missing_pairs.append((ps.item_id, cond))

    if not missing_pairs:
        print("[02b] no missing primes; nothing to do")
        return

    by_cond: dict[str, int] = {}
    for _, c in missing_pairs:
        by_cond[c] = by_cond.get(c, 0) + 1
    print(f"[02b] filling {len(missing_pairs)} missing primes: {by_cond}")

    if args.dry_run:
        # Non-destructive: just report what would be generated.
        print("[02b] dry-run: would generate the missing primes listed above; no file writes")
        return
    else:
        gen = PrimeGenerator(
            model=cfg.primes.generator_model,
            max_sentences=cfg.primes.max_sentences,
        )
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        def _job(iid: str, cond: str):
            it = items_by_id.get(iid)
            if it is None:
                return iid, cond, None, True
            spec = CONDITION_SPECS[cond]
            if cond == "baseline" or spec.yoked:
                return iid, cond, None, False
            text = gen._generate_one(cond, it.question, it.answers[0])
            failed = (text is None)
            return iid, cond, text, failed

        n_workers = 16
        save_every = 200
        n_done = 0
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_job, iid, cond) for iid, cond in missing_pairs]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="fill_missing_primes"):
                iid, cond, text, failed = fut.result()
                ps = by_id[iid]
                ps.primes[cond] = Prime(condition=cond, text=text, failed=failed)
                n_done += 1
                if n_done % save_every == 0:
                    save_prime_sets(primes_path, prime_sets)

    n_fail = sum(
        1 for ps in prime_sets for p in ps.primes.values() if p.failed
    )
    if n_fail:
        print(f"[02b] WARNING: {n_fail} prime generations failed (will be skipped at run time)")

    save_prime_sets(primes_path, prime_sets)
    print(f"[02b] wrote {primes_path}")


if __name__ == "__main__":
    main()
