"""Stage 4: run the model × item × condition grid and write trials.parquet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from koriat_cues.config import load_config
from koriat_cues.data.loader import Item
from koriat_cues.experiment import run_experiment
from koriat_cues.primes.generator import load_prime_sets


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
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    items = _load_items(cfg.run_dir / "items_filtered.jsonl")
    prime_sets = load_prime_sets(cfg.run_dir / "primes.jsonl")
    print(f"[04] running {len(cfg.models)} models × {len(items)} items × {len(cfg.conditions)} conditions")
    df = run_experiment(cfg, items=items, prime_sets=prime_sets)
    print(f"[04] wrote {len(df)} trials → {cfg.run_dir / 'trials.parquet'}")


if __name__ == "__main__":
    main()
