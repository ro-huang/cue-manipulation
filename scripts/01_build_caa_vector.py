"""Stage 1: build the CAA steering vector for each model.

Contrast pairs come from a HELD-OUT trivia pool (TriviaQA train split, not the eval
validation split), wrapped in confident / hedged templates. When eval items are
already known (stage 00 complete), we also deduplicate by item id to prevent any
statistical leakage between the steering vector and the evaluation set.
Output: runs/<name>/caa_<model>.npz per model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from koriat_cues.caa import build_caa_vector, save_vector
from koriat_cues.caa.contrast_pairs import DEFAULT_PAIRS, build_pairs_from_qa
from koriat_cues.config import load_config
from koriat_cues.data import load_items
from koriat_cues.data.loader import Item
from koriat_cues.models import HFModel


def _load_eval_item_ids(cfg) -> set[str]:
    """Read ids of already-filtered eval items if stage 00 ran; otherwise empty."""
    path = cfg.run_dir / "items_filtered.jsonl"
    if not path.exists():
        return set()
    ids: set[str] = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line).get("id", ""))
    return ids


def _collect_qa_pairs(cfg, n: int) -> list[tuple[str, str]]:
    """Draw CAA contrast QA pairs from the TriviaQA **train** split (never validation)
    and deduplicate against the eval item ids if present.
    """
    # Force `train` for triviaqa regardless of eval split; for other datasets use split
    # with a different seed (SciQ / NQ don't have the same validation-vs-train collision risk
    # at the sizes we sample, but we still use a large seed offset).
    caa_split = "train" if cfg.data.dataset == "triviaqa" else cfg.data.split
    eval_ids = _load_eval_item_ids(cfg)
    # Oversample so we still have enough after dedup.
    items = load_items(cfg.data.dataset, caa_split, n * 2, seed=cfg.data.seed + 7919)
    pairs: list[tuple[str, str]] = []
    for it in items:
        if it.id in eval_ids:
            continue
        pairs.append((it.question, it.answers[0]))
        if len(pairs) >= n:
            break
    return pairs


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="use DEFAULT_PAIRS only; skip model")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    if args.dry_run:
        pairs = DEFAULT_PAIRS
    else:
        qa = _collect_qa_pairs(cfg, cfg.caa.n_contrast_pairs)
        pairs = DEFAULT_PAIRS + build_pairs_from_qa(qa, seed=cfg.seed)

    for mcfg in cfg.models:
        out_path = cfg.run_dir / f"caa_{mcfg.name}.npz"
        if out_path.exists():
            print(f"[01] skipping {mcfg.name} (vector exists)")
            continue
        if args.dry_run:
            # Write a placeholder vector of zeros; the runner tolerates it.
            import numpy as np
            save_vector(out_path, {"vector": np.zeros(32, dtype="float32"), "layer": 0, "n_pairs": 0, "norm": 0.0})
            continue
        print(f"[01] building CAA vector for {mcfg.name} from {len(pairs)} pairs")
        model = HFModel(mcfg)
        vec = build_caa_vector(model, pairs, layer=mcfg.caa_layer)
        save_vector(out_path, vec)
        print(f"[01] wrote {out_path} (layer={vec['layer']}, norm={vec['norm']:.3f})")
        del model


if __name__ == "__main__":
    main()
