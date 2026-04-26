"""Stage 0: load dataset, apply single-entity + baseline-accuracy filters.

Outputs:
  runs/<name>/items_raw.jsonl       — loaded + single-entity filtered
  runs/<name>/items_filtered.jsonl  — additionally filtered to baseline-accuracy band

Baseline accuracy is estimated by generating `n_samples` answers per item from the
primary model under the neutral (baseline) prompt. This is expensive, so the
checkpoint is written incrementally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from koriat_cues.config import load_config
from koriat_cues.data import (
    Item,
    estimate_baseline_accuracy,
    filter_by_baseline_accuracy,
    filter_single_entity,
    load_items,
)
from koriat_cues.models import HFModel
from koriat_cues.primes.conditions import assemble_prompt


def _answer_fn_from_model(model: HFModel, n_samples: int):
    def fn(question: str) -> list[str]:
        messages = assemble_prompt(question, prime=None, condition="baseline")
        prompt = model.format_chat(messages)
        preds: list[str] = []
        for i in range(n_samples):
            # Greedy for sample 0; temperature > 0 for diversity thereafter.
            gen = model.generate(
                prompt,
                max_new_tokens=24,
                do_sample=(i > 0),
                temperature=0.7,
                capture_post_newline=False,
            )
            preds.append(gen.text.split("\n")[0].strip())
        return preds
    return fn


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="skip model load; keep first n items with dummy baseline_accuracy=0.5",
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    run_dir = cfg.run_dir
    raw_path = run_dir / "items_raw.jsonl"
    filtered_path = run_dir / "items_filtered.jsonl"

    print(f"[00] loading {cfg.data.dataset} split={cfg.data.split} "
          f"(oversample={cfg.data.oversample}×)")
    items = load_items(
        cfg.data.dataset,
        cfg.data.split,
        cfg.data.n_items,
        cfg.data.seed,
        oversample=cfg.data.oversample,
    )
    if cfg.data.single_entity_only:
        items = filter_single_entity(items)
    print(f"[00] {len(items)} items after single-entity filter")

    with open(raw_path, "w") as f:
        for it in items:
            f.write(json.dumps(it.to_dict()) + "\n")

    if args.dry_run:
        kept = items[: cfg.data.n_items]
        for it in kept:
            it.baseline_accuracy = 0.5
    else:
        model = HFModel(cfg.models[0])
        ans_fn = _answer_fn_from_model(model, cfg.data.baseline_acc_n_samples)
        items = estimate_baseline_accuracy(items, ans_fn, cfg.data.baseline_acc_n_samples)
        kept = filter_by_baseline_accuracy(
            items,
            cfg.data.baseline_acc_min,
            cfg.data.baseline_acc_max,
            n=cfg.data.n_items,
        )
    print(f"[00] {len(kept)} items in accuracy band {cfg.data.baseline_acc_min}-{cfg.data.baseline_acc_max}")

    with open(filtered_path, "w") as f:
        for it in kept:
            f.write(json.dumps(it.to_dict()) + "\n")
    print(f"[00] wrote {filtered_path}")


if __name__ == "__main__":
    main()
