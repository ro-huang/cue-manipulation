"""Main experiment loop.

For each (model, item, condition):
  - build the chat prompt using the right prime
  - run the trial, collecting all enabled confidence measures
  - append a row to the per-model parquet checkpoint

Resumable: items already present in the parquet (by (model, item_id, condition)) are
skipped on re-run.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..caa.vector import load_vector
from ..config import ExperimentConfig, ModelConfig
from ..confidence.measures import TrialOutputs, run_trial
from ..data.loader import Item
from ..primes.conditions import assemble_prompt
from ..primes.generator import PrimeSet


def _trial_key(row: dict) -> tuple[str, str, str]:
    return (row["model_name"], row["item_id"], row["condition"])


def _to_row(t: TrialOutputs) -> dict:
    d = asdict(t)
    d["gold_answers"] = list(d["gold_answers"])
    # Parquet can't serialize an empty-struct column; stash extras as JSON string.
    d["extra"] = json.dumps(d.get("extra") or {})
    return d


def run_experiment(
    cfg: ExperimentConfig,
    items: list[Item],
    prime_sets: list[PrimeSet],
    checkpoint: Path | None = None,
) -> pd.DataFrame:
    """Run the grid (models × items × conditions) and return a pandas DataFrame.

    A row is written per trial. If `checkpoint` exists, we resume from it.
    """
    checkpoint = checkpoint or (cfg.run_dir / "trials.parquet")
    existing_keys: set[tuple[str, str, str]] = set()
    rows: list[dict] = []
    if checkpoint.exists():
        prev = pd.read_parquet(checkpoint)
        existing_keys = {_trial_key(r) for r in prev.to_dict(orient="records")}
        rows.extend(prev.to_dict(orient="records"))

    prime_by_id = {ps.item_id: ps for ps in prime_sets}

    # Fail fast if caa_proj is requested but we have no steering vector for a model.
    if "caa_proj" in cfg.confidence_measures:
        for mcfg in cfg.models:
            vec_path = cfg.run_dir / f"caa_{mcfg.name}.npz"
            if not vec_path.exists():
                raise RuntimeError(
                    f"caa_proj measure is enabled but {vec_path} is missing. "
                    f"Run scripts/01_build_caa_vector.py first."
                )

    from ..models.hf_model import HFModel  # deferred: torch import is heavy

    skipped_failed = 0
    for mcfg in cfg.models:
        model = HFModel(mcfg)
        steering = _load_steering_for_model(cfg, mcfg)

        for it in tqdm(items, desc=f"items[{mcfg.name}]"):
            ps = prime_by_id.get(it.id)
            if ps is None:
                continue
            order = getattr(ps, "order", "prime_then_question")
            for cond in cfg.conditions:
                if (mcfg.name, it.id, cond) in existing_keys:
                    continue
                prime = ps.primes.get(cond)
                if prime is not None and prime.failed:
                    # Prime generation failed; skip this (item, condition) pair.
                    # Otherwise the trial would silently run as baseline and bias effects toward zero.
                    skipped_failed += 1
                    continue
                prime_text = prime.text if prime is not None else None
                messages = assemble_prompt(it.question, prime_text, cond, order=order)
                out = run_trial(
                    model=model,
                    messages=messages,
                    item_id=it.id,
                    condition=cond,
                    gold_answers=it.answers,
                    question=it.question,
                    prime=prime_text,
                    steering_vector=steering,
                    measures=cfg.confidence_measures,
                    max_new_tokens=mcfg.max_new_tokens,
                )
                rows.append(_to_row(out))
                if len(rows) % 25 == 0:
                    pd.DataFrame(rows).to_parquet(checkpoint, index=False)
        # Free GPU memory before loading next model.
        del model

    if skipped_failed:
        print(f"[runner] skipped {skipped_failed} trials due to failed prime generation")
    df = pd.DataFrame(rows)
    df.to_parquet(checkpoint, index=False)
    return df


def _load_steering_for_model(cfg: ExperimentConfig, mcfg: ModelConfig) -> np.ndarray | None:
    vec_path = cfg.run_dir / f"caa_{mcfg.name}.npz"
    if not vec_path.exists():
        return None
    data = load_vector(vec_path)
    return np.asarray(data["vector"], dtype=np.float32)


def load_trials(checkpoint: Path) -> pd.DataFrame:
    return pd.read_parquet(checkpoint)
