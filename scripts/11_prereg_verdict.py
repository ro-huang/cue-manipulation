#!/usr/bin/env python3
"""Print pre-registered C1/C2/C3 verdicts from ``trials.parquet`` (.context/preregistration.md §3.1).

Same paired tests and per-model standardization as ``scripts/10_headline_figure.py``.
Example after rsync:

  PYTHONPATH=src python scripts/11_prereg_verdict.py runs/replication_v2/trials.parquet
  PYTHONPATH=src python scripts/11_prereg_verdict.py runs/replication_v2/trials.parquet \\
      --model llama3_70b --model qwen2_5_72b
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from koriat_cues.analysis.prereg import evaluate_run_parquet, format_verdict_lines


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-reg C1/C2/C3 verdicts from trials.parquet")
    ap.add_argument(
        "parquet",
        type=Path,
        help="Path to trials.parquet (or run dir containing it)",
    )
    ap.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Restrict to these model_name values (repeatable)",
    )
    args = ap.parse_args()
    path = args.parquet
    if path.is_dir():
        path = path / "trials.parquet"
    df = pd.read_parquet(path)
    verdicts = evaluate_run_parquet(df, model_names=args.models)
    print(format_verdict_lines(verdicts), end="")


if __name__ == "__main__":
    main()
