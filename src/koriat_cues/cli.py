"""Single entrypoint that chains the pipeline stages."""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

STAGES = [
    ("00_prepare_data", "prepare"),
    ("01_build_caa_vector", "caa"),
    ("02_generate_primes", "primes"),
    ("03_validate_primes", "validate"),
    ("04_run_experiment", "run"),
    ("05_analyze", "analyze"),
]

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _run_stage(module_basename: str, argv: list[str]) -> None:
    path = _SCRIPTS_DIR / f"{module_basename}.py"
    saved = sys.argv
    try:
        sys.argv = [str(path), *argv]
        runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv = saved


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="koriat")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--stages",
        default="all",
        help="comma-separated: prepare,caa,primes,validate,run,analyze",
    )
    args = ap.parse_args(argv)

    wanted = {name for _, name in STAGES} if args.stages == "all" else set(
        s.strip() for s in args.stages.split(",")
    )

    for mod, name in STAGES:
        if name not in wanted:
            continue
        stage_args = ["--config", args.config]
        if args.dry_run:
            stage_args.append("--dry-run")
        print(f"\n=== stage: {name} ===")
        _run_stage(mod, stage_args)


if __name__ == "__main__":
    main()
