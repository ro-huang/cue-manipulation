#!/usr/bin/env bash
# Replication v2: Llama-3.1-70B + Qwen-2.5-72B on the locked design (see config).
# Designed to be invoked on a GPU pod with sufficient memory (e.g. H100 80GB).
#
# Notes:
# - 70B/72B use int4 (bitsandbytes) to fit on a single 80GB GPU (see prereg §5).
# - Reuses items + primes from runs/mve_llama8b/.
# - Wall-clock depends on download + model; resumable via the runner's parquet logic.
set -euo pipefail

cd "${KORIAT_ROOT:-/workspace/koriat}"

[[ -f .envrc ]] && set -a && . .envrc && set +a

echo "=== bootstrap items + primes ==="
./scripts/bootstrap_replication.sh runs/mve_llama8b runs/replication_v2

echo "=== installing bitsandbytes (int4) ==="
pip install --quiet --break-system-packages bitsandbytes >/dev/null 2>&1 || pip install --quiet bitsandbytes

echo "=== stage 04: run experiment (per configs/replication_v2.yaml) ==="
PYTHONPATH=src python3 scripts/04_run_experiment.py --config configs/replication_v2.yaml

echo "=== stage 05: analyze ==="
PYTHONPATH=src python3 scripts/05_analyze.py --config configs/replication_v2.yaml

echo "=== done; outputs in runs/replication_v2/ ==="
ls -lh runs/replication_v2/
