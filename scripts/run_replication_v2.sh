#!/usr/bin/env bash
# Replication v2: Llama-70B + Qwen-72B + Gemma-9B + Gemma-27B on the locked design.
# Designed to be invoked on a GPU pod with sufficient memory (e.g. H100 80GB).
#
# Notes:
# - 9B and 27B run in bf16; 70B and 72B run in int8 to fit on a single 80GB GPU.
# - Reuses items + primes from runs/mve_llama8b/.
# - The full run is ~3 hours wall-clock on a single H100; resumable via the
#   runner's parquet-checkpoint logic.
set -euo pipefail

cd "${KORIAT_ROOT:-/workspace/koriat}"

[[ -f .envrc ]] && set -a && . .envrc && set +a

echo "=== bootstrap items + primes ==="
./scripts/bootstrap_replication.sh runs/mve_llama8b runs/replication_v2

echo "=== installing extra deps for int8 quantization ==="
pip install --quiet --break-system-packages bitsandbytes >/dev/null 2>&1 || pip install --quiet bitsandbytes

echo "=== stage 04: run experiment on 4 models (bf16 for ≤27B, int8 for 70B-class) ==="
PYTHONPATH=src python3 scripts/04_run_experiment.py --config configs/replication_v2.yaml

echo "=== stage 05: analyze ==="
PYTHONPATH=src python3 scripts/05_analyze.py --config configs/replication_v2.yaml

echo "=== done; outputs in runs/replication_v2/ ==="
ls -lh runs/replication_v2/
