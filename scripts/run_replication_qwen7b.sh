#!/usr/bin/env bash
# Replication run: Qwen-2.5-7B-Instruct, on the same items + primes as runs/mve_llama8b.
# Designed to be invoked on a GPU pod after `pod_bootstrap.sh` has installed deps.
#
# Steps:
#   1. Copy items_filtered.jsonl + primes.jsonl from runs/mve_llama8b → runs/replication_qwen7b.
#   2. Run experiment (stage 04) on Qwen-2.5-7B for all 8 conditions × 461 items.
#   3. Analyze (stage 05).
#   4. Render headline figure with both models on it (stage 10).
#
# Skipped vs the original pipeline:
#   - 00 prepare_data:   items reused.
#   - 01 build_caa:      caa_proj not in measures.
#   - 02 generate_primes: primes reused (model-independent text).
#   - 03 validate_primes: prime leaks judged by claude-opus-4-7, model-independent.
set -euo pipefail

cd "${KORIAT_ROOT:-/workspace/koriat}"

[[ -f .envrc ]] && set -a && . .envrc && set +a

echo "=== bootstrap items + primes ==="
./scripts/bootstrap_replication.sh runs/mve_llama8b runs/replication_qwen7b

echo "=== stage 04: run experiment on Qwen-2.5-7B ==="
PYTHONPATH=src python3 scripts/04_run_experiment.py --config configs/replication_qwen7b.yaml

echo "=== stage 05: analyze ==="
PYTHONPATH=src python3 scripts/05_analyze.py --config configs/replication_qwen7b.yaml

echo "=== stage 10: headline figure (Llama-3.1-8B + Qwen-2.5-7B) ==="
PYTHONPATH=src python3 scripts/10_headline_figure.py \
    --run-dir runs/mve_llama8b \
    --run-dir runs/replication_qwen7b \
    --out runs/replication_qwen7b/plots/headline.png \
    --title-suffix "discovery + Qwen replication"

echo "=== done ==="
ls -lh runs/replication_qwen7b/
