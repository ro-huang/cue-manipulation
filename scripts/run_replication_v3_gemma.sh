#!/usr/bin/env bash
# Replication v3: Gemma-2-9B-IT + Gemma-2-27B-IT on the locked items + primes.
# Third model family (alongside Llama / Qwen); bf16 for both.
#
# Requires HF_TOKEN with Gemma download access (accept licenses on Hugging Face).
# Intended for a GPU pod (e.g. H100/A100 80GB). Stage 10 is omitted here because
# many pods lack matplotlib; render the headline figure locally after rsync.
set -euo pipefail

cd "${KORIAT_ROOT:-/workspace/koriat}"

[[ -f .envrc ]] && set -a && . .envrc && set +a

echo "=== bootstrap items + primes ==="
./scripts/bootstrap_replication.sh runs/mve_llama8b runs/replication_v3_gemma

echo "=== stage 04: Gemma-2 9B + 27B ==="
PYTHONPATH=src python3 scripts/04_run_experiment.py --config configs/replication_v3_gemma.yaml

echo "=== stage 05: analyze ==="
PYTHONPATH=src python3 scripts/05_analyze.py --config configs/replication_v3_gemma.yaml

echo "=== pre-reg verdict (stdout) ==="
PYTHONPATH=src python3 scripts/11_prereg_verdict.py --trials runs/replication_v3_gemma/trials.parquet

echo "=== done; outputs in runs/replication_v3_gemma/ ==="
echo "Local headline figure example:"
echo "  PYTHONPATH=src python scripts/10_headline_figure.py \\"
echo "    --run-dir runs/mve_llama8b --run-dir runs/replication_qwen7b \\"
echo "    --run-dir runs/replication_v3_gemma \\"
echo "    --out runs/replication_v3_gemma/plots/headline.png \\"
echo "    --title-suffix \"... + Gemma-9B + Gemma-27B\""
ls -lh runs/replication_v3_gemma/
