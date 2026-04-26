#!/usr/bin/env bash
set -e
cd /workspace/koriat
# Container secrets (HF_TOKEN, etc.) live on PID 1; SSH sessions don't inherit them.
[ -f /workspace/koriat/.envrc ] && set -a && . /workspace/koriat/.envrc && set +a
echo "HF_TOKEN length: ${#HF_TOKEN}"
echo "=== stage 01 build CAA correctness partition ==="
python3 scripts/01_build_caa_vector.py --config configs/mve.yaml --partition correctness 2>&1
echo "=== stage 04 run experiment ==="
python3 scripts/04_run_experiment.py --config configs/mve.yaml 2>&1
echo "=== stage 05 analyze ==="
python3 scripts/05_analyze.py --config configs/mve.yaml 2>&1
echo "=== ALL DONE ==="
