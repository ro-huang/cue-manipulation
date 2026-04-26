#!/usr/bin/env bash
# Bootstraps a fresh RunPod PyTorch pod to run the koriat_cues pilot.
# Expects ANTHROPIC_API_KEY and HF_TOKEN to already be exported.
set -euo pipefail

cd /workspace/koriat

echo "=== python version ==="
python3 --version

echo "=== installing pip deps ==="
pip install --break-system-packages --quiet --upgrade pip
pip install --break-system-packages --quiet \
    "transformers>=4.44" \
    "accelerate>=0.30" \
    "datasets>=2.18" \
    "anthropic>=0.40" \
    "pydantic>=2.5" \
    "pyyaml>=6.0" \
    "pandas>=2.1" \
    "numpy>=1.26" \
    "scipy>=1.11" \
    "scikit-learn>=1.4" \
    "statsmodels>=0.14" \
    "pyarrow>=15.0" \
    "tqdm>=4.66" \
    "tenacity>=8.2" \
    "click>=8.1"

echo "=== sanity: torch + CUDA ==="
python3 -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

echo "=== HF login (non-interactive via env) ==="
huggingface-cli whoami || echo "(unauthenticated, token from env will be used)"

echo "=== running unit tests before real run ==="
PYTHONPATH=src python3 -m pytest tests/ -q

echo "=== stage 00: prepare data (baseline-accuracy filter) ==="
PYTHONPATH=src python3 scripts/00_prepare_data.py --config configs/pilot.yaml

echo "=== stage 01: build CAA vector ==="
PYTHONPATH=src python3 scripts/01_build_caa_vector.py --config configs/pilot.yaml

echo "=== stage 02: generate primes via Claude ==="
PYTHONPATH=src python3 scripts/02_generate_primes.py --config configs/pilot.yaml

echo "=== stage 03: validate primes (leak judge) ==="
PYTHONPATH=src python3 scripts/03_validate_primes.py --config configs/pilot.yaml

echo "=== stage 04: run experiment ==="
PYTHONPATH=src python3 scripts/04_run_experiment.py --config configs/pilot.yaml

echo "=== stage 05: analyze ==="
PYTHONPATH=src python3 scripts/05_analyze.py --config configs/pilot.yaml

echo "=== done; outputs in runs/pilot_llama8b/ ==="
ls -la runs/pilot_llama8b/
