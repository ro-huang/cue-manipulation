#!/usr/bin/env bash
# Copy already-validated items + primes from a discovery run into a replication
# run dir, so the replication run only needs to execute stages 04 + 05 (and 10
# for the headline figure).
#
# Usage:
#   ./scripts/bootstrap_replication.sh <src_run_dir> <dest_run_dir>
# Example:
#   ./scripts/bootstrap_replication.sh runs/mve_llama8b runs/replication_qwen7b
set -euo pipefail

src=${1:?"usage: $0 <src_run_dir> <dest_run_dir>"}
dest=${2:?"usage: $0 <src_run_dir> <dest_run_dir>"}

if [[ ! -f "$src/items_filtered.jsonl" ]]; then
    echo "ERROR: $src/items_filtered.jsonl not found" >&2
    exit 1
fi
if [[ ! -f "$src/primes.jsonl" ]]; then
    echo "ERROR: $src/primes.jsonl not found" >&2
    exit 1
fi

mkdir -p "$dest"
cp "$src/items_filtered.jsonl" "$dest/items_filtered.jsonl"
cp "$src/primes.jsonl" "$dest/primes.jsonl"
# items_raw and prime_leaks are diagnostic, copy if present.
[[ -f "$src/items_raw.jsonl" ]] && cp "$src/items_raw.jsonl" "$dest/items_raw.jsonl" || true
[[ -f "$src/prime_leaks.json" ]] && cp "$src/prime_leaks.json" "$dest/prime_leaks.json" || true

echo "bootstrapped $dest from $src:"
ls -lh "$dest"/items_filtered.jsonl "$dest"/primes.jsonl
