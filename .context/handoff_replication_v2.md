# Handoff: replication_v2 (Llama-3.1-70B + Qwen-2.5-72B)

**Status as of 2026-04-26.** A run is in progress on a RunPod GPU pod. The user
asked for a handoff packet so any agent (or the user on a different machine) can
take over.

---

## TL;DR — what's left

1. Wait for the in-progress run to finish on the pod. (Or detect that it's done.)
2. Pull `runs/replication_v2/` artifacts back to local.
3. Render the cross-model headline figure locally (the pod has no matplotlib).
4. Stop the H100 pod (it's billed at $2.99/hr).
5. Report C1 / C2 / C3 verdict per new model against the pre-registration.

---

## Current state on the pod

**Pod:** `5arx3v799tky85` ("koriat-replication-v2")
- SSH: `ssh -i /Users/rowanhuang/.runpod/ssh/RunPod-Key-Go root@64.247.201.44 -p 18210`
- 1× NVIDIA H100 80GB, $2.99/hr, US, 200GB container disk
- Working dir: `/workspace/koriat`
- HF token cached at `~/.cache/huggingface/token` (account: `yes-man-today`)
- Already-installed deps: torch 2.5.1+cu124, transformers, bitsandbytes (latest), all project deps

**What's done in `runs/replication_v2/trials.parquet`:**
- `llama3_70b` model, 460/461 items × 8 conditions = 3675 rows.
  (1 item lost to a failed-prime skip; expected and harmless.)

**What's pending:**
- `qwen2_5_72b` trials. The runner restarted after a disk-full crash on the
  first attempt; the second attempt resumes the partial download (~57GB cached
  out of ~140GB) and then runs ~461 items × 8 conditions = ~3.7k more trials.
- Stage 05 (`scripts/05_analyze.py`) auto-runs after stage 04 succeeds.
- Stage 10 (the headline figure) **fails on the pod** because matplotlib isn't
  installed there — that's expected. Render locally instead (see §"Rendering").

**Approximate ETA:**
- Qwen download finish: a few min from the relaunch.
- Qwen trials: ~60 min on the H100.
- So the run should complete ~60–75 min after relaunch.

---

## How to detect "done"

```bash
ssh -i /Users/rowanhuang/.runpod/ssh/RunPod-Key-Go root@64.247.201.44 -p 18210 \
  'cd /workspace/koriat && \
   (ps -ef | grep -E "(04_run|run_repli)" | grep -v grep || echo NO_PROC); \
   echo "===log==="; tail -30 logs/replication_v2.log | tr -s "\r" "\n" | tail -10; \
   echo "===parquet rows by model==="; \
   python3 -c "import pandas as pd; df = pd.read_parquet(\"runs/replication_v2/trials.parquet\"); print(df.groupby(\"model_name\").size())"'
```

**Done iff:**
- `NO_PROC` returned, AND
- log tail contains `=== done; outputs in runs/replication_v2/` (the last line
  of `scripts/run_replication_v2.sh`), AND
- both `llama3_70b` and `qwen2_5_72b` show ~3675 rows in the parquet.

If the log shows a Python traceback or "No space left on device", see
§"Common failure modes" below.

---

## Pulling artifacts back

```bash
mkdir -p runs/replication_v2/plots
rsync -av -e "ssh -i /Users/rowanhuang/.runpod/ssh/RunPod-Key-Go -p 18210" \
  --exclude='items_filtered.jsonl' \
  --exclude='primes.jsonl' \
  --exclude='prime_leaks.json' \
  root@64.247.201.44:/workspace/koriat/runs/replication_v2/ \
  runs/replication_v2/
```

The pod has the items + primes (copied from `runs/mve_llama8b/`); we don't need
to pull them back since the local already has the originals.

---

## Rendering the headline figure (locally)

```bash
PYTHONPATH=src ./.venv/bin/python scripts/10_headline_figure.py \
  --run-dir runs/mve_llama8b \
  --run-dir runs/replication_qwen7b \
  --run-dir runs/replication_v2 \
  --out runs/replication_v2/plots/headline.png \
  --title-suffix "discovery (8B) + Qwen-7B + Llama-70B + Qwen-72B"
```

Result will be a 4-row figure: 8B, Qwen-7B, Llama-70B (int4), Qwen-72B (int4).
Each row has paired Δ-from-baseline bars for {log_prob, verbal_cat, p_true,
accuracy} × {cue_fam, partial_acc, target, rand_para, whitespace}.

For more views (all 7 conditions, per-item Koriat overconfidence scatter), see
the inline script at `runs/replication_qwen7b/plots/` — replicate the pattern
or grep prior conversation for the `extra_figures.py` snippet.

---

## Pre-registration criteria (the "did it pass" check)

Authoritative doc: **`.context/preregistration.md`**. Pre-reg pass criteria
(§3.1):

- **C1** `cue_familiarity_priming`: log_prob ↑ (p<.05), verbal_cat ↑ (p<.05),
  p_true ≈ 0 (n.s. or |Δ|<0.02), accuracy ≈ 0 (n.s. or |Δ|<0.03).
- **C2** `partial_accessibility`: log_prob ↑ (p<.05), verbal_cat ↑ (p<.05),
  p_true flat or smaller-magnitude than log_prob, accuracy ↓ (p<.05).
- **C3** `target_priming`: accuracy ↑ (p<.05).

Headline claim passes overall iff it passes on Llama-3.1-8B (already done — it
does) AND at least one replication model. Qwen-2.5-7B already failed C1 (see
deviation log §5). The new question is whether Llama-70B or Qwen-72B clears
C1 ∧ C2 ∧ C3.

The cleanest cross-model finding is **C2's partial-accessibility signature**
(log_prob inflation paired with accuracy collapse). Already replicates on 8B
and 7B; expect to replicate on 70B and 72B too.

---

## Stopping the pod

After artifacts are pulled and the figure renders cleanly:

```bash
runpodctl pod stop 5arx3v799tky85
# OR delete entirely (no preserved state):
runpodctl pod delete 5arx3v799tky85
```

The pod has no persistent volume (`volumeInGb: 0`) — once stopped/deleted, all
of `/workspace/koriat` is gone. Make sure the pull succeeded first.

---

## Common failure modes (in order of likelihood)

1. **Disk full ("No space left on device")** during a model download.
   - Symptom: log shows `RuntimeError: ... No space left on device`, then SegFault.
   - Fix: SSH in, check `df -h /` and `du -sh ~/.cache/huggingface/hub/models--*/`,
     then delete cached models that are already done. Llama-70B's cache (~132GB
     after download — bitsandbytes int4 quantizes at load time, the disk holds
     bf16 weights) was deleted to make room for Qwen-72B. If you need to free
     more: `rm -rf ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct`.
   - Then relaunch: `cd /workspace/koriat && nohup bash scripts/run_replication_v2.sh > logs/replication_v2.log 2>&1 < /dev/null & disown`.
   - The runner is **resumable** — it dedups by `(model_name, item_id, condition)`
     against the existing parquet, so already-done trials don't re-run.

2. **`AttributeError: ... no attribute 'set_submodule'`** when loading int4/int8.
   - Cause: torch < 2.5 + recent transformers BNB integration mismatch.
   - Fix: `pip install --break-system-packages torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124`
     and reinstall bitsandbytes. The pod already has this fixed.

3. **`ValueError: Some modules are dispatched on CPU or disk` (int8 path)**.
   - Cause: 70B int8 doesn't reliably fit on a single 80GB H100 at load time.
   - Fix: switch to `dtype: int4` in the config (`configs/replication_v2.yaml`).
     Already done on this run.

4. **Gemma 403 Forbidden / GatedRepoError**.
   - Cause: The HF token has metadata access but not download access.
   - Fix: User must visit https://huggingface.co/google/gemma-2-9b-it and
     https://huggingface.co/google/gemma-2-27b-it and accept the license, then
     re-add the Gemma entries to `configs/replication_v2.yaml`. **As of this
     handoff the user has dropped Gemma; this is informational only.**

---

## Code & files (post-Apr 26 snapshot)

If cloning fresh, ensure these files are present (some may have been added
after the last `origin/main` push that you see):

- `configs/replication_v2.yaml` (Llama-70B + Qwen-72B, int4)
- `configs/replication_qwen7b.yaml` (the prior Qwen-7B replication)
- `scripts/10_headline_figure.py` (the figure tool)
- `scripts/bootstrap_replication.sh` (copy items + primes between run dirs)
- `scripts/run_replication_v2.sh` (the bootstrap → 04 → 05 sequence)
- `scripts/run_replication_qwen7b.sh` (parallel for the Qwen-7B run)
- `.context/preregistration.md` (authoritative design doc; deviation log lives at §5)

`.context/` is gitignored by Conductor's convention; if anything in there is
missing locally, force-add (`git add -f`) when committing.

---

## SSH access

The pod is keyed to the SSH private key at
`/Users/rowanhuang/.runpod/ssh/RunPod-Key-Go` on the user's primary machine.
A different machine needs either:

- (a) A copy of that private key file at the same path, OR
- (b) Their own public key added to the pod via `runpodctl pod` (see the runpod
  docs for editing pod env / SSH keys), OR
- (c) Use the user's RunPod account directly with `runpodctl ssh 5arx3v799tky85`,
  which uses the account-level managed key.

Option (c) is simplest if the receiving machine has `runpodctl` and the user's
RunPod API key.

---

## Account context

- RunPod account email: `rowan@fulcrum.inc`
- Recent pod balance: ~$57 (as of 2026-04-26). Each H100 hour is $2.99.
- An older A40 pod ("koriat-rerun", `2es7p02rs2qv09`) is also running at
  $0.44/hr — used for prior 8B/7B runs. Still up; user hasn't asked to stop it.
