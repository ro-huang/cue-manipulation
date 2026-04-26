# Spec: Causal "feel-known" axis in Llama-3-8B activations

**Status:** open task. Should be done on a separate branch off `koriat-cues` (the `koriat-cues` branch is dedicated to tightening the existing behavioral results in `runs/mve_llama8b/` and should not be touched by this work).

**Owner branch suggestion:** `feel-known-axis` off `koriat-cues` (you need the existing `runs/mve_llama8b/` artifacts; they live in that branch's working tree, not in git history — coordinate with the workspace owner to copy `runs/mve_llama8b/` over if you start from a fresh checkout).

---

## 1. Background and motivation

The `koriat_cues` project replicates Koriat-style metamemory cue-manipulation paradigms on small LLMs. Items are TriviaQA questions; each item is run under eight conditions that vary how a "prime" is prepended to the question (`src/koriat_cues/primes/conditions.py`). Three of those conditions are designed to raise *feeling-of-knowing* (FOK) without raising actual knowledge:

- `cue_familiarity_priming` — repeats surface terms of the question, no answer info (Schwartz & Metcalfe 1992).
- `partial_accessibility` — plausible-sounding wrong info on the topic (Koriat 1993).
- `illusory_tot` — semantically adjacent passage, dense with proper nouns (Schwartz 1998).

Behavioral results (`runs/mve_llama8b/summary_per_condition.csv`, `regression_clustered.txt`) show that these manipulations *do* raise the model's confidence (most clearly on `log_prob` and `verbal_cat`), and `partial_accessibility` produces a striking accuracy collapse (−0.34) with simultaneous confidence rise. But the five confidence channels (log_prob, verbal_cat, verbal_num, caa_proj, p_true) disagree with each other across conditions, so a behavioral statement of "the model has FOK" is unstable — it depends which operationalization you pick.

**Hypothesis (this work):** there is a low-dimensional direction (or low-rank subspace) in Llama-3-8B's residual stream such that:

1. Adding it to baseline-trial activations raises model confidence **without** raising accuracy.
2. Subtracting it from primed-trial activations eliminates the confidence lift.
3. The same direction works (with degraded but non-trivial transfer) across `cue_familiarity_priming`, `partial_accessibility`, and `illusory_tot`.

If found, this is a *causal* axis for "feels-known" in the model — a much stronger claim than any behavioral one, and the right scientific contribution.

**Why the existing CAA axis is not this.** The current `caa_proj` measure is built on confident-vs-hedged *completion* pairs (`src/koriat_cues/caa/contrast_pairs.py`). That contrast confounds "knowing" with "feels-known" — both make the model say confident things. Empirically `caa_proj` barely moves under any FOK manipulation (most condition p-values > 0.1 in `regression_clustered.txt`), which is consistent with it tracking real knowing rather than felt-knowing. Do not reuse this axis. Build a new one on the right contrast.

---

## 2. Existing artifacts you should reuse

Located at `runs/mve_llama8b/` (untracked; copy the directory wholesale):

- `items_filtered.jsonl` — 461 TriviaQA items that survived leak-validation; the item set this whole project uses.
- `primes.jsonl` — generated prime text per (item, condition).
- `trials.parquet` — per-trial outputs including all five confidence measures and accuracy. **Use this to identify which (item, condition) trials raised confidence without raising accuracy** — those are the FOK-positive trials, and they're the contrast you build the axis on.
- `caa_llama3_8b_layer_scan.csv` — layer scan for the *old* CAA contrast. Best AUC is at layers 12–15 (out of 32). Use as a prior on where to look first; don't assume the FOK axis lives at the same layer.
- `summary_per_condition.csv`, `compare_measures.csv`, `regression_clustered.txt` — reference for the behavioral effects you're trying to causally reproduce.

You will **need to re-extract activations** — the existing `caa_llama3_8b.npz` is built on the wrong contrast. Reuse `src/koriat_cues/caa/vector.py`'s read-position helper (`_post_newline_position`) and the `HFModel` wrapper in `src/koriat_cues/models/`, but the contrast / averaging logic is new.

Model: `meta-llama/Llama-3.1-8B-Instruct`, bfloat16. System prompt and `assemble_prompt` are in `src/koriat_cues/primes/conditions.py`.

---

## 3. Task

Build, validate, and (if it survives validation) test the generality of a causal "feel-known" axis.

The work is staged into four phases with explicit go/no-go gates between them. **Stop at the gate if the criterion fails — that is itself a finding, and pushing past it produces nonsense.**

### Phase 1 — Construct candidate axes (cheap; ~1 day)

For each FOK condition C ∈ {cue_familiarity_priming, partial_accessibility, illusory_tot}:

1. Identify the **FOK-positive item subset** for C: items where `confidence_shift_log_prob > 0` AND `accuracy_shift ≤ 0` (i.e., confidence rose, accuracy did not improve). For `partial_accessibility` you should additionally have a stricter "Koriat" subset where `accuracy_shift < 0`. Pull these from `trials.parquet`.
2. Re-run forward passes on (primed_i, baseline_i) pairs for those items. Capture residual-stream activations at the **post-newline position** (the position right before the answer is generated; see `_post_newline_position` in `src/koriat_cues/caa/vector.py`) at every layer.
3. Compute the per-condition mean delta vector at each layer:
   `v_C(L) = mean_i [ act_L(primed_i) − act_L(baseline_i) ]`.
4. Compute a **whitespace-residualized** version: `v_C^*(L) = v_C(L) − v_whitespace(L)` where `v_whitespace(L)` is the same mean delta computed on the `whitespace` condition (length-only artifact control). This removes generic "any-prefix-present" effects.
5. Compute a **pooled** axis `v_FOK(L) = mean over C in {cue_fam, partial_acc, illusory_tot}` of `v_C^*(L)`, weighting equally.

Deliverables:
- A cosine-similarity heatmap between the three per-condition residualized axes, at the layer where the old CAA AUC peaks (layer 12–15) **and** at one early (layer 6) and one late (layer 24) layer for sanity. Include a written interpretation of whether you see block structure (≥0.5 cosine within FOK, ≪0.5 to whitespace and to a `target_priming` axis computed the same way as a negative reference).
- Saved tensors of `v_C^*(L)` and `v_FOK(L)` for L ∈ scanned layers.
- A short markdown report at `runs/feel_known/phase1_report.md`.

**Gate G1:** the three FOK axes are cosine-similar to each other (cosine ≥ 0.4 at *some* layer, *and* meaningfully larger than their cosine to `target_priming`'s axis or to `whitespace`'s axis). If this fails uniformly across layers, the construct probably isn't a single direction in this model. Stop and write that up; do not proceed to phase 2.

### Phase 2 — Causal validation by steering (~3–5 days)

Choose the layer L* with the cleanest phase-1 block structure (not necessarily the old CAA peak). Use the pooled axis `v_FOK(L*)`.

**Steering test:**
1. On held-out items (split the 461-item set 70/30 by `item_id` before phase 1; never let the held-out 30% participate in axis construction), run baseline trials with an activation hook that adds `α · v_FOK(L*)` at layer L*, post-newline position, for a sweep of magnitudes α ∈ {0, 0.5, 1, 2, 4, 8} × ||v_FOK(L*)||.
2. For each α, recompute all five confidence measures and accuracy on the held-out set.
3. Required pattern (the falsifiable claim):
   - `log_prob` confidence rises monotonically with α over some range.
   - `verbal_cat` rises (less monotonically is OK; verbal channels are noisier).
   - **Accuracy is statistically flat** (no significant rise, paired test on item_id, p > 0.05) over the range where confidence is rising.
4. Report the (Δconfidence, Δaccuracy) curve. The "win" condition is finding a magnitude α* where the held-out steered distribution overlaps statistically with the held-out `cue_familiarity_priming` primed distribution (use a 2-sample test on the joint (Δconfidence, Δaccuracy) outcome; KS or energy distance).

**Subtraction test:**
1. On held-out `cue_familiarity_priming` primed trials, run with a hook that *subtracts* `α · v_FOK(L*)` at L*.
2. Required pattern: confidence shift collapses toward zero (no significant lift over baseline). If subtraction also lowers accuracy, the axis is entangled with knowing — report this honestly.

Deliverables:
- `runs/feel_known/steering_curve.csv` and matching plots.
- `runs/feel_known/subtraction_results.csv`.
- `runs/feel_known/phase2_report.md`: a written go/no-go decision.

**Gate G2:** steering raises log_prob confidence by ≥ 50% of the cue_familiarity_priming primed lift, with accuracy not significantly different from baseline (paired test, p > 0.05), at some α*. AND subtraction reduces the cue_familiarity_priming confidence lift by ≥ 50%. If either fails, the axis is not causal in the required sense — stop and write up the negative result. **Do not** silently retune the contrast or layer to make the gate pass; that's p-hacking on causal claims.

### Phase 3 — Refinement via DAS (optional but recommended; ~3–5 days)

If gate G2 passes, refine the axis from a single direction to a low-rank subspace using **Distributed Alignment Search** (Geiger et al.; standard implementation in [`pyvene`](https://github.com/stanfordnlp/pyvene)).

The training objective: find a rank-k rotation R such that patching the projected subspace from `act(primed_i)` into `act(baseline_i)` at layer L* reproduces the primed-condition behavior (confidence and accuracy distributions) on training items, evaluated at k ∈ {1, 4, 16}.

Deliverables:
- Trained DAS rotation per k.
- Comparison: DAS subspace vs phase-1 mean-delta direction on the same steering and subtraction tests as phase 2 (held-out items).
- `runs/feel_known/phase3_report.md`.

**Gate G3:** rank-1 DAS does at least as well as the mean-delta axis on phase-2 metrics, and rank-4 does meaningfully better (≥ 20% improvement in held-out steering effect for the same accuracy-flatness constraint). If rank-1 already saturates, the construct really is a single direction; if rank-16 is required, FOK is more like a manifold.

### Phase 4 — Generality (publishable test; ~1 week)

Two transfer tests, in priority order:

1. **Cross-condition transfer.** Fit the axis on `cue_familiarity_priming` items only; test the steering and subtraction conditions on `partial_accessibility` and `illusory_tot` held-out items. The win: the axis fit on one FOK condition reproduces the others' behavioral signature (confidence ↑ on all; accuracy *drop* specifically when applied to partial_accessibility-typical items).
2. **Cross-task transfer.** Apply the axis (without refitting) on a held-out task — recommended: a 1k-item slice of MMLU, with confidence elicited the same way (log_prob of the answer choice). The win: steering raises confidence without raising accuracy on MMLU too. Failure here means the axis is TriviaQA-specific, which is a real but lesser finding.

Deliverables:
- `runs/feel_known/phase4_cross_condition.csv`, `phase4_mmlu.csv`, plots, written report.
- A figure that summarizes: rows = held-out conditions / held-out tasks, columns = steering effect on confidence and accuracy, with CIs. This is the headline figure of any writeup.

---

## 4. Implementation notes

- **Hooks.** Use `transformer-lens` if you want a clean abstraction, or raw PyTorch `register_forward_hook`s on the residual stream of the relevant `LlamaDecoderLayer`. The existing project uses raw HF + manual capture (see `src/koriat_cues/caa/vector.py`); you can follow that pattern for capture and add a separate "steering hook" path for additive interventions.
- **Read position.** Match the existing project: post-newline of the question/context block, immediately before the answer. `_post_newline_position` in `caa/vector.py` already implements this; reuse it. **Do not** read at the answer-final token — that's a behavioral readout, not a metacognitive computation.
- **Layer choice.** Prior: layers 12–15 are where the old confident-vs-hedged signal lives most cleanly (AUC ≈ 0.80). FOK may live at the same layers or earlier — let phase 1 decide. Don't pre-commit.
- **Magnitudes.** ||v_FOK|| will be small in absolute terms (a single direction in residual stream of dim 4096). The natural unit is "in multiples of ||v_FOK||"; sweep α from 0 to ~8 of those units. Beyond that you usually break fluency.
- **Item splitting.** Pre-register the 70/30 train/held-out split by `item_id` at phase 1 time and don't change it. Use `seed=0` for reproducibility (matches `configs/mve.yaml`).
- **Statistical tests.** Paired tests on `item_id` for accuracy and confidence shifts. Cluster-robust SEs for any regression (matches the existing `regression_clustered.txt` style).
- **Bootstrapping.** Use 1000-draw bootstrap CIs for the steering curve and any condition-level effect — do not rely on parametric SEs alone for the headline figure.
- **Compute budget.** A single Llama-3-8B forward pass over 461 items is cheap (minutes on a single A100). The whole project, end-to-end, should fit in <100 GPU-hours including the MMLU transfer.

---

## 5. Risks and how to handle them

- **No shared direction (gate G1 fails).** Real outcome. The construct fragments. Write up the per-condition axes, their cosine matrix, and the implication: "feel-known" is not represented as a unified direction in Llama-3-8B at this scale. Worth its own short paper.
- **Steering raises both confidence and accuracy.** You found a "general capability" axis, not FOK. Report and stop; don't redefine the axis to hide it.
- **Subtraction also tanks accuracy.** The axis is entangled with knowing. Try a "knowing-orthogonalized FOK axis": project `v_FOK` onto the orthogonal complement of the old confident-vs-hedged CAA direction. Test the orthogonalized version. If *that* still tanks accuracy, the axes really are entangled.
- **Steering breaks fluency at small α.** The axis you found is dominated by a non-FOK component (e.g., a noun-density direction). Tighten the contrast: restrict to FOK-positive items where the prime did not change generation length or token-type distribution.
- **8B is too small for a clean axis.** Possible. Replicate the phase-1 cosine-heatmap analysis on a second model (Qwen-7B or Mistral-7B) before concluding. Cross-model agreement is strong evidence, disagreement narrows the claim.
- **Token-content confound.** Even within an item, primed and baseline trials have different prefix tokens, and the post-newline activation reflects that. Mitigations: (a) include `cue_familiarity_priming` vs `random_paragraph` as a content-matched contrast for sanity; (b) use DAS in phase 3, which optimizes for *causal* patching and is less sensitive to incidental representational differences than mean deltas.

---

## 6. What "done" looks like

Three possible end-states, in decreasing order of strength:

1. **Best:** Gates G1, G2, G3 pass. Cross-condition transfer in phase 4 passes. MMLU transfer passes. → Writeup: "A causal feel-known axis in Llama-3-8B."
2. **Solid:** G1, G2 pass; G3 ambiguous (rank-1 saturates); cross-condition transfer partial; MMLU transfer fails. → Writeup: "A causal axis for trivia-domain illusory FOK."
3. **Negative:** G1 or G2 fails. → Writeup: "Feel-known does not factor as a low-dimensional direction in Llama-3-8B residual stream — implications for the representational basis of LLM metacognition."

All three are publishable. Don't bend the methodology to convert a #2 into a #1, or a #3 into a #2.

---

## 7. Out of scope (for this branch)

- Re-running the behavioral experiment, regenerating primes, or re-tuning the leak validator. Those are the `koriat-cues` branch's job.
- Multi-model replication of the *behavioral* paradigm — only replicate phase 1's cosine heatmap on a second model if needed for risk mitigation in §5.
- Building new behavioral conditions or new confidence measures. Use the existing eight conditions and five confidence measures unchanged.
- Any work on `fluency_degradation` — its behavioral effect is opposite-signed and probably reflects a different mechanism; not part of FOK pooling.

---

## 8. Coordination with the `koriat-cues` branch

- The behavioral analyses on `koriat-cues` may produce a tighter FOK-positive item subset (e.g., by recomputing leak rates or filtering low-quality primes). If that lands before phase 2, re-run phase 1 with the updated subset; otherwise use what's in `runs/mve_llama8b/` as of branch-off.
- Do **not** modify `src/koriat_cues/primes/`, `src/koriat_cues/confidence/`, or `configs/*.yaml` — those are shared. Add new code under a new `src/koriat_cues/feel_known/` module and new scripts under `scripts/feel_known/`.
- All outputs go under `runs/feel_known/` so they don't collide with the behavioral run.

---

## 9. Concrete first steps

1. Branch off `koriat-cues` as `feel-known-axis`. Confirm `runs/mve_llama8b/` is present locally.
2. Create `src/koriat_cues/feel_known/` skeleton: `axis.py` (construction), `steering.py` (hooks), `eval.py` (held-out evaluation).
3. Write `scripts/feel_known/01_extract_activations.py` — reuses `HFModel` and `_post_newline_position`, captures residual-stream activations at all layers for primed/baseline pairs on the FOK conditions and on whitespace + target_priming (negative references).
4. Write `scripts/feel_known/02_build_axes.py` — implements §3 phase 1, produces the cosine heatmap and saves axes.
5. Decide gate G1, write `phase1_report.md`. Stop and check in with the workspace owner before phase 2.
