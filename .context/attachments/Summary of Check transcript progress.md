This is a partial transcript of a conversation between a user and an AI assistant. Intermediate AI messages have been elided.

## User

@Summary of Explain CAA whitespace.md What's going on with this transcript progress?

## Assistant

Here's where things actually stand vs. what the transcript implied.

## Status

The transcript cut off mid-stream while the assistant was retrying a nohup launch on the pod. Looking at the workspace, **the run did finish — but only partially as planned.**

**What completed (Apr 24 22:33–22:34):**
- ✅ Correctness-partitioned CAA rebuild — `caa_llama3_8b_layer_scan.csv` shows layer 31 picked, AUC 0.765 (vs the v2 log-prob version at 0.77 — basically same layer/quality, different basis).
- ✅ Full 5-condition × 489-item trial run (2445 rows in `trials.parquet`).
- ✅ Analysis CSVs + plot regeneration (plots dir Apr 25 07:58).

**What did NOT happen:**
- ❌ **`p_true` is missing from the parquet.** Columns are `log_prob, verbal_cat, verbal_num, caa_proj` only. The `extra` field is empty. No `p_true` in `summary_per_condition.csv` or regression output either.

**Why:** mtimes give it away. `trials.parquet` is Apr 24 22:34, but `src/koriat_cues/confidence/measures.py` is Apr 25 08:30 and `configs/mve.yaml` is Apr 25 09:34 — both edited *after* the rerun. So the pod ran a code version that didn't have the Kadavath p_true wiring yet (or the config without `p_true` in the measures list). The transcript's "All four tasks complete" was the local code state, but the pod run that followed used a stale checkout.

## What the existing results actually say (correctness-CAA only)

For `cue_familiarity_priming` (n=489):

| measure | conf shift | item-corr w/ accuracy | DI ratio |
|---|---|---|---|
| log_prob | 0.063 | 0.20 | 1.63 |
| verbal_cat | 0.050 | 0.29 | 1.29 |
| verbal_num | 0.038 | 0.29 | 0.98 |
| caa_proj | 0.021 | **0.31** | 0.55 |

The correctness-partitioned CAA tracks accuracy more tightly than any other measure (item_corr 0.31) and shows the *smallest* cue-familiarity inflation — which is the cleanly-grounded behavior the previous discussion argued was the honest version of CAA. Whitespace is also down to +0.037 on caa_proj (still nonzero — the centering issue from the prior turn was never fixed, only the contrast basis was).

## Suggested next step

The interesting Kadavath comparison is what's missing. Two options:
1. **Re-run on the pod** with the current code+config to get `p_true` populated (~$1–2 GPU per the prior estimate; only `04_run_experiment.py` needs to rerun, the CAA vector is already correctness-built).
2. Or partial-rerun `p_true` only and merge into the existing parquet, if you want to skip re-doing everything else.

Want me to push the current code/config to the pod and kick off (1)?

## User

I'd like to get p_true please.

## Assistant

Done — `p_true` is in the data and all analysis is updated.

## What you now have

`runs/mve_llama8b/trials.parquet` (2445 rows × 5 conditions × 489 items) now includes both `p_true` and `p_true_raw` columns. All CSVs (summary, dissociation, regression) and all plots in `plots/` were regenerated.

## Headline result (cue_familiarity_priming, n=489)

| measure | conf shift | acc shift | DI ratio | item_corr w/ acc | regression p |
|---|---|---|---|---|---|
| log_prob | +0.063 | +0.039 | **1.63** ⬆ | 0.20 | <1e-12 |
| verbal_cat | +0.050 | +0.039 | 1.29 ⬆ | 0.29 | 3e-4 |
| verbal_num | +0.038 | +0.039 | 0.98 | 0.29 | 3e-5 |
| **caa_proj** (correctness) | +0.006 | +0.039 | 0.15 | **0.38** | 0.10 (n.s.) |
| **p_true** | −0.006 | +0.039 | −0.16 | **0.35** | 0.96 (n.s.) |

The two correctness-grounded probes (correctness-partitioned CAA, P(True)) **do not show the cue-familiarity inflation**. And they have the *highest* item-level correlation with accuracy of any measure. log_prob and verbal_cat both show inflation; verbal_num matches accuracy exactly.

## Side notes

- The CAA layer scan now picks **layer 12** at AUC 0.807 (with the correctness partition + larger contrast pool of 308 vs 172 items). Yesterday's correctness vector picked layer 31 at 0.765. The shallower layer + cleaner partition produces a much more accuracy-aligned probe.
- Whitespace effect on caa_proj: now −0.016 (regression p=0.004) — still significant but flipped sign and smaller magnitude than before. The whitespace-on-p_true effect is +0.022 (p=0.010). Still a centering issue, but smaller story than the v2 logprob CAA.
- Plot `summary_headline.png` is the easiest one to look at first.

The honest read: **the Koriat-style cue-familiarity inflation lives in log_prob and verbal_cat, not in P(True) or correctness-CAA.** That's now testable end-to-end with one figure.