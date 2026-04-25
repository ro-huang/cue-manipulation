# Pilot run — 2026-04-24

Configuration: `configs/pilot.yaml` (Llama-3.1-8B-Instruct, 50-item target, 3 conds, 2 measures).
GPU: RunPod A40 (48GB), ~14 min, ~$0.10 compute.
Prime generator + judge: Claude Sonnet 4.6.

## Items that survived filtering

- TriviaQA validation, oversampled 200 items
- 193 passed single-entity filter
- **24** passed the baseline-accuracy band [0.2, 0.8]
- **23** survived the cue-familiarity leak judge (1 item had a ≥5% leak rate and was dropped)

N=23 is smaller than the 50-item target. TriviaQA is bimodal difficulty-wise: most items are either trivial or out of range for 8B. Widening the band or oversampling more would recover the target count.

## Per-condition raw means

| condition                | n  | accuracy | verbal_cat | caa_proj |
|--------------------------|----|----------|------------|----------|
| baseline                 | 23 | 0.652    | 0.895      | 2.866    |
| cue_familiarity_priming  | 23 | 0.652    | 0.941      | 2.753    |
| target_priming           | 23 | 0.696    | 0.956      | 2.824    |

## Per-item paired deltas vs baseline

| condition                | Δaccuracy | Δverbal_cat | Δcaa_proj |
|--------------------------|-----------|-------------|-----------|
| cue_familiarity_priming  | +0.000    | +0.062      | -0.112    |
| target_priming           | +0.043    | +0.074      | -0.042    |

## Read

**H1 — cue familiarity inflates confidence without accuracy (Schwartz & Metcalfe 1992):**
Directionally confirmed on verbalized confidence. Accuracy delta is *exactly* zero (same items
correct under both conditions), verbal_cat +0.062. Regression p=0.064 on the cue-familiarity
intercept — close but not significant at this N.

**H3 — dissociation of cue familiarity from target retrievability:**
Target priming raises accuracy modestly (+4.3pp) and verbal confidence (+0.074). Cue-familiarity
raises verbal confidence by nearly as much (+0.062) *without any accuracy gain*. That is the
Schwartz-Metcalfe pattern.

**CAA projection moves opposite (mechanistic surprise):**
Both priming conditions lower the CAA projection (-0.112 cue-familiarity, -0.042 target). This
is the finding SPEC.md flagged as plausible ("CAA-projection and log-probs might differ, which
would be a useful mechanistic finding in itself"). Possible explanations:

1. Contextual hidden states at the post-newline position respond to *any* added context with
   a small confidence dampening (the random-paragraph yoked control would disambiguate).
2. The CAA vector captures something other than verbalized confidence — this is itself a
   second-order finding worth following up.
3. Steering vector needs more pairs (205 used; 200 was the spec target) or a different layer
   than -12.

## Caveats

- **N=23** — directional, not significant. p≈0.064 on verbal_cat intercept.
- **Ceiling at 0.89** for baseline verbal_cat — Llama-3.1-8B-Instruct is already reporting "high"
  confidence by default; there's little room upward.
- **No random-paragraph control** was run — pilot config dropped it; the yoked control is
  essential for interpreting the CAA sign reversal.
- **Single model** — scale effects (Llama 70B vs 8B) were not tested.

## Scale-up path

Full MVE (500 items, 3 conds, 2 measures):
- Widen baseline-accuracy band to 0.15–0.85 to recover more items, **or** oversample 8× (n*8).
- Add `random_paragraph` yoked control (cheap, no extra Claude calls).
- Add `verbal_num` measure (free — one extra forward pass per trial).
- Estimated cost: ~$1 GPU + ~$5 Claude Sonnet.
