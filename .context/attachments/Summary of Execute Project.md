# Methodology — Koriat-Style Cue Manipulations of LLM Confidence

This document describes, in implementation-faithful detail, the experimental pipeline that lives in this repository. The pipeline is designed to test whether LLM-reported confidence is driven by **Koriat-style heuristic cues** (cue familiarity, partial accessibility, perceptual fluency) — i.e., whether the model's "feeling of knowing" is computed from surface features of the prompt rather than from privileged access to a correctness signal.

The entire pipeline is six executable stages (`scripts/00_…` through `scripts/05_…`), each consuming the previous stage's checkpoint, plus a non-ML smoke test (`scripts/smoke_dry_run.py`) and 24 unit tests. Configuration lives in `configs/{pilot,mve,full}.yaml` and is pydantic-validated by `src/koriat_cues/config.py`.

---

## 1. Theoretical scaffolding

The core dependent variable is a **confidence shift relative to baseline**, paired item-by-item. The core independent variable is the **cue manipulation condition**, which targets a specific Koriat heuristic. The seven conditions and the predictions they encode are:

| condition                  | what the prime contains                                            | prediction                          | source                       |
|----------------------------|--------------------------------------------------------------------|-------------------------------------|------------------------------|
| `baseline`                 | nothing                                                            | reference                           | —                            |
| `cue_familiarity_priming`  | surface entities/terms from the question, **no** answer-relevant info | ↑ confidence, **=** accuracy        | Schwartz & Metcalfe (1992)   |
| `target_priming`           | the correct answer slipped into an unrelated context               | ↑ confidence, ↑ accuracy            | semantic-priming control     |
| `partial_accessibility`    | plausible-sounding but **wrong** topic-adjacent facts              | ↑ confidence, **=** accuracy        | Koriat (1993)                |
| `illusory_tot`             | semantically adjacent but irrelevant entity-dense passage          | ↑ confidence, ↓ accuracy            | Schwartz (1998)              |
| `fluency_degradation`      | the question rephrased in awkward syntax with rare synonyms        | ↓ confidence, **=** accuracy        | fluency heuristic            |
| `random_paragraph`         | length-matched neutral noise                                       | yoked control for "any-context" effects | —                        |

Three primary hypotheses fall out:

- **H1**: cue_familiarity raises confidence without raising accuracy (illusion of knowing).
- **H2**: partial_accessibility raises confidence without raising accuracy (distinct mechanistic route).
- **H3**: cue_familiarity and target_priming produce **comparable confidence lifts via dissociated routes** — one heuristic, one substantive.

---

## 2. Stage 0 — Data preparation (`scripts/00_prepare_data.py`)

Implemented in `src/koriat_cues/data/{loader,filter}.py`.

**Source datasets.** TriviaQA (default; `rc.nocontext` config), Natural Questions, or SciQ via `load_items()` (`data/loader.py`). The MVE and pilot use TriviaQA validation split. For each item we keep the canonical answer plus all aliases, deduplicated by normalized lowercase comparison.

**Single-entity filter.** `filter_single_entity()` (`data/filter.py:13`) drops any item whose first answer string contains `,`, `;`, `and`, or `&`, or runs longer than 6 tokens. This excludes list-valued answers ("Roosevelt and Churchill") that confound exact-match grading. The filter is coarse (it incorrectly drops "Washington, D.C.") but cheap.

**Baseline-accuracy estimation.** For each item we generate `baseline_acc_n_samples` answers (default 5) under the `baseline` prompt assembly. Sample 0 is greedy; samples 1–4 are stochastic at `temperature=0.7`, `max_new_tokens=24`. Each prediction is graded against all aliases via normalized exact match. The item's `baseline_accuracy` is the hit fraction (`scripts/00_prepare_data.py:34-53`).

**Accuracy-band filter.** `filter_by_baseline_accuracy()` keeps items with `0.2 ≤ baseline_accuracy ≤ 0.8` (configurable). Items above the upper bound have no headroom for confidence inflation; items below the lower bound contribute too much noise. The loader oversamples by 4× (`min(n*4, len(ds))`) to absorb filter losses; in practice the pilot's pass rate through this band was ~12% on TriviaQA, which the audit flagged as too tight a margin for N=500.

**Outputs.** `runs/<run>/items_raw.jsonl` (post single-entity), `runs/<run>/items_filtered.jsonl` (post accuracy band, truncated to `n_items`).

---

## 3. Stage 1 — CAA steering vector (`scripts/01_build_caa_vector.py`)

Implemented in `src/koriat_cues/caa/{contrast_pairs,vector}.py`. The construction follows Rimsky et al.'s contrastive activation addition.

**Contrast pairs.** Each pair is `(prompt, confident_completion, hedged_completion)` over the same QA. The pool is built from:

- 5 hand-authored canonical pairs (`DEFAULT_PAIRS`, `contrast_pairs.py:24-50`) covering authors, capitals, painters, etc.
- `n_contrast_pairs - 5` template-expanded pairs (default 195) drawn from a held-out QA pool. The held-out pool is loaded with `seed = cfg.data.seed + 7919` to reduce overlap with the eval set (the audit notes that explicit item-id deduplication would be safer).

For each held-out QA the builder samples one `confident_template` and one `hedged_template`:

- Confident (3 variants): `" {ans}.\n"`, `" The answer is {ans}.\n"`, `" It's {ans}.\n"`.
- Hedged (4 variants): `" I'm not sure, maybe {ans}?\n"`, `" Possibly {ans}, but I could be mistaken.\n"`, `" I think it might be {ans}, though I'm not certain.\n"`, `" I could be wrong, but perhaps {ans}.\n"`.

**Hidden-state extraction.** For each pair, `build_caa_vector()` (`caa/vector.py:40`) runs two forward passes — `prompt + confident_completion` and `prompt + hedged_completion` — and reads the residual stream at the **post-answer-newline token** at a configurable layer (`cfg.models[*].caa_layer`; defaults: `-12` for Llama-8B, `-20` for Llama-70B, `-8` / `-16` for Gemma-3 4B/27B). Negative indices are resolved against `num_hidden_layers + 1` (the `+1` accounts for the embedding layer at index 0). The token position is found by `_post_newline_position()`, which decodes step-by-step and returns the index of the final `\n` in the completion.

**Vector computation.** The steering vector is the **mean of confident-minus-hedged differences** across all pairs: `v = mean_i (h_confident,i − h_hedged,i)`. The output `.npz` stores `{vector, layer, n_pairs, norm}`.

**Output.** `runs/<run>/caa_<model_name>.npz`.

---

## 4. Stage 2 — Prime generation (`scripts/02_generate_primes.py`)

Implemented in `src/koriat_cues/primes/{conditions,generator}.py`.

**Per-condition generator instructions** (`conditions.py:24-99`) are sent to Claude (`generator_model`, default `claude-opus-4-7`; pilot used `claude-sonnet-4-6` for cost). Each prime is generated independently per (item, condition) so leak risk and length distributions can be diagnosed per condition. The instructions are quoted verbatim in `conditions.py`; the salient excerpts are:

- **cue_familiarity_priming**: *"Write a 2-3 sentence passage that mentions the key entities / terms in the question but contains NO information that helps answer it. The passage must not name or describe the target answer, directly or by paraphrase."*
- **target_priming**: *"Write a 2-3 sentence passage that mentions the correct answer `{answer}` in an unrelated context, without flagging that it answers any question."*
- **partial_accessibility**: *"Write a 2-3 sentence passage that presents plausible-sounding information related to the question's topic but that does NOT support the correct answer. It may cite incorrect or tangential facts that sound confident and authoritative."*
- **illusory_tot**: *"Write a 2-3 sentence passage on a topic semantically adjacent to the question (same era, same field, same category of entity)… Keep it dense with confident-sounding proper nouns."*
- **fluency_degradation**: *"Rephrase the following question using uncommon synonyms and awkward (but grammatical) syntax. Preserve the exact semantic content."* This condition has `rephrases_question = True`; the prime *replaces* the question rather than prepending to it.
- **random_paragraph**: yoked control, no LLM call. Filled by `yoke_random_paragraphs()` (`generator.py:158`), which length-matches each item's longest non-null prime by splicing from a 6-paragraph neutral pool (`_RANDOM_POOL`, lines 27-46) and truncating on word boundaries.

**Generator framing.** A fixed system prompt (`generator.py:113-118`) tells Claude this is a metacognition experiment, asks for concise factual-sounding output, and forbids revealing the answer unless explicitly instructed. The user message interpolates `{instruction}`, the question, the gold answer (with the explicit warning not to reveal it unless the instruction permits), and a length cap (`primes.max_sentences`, default 3). Calls retry up to 4 times with exponential backoff. Failures return `None` — the audit flagged that the runner currently treats `None` primes as silent baseline fallbacks, biasing effects toward zero.

**Post-processing** strips boilerplate preambles ("Passage:", "Here's…") and surrounding quotes.

**Output.** `runs/<run>/primes.jsonl`, one PrimeSet per item, each PrimeSet holding all condition→prime mappings.

---

## 5. Stage 3 — Prime leak validation (`scripts/03_validate_primes.py`)

Implemented in `src/koriat_cues/primes/validator.py`.

**The leak threat.** A `cue_familiarity_priming` prime is only a clean test of the heuristic if it actually contains no answer-relevant information. If Claude inadvertently leaks the answer, any confidence lift could be substantive rather than heuristic.

**Judge mechanism.** `PrimeJudge._guess()` (`validator.py:33`) sends the prime + question to Claude (`judge_model`, default `claude-opus-4-7`) with the system instruction: *"Using ONLY the passage (not your own knowledge), decide whether the passage contains or strongly implies the answer to the question. If it does, output the answer on a single line. If it does not, output exactly: UNKNOWN."* The judge is called `n_attempts` times (default 3) per (item, prime); each guess is graded against the gold answers via normalized exact match. The leak rate is `hits / n_attempts`.

**Filtering.** Items with leak rate > `primes.max_leak_rate` (default 0.05) are dropped. `items_filtered.jsonl` and `primes.jsonl` are rewritten in place. Currently only `cue_familiarity_priming` is judged; the audit notes `partial_accessibility` and `illusory_tot` should be judged too, as their generator instructions also create leak risk.

**Output.** Filtered checkpoints + `runs/<run>/prime_leaks.json` with the per-item leak rates.

---

## 6. Stage 4 — Experiment run (`scripts/04_run_experiment.py`)

Implemented in `src/koriat_cues/experiment/runner.py`, with the prompt assembly in `primes/conditions.py:assemble_prompt` and the trial loop in `confidence/measures.py:run_trial`.

### 6.1 Model wrapper

`src/koriat_cues/models/hf_model.py` wraps a HuggingFace causal LM. It supports bf16/fp16/fp32 plus int8 / int4 quantization via `BitsAndBytesConfig`, applies the model's chat template via `tokenizer.apply_chat_template(..., add_generation_prompt=True)`, and exposes a single `generate()` method that returns:

- the decoded text,
- the **first-token log-probability** (computed as `log_softmax(scores[0])[token_id]`),
- optionally, the **post-answer-newline hidden state** at a specified layer.

The post-newline token is located by `_find_post_answer_newline_step()` (`hf_model.py:146`): step through generated tokens, mark when a non-whitespace token has been seen, then return the first subsequent token containing `\n`. This locates the residual stream **right after the model has committed to an answer** — the position Kumaran et al. (2026) identify as carrying second-order confidence.

### 6.2 Prompt assembly per condition

`assemble_prompt()` (`conditions.py:102`) returns a chat-format message list with a fixed system prompt — *"You are a careful trivia assistant. When asked a question, respond with the answer on a single line, followed by a newline. Be concise: name the entity only, no explanation."* — and a user turn that depends on the condition:

- **baseline**: `"Question: {q}\nAnswer:"`
- **fluency_degradation**: `"Question: {prime_or_q}\nAnswer:"` (the prime IS the rephrased question)
- **all other conditions**, default order: `"Context: {prime}\n\nQuestion: {q}\nAnswer:"`
- **all other conditions**, counterbalance order: `"Question: {q}\n\nContext: {prime}\nAnswer:"` (controlled by `primes.counterbalance_order`; full.yaml enables this, MVE and pilot disable it).

### 6.3 The four confidence measures

`run_trial()` (`measures.py:95`) collects all four measures from a single answer generation plus, for the verbal measures, follow-up turns appended to the chat:

1. **`log_prob`** — first-token log-probability of the model's answer, captured during the answer generation. No additional forward pass.
2. **`caa_proj`** — the post-answer-newline hidden state at the CAA layer, projected onto the steering vector via `project()` (dot-product, normalized). Requires `cfg.confidence.caa_layer == cfg.models[*].caa_layer`.
3. **`verbal_cat`** — a follow-up user turn (`_verbalize`, categorical=True; `measures.py:186-193`) asks the model to *"Rate your confidence that your answer is correct on this five-point scale: very low, low, medium, high, very high."* with five short rubric examples. Generated greedily, max 24 tokens. Parsed by `_parse_categorical()`: regex word-boundary search for the five scale labels (longest match preferred); on failure, falls back to a hedge-word lookup (`_HEDGE_FALLBACK`: `certain`/`definitely`→1.0, `confident`→0.75, `probably`/`likely`→0.5, `unsure`→0.25, `no idea`→0.0, etc.). The scale is mapped to `{very low: 0.0, low: 0.25, medium: 0.5, high: 0.75, very high: 1.0}`. The raw response is stored in `verbal_cat_raw` (the audit noted that earlier code threw away the raw text on parse failures, blocking diagnosis — this should be fixed before scaled runs).
4. **`verbal_num`** — follow-up turn (`measures.py:195-200`) asks for *"an integer from 0 to 100, where 0 means no confidence and 100 means complete certainty"* with five rubric examples. Greedy, max 16 tokens. Parsed by `_parse_numeric()`: extract first 1–3 digit (optionally decimal) match, clamp to `[0, 100]`, divide by 100.

After the run, `standardize_per_model()` (`measures.py:216`) min-max scales each measure column **within each model** so the four measures are on a comparable [0, 1] scale before regression.

### 6.4 Grid + checkpointing

The runner iterates `models × items × conditions`, skipping any (model, item, condition) triple already present in the parquet checkpoint (loaded at startup). Each row is the full `TrialOutputs` dataclass (`measures.py:48`): identifiers, inputs, prediction, correctness flag, all four measures, raw verbal responses, and an `extra` dict (JSON-serialized for parquet compatibility — a fix introduced after the pilot crashed on empty pyarrow structs). The checkpoint is flushed every 25 rows. After all items for a model are processed, the model is `del`'d to free GPU memory before loading the next model.

### 6.5 Answer grading

`src/koriat_cues/eval/grader.py` implements normalized exact match: NFKD-strip accents, lowercase, strip punctuation, remove `{a, an, the}`, collapse whitespace. A multi-token gold answer (`len(gold.split()) >= 2`) also matches if it appears as a substring of the prediction (handles "Sir Isaac Newton" containing "Isaac Newton"); single-token gold requires exact equality (avoids "snowball" matching "no"). An optional LLM-judge fallback (`Grader.grade`) is wired but not currently invoked from the runner.

**Output.** `runs/<run>/trials.parquet`.

---

## 7. Stage 5 — Analysis (`scripts/05_analyze.py`)

Implemented in `src/koriat_cues/analysis/{shifts,regression,dissociation}.py`.

**Pairing.** `compute_shifts()` (`shifts.py:10`) pivots on `(model_name, item_id)` to attach each non-baseline trial to its baseline counterpart. For every (item, non-baseline-condition) pair it computes:

- `accuracy_shift ∈ {-1, 0, +1}`: `+1` if the condition rescued a baseline-wrong item, `-1` if it broke a baseline-right item, `0` otherwise.
- `confidence_shift_<m> = condition_value[m] - baseline_value[m]` for each measure.

**Per-condition summary.** `per_condition_summary()` aggregates shifts by `(model, condition)` reporting mean / std / count for each measure.

**Regression.** `run_regression()` fits OLS per (model, measure) with the formula `confidence_shift_<m> ~ accuracy_shift * C(condition)`. The intercept and condition contrasts test whether each condition produces a confidence shift above and beyond what its accuracy shift would predict — i.e., whether confidence is dissociated from correctness in the direction Koriat predicts. The audit flagged that this OLS treats every trial as independent even though each item contributes one row per non-baseline condition, inflating Type I error; switching to MixedLM with `groups=item_id` (or per-condition OLS with item-clustered SEs) is the standing recommendation before the full MVE.

**Dissociation index.** `dissociation_index()` (`dissociation.py:21`) computes `DI_<m> = mean(confidence_shift_<m>) / mean(accuracy_shift)` per condition. Returns NaN when `|mean(accuracy_shift)| < 1e-3`, which is exactly the Koriat-positive case (cue_familiarity is *predicted* to give zero accuracy shift). The audit recommends additionally reporting the raw difference `mean_conf − mean_acc` and per-item `ρ(Δconf, Δacc)`, which remain interpretable when the denominator is near zero. `compare_measures()` produces a per-(model, measure, condition) table for direct cross-measure comparison.

**Outputs.** `runs/<run>/summary_per_condition.csv`, `regression.txt`, `dissociation_index.csv`, `compare_measures.csv`.

---

## 8. Configurations

Three pydantic-validated configs share schema and differ only in scope.

| field                       | pilot                              | mve                                | full                                                                 |
|-----------------------------|------------------------------------|------------------------------------|----------------------------------------------------------------------|
| models                      | Llama-3.1-8B (`caa_layer=-12`)     | Llama-3.1-8B                       | Llama-3.1-8B & 70B (`-12`/`-20`); Gemma-3 4B & 27B (`-8`/`-16`)      |
| `data.n_items`              | 50                                 | 500                                | 2000                                                                 |
| baseline_acc band           | [0.2, 0.8]                         | [0.2, 0.8]                         | [0.2, 0.8]                                                           |
| `baseline_acc_n_samples`    | 5                                  | 5                                  | 5                                                                    |
| primes generator/judge      | `claude-sonnet-4-6` (cheaper)      | `claude-opus-4-7`                  | `claude-opus-4-7`                                                    |
| `max_leak_rate`             | 0.05                               | 0.05                               | 0.05                                                                 |
| `counterbalance_order`      | false                              | false                              | true                                                                 |
| conditions                  | baseline, cue_familiarity, target  | baseline, cue_familiarity, target  | all 6 + random_paragraph                                             |
| confidence_measures         | verbal_cat, caa_proj               | verbal_cat, caa_proj               | log_prob, verbal_cat, verbal_num, caa_proj                           |
| `caa.n_contrast_pairs`      | 200                                | 200                                | 200                                                                  |

The MVE is sized to cleanly test H1 (cue-familiarity illusion) and H3 (dissociation of cue from target) at one model and scale; the full design adds H2 (partial accessibility), the fluency and TOT manipulations, the yoked random-paragraph control, and the cross-scale / cross-family replication that any external venue would expect.

---

## 9. Verification without GPU/API

`scripts/smoke_dry_run.py` wires the entire non-ML path on five hardcoded synthetic items with embedded Koriat-pattern noise (baseline acc ≈ 0.5, conf ≈ 0.45; cue_familiarity acc ≈ 0.5, conf ≈ 0.70; target acc ≈ 0.85, conf ≈ 0.55). It asserts that `compute_shifts → per_condition_summary → dissociation_index` correctly recovers the prediction `confidence_shift[cue_familiarity] > confidence_shift[target_priming]`. Twenty-four pytest unit tests cover the grader, condition specs, prompt assembly, confidence parsers (categorical scale + hedge fallback + numeric clamp), prime serialization, length-yoking, shift computation, dissociation NaN handling, and per-model standardization bounds. No test currently exercises a real-model forward pass, so issues like the parquet `extra`-struct serialization bug only surfaced during the pilot run.

---

## 10. Pilot run as a worked example (`runs/pilot_llama8b/`)

The pilot exercised the pipeline end-to-end on an A40 (~14 min GPU, <$0.50 total) with the `pilot.yaml` config: 50 items requested, **23 surviving** the 0.2–0.8 band on TriviaQA, 3 conditions (baseline / cue_familiarity / target), 2 measures (verbal_cat / caa_proj). Results were directionally consistent with H1 and H3 — cue_familiarity raised verbal_cat by +0.062 with **zero** accuracy shift, while target_priming raised both accuracy (+0.043) and verbal_cat (+0.074) — and produced an unanticipated mechanistic finding: the CAA projection moved in the **opposite** direction of verbalized confidence under priming (−0.112 cue_familiarity, −0.042 target). This is the dissociation that SPEC.md flagged as scientifically interesting in its own right and that the full MVE — with the yoked random_paragraph control — is designed to either confirm or attribute to a generic "any context dampens post-newline hidden state" effect.

---

## 11. Known gaps surfaced by the audit

In priority order, with file pointers:

1. `_parse_categorical` discards the raw model text on parse failure — should always retain `verbal_cat_raw` (`confidence/measures.py`). *(flagged P0)*
2. OLS in `run_regression` ignores within-item dependence — switch to MixedLM grouped by `item_id` or cluster SEs (`analysis/regression.py`).
3. Dissociation index is undefined precisely when the Koriat prediction holds — also report raw `Δconf − Δacc` and per-item correlation (`analysis/dissociation.py`).
4. Loader oversample factor (`n*4`) undershoots for N=500 at the observed 12% pass rate — make oversample configurable and/or widen the band to [0.15, 0.85] (`data/loader.py:51`).
5. `judge_prime_leakage` only checks `cue_familiarity_priming` — extend to `partial_accessibility` and `illusory_tot` (`primes/validator.py`).
6. `counterbalance_order` config flag is a silent no-op in the runner — wire it through `assemble_prompt` (`experiment/runner.py:76`).
7. Missing CAA vector currently fails silently at trial time — should raise at runner startup if `caa_proj` is requested but no `.npz` is found.
8. Failed prime generations fall through to baseline silently — should record explicit failures and exclude from analysis (`primes/generator.py:_generate_one` → runner).
9. CAA contrast-pair pool is drawn from the same TriviaQA split as the eval set — explicit item-id deduplication or use the train split.
10. `estimate_baseline_accuracy` is unbatched — batch to 16–32 to roughly halve stage-0 GPU time on the full MVE.

A scoped patch covering items 1–4 (P0) and 5, 7, 8 (P1 correctness) is the recommended pre-flight for the 500-item run.
