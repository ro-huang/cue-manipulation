# Koriat-style cue manipulations of LLM confidence

Experimental pipeline that tests whether LLM confidence is driven by Koriat-style
heuristic cues (cue familiarity, partial accessibility) rather than direct access to
a correctness signal. See `SPEC.md` for the research design.

## Layout

```
configs/
  mve.yaml            # Minimum viable experiment (1 model, 500 items, 3 conds, 2 measures)
  full.yaml           # Full design (4 models, 2000 items, 6 conds, 4 measures)
src/koriat_cues/
  data/               # TriviaQA / NQ / SciQ loaders, baseline-accuracy filter
  primes/             # 6 cue-manipulation conditions + generator + judge
  models/             # HuggingFace wrapper with hidden-state capture
  confidence/         # log-prob, verbalized categorical + numeric, CAA projection
  caa/                # Contrastive-activation steering-vector computation
  experiment/         # Runner, prompt assembly, checkpointing
  analysis/           # Regression, dissociation index, per-measure comparison
  eval/               # Answer grading (normalized EM + judge fallback)
scripts/
  00_prepare_data.py      # Load + single-entity filter + baseline-accuracy filter
  01_build_caa_vector.py  # Collect contrast activations, save steering vector
  02_generate_primes.py   # Generate primes for all conditions via Claude
  03_validate_primes.py   # External-judge leak check; filter items
  04_run_experiment.py    # Run full condition × item × measure grid
  05_analyze.py           # Produce tables + the dissociation-index report
```

## Run the MVE

```bash
pip install -e .
export ANTHROPIC_API_KEY=...          # for prime generation + prime judge + answer judge
export HF_TOKEN=...                    # for gated Llama weights

python scripts/00_prepare_data.py    --config configs/mve.yaml
python scripts/01_build_caa_vector.py --config configs/mve.yaml
python scripts/02_generate_primes.py --config configs/mve.yaml
python scripts/03_validate_primes.py --config configs/mve.yaml
python scripts/04_run_experiment.py  --config configs/mve.yaml
python scripts/05_analyze.py         --config configs/mve.yaml
```

Each stage writes a parquet/jsonl checkpoint under `runs/<run-name>/`, so the later
stages can resume without re-running the earlier ones. Scripts accept `--dry-run`
to exercise the pipeline with tiny samples and no GPU/API calls.

## Conditions

Following SPEC.md §Cue manipulation conditions:

| idx | name                      | prediction                                |
|-----|---------------------------|-------------------------------------------|
| 0   | baseline                  | reference                                 |
| 1   | cue_familiarity_priming   | ↑ confidence, = accuracy (Schwartz-Metcalfe) |
| 2   | target_priming            | = confidence, ↑ accuracy                   |
| 3   | partial_accessibility     | ↑ confidence, = accuracy (Koriat 1993)     |
| 4   | illusory_tot              | ↑ confidence, ↓ accuracy (Schwartz 1998)   |
| 5   | fluency_degradation       | ↓ confidence, = accuracy                   |
| ctrl| random_paragraph          | yoked length-matched noise                 |

## Confidence measures

1. **log_prob**  – log-prob of the first answer token.
2. **verbal_cat** – verbalized categorical rating ("very low"…"very high"), Yoon 2025 style.
3. **verbal_num** – verbalized 0–100 numeric rating.
4. **caa_proj**   – projection of the post-answer-newline hidden state onto a CAA steering
                    vector computed from confident-vs-hedged contrast pairs (Rimsky-style).

Each is standardized per-model to [0, 1] before the regression so the four measures are
comparable.

## MVE scope

Following SPEC.md §Minimum viable experiment: Llama-3.1-8B-Instruct, 500 TriviaQA items
(filtered to 0.2 ≤ baseline_acc ≤ 0.8), conditions {baseline, cue_familiarity, target},
measures {verbal_cat, caa_proj}. Cleanly tests H1 and H3.

## Extending to the full design

Change `configs/mve.yaml` → `configs/full.yaml`; no code changes required. Add more model
names to `models:`, more conditions to `conditions:`, more measures to `confidence_measures:`.

## Verifying the pipeline without GPU/API

```bash
pip install pydantic pyyaml pandas numpy statsmodels tqdm tenacity pytest
PYTHONPATH=src python -m pytest tests/ -q      # 24 unit tests
PYTHONPATH=src python scripts/smoke_dry_run.py  # end-to-end synthetic run
```

The smoke test exercises data → primes → assembly → shifts → regression → dissociation-index
on a synthetic 5-item dataset with built-in Koriat-pattern noise, and asserts that the
analysis correctly flags higher confidence shift under cue-familiarity than target priming.

