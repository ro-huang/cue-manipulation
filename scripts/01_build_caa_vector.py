"""Stage 1 (rigorous): build a held-out-validated CAA confidence steering vector.

The contrast pairs come from the model's own behavior on a held-out trivia pool.
Two partition methods are supported:

  - "correctness" (default, Azaria & Mitchell 2023): "knows" = items the model
     answered correctly on the un-primed baseline; "doesn't know" = items it got
     wrong. The label is external (gold answers), so the resulting direction is
     not tautologically tied to the model's own log-prob.
  - "logprob" (legacy): "confident" = top quartile of first-token log-prob,
     "unsure" = bottom quartile. Provided for backwards comparison; this version
     is circular if `log_prob` is also being analyzed as a confidence channel.

We then scan every layer at the post-answer-newline position, build a candidate
mean-difference vector per layer, and validate each on a separate held-out fold:
  - Pearson r with first-token log-prob (sanity)
  - AUC for predicting accuracy

The layer with highest validated AUC is selected. The vector is saved with the
full per-layer validation table so the choice is transparent. The chosen
partition method is recorded in the npz metadata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from koriat_cues.config import load_config
from koriat_cues.data import load_items
from koriat_cues.eval.grader import grade_prediction
from koriat_cues.primes.conditions import assemble_prompt


N_HELDOUT = 800        # total items for activation collection
TRAIN_FRAC = 0.6       # 60% to build vectors, 40% to validate
TOP_QUANTILE = 0.25    # confident = top 25% log_prob
BOTTOM_QUANTILE = 0.25 # unsure = bottom 25% log_prob


def _grade(pred: str, gold: list[str]) -> bool:
    return grade_prediction(pred.split("\n")[0], gold)


def collect_activations(model, items, max_new_tokens: int):
    """Run model on each item and capture all-layer post-newline hidden states."""
    n_layers = model.model.config.num_hidden_layers + 1
    hidden_dim = model.model.config.hidden_size

    H = np.zeros((len(items), n_layers, hidden_dim), dtype=np.float32)
    log_probs = np.zeros(len(items), dtype=np.float32)
    correct = np.zeros(len(items), dtype=bool)
    valid = np.zeros(len(items), dtype=bool)

    for i, it in enumerate(tqdm(items, desc="caa_collect")):
        messages = assemble_prompt(it.question, prime=None, condition="baseline")
        prompt = model.format_chat(messages)
        gen = model.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            capture_post_newline=False,
            capture_all_layers=True,
        )
        if gen.post_newline_hidden_all_layers is None:
            continue
        H[i] = gen.post_newline_hidden_all_layers.numpy()
        log_probs[i] = gen.first_token_logprob
        correct[i] = _grade(gen.text, it.answers)
        valid[i] = True
    return H[valid], log_probs[valid], correct[valid], np.array([it.id for it in items])[valid]


def evaluate_layers(
    H_train, lp_train, correct_train, H_val, lp_val, correct_val,
    partition: str, top_q: float, bot_q: float,
):
    """For each layer build a mean-difference vector and validate on (val) split.

    `partition` selects how the train items are split into the two contrast classes:
      - "correctness": class membership = ground-truth correctness (Azaria & Mitchell).
      - "logprob": top vs bottom quartile of first-token log-prob (legacy).
    """
    from sklearn.metrics import roc_auc_score
    from scipy.stats import pearsonr

    if partition == "correctness":
        confident = correct_train.astype(bool)
        unsure = ~confident
    elif partition == "logprob":
        cutoff_top = np.quantile(lp_train, 1 - top_q)
        cutoff_bot = np.quantile(lp_train, bot_q)
        confident = lp_train >= cutoff_top
        unsure = lp_train <= cutoff_bot
    else:
        raise ValueError(f"unknown partition: {partition!r}")
    if confident.sum() < 5 or unsure.sum() < 5:
        raise RuntimeError(
            f"partition {partition!r} produced too few items per class: "
            f"confident={int(confident.sum())} unsure={int(unsure.sum())}"
        )
    n_layers = H_train.shape[1]

    rows: list[dict] = []
    vectors: dict[int, np.ndarray] = {}
    for L in range(n_layers):
        v = H_train[confident, L].mean(0) - H_train[unsure, L].mean(0)
        vectors[L] = v
        norm_v = float(np.linalg.norm(v))
        proj_val = (H_val[:, L] @ v) / (norm_v + 1e-9)
        if np.std(proj_val) < 1e-9 or len(np.unique(correct_val)) < 2:
            continue
        r_lp, _ = pearsonr(proj_val, lp_val)
        try:
            auc = float(roc_auc_score(correct_val, proj_val))
        except Exception:
            auc = float("nan")
        rows.append({
            "layer": L,
            "vector_norm": norm_v,
            "n_confident": int(confident.sum()),
            "n_unsure": int(unsure.sum()),
            "r_logprob": float(r_lp),
            "auc_correct": auc,
        })
    return pd.DataFrame(rows), vectors


def _shuffle_split(N: int, train_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    cut = int(round(train_frac * N))
    return idx[:cut], idx[cut:]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--partition",
        choices=["correctness", "logprob"],
        default="correctness",
        help=(
            "How to label train items as 'confident' vs 'unsure' for the contrast. "
            "'correctness' (default) uses ground-truth correctness on the un-primed "
            "baseline (Azaria & Mitchell 2023). 'logprob' uses top vs bottom quartile "
            "of first-token log-prob (legacy, circular against the log_prob measure)."
        ),
    )
    args = ap.parse_args(argv)

    cfg = load_config(args.config)

    if args.dry_run:
        from koriat_cues.caa import save_vector
        for mcfg in cfg.models:
            save_vector(
                cfg.run_dir / f"caa_{mcfg.name}.npz",
                {"vector": np.zeros(32, dtype=np.float32), "layer": 0, "n_pairs": 0, "norm": 0.0},
            )
        return

    eval_ids: set[str] = set()
    p = cfg.run_dir / "items_filtered.jsonl"
    if p.exists():
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    eval_ids.add(json.loads(line).get("id", ""))

    caa_split = "train" if cfg.data.dataset == "triviaqa" else cfg.data.split
    items = load_items(
        cfg.data.dataset, caa_split, N_HELDOUT * 2,
        seed=cfg.data.seed + 7919, oversample=2,
    )
    items = [it for it in items if it.id not in eval_ids][:N_HELDOUT]
    print(f"[01-rigorous] using {len(items)} held-out items "
          f"(eval-set ids excluded: {len(eval_ids)})")

    from koriat_cues.caa import save_vector
    from koriat_cues.models import HFModel

    for mcfg in cfg.models:
        out_path = cfg.run_dir / f"caa_{mcfg.name}.npz"
        v1_path = cfg.run_dir / f"caa_v1_{mcfg.name}.npz"
        if out_path.exists() and not v1_path.exists():
            # Preserve any earlier (template-based) vector for comparison.
            np.savez(v1_path, **dict(np.load(out_path, allow_pickle=True)))
            print(f"[01-rigorous] backed up legacy vector to {v1_path}")
        # Also preserve any prior real-trace vector under its method tag, so a
        # rerun with a different --partition doesn't silently clobber it.
        if out_path.exists():
            existing = dict(np.load(out_path, allow_pickle=True))
            prior_method = existing.get("method", "")
            if hasattr(prior_method, "item"):
                prior_method = prior_method.item()
            if isinstance(prior_method, bytes):
                prior_method = prior_method.decode()
            tag = "logprob" if "logprob" in str(prior_method).lower() else (
                "correctness" if "correct" in str(prior_method).lower() else "prev"
            )
            tagged_path = cfg.run_dir / f"caa_{mcfg.name}_{tag}.npz"
            if not tagged_path.exists():
                np.savez(tagged_path, **existing)
                print(f"[01-rigorous] preserved existing vector → {tagged_path}")

        print(f"[01-rigorous] loading model {mcfg.name}")
        model = HFModel(mcfg)
        print(f"[01-rigorous] collecting activations on {len(items)} items at all "
              f"{model.model.config.num_hidden_layers + 1} layers")
        H, lp, correct, item_ids = collect_activations(model, items, mcfg.max_new_tokens)
        print(f"[01-rigorous] collected {len(H)} valid activations  "
              f"(accuracy on this set: {correct.mean():.3f})")

        # Train / validation split
        idx_t, idx_v = _shuffle_split(len(H), TRAIN_FRAC, seed=cfg.seed)
        H_t, H_v = H[idx_t], H[idx_v]
        lp_t, lp_v = lp[idx_t], lp[idx_v]
        c_v = correct[idx_v]

        c_t = correct[idx_t]
        results, vectors = evaluate_layers(
            H_t, lp_t, c_t, H_v, lp_v, c_v,
            partition=args.partition, top_q=TOP_QUANTILE, bot_q=BOTTOM_QUANTILE,
        )
        results = results.sort_values("auc_correct", ascending=False).reset_index(drop=True)
        report_path = cfg.run_dir / f"caa_{mcfg.name}_layer_scan.csv"
        results.to_csv(report_path, index=False)
        print(f"[01-rigorous] wrote per-layer validation table → {report_path}")

        # Pick best layer by AUC, with sanity floor.
        best = results.iloc[0]
        print(f"[01-rigorous] best layer = {int(best.layer)}  AUC={best.auc_correct:.3f}  "
              f"r_logprob={best.r_logprob:+.3f}")
        # Bottom-3 layers for sanity
        worst = results.iloc[-3:]
        print(f"[01-rigorous] bottom layers for comparison:")
        print(worst[["layer", "auc_correct", "r_logprob"]].round(3).to_string(index=False))

        if best.auc_correct < 0.55:
            print("[01-rigorous] WARNING: best layer's AUC < 0.55. "
                  "The CAA vector may not meaningfully predict correctness.")

        chosen_v = vectors[int(best.layer)]
        method_str = (
            "real-trace mean-diff (correct vs incorrect on baseline; Azaria & Mitchell 2023)"
            if args.partition == "correctness"
            else "real-trace mean-diff (top vs bottom log-prob quartile)"
        )
        payload = {
            "vector": chosen_v.astype(np.float32),
            "layer": int(best.layer),
            "n_pairs": int(best.n_confident + best.n_unsure),
            "norm": float(np.linalg.norm(chosen_v)),
            "auc": float(best.auc_correct),
            "r_logprob": float(best.r_logprob),
            "method": method_str,
            "partition": args.partition,
        }
        save_vector(out_path, payload)
        # Also save under a method-tagged path so a follow-up rebuild with a
        # different --partition leaves both vectors on disk.
        save_vector(cfg.run_dir / f"caa_{mcfg.name}_{args.partition}.npz", payload)
        print(f"[01-rigorous] wrote {out_path} (partition={args.partition})")
        del model


if __name__ == "__main__":
    main()
