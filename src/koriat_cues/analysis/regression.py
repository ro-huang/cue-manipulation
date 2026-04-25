"""Regression of confidence shift on accuracy shift × condition.

Rows in `shifts` are (model × item × non-baseline condition). Every item appears in
multiple rows, so residuals are correlated within item and plain-OLS standard errors
are too small — Type I error inflates.

Two corrections are available:
  - `run_regression` (default): OLS with **cluster-robust** standard errors clustered
    on `item_id`. Handles repeated items without convergence risk; appropriate for
    the small-to-moderate N expected in MVE / full-design runs.
  - `run_mixed_regression`: `statsmodels.MixedLM` with a random item intercept.
    Conceptually the right model but can fail to converge with few items or singular
    random effects. Use when N is large and you want the random-effect variance
    estimate itself as a number.

Per SPEC §Core analysis: the Koriat prediction is that cue-familiarity, partial-
accessibility, and illusory-TOT conditions produce positive confidence shifts with
non-significant or smaller accuracy shifts.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd
import statsmodels.formula.api as smf


def run_regression(
    shifts: pd.DataFrame,
    measure_cols: Iterable[str],
    by_model: bool = True,
    cluster_var: str = "item_id",
) -> dict[str, dict[str, "smf.OLS"]]:
    """OLS with cluster-robust SEs on `cluster_var` (default: `item_id`).

    Equivalent to treating items as fixed clusters — SEs are inflated to account for
    within-item correlation across conditions, which plain OLS ignores.
    """
    results: dict[str, dict] = {}
    models = shifts["model_name"].unique() if by_model else ["__pooled__"]
    for mdl in models:
        sub = shifts if mdl == "__pooled__" else shifts[shifts["model_name"] == mdl]
        results[mdl] = {}
        for m in measure_cols:
            col = f"confidence_shift_{m}"
            if col not in sub.columns:
                continue
            work = sub[[col, "accuracy_shift", "condition", cluster_var]].dropna()
            if len(work) < 10 or work["condition"].nunique() < 2:
                continue
            model_fit = smf.ols(f"{col} ~ accuracy_shift * C(condition)", data=work).fit(
                cov_type="cluster",
                cov_kwds={"groups": work[cluster_var]},
            )
            results[mdl][m] = model_fit
    return results


def run_mixed_regression(
    shifts: pd.DataFrame,
    measure_cols: Iterable[str],
    by_model: bool = True,
) -> dict[str, dict[str, object]]:
    """MixedLM with a random item intercept. Returns `{model: {measure: fit_or_None}}`.

    Fits that fail to converge are stored as `None` so downstream code can fall back
    to the cluster-robust OLS fit.
    """
    results: dict[str, dict] = {}
    models = shifts["model_name"].unique() if by_model else ["__pooled__"]
    for mdl in models:
        sub = shifts if mdl == "__pooled__" else shifts[shifts["model_name"] == mdl]
        results[mdl] = {}
        for m in measure_cols:
            col = f"confidence_shift_{m}"
            if col not in sub.columns:
                continue
            work = sub[[col, "accuracy_shift", "condition", "item_id"]].dropna()
            if len(work) < 20 or work["item_id"].nunique() < 5:
                results[mdl][m] = None
                continue
            try:
                fit = smf.mixedlm(
                    f"{col} ~ accuracy_shift * C(condition)",
                    data=work,
                    groups=work["item_id"],
                ).fit(method="lbfgs", reml=True)
            except Exception:
                fit = None
            results[mdl][m] = fit
    return results


def regression_summary(fit) -> pd.DataFrame:
    """Convert a statsmodels fit (OLS cluster-robust or MixedLM) to a tidy table."""
    return pd.DataFrame(
        {
            "term": fit.params.index,
            "estimate": fit.params.values,
            "std_err": fit.bse.values,
            "t": fit.tvalues.values,
            "p": fit.pvalues.values,
        }
    )
