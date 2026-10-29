"""Phase 2 driver: train S/T/X/DR-Learner meta-learners + Causal Forest on
the stratified modeling sample, predict CATE on a held-out test split, and
evaluate with Qini / AUUC / gain curves / calibration by decile.

Persists test-set predictions (y, t, and every model's CATE score) to
data/processed/test_predictions.parquet so the policy-simulation phase
(script 03) does not need to retrain anything.
"""
from __future__ import annotations

import time

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation.uplift_metrics import (
    calibration_by_decile,
    evaluate_all_models,
    plot_calibration,
    plot_gain_curves,
    plot_qini_curves,
)
from src.uplift.meta_learners.learners import FITTERS
from src.uplift.tree_models.causal_forest import fit_causal_forest
from src.utils.config import (
    CONVERSION_COL,
    DATA_PROCESSED_DIR,
    FEATURE_COLS,
    FIGURES_DIR,
    RANDOM_SEED,
    SAMPLE_PARQUET,
    TABLES_DIR,
    TRAIN_TEST_SPLIT_RATIO,
    TREATMENT_COL,
)
from src.visualization.style import set_style

set_style()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("Loading modeling sample...")
    df = pd.read_parquet(SAMPLE_PARQUET)
    log(f"Sample shape: {df.shape}")

    train_df, test_df = train_test_split(
        df, test_size=1 - TRAIN_TEST_SPLIT_RATIO, stratify=df[TREATMENT_COL], random_state=RANDOM_SEED
    )
    log(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    log(f"Train conversions: {int(train_df[CONVERSION_COL].sum())} | Test conversions: {int(test_df[CONVERSION_COL].sum())}")

    X_train = train_df[FEATURE_COLS].to_numpy()
    T_train = train_df[TREATMENT_COL].to_numpy()
    Y_train = train_df[CONVERSION_COL].to_numpy()
    X_test = test_df[FEATURE_COLS].to_numpy()

    scores = {}
    fit_times = {}

    for name, fitter in FITTERS.items():
        log(f"Fitting {name}...")
        t0 = time.time()
        model = fitter(X_train, T_train, Y_train)
        fit_times[name] = time.time() - t0
        log(f"{name} fit in {fit_times[name]:.1f}s. Predicting CATE on test set...")
        scores[name] = model.effect(X_test).flatten()
        log(f"{name} done. mean CATE={scores[name].mean():.5f}")

    log("Fitting Causal Forest...")
    t0 = time.time()
    # n_estimators=150 (not the 300 used elsewhere) -- benchmarked at 500k rows
    # (~105s for 100 trees, ~linear scaling with both rows and trees); 150 trees
    # on the ~2.1M-row training split keeps this to single-digit minutes while
    # still being a reasonably powerful forest.
    cf_model = fit_causal_forest(X_train, T_train, Y_train, n_estimators=150)
    fit_times["Causal Forest"] = time.time() - t0
    log(f"Causal Forest fit in {fit_times['Causal Forest']:.1f}s. Predicting CATE on test set...")
    scores["Causal Forest"] = cf_model.effect(X_test).flatten()
    log(f"Causal Forest done. mean CATE={scores['Causal Forest'].mean():.5f}")

    pd.Series(fit_times).to_csv(TABLES_DIR / "uplift_model_fit_times.csv", header=["fit_time_seconds"])

    # -------------------------------------------------------------- persist
    out = test_df[[TREATMENT_COL, CONVERSION_COL] + FEATURE_COLS].copy().reset_index(drop=True)
    for name, s in scores.items():
        out[f"score__{name}"] = s
    out.to_parquet(DATA_PROCESSED_DIR / "test_predictions.parquet", index=False)
    log("Saved test-set predictions.")

    # ------------------------------------------------------------- evaluate
    y_test = test_df[CONVERSION_COL].to_numpy()
    t_test = test_df[TREATMENT_COL].to_numpy()

    eval_table = evaluate_all_models(y_test, t_test, scores)
    eval_table.to_csv(TABLES_DIR / "uplift_model_comparison.csv", index=False)
    log("\n" + eval_table.to_string(index=False))

    plot_qini_curves(y_test, t_test, scores, save_path=FIGURES_DIR / "qini_curves_all_models.png")
    plot_gain_curves(y_test, t_test, scores, save_path=FIGURES_DIR / "gain_curves_all_models.png")

    for name, s in scores.items():
        calib = calibration_by_decile(y_test, t_test, s)
        calib.to_csv(TABLES_DIR / f"calibration_decile_{name.replace(' ', '_')}.csv", index=False)
        plot_calibration(calib, name, save_path=FIGURES_DIR / f"calibration_{name.replace(' ', '_')}.png")

    log("Phase 2 complete.")


if __name__ == "__main__":
    main()
