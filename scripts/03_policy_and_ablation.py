"""Phase 3 driver: targeting-policy simulation (treat all/none/random/
positive-uplift/top-10%/top-20%/profit-maximizing top-K), a cost
sensitivity sweep, and the estimator-family ablation study.

Uses the best-performing uplift model (by Qini coefficient, from script 02)
to rank customers for the uplift-based policies.
"""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.evaluation.uplift_metrics import (
    bootstrap_uplift_metrics,
    calibration_by_decile,
    evaluate_all_models,
    plot_calibration,
    segment_profile_by_uplift,
)
from src.policy.policies import (
    build_policies,
    cost_sensitivity_sweep,
    evaluate_policies,
    find_profit_maximizing_threshold,
    policy_profit_curve,
    top_fraction_mask,
)
from src.utils.config import (
    ASSUMED_COST_PER_CONTACT,
    ASSUMED_VALUE_PER_CONVERSION,
    CONVERSION_COL,
    DATA_PROCESSED_DIR,
    FEATURE_COLS,
    FIGURES_DIR,
    RANDOM_SEED,
    REPORT_DIR,
    TABLES_DIR,
    TREATMENT_COL,
    UPLIFT_BOOTSTRAP_REPS,
)
from src.visualization.style import set_style

set_style()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("Loading test-set predictions from Phase 2...")
    df = pd.read_parquet(DATA_PROCESSED_DIR / "test_predictions.parquet")
    score_cols = [c for c in df.columns if c.startswith("score__")]
    model_names = [c.replace("score__", "") for c in score_cols]

    eval_table = pd.read_csv(TABLES_DIR / "uplift_model_comparison.csv")
    stratification = df[TREATMENT_COL].astype(str) + "_" + df[CONVERSION_COL].astype(str)
    selection_df, evaluation_df = train_test_split(
        df,
        test_size=0.5,
        stratify=stratification,
        random_state=RANDOM_SEED,
    )
    selection_df = selection_df.reset_index(drop=True)
    evaluation_df = evaluation_df.reset_index(drop=True)
    log(
        f"Leakage-free policy split: validation={len(selection_df):,} rows | "
        f"test={len(evaluation_df):,} rows"
    )

    selection_scores = {
        name: selection_df[f"score__{name}"].to_numpy()
        for name in model_names
    }
    policy_model_selection = evaluate_all_models(
        selection_df[CONVERSION_COL].to_numpy(),
        selection_df[TREATMENT_COL].to_numpy(),
        selection_scores,
    )
    policy_model_selection.to_csv(TABLES_DIR / "policy_model_selection_validation.csv", index=False)
    best_model = policy_model_selection.iloc[0]["model"]
    log(f"Policy model selected by validation-set Qini: {best_model}")

    y_selection = selection_df[CONVERSION_COL].to_numpy()
    t_selection = selection_df[TREATMENT_COL].to_numpy()
    score_selection = selection_df[f"score__{best_model}"].to_numpy()
    y_evaluation = evaluation_df[CONVERSION_COL].to_numpy()
    t_evaluation = evaluation_df[TREATMENT_COL].to_numpy()
    score_evaluation = evaluation_df[f"score__{best_model}"].to_numpy()
    e_treatment = float(df[TREATMENT_COL].mean())
    log(f"Empirical treatment probability (used as known propensity for IPS policy evaluation): {e_treatment:.4f}")

    # --------------------------------------- uplift metric uncertainty / CIs
    log(f"Running {UPLIFT_BOOTSTRAP_REPS} paired stratified bootstrap repetitions for Qini/AUUC...")
    full_scores = {name: df[f"score__{name}"].to_numpy() for name in model_names}
    uplift_uncertainty, uplift_pairwise = bootstrap_uplift_metrics(
        df[CONVERSION_COL].to_numpy(),
        df[TREATMENT_COL].to_numpy(),
        full_scores,
        n_bootstrap=UPLIFT_BOOTSTRAP_REPS,
    )
    uplift_uncertainty.to_csv(TABLES_DIR / "uplift_model_uncertainty.csv", index=False)
    uplift_pairwise.to_csv(TABLES_DIR / "uplift_model_pairwise_bootstrap.csv", index=False)
    log("\n" + uplift_uncertainty.to_string(index=False))

    # Regenerate calibration artifacts so current CSVs include Newcombe CIs.
    for name, score in full_scores.items():
        calibration = calibration_by_decile(
            df[CONVERSION_COL].to_numpy(),
            df[TREATMENT_COL].to_numpy(),
            score,
        )
        safe_name = name.replace(" ", "_")
        calibration.to_csv(TABLES_DIR / f"calibration_decile_{safe_name}.csv", index=False)
        plot_calibration(calibration, name, save_path=FIGURES_DIR / f"calibration_{safe_name}.png")

    # ---------------------------------------------------------- core policies
    policies = build_policies(score_evaluation)
    policy_table = evaluate_policies(
        y_evaluation,
        t_evaluation,
        policies,
        e_treatment,
        ASSUMED_VALUE_PER_CONVERSION,
        ASSUMED_COST_PER_CONTACT,
    )

    profit_max = find_profit_maximizing_threshold(
        y_selection,
        t_selection,
        score_selection,
        e_treatment,
        ASSUMED_VALUE_PER_CONVERSION,
        ASSUMED_COST_PER_CONTACT,
    )
    selected_test_mask = top_fraction_mask(score_evaluation, profit_max["k_star_fraction"])
    profit_row = evaluate_policies(
        y_evaluation,
        t_evaluation,
        {"Top-K Selected on Validation": selected_test_mask},
        e_treatment,
        ASSUMED_VALUE_PER_CONVERSION,
        ASSUMED_COST_PER_CONTACT,
    )
    policy_table = pd.concat([policy_table, profit_row], ignore_index=True).sort_values("net_profit", ascending=False).reset_index(drop=True)
    policy_table.to_csv(TABLES_DIR / "policy_comparison.csv", index=False)
    log("\n" + policy_table.to_string(index=False))

    # -------------------------------------------------------- profit curve
    selection_curve = profit_max["curve"]
    evaluation_curve = policy_profit_curve(
        y_evaluation,
        t_evaluation,
        score_evaluation,
        e_treatment,
        ASSUMED_VALUE_PER_CONVERSION,
        ASSUMED_COST_PER_CONTACT,
        candidate_fractions=selection_curve["population_fraction"].to_numpy(),
    )
    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax1.plot(
        selection_curve["population_fraction"],
        selection_curve["profit"],
        color="#2563eb",
        linewidth=2,
        label="Validation profit (threshold selection)",
    )
    ax1.plot(
        evaluation_curve["population_fraction"],
        evaluation_curve["profit"],
        color="#d97706",
        linewidth=2,
        label="Held-out test profit (evaluation only)",
    )
    ax1.axvline(profit_max["k_star_fraction"], color="#dc2626", linestyle="--", linewidth=1.2,
                label=f"Validation-selected coverage={profit_max['k_star_fraction']:.1%}")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.set_xlabel("Fraction of customers targeted (ranked by predicted uplift)")
    ax1.set_ylabel(f"Net profit (value=USD {ASSUMED_VALUE_PER_CONVERSION:.0f}/conv, cost=USD {ASSUMED_COST_PER_CONTACT:.2f}/contact)")
    ax1.set_title(f"Leakage-Free Policy Selection ({best_model} uplift ranking)")
    ax1.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "profit_maximizing_curve.png")

    # ---------------------------------------------------- cost sensitivity
    log("Running cost-sensitivity sweep...")
    value_grid = [10.0, 25.0, 50.0, 100.0, 200.0]
    cost_grid = [0.1, 0.25, 0.5, 1.0, 2.0]
    sweep = cost_sensitivity_sweep(
        y_selection,
        t_selection,
        score_selection,
        y_evaluation,
        t_evaluation,
        score_evaluation,
        e_treatment,
        value_grid,
        cost_grid,
    )
    sweep.to_csv(TABLES_DIR / "cost_sensitivity_sweep.csv", index=False)

    pivot = sweep.pivot(
        index="value_per_conversion",
        columns="cost_per_contact",
        values="selected_coverage_validation",
    )
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(pivot.to_numpy(), cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"${c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"${v:.0f}" for v in pivot.index])
    ax.set_xlabel("Cost per contact")
    ax.set_ylabel("Value per conversion")
    ax.set_title("Optimal Targeting Coverage (%) -- Sensitivity")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.0%}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Optimal coverage fraction")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "cost_sensitivity_heatmap.png")
    log("Cost sensitivity sweep done.")

    # --------------------------------------------------- who benefits/harmed
    log("Profiling top vs. bottom predicted-uplift segments...")
    segment_profile = segment_profile_by_uplift(
        evaluation_df, score_col=f"score__{best_model}", feature_cols=FEATURE_COLS,
        outcome_col=CONVERSION_COL, treatment_col=TREATMENT_COL,
    )
    segment_profile.to_csv(TABLES_DIR / "segment_profile_top_vs_bottom_uplift.csv", index=False)
    log("\n" + segment_profile[["segment", "n", "mean_predicted_uplift", "actual_uplift"]].to_string(index=False))

    # ------------------------------------------------------------- ablation
    log("Building ablation study...")
    ate_conv = pd.read_csv(TABLES_DIR / "ate_estimators_conversion.csv")
    ate_confounded = pd.read_csv(TABLES_DIR / "ate_estimators_confounded_sample.csv")
    uplift_comparison = eval_table

    ablation = {
        "ate_estimator_ablation_rct_data": ate_conv[["method", "ate", "se", "ci_low", "ci_high"]].to_dict(orient="records"),
        "ate_estimator_ablation_confounded_data": ate_confounded[
            ["method", "ate", "bias_vs_target_population_rct"]
        ].to_dict(orient="records"),
        "uplift_model_ablation": uplift_comparison[["model", "qini_coefficient", "auuc_score"]].to_dict(orient="records"),
        "uplift_model_uncertainty": uplift_uncertainty.to_dict(orient="records"),
        "policy_model_selected_on_validation": best_model,
        "policy_validation_rows": len(selection_df),
        "policy_test_rows": len(evaluation_df),
        "profit_maximizing_summary": {
            "selected_coverage_validation": profit_max["k_star_fraction"],
            "selected_customers_validation": profit_max["k_star_customers"],
            "selection_profit_validation": profit_max["expected_profit"],
            "targeted_customers_test": int(selected_test_mask.sum()),
            "evaluation_profit_test": float(profit_row.iloc[0]["net_profit"]),
            "evaluation_profit_ci_low_test": float(profit_row.iloc[0]["net_profit_ci_low"]),
            "evaluation_profit_ci_high_test": float(profit_row.iloc[0]["net_profit_ci_high"]),
        },
    }
    (REPORT_DIR / "phase3_ablation.json").write_text(json.dumps(ablation, indent=2, default=float))
    log("Phase 3 complete.")


if __name__ == "__main__":
    main()
