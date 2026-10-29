"""Phase 1 driver: preprocessing validation, EDA, treatment-assignment
analysis (balance/SMD/propensity), causal DAG, and ATE estimation
(Difference in Means, Regression Adjustment, IPW, Doubly Robust) -- run on
the FULL 13,979,592-row Criteo dataset.

Also runs the synthetic-observational confounding-injection experiment
(Section 6.3 Scenario 2 + Section 7.4 ablation).

All figures land in report/figures/, all tables in report/tables/, and a
consolidated results.json in report/.
"""
from __future__ import annotations

import json
import time

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from src.causal.ate.confounding_simulation import (
    confounding_diagnostics,
    make_confounded_observational_sample,
    target_population_rct_reference,
)
from src.causal.ate.estimators import run_all_estimators
from src.causal.dag.dag import plot_dag, DISCUSSION as DAG_DISCUSSION
from src.causal.propensity.propensity import (
    fit_propensity_model,
    plot_propensity_diagnostics,
    plot_smd_love_plot,
    standardized_mean_difference,
    treatment_assignment_verdict,
)
from src.preprocessing.eda import (
    dataset_overview,
    feature_summary_stats,
    naive_lift_summary,
    plot_feature_correlation,
    plot_feature_distributions,
    plot_treatment_and_outcome_distribution,
)
from src.preprocessing.load_data import load_clean
from src.utils.config import FIGURES_DIR, REPORT_DIR, TABLES_DIR
from src.visualization.style import set_style

set_style()
results = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("Loading cleaned full dataset...")
    df = load_clean()
    log(f"Loaded {len(df):,} rows.")

    # ---------------------------------------------------------------- EDA
    log("Running EDA...")
    overview = dataset_overview(df)
    overview.to_csv(TABLES_DIR / "dataset_overview.csv", index=False)
    results["dataset_overview"] = dict(zip(overview["metric"], overview["value"]))

    naive_conv = naive_lift_summary(df, "conversion")
    naive_visit = naive_lift_summary(df, "visit")
    results["naive_lift_conversion"] = naive_conv
    results["naive_lift_visit"] = naive_visit

    feature_summary_stats(df).to_csv(TABLES_DIR / "feature_summary_stats.csv")

    plot_treatment_and_outcome_distribution(df, save_path=FIGURES_DIR / "treatment_outcome_distribution.png")
    plot_feature_distributions(df, save_path=FIGURES_DIR / "feature_distributions_by_arm.png")
    _, corr = plot_feature_correlation(df, save_path=FIGURES_DIR / "feature_correlation_matrix.png")
    corr.to_csv(TABLES_DIR / "feature_correlation_matrix.csv")
    log("EDA done.")

    # ---------------------------------------------------- Treatment assignment
    log("Computing covariate balance (SMD)...")
    smd_table = standardized_mean_difference(df)
    smd_table.to_csv(TABLES_DIR / "covariate_balance_smd.csv", index=False)
    plot_smd_love_plot(smd_table, save_path=FIGURES_DIR / "smd_love_plot.png")

    log("Fitting propensity model for treatment-prediction diagnostic (AUC)...")
    prop_diag = fit_propensity_model(df, cv_folds=3, max_rows_for_cv=2_000_000)
    plot_propensity_diagnostics(prop_diag, save_path=FIGURES_DIR / "propensity_score_diagnostics.png")
    prop_diag["coef_table"].to_csv(TABLES_DIR / "propensity_model_coefficients.csv", index=False)

    verdict = treatment_assignment_verdict(prop_diag["auc"], smd_table)
    results["treatment_assignment_verdict"] = verdict
    log(f"Treatment assignment verdict: {verdict}")

    log("Rendering causal DAG...")
    plot_dag(save_path=FIGURES_DIR / "causal_dag.png")
    (REPORT_DIR / "dag_discussion.json").write_text(json.dumps(DAG_DISCUSSION, indent=2))

    # ---------------------------------------------------------------- ATE
    log("Running ATE estimators on FULL data (outcome=conversion)... this fits cross-fitted "
        "propensity + outcome nuisance models (~4 min).")
    t0 = time.time()
    ate_table_conv, nuisances_conv = run_all_estimators(df, outcome_col="conversion")
    log(f"ATE (conversion) done in {time.time()-t0:.1f}s")
    ate_table_conv.to_csv(TABLES_DIR / "ate_estimators_conversion.csv", index=False)

    log("Running ATE estimators on FULL data (outcome=visit)...")
    t0 = time.time()
    ate_table_visit, _ = run_all_estimators(df, outcome_col="visit")
    log(f"ATE (visit) done in {time.time()-t0:.1f}s")
    ate_table_visit.to_csv(TABLES_DIR / "ate_estimators_visit.csv", index=False)

    results["stabilized_ipw_conversion"] = {
        "ate": nuisances_conv["stabilized_ipw"].ate,
    }

    # ------------------------------------------- Confounding-injection ablation
    log("Building synthetic-observational confounded sample...")
    target_reference = target_population_rct_reference(df, outcome_col="conversion")
    confounded_df = make_confounded_observational_sample(df, max_rows=2_000_000)
    conf_diag = confounding_diagnostics(df, confounded_df)
    results["confounding_injection_diagnostics"] = conf_diag
    log(f"Confounded sample: n={len(confounded_df):,}, "
        f"treatment_ratio={conf_diag['confounded_treatment_ratio']:.3f}, "
        f"confounder SMD original={conf_diag['original_smd_confounder']:.3f} -> "
        f"confounded={conf_diag['confounded_smd_confounder']:.3f}")

    log("Re-running ATE estimators on the CONFOUNDED sample (outcome=conversion)...")
    t0 = time.time()
    ate_table_confounded, _ = run_all_estimators(confounded_df, outcome_col="conversion")
    log(f"Confounded ATE done in {time.time()-t0:.1f}s")
    ate_table_confounded["bias_vs_target_population_rct"] = (
        ate_table_confounded["ate"] - target_reference["ate"]
    )
    ate_table_confounded.to_csv(TABLES_DIR / "ate_estimators_confounded_sample.csv", index=False)
    results["confounding_target_population_rct_reference"] = target_reference

    (REPORT_DIR / "phase1_results.json").write_text(json.dumps(results, indent=2, default=float))
    log("Phase 1 complete. Results written to report/phase1_results.json")
    print("\n=== ATE (conversion, full RCT data) ===")
    print(ate_table_conv.to_string(index=False))
    print("\n=== ATE (confounded synthetic-observational sample) ===")
    print(ate_table_confounded.to_string(index=False))


if __name__ == "__main__":
    main()
