"""Assemble the four narrative notebooks in notebooks/ from the already-
computed report/tables + report/figures artifacts (produced by scripts
01-03). This script does NOT recompute anything expensive -- it loads
cached CSVs/PNGs and a couple of tiny pandas ops, then executes the
notebook in place so reviewers see real, already-run output.
"""
from __future__ import annotations

import nbformat as nbf
from nbclient import NotebookClient

from src.utils.config import PROJECT_ROOT

NB_DIR = PROJECT_ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


PREAMBLE = """
import pandas as pd
from IPython.display import Image, display
pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 140)
from src.utils.config import TABLES_DIR, FIGURES_DIR, REPORT_DIR
"""


def build_01_eda():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# 01 - Data Preprocessing and Exploratory Data Analysis

Dataset: Criteo Uplift Prediction Dataset v2.1 (13,979,592 observations from a randomized
incrementality experiment, downloaded from Criteo's public research mirror).

This notebook presents the outputs of `scripts/01_eda_and_causal_inference.py` (run on the FULL dataset).
"""),
        code(PREAMBLE),
        md("## Dataset Overview"),
        code('pd.read_csv(TABLES_DIR / "dataset_overview.csv")'),
        md("""
## Preprocessing Note: Duplicate Rows Are Not Data Errors

Raw validation found about 1.66M exact-duplicate rows (11.8%). Several anonymized features have low
cardinality (`f1`: 60 unique values, `f4`: 260, `f11`: 136), so repeated feature combinations are expected
in a dataset of this size. The analysis retains these observations because there is no row identifier that
would establish that they are accidental duplicate records.
"""),
        md("## Naive (Unadjusted) Treatment-vs-Control Gap"),
        code("""
import json
results = json.loads((REPORT_DIR / "phase1_results.json").read_text())
pd.DataFrame([results["naive_lift_conversion"], results["naive_lift_visit"]])
"""),
        md("## Treatment & Outcome Distributions"),
        code('display(Image(filename=str(FIGURES_DIR / "treatment_outcome_distribution.png")))'),
        md("## Feature Distributions by Arm"),
        code('display(Image(filename=str(FIGURES_DIR / "feature_distributions_by_arm.png")))'),
        md("## Feature / Treatment / Outcome Correlation"),
        code('display(Image(filename=str(FIGURES_DIR / "feature_correlation_matrix.png")))'),
        code('pd.read_csv(TABLES_DIR / "feature_correlation_matrix.csv", index_col=0)["conversion"].drop(["conversion","visit"], errors="ignore").sort_values(key=abs, ascending=False)'),
        md("""
## Discussion: Potential Biases Observed

- Treatment is imbalanced (85% treated and 15% control). The smaller control arm increases the uncertainty
  of estimates that depend heavily on control observations.
- Conversion is rare (about 0.29% overall). This limits the precision of subgroup and uplift estimates even
  with a large dataset.
- `f11` and `f4` have the largest marginal correlations with conversion (approximately -0.31 and +0.30).
  They are used in the semi-synthetic confounding experiment in notebook 02.
"""),
    ]
    (nb_path := NB_DIR / "01_eda.ipynb").write_text(nbf.writes(nb))
    return nb_path


def build_02_causal_inference():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# 02 - Treatment Assignment, Causal DAG, and ATE Estimation

Answers Section 6 of the assignment: does treatment look randomized, what does the causal graph imply
about identifiability, and what is the campaign's Average Treatment Effect under four primary estimators
and a stabilized-IPW robustness check?
"""),
        code(PREAMBLE),
        md("## 6.1 — Treatment Assignment Analysis: Covariate Balance (SMD)"),
        code('pd.read_csv(TABLES_DIR / "covariate_balance_smd.csv")'),
        code('display(Image(filename=str(FIGURES_DIR / "smd_love_plot.png")))'),
        md("## 6.1 — Can Customer Features Predict Treatment Assignment?"),
        code("""
import json
results = json.loads((REPORT_DIR / "phase1_results.json").read_text())
results["treatment_assignment_verdict"]
"""),
        code('display(Image(filename=str(FIGURES_DIR / "propensity_score_diagnostics.png")))'),
        md("""
Treatment assignment is consistent with randomization. All 12 covariates have |SMD| < 0.1, and a logistic
regression predicting treatment from those covariates has AUC 0.509. Estimated propensity scores for both
arms are concentrated near the empirical treatment rate of 0.85.
"""),
        md("## 6.2 — Causal DAG"),
        code('display(Image(filename=str(FIGURES_DIR / "causal_dag.png")))'),
        code("""
dag_discussion = json.loads((REPORT_DIR / "dag_discussion.json").read_text())
for k, v in dag_discussion.items():
    print(f"### {k}\\n{v}\\n")
"""),
        md("## 6.3 — Observable vs. Unobservable Confounding: Two Scenarios"),
        md("""
### Scenario 1: randomized treatment

With randomized treatment, Difference-in-Means identifies the ATE without covariate adjustment. Regression
and propensity-based estimators are included as precision adjustments and robustness checks.

### Scenario 2: observational targeting

If targeting depends on `X`, the path `T <- X -> Y` is open and unadjusted Difference-in-Means can be
biased. Regression adjustment, IPW, and AIPW require measured covariates to capture the confounding. The
semi-synthetic experiment below changes sample retention while preserving the original randomized outcomes.
"""),
        md("## 6.4 — Estimator Comparison (on the REAL, randomized data)"),
        code('pd.read_csv(TABLES_DIR / "ate_estimators_conversion.csv")'),
        code('pd.read_csv(TABLES_DIR / "ate_estimators_visit.csv")'),
        md("""
The point estimates are similar, as expected under randomization. The Regression Adjustment interval
uses a plug-in variance that omits uncertainty from fitting the outcome models, so it is too narrow for
inference. The AIPW interval is based on its influence function and includes residual outcome variation.
"""),
        md("## Synthetic Observational Scenario: Confounding-Injection Ablation"),
        code("""
conf_diag = results["confounding_injection_diagnostics"]
conf_diag
"""),
        code('pd.read_csv(TABLES_DIR / "ate_estimators_confounded_sample.csv")'),
        code('results["confounding_target_population_rct_reference"]'),
        md("""
The retained sample changes the estimand, so the benchmark is the retention-weighted ATE from the original
randomized trial. Weighting by `q(X)=e*s1(X)+(1-e)*s0(X)` targets that same population. Difference-in-Means
is biased after selection, while Regression Adjustment and AIPW are close to the reference. Standalone IPW
is unstable because fitted control weights reach the clipping threshold.
"""),
        md("""
## 6.5 — Interpretation

The estimated conversion ATE is about 0.10 percentage points. Because the control conversion rate is low,
this corresponds to a relative increase of approximately 50% to 59%, depending on the estimator. The
absolute effect and its uncertainty are the relevant inputs for the policy analysis in notebook 04.
"""),
    ]
    (nb_path := NB_DIR / "02_causal_inference.ipynb").write_text(nbf.writes(nb))
    return nb_path


def build_03_uplift_modeling():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# 03 - Uplift / CATE Modeling

Trains S-Learner, T-Learner, X-Learner, DR-Learner (EconML meta-learners, LightGBM base learners) and a
Causal Forest (EconML `CausalForestDML`) on a 3,000,000-row stratified sample. Evaluation uses a held-out
30% split. Qini and AUUC are reported per 100,000 observations with paired bootstrap intervals.
"""),
        code(PREAMBLE),
        md("## Model Fit Times"),
        code('pd.read_csv(TABLES_DIR / "uplift_model_fit_times.csv", index_col=0)'),
        md("## Model Comparison: Qini Coefficient & AUUC"),
        code('pd.read_csv(TABLES_DIR / "uplift_model_comparison.csv")'),
        md("## Bootstrap Uncertainty"),
        code('pd.read_csv(TABLES_DIR / "uplift_model_uncertainty.csv")'),
        code("""
pairs = pd.read_csv(TABLES_DIR / "uplift_model_pairwise_bootstrap.csv")
pairs[
    ((pairs["model_a"] == "S-Learner") & pairs["model_b"].isin(["DR-Learner", "X-Learner", "T-Learner"]))
    | ((pairs["model_a"] == "DR-Learner") & pairs["model_b"].isin(["X-Learner", "T-Learner"]))
]
"""),
        md("""
Raw Qini and AUUC scale with the number of observations, so comparisons across datasets use the per-100,000
columns. S-Learner and DR-Learner have nearly identical point estimates; their paired bootstrap difference
includes zero. Both separate more clearly from X-Learner and T-Learner. These percentile intervals condition
on the fitted scores and do not include model-refitting uncertainty.
"""),
        md("## Qini Curves"),
        code('display(Image(filename=str(FIGURES_DIR / "qini_curves_all_models.png")))'),
        md("## Incremental Gain Curves"),
        code('display(Image(filename=str(FIGURES_DIR / "gain_curves_all_models.png")))'),
        md("## Who Benefits, Who May Be Harmed"),
        code('pd.read_csv(TABLES_DIR / "segment_profile_top_vs_bottom_uplift.csv")'),
        code('display(Image(filename=str(FIGURES_DIR / "calibration_S-Learner.png")))'),
        md("""
The top S-Learner decile has positive observed uplift and a 95% interval above zero. The bottom-decile point
estimate is 0.042 percentage points and its interval includes zero. Several calibration deciles have small
negative point estimates, but their intervals also include zero; the data therefore do not establish either
harm or safety within those deciles.
"""),
    ]
    (nb_path := NB_DIR / "03_uplift_modeling.ipynb").write_text(nbf.writes(nb))
    return nb_path


def build_04_policy():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("""
# 04 - Campaign Targeting Policy and Business Recommendation

Simulates and offline-evaluates 7 targeting policies (treat all / none / random / positive-uplift / top-10%
/ top-20% / validation-selected top-K). The original model holdout is split into policy-validation and
policy-test halves. Model/coverage selection uses validation outcomes only; fixed policies are evaluated
once on untouched test outcomes with Inverse Propensity Scoring and profit confidence intervals.
"""),
        code(PREAMBLE),
        md("## Validation-Set Model Selection"),
        code('pd.read_csv(TABLES_DIR / "policy_model_selection_validation.csv")'),
        md("## Policy Comparison"),
        code('pd.read_csv(TABLES_DIR / "policy_comparison.csv")'),
        md("## Profit-Maximizing Targeting Curve"),
        code('display(Image(filename=str(FIGURES_DIR / "profit_maximizing_curve.png")))'),
        md("## Cost / Value Sensitivity"),
        code('display(Image(filename=str(FIGURES_DIR / "cost_sensitivity_heatmap.png")))'),
        code('pd.read_csv(TABLES_DIR / "cost_sensitivity_sweep.csv")'),
        md("## Ablation Study & Final Business Recommendation"),
        code("""
import json
ablation = json.loads((REPORT_DIR / "phase3_ablation.json").read_text())
ablation["profit_maximizing_summary"]
"""),
        md("""
Validation selects S-Learner and a 2% targeting rate. Applying that fixed rule once to the untouched policy
test set yields positive estimated profit, while treating everyone is unprofitable under the stated value
and contact-cost assumptions. These figures are scenario results rather than observed accounting outcomes.
"""),
    ]
    (nb_path := NB_DIR / "04_policy_and_business.ipynb").write_text(nbf.writes(nb))
    return nb_path


def execute_notebook(path):
    notebook = nbf.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=600,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    nbf.write(notebook, path)


if __name__ == "__main__":
    paths = [build_01_eda(), build_02_causal_inference(), build_03_uplift_modeling(), build_04_policy()]
    for p in paths:
        execute_notebook(p)
        print("wrote and executed", p)
