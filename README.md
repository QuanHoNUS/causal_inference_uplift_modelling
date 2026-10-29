# Marketing Campaign Optimization with Causal Inference and Uplift Modeling

This project estimates the incremental effect of a marketing campaign and uses predicted treatment effects
to design a targeting policy. The analysis uses the Criteo Uplift Prediction Dataset v2.1, a randomized
incrementality experiment with 13,979,592 observations.

The full analysis is in [report/REPORT.md](report/REPORT.md). The notebooks provide a shorter walkthrough
with executed outputs.

## Main Results

| Question | Result |
|---|---|
| Does the campaign increase conversion? | AIPW ATE = 0.000966, 95% CI [0.000898, 0.001033] |
| Which uplift model ranks customers best? | S-Learner and DR-Learner have nearly identical Qini estimates; their paired bootstrap difference includes zero |
| Which policy is selected? | S-Learner and top-2% coverage, selected on a 450,000-row validation set |
| Does that policy generalize? | Held-out test profit = $8,441, 95% CI [$1,504, $15,377], under illustrative economics |
| Is broad targeting profitable? | Treat-all test profit = -$203,569, 95% CI [-$212,295, -$194,843] |

The campaign has a positive average effect, but the effect is too small to support broad targeting at the
assumed value of $50 per conversion and cost of $0.50 per contact. A narrow uplift-based policy performs
better on the held-out policy test set.

## Data

The dataset contains 12 anonymized customer features and four binary variables:

- `treatment`: randomized treatment assignment
- `exposure`: realized ad exposure
- `visit`: site visit outcome
- `conversion`: primary outcome

The raw data has no missing values. Exact duplicate rows are retained because several anonymized features
have low cardinality; identical feature vectors can represent separate experimental units.

The data is not intended to be committed to Git. Download it with:

```bash
mkdir -p data/raw
curl -L -o data/raw/criteo-uplift-v2.1.csv.gz \
  https://criteostorage.blob.core.windows.net/criteo-research-datasets/criteo-uplift-v2.1.csv.gz
```

## Methods

| Stage | Methods |
|---|---|
| Treatment assignment | Standardized mean differences, propensity diagnostics, treatment-prediction AUC |
| Average treatment effect | Difference in Means, Regression Adjustment, HT/Hajek IPW, AIPW |
| Confounding stress test | Biased retention with a population-aligned weighted RCT reference |
| Conditional effects | S-, T-, X-, and DR-Learners; Causal Forest |
| Uplift evaluation | Qini, AUUC, gain curves, calibration by decile, paired stratified bootstrap |
| Policy evaluation | Validation-selected model and coverage, held-out IPS value, profit confidence intervals |

Raw Qini and AUUC values depend on evaluation-set size. The report therefore also gives normalized values
per 100,000 customers and 95% intervals from 200 paired, treatment-stratified bootstrap samples.

## Project Layout

```text
.
├── data/                         # raw and processed data; ignored by Git
├── notebooks/                    # four executed analysis notebooks
├── report/
│   ├── figures/
│   ├── tables/
│   └── REPORT.md
├── scripts/                      # reproducible phase drivers
├── src/
│   ├── causal/
│   ├── evaluation/
│   ├── policy/
│   ├── preprocessing/
│   ├── uplift/
│   └── visualization/
├── tests/
├── pyproject.toml
└── requirements.txt
```

## Reproduce the Analysis

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

python -c "from src.preprocessing.load_data import load_clean; load_clean()"
python -m scripts.01_eda_and_causal_inference
python -m scripts.02_uplift_modeling
python -m scripts.03_policy_and_ablation
python -m scripts.build_notebooks

pytest -q
```

Phase 2 trains the five uplift models and is the longest modeling step. Phase 3 reuses saved predictions,
runs the policy analysis, and computes 200 bootstrap repetitions for uplift-metric uncertainty.

## Scope and Limitations

- Campaign value and contact cost are assumptions, not fields in the Criteo data. Policy conclusions should
  be recalculated with actual business economics.
- Regression Adjustment uses a plug-in variance that omits first-stage model uncertainty. AIPW is used for
  the main ATE interval.
- The Qini/AUUC bootstrap conditions on fitted predictions. It measures evaluation-sample uncertainty but
  does not include variation from retraining the models on a different training sample.
- Some uplift deciles have negative point estimates, but every corresponding 95% interval contains zero.
  The analysis does not establish either the presence or absence of a harmed subgroup.
- Uplift models are trained on a 3,000,000-row stratified sample for computational tractability. Full-data
  ATE estimation uses all 13,979,592 rows.
- Propensity Score Matching was not used. With nearly constant treatment propensity and a very large sample,
  matching would discard information without adding a useful contrast to IPW and AIPW.
- R-Learner and DragonNet were outside the selected model set. The submitted comparison covers four
  meta-learners and one tree-based estimator.

## References

- Diemert, E., Betlei, A., Renaudin, C., and Amini, M. R. (2018). *A Large Scale Benchmark for Uplift Modeling*.
- Robins, J. M., Rotnitzky, A., and Zhao, L. P. (1994). *Estimation of Regression Coefficients When Some Regressors Are Not Always Observed*.
- Radcliffe, N. J. (2007). *Using Control Groups to Target on Predicted Lift*.
- Kunzel, S. R., Sekhon, J. S., Bickel, P. J., and Yu, B. (2019). *Metalearners for Estimating Heterogeneous Treatment Effects Using Machine Learning*.
