"""Project-wide configuration: paths, schema, and modeling constants.

Single source of truth so every module (EDA, causal estimators, uplift
models, policy simulation) agrees on column names, sample sizes, and seeds.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "report"
FIGURES_DIR = REPORT_DIR / "figures"
TABLES_DIR = REPORT_DIR / "tables"

RAW_CSV_GZ = DATA_RAW_DIR / "criteo-uplift-v2.1.csv.gz"
RAW_PARQUET = DATA_PROCESSED_DIR / "criteo_uplift_full.parquet"
SAMPLE_PARQUET = DATA_PROCESSED_DIR / "criteo_uplift_sample.parquet"
CONFOUNDED_PARQUET = DATA_PROCESSED_DIR / "criteo_uplift_confounded_observational.parquet"

for _d in (DATA_RAW_DIR, DATA_PROCESSED_DIR, FIGURES_DIR, TABLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Schema (Criteo Uplift Prediction Dataset v2.1)
# https://ailab.criteo.com/criteo-uplift-prediction-dataset/
# --------------------------------------------------------------------------
FEATURE_COLS = [f"f{i}" for i in range(12)]
TREATMENT_COL = "treatment"
VISIT_COL = "visit"
CONVERSION_COL = "conversion"
EXPOSURE_COL = "exposure"

# Primary outcome for this capstone. The assignment frames the business
# question around "conversion"; visit is kept as a secondary, less-rare
# outcome used for sanity checks (conversion is extremely rare, see EDA).
OUTCOME_COL = CONVERSION_COL

ALL_RAW_COLS = FEATURE_COLS + [TREATMENT_COL, CONVERSION_COL, VISIT_COL, EXPOSURE_COL]

# --------------------------------------------------------------------------
# Reproducibility & sampling
# --------------------------------------------------------------------------
RANDOM_SEED = 42

# The full dataset has ~13,979,592 rows: 85.0% treated / 15.0% control, with
# an overall conversion rate of only 0.29% (0.309% treated vs 0.194%
# control -- ~40.8k conversions total). EDA, balance/SMD diagnostics,
# propensity modeling, and the closed-form ATE estimators (Diff-in-Means,
# Regression Adjustment, IPW, AIPW/Doubly-Robust) all run in closed form or
# with a single sklearn fit and comfortably scale to the FULL dataset on a
# laptop (see README "Compute strategy"). CATE/uplift model training
# (meta-learners + Causal Forest, several models each cross-fit) is far more
# expensive, and -- because conversion is such a rare event -- also needs
# enough positive labels per arm to evaluate Qini/AUUC reliably. We use a
# 3,000,000-row stratified-by-treatment subsample (~21% of the data),
# which preserves the 85/15 treatment ratio and yields an expected ~8.7k
# conversions (~2.6k in the held-out test split) -- enough for stable
# uplift curves while keeping per-model training under a few minutes.
UPLIFT_MODELING_SAMPLE_SIZE = 3_000_000
USE_FULL_DATA_FOR_ATE = True

TRAIN_TEST_SPLIT_RATIO = 0.7  # train fraction for uplift modeling
UPLIFT_BOOTSTRAP_REPS = 200

# --------------------------------------------------------------------------
# Business / policy assumptions (explicit, revisited in sensitivity analysis)
# --------------------------------------------------------------------------
# Criteo does not publish a monetary value per conversion or a per-contact
# cost, so we adopt illustrative, clearly-labeled placeholder economics and
# sweep them in a sensitivity analysis rather than presenting a single
# "true" number as fact.
ASSUMED_VALUE_PER_CONVERSION = 50.0   # currency units, illustrative
ASSUMED_COST_PER_CONTACT = 0.5        # currency units, illustrative
