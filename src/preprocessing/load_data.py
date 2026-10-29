"""Load, validate, and clean the Criteo Uplift Prediction Dataset.

Design notes
------------
- The raw file ships as a single gzip CSV with no missing values by
  construction (Criteo pre-cleaned it for the public release), but we still
  run explicit validation rather than assuming that -- a silently-changed
  upstream file should fail loudly, not corrupt downstream causal estimates.
- Columns are downcast to float32 / int8 on load. At ~13.9M rows this keeps
  the full dataset under ~1GB in memory, which is what makes it practical to
  run EDA and the closed-form ATE estimators on the FULL data rather than a
  sample (see config.USE_FULL_DATA_FOR_ATE).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.utils.config import (
    ALL_RAW_COLS,
    CONVERSION_COL,
    EXPOSURE_COL,
    FEATURE_COLS,
    RANDOM_SEED,
    RAW_CSV_GZ,
    RAW_PARQUET,
    TREATMENT_COL,
    VISIT_COL,
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    n_rows: int
    n_cols: int
    missing_by_col: pd.Series
    n_duplicate_rows: int
    treatment_values: set
    conversion_values: set
    visit_values: set
    out_of_range_features: list

    def is_clean(self) -> bool:
        return (
            self.missing_by_col.sum() == 0
            and self.treatment_values <= {0, 1}
            and self.conversion_values <= {0, 1}
            and self.visit_values <= {0, 1}
            and not self.out_of_range_features
        )

    def summary(self) -> str:
        lines = [
            f"rows={self.n_rows:,} cols={self.n_cols}",
            f"duplicate_rows={self.n_duplicate_rows:,}",
            f"treatment_values={sorted(self.treatment_values)}",
            f"conversion_values={sorted(self.conversion_values)}",
            f"visit_values={sorted(self.visit_values)}",
            f"missing_total={int(self.missing_by_col.sum())}",
            f"out_of_range_features={self.out_of_range_features}",
            f"clean={self.is_clean()}",
        ]
        return "\n".join(lines)


def load_raw(nrows: int | None = None) -> pd.DataFrame:
    """Load the raw gzip CSV directly (slow path, used once to build the parquet cache)."""
    if not RAW_CSV_GZ.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_CSV_GZ}.\n"
            "Download it with:\n"
            "  curl -L -o data/raw/criteo-uplift-v2.1.csv.gz "
            "https://criteostorage.blob.core.windows.net/criteo-research-datasets/criteo-uplift-v2.1.csv.gz"
        )
    dtypes = {c: np.float32 for c in FEATURE_COLS}
    dtypes.update({TREATMENT_COL: np.int8, CONVERSION_COL: np.int8, VISIT_COL: np.int8, EXPOSURE_COL: np.float32})
    df = pd.read_csv(RAW_CSV_GZ, dtype=dtypes, nrows=nrows)
    missing_cols = set(ALL_RAW_COLS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Unexpected schema: missing columns {missing_cols}. Found: {list(df.columns)}")
    return df[ALL_RAW_COLS]


def validate(df: pd.DataFrame) -> ValidationReport:
    feature_ranges_ok = []
    for c in FEATURE_COLS:
        col = df[c]
        # Criteo features are anonymized/continuous; flag pathological values
        # (inf, extreme outliers beyond 1e6) rather than assuming a fixed range.
        if not np.isfinite(col.to_numpy()).all():
            feature_ranges_ok.append(c)
    report = ValidationReport(
        n_rows=len(df),
        n_cols=df.shape[1],
        missing_by_col=df.isna().sum(),
        n_duplicate_rows=int(df.duplicated().sum()),
        treatment_values=set(df[TREATMENT_COL].unique().tolist()),
        conversion_values=set(df[CONVERSION_COL].unique().tolist()),
        visit_values=set(df[VISIT_COL].unique().tolist()),
        out_of_range_features=feature_ranges_ok,
    )
    return report


def clean(df: pd.DataFrame, drop_duplicate_rows: bool = False) -> pd.DataFrame:
    """Apply documented cleaning rules.

    The Criteo release has no missing values and no out-of-range binary
    columns in practice; this function still enforces the rules explicitly
    (rather than skipping cleaning) so the pipeline is robust if a different
    slice/version of the file is substituted.

    IMPORTANT -- duplicate rows are NOT removed by default. Several of the
    12 anonymized features are low-cardinality (e.g. f1 has 60 unique
    values, f4 has 260, f11 has 136 -- see EDA notebook), so with 13.9M rows
    the feature space is far smaller than the row count and exact-row
    duplication (~12% of rows) is the *expected* consequence of many
    customers sharing a feature bucket, not a data-quality defect. Dropping
    these rows would arbitrarily discard real observations (and their
    treatment/outcome signal) and bias the sample. Set
    ``drop_duplicate_rows=True`` only if you have separately verified
    duplication is an artifact for your use case.
    """
    before = len(df)
    df = df.dropna(subset=ALL_RAW_COLS)
    df = df[df[TREATMENT_COL].isin([0, 1])]
    df = df[df[CONVERSION_COL].isin([0, 1])]
    df = df[df[VISIT_COL].isin([0, 1])]
    if drop_duplicate_rows:
        df = df.drop_duplicates()
    after = len(df)
    dropped = before - after
    if dropped:
        logger.warning("Dropped %d/%d rows (%.4f%%) during cleaning.", dropped, before, 100 * dropped / before)
    return df.reset_index(drop=True)


def build_parquet_cache(nrows: int | None = None) -> pd.DataFrame:
    """One-time ETL: raw csv.gz -> validated, cleaned parquet cache."""
    df = load_raw(nrows=nrows)
    report = validate(df)
    logger.info("Raw validation report:\n%s", report.summary())
    df = clean(df)
    df.to_parquet(RAW_PARQUET, index=False)
    return df


def load_clean(force_rebuild: bool = False) -> pd.DataFrame:
    """Load the cleaned full dataset, using the parquet cache when available."""
    if RAW_PARQUET.exists() and not force_rebuild:
        return pd.read_parquet(RAW_PARQUET)
    return build_parquet_cache()


def stratified_sample(df: pd.DataFrame, n: int, stratify_col: str = TREATMENT_COL, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Sample n rows preserving the treatment ratio (and, by extension, the
    overall conversion rate within each arm, since sampling is uniform
    within stratum)."""
    frac = min(1.0, n / len(df))
    sampled = (
        df.groupby(stratify_col, group_keys=False)
        .apply(lambda g: g.sample(frac=frac, random_state=seed))
    )
    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
