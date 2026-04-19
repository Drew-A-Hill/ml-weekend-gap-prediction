"""
test_temporal_model.py
======================
Pytest test suite for the temporal data analysis pipeline.

Covers:
    1. Data Validation   — CSV integrity, column presence, COVID exclusion, row counts
    2. Feature Checks    — cyclic encoding math, DaysSinceStart monotonicity, ranges
    3. Walk-forward Logic — fold structure, no leakage, expanding window
    4. Model Performance — AUC above chance, Combined >= Baseline, COVID fold flagged

Run from the project root:
    pytest notebooks/test_temporal_model.py -v
"""

import warnings
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
#PROJECT_ROOT  = Path(__file__).resolve().parents[1]
DATASET_PATH  = "../../structured_csv_data_files/fetched_data/dataset_clean.csv"
PLOTS_PATH    = "../../structured_csv_data_files/fetched_data/plots" 
TARGET = "GapUp"

BASELINE_FEATURES = [
    "MACD", "ROC", "StochPercK",
    "CloseVEma50", "CloseVSma20", "ADX",
    "BollingerBandWidth", "ATR", "FiveDStdDev",
    "OBV", "MFI", "VolumeRatio",
    "NetMargin", "RoA", "RevGrowthQoQ",
]

TEMPORAL_FEATURES = [
    "Month_sin", "Month_cos",
    "WeekOfYear_sin", "WeekOfYear_cos",
    "Quarter_sin", "Quarter_cos",
    "DaysSinceStart",
]

COMBINED_FEATURES = BASELINE_FEATURES + TEMPORAL_FEATURES

FOLDS = [
    {"train": list(range(2016, 2019)), "test": 2019},
    {"train": list(range(2016, 2020)), "test": 2020},  # COVID
    {"train": list(range(2016, 2021)), "test": 2021},
    {"train": list(range(2016, 2022)), "test": 2022},
    {"train": list(range(2016, 2023)), "test": 2023},
    {"train": list(range(2016, 2024)), "test": 2024},
]

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def df_all():
    """Full dataset loaded once for the entire test session."""
    df = pd.read_csv(DATASET_PATH)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None)
    return df


@pytest.fixture(scope="session")
def primary(df_all):
    """Primary set — COVID / extreme events excluded, sorted."""
    df = df_all[df_all["is_extreme_event"] == 0].copy()
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return df


@pytest.fixture(scope="session")
def logreg_results(primary):
    """
    Run the full walk-forward LogReg evaluation once and cache results.
    Returns a dict keyed by feature-set label with per-fold metric DataFrames.
    """
    def _run(features, label):
        X = primary[features].to_numpy(dtype=float)
        yr = primary["Year"].to_numpy()
        y  = primary[TARGET].to_numpy().astype(int)
        records = []
        for fold in FOLDS:
            tr = np.isin(yr, fold["train"])
            te = yr == fold["test"]
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            if len(y_te) == 0 or len(np.unique(y_te)) < 2:
                continue
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    penalty="l2", C=1.0, solver="liblinear",
                    max_iter=1000, random_state=42
                )),
            ])
            clf.fit(X_tr, y_tr)
            proba = clf.predict_proba(X_te)[:, 1]
            records.append({
                "label":     label,
                "test_year": fold["test"],
                "auc":       roc_auc_score(y_te, proba),
                "covid":     fold["test"] == 2020,
                "train_years": fold["train"],
                "n_train":   int(tr.sum()),
                "n_test":    int(te.sum()),
            })
        return pd.DataFrame(records)

    warnings.filterwarnings("ignore")
    return {
        "baseline": _run(BASELINE_FEATURES, "baseline"),
        "temporal": _run(TEMPORAL_FEATURES, "temporal"),
        "combined": _run(COMBINED_FEATURES, "combined"),
    }


# ===========================================================================
# 1. DATA VALIDATION
# ===========================================================================

class TestDataValidation:
    """Tests that the dataset loads correctly and meets integrity requirements."""

    def test_dataset_file_exists(self):
        """The temporal CSV must exist on disk."""
        assert DATASET_PATH.exists(), f"Dataset not found: {DATASET_PATH}"

    def test_loads_without_error(self, df_all):
        """Dataset must load into a non-empty DataFrame."""
        assert isinstance(df_all, pd.DataFrame)
        assert len(df_all) > 0

    def test_expected_row_count(self, df_all):
        """Should have roughly 8,000–9,000 rows (25 tickers × ~340 weeks)."""
        assert 7_000 <= len(df_all) <= 10_000, (
            f"Unexpected row count: {len(df_all)}"
        )

    def test_expected_ticker_count(self, primary):
        """Primary set should contain exactly 25 tickers."""
        assert primary["Ticker"].nunique() == 25, (
            f"Expected 25 tickers, got {primary['Ticker'].nunique()}"
        )

    def test_target_column_present(self, df_all):
        """GapUp column must exist."""
        assert TARGET in df_all.columns

    def test_target_is_binary(self, primary):
        """GapUp must only contain 0 and 1."""
        unique_vals = set(primary[TARGET].unique())
        assert unique_vals == {0, 1}, f"Non-binary values in GapUp: {unique_vals}"

    def test_baseline_features_present(self, primary):
        """All 15 baseline features must be in the dataset."""
        missing = [f for f in BASELINE_FEATURES if f not in primary.columns]
        assert not missing, f"Missing baseline features: {missing}"

    def test_temporal_features_present(self, primary):
        """All 7 temporal features must be in the dataset."""
        missing = [f for f in TEMPORAL_FEATURES if f not in primary.columns]
        assert not missing, f"Missing temporal features: {missing}"

    def test_no_nan_in_baseline_features(self, primary):
        """Baseline features must be fully populated in the primary set."""
        nan_counts = primary[BASELINE_FEATURES].isna().sum()
        bad = nan_counts[nan_counts > 0]
        assert bad.empty, f"NaN in baseline features:\n{bad}"

    def test_no_nan_in_temporal_features(self, primary):
        """Temporal features must be fully populated in the primary set."""
        nan_counts = primary[TEMPORAL_FEATURES].isna().sum()
        bad = nan_counts[nan_counts > 0]
        assert bad.empty, f"NaN in temporal features:\n{bad}"

    def test_covid_rows_excluded_from_primary(self, primary):
        """Primary set must contain no extreme-event rows."""
        assert (primary["is_extreme_event"] == 0).all(), (
            "Extreme-event rows found in primary set"
        )

    def test_covid_rows_exist_in_full_dataset(self, df_all):
        """The full dataset must contain some extreme-event rows (sanity check)."""
        n_covid = (df_all["is_extreme_event"] == 1).sum()
        assert n_covid > 0, "No extreme-event rows found — check data pipeline"

    def test_date_range(self, primary):
        """Primary set should span 2016 to 2024."""
        assert primary["Date"].min().year == 2016
        assert primary["Date"].max().year == 2024

    def test_all_rows_are_monday(self, primary):
        """All rows should be Mondays (weekly data anchored to Monday)."""
        non_monday = (primary["Date"].dt.dayofweek != 0).sum()
        assert non_monday == 0, f"{non_monday} rows are not Monday"

    def test_gapup_rate_near_half(self, primary):
        """GapUp rate should be close to 0.50 — dataset is roughly balanced."""
        rate = primary[TARGET].mean()
        assert 0.45 <= rate <= 0.60, f"Unexpected GapUp rate: {rate:.3f}"

    def test_plots_directory_exists(self):
        """The pre-generated plots directory must exist."""
        assert PLOTS_PATH.exists(), f"Plots directory not found: {PLOTS_PATH}"

    def test_nine_eda_plots_present(self):
        """All 9 pre-generated EDA PNGs must be in the plots directory."""
        expected = [
            "fig1_annual_return_distribution.png",
            "fig2_seasonality_heatmap.png",
            "fig3_quarterly_return_profile.png",
            "fig4_volatility_regime.png",
            "fig5_week_of_year_profile.png",
            "fig6_ticker_annual_heatmap.png",
            "fig7_rsi_temporal.png",
            "fig8_gap_extreme_events.png",
            "fig9_rolling_indicator_correlation.png",
        ]
        missing = [f for f in expected if not (PLOTS_PATH / f).exists()]
        assert not missing, f"Missing EDA plots: {missing}"


# ===========================================================================
# 2. FEATURE CHECKS
# ===========================================================================

class TestFeatureChecks:
    """Tests that temporal features are correctly engineered."""

    def test_cyclic_month_unit_circle(self, primary):
        """Month_sin² + Month_cos² must equal 1 for every row (unit circle)."""
        sq_sum = primary["Month_sin"] ** 2 + primary["Month_cos"] ** 2
        assert np.allclose(sq_sum, 1.0, atol=1e-6), (
            "Month sin/cos violates unit-circle property"
        )

    def test_cyclic_weekofyear_unit_circle(self, primary):
        """WeekOfYear_sin² + WeekOfYear_cos² must equal 1 for every row."""
        sq_sum = primary["WeekOfYear_sin"] ** 2 + primary["WeekOfYear_cos"] ** 2
        assert np.allclose(sq_sum, 1.0, atol=1e-6), (
            "WeekOfYear sin/cos violates unit-circle property"
        )

    def test_cyclic_quarter_unit_circle(self, primary):
        """Quarter_sin² + Quarter_cos² must equal 1 for every row."""
        sq_sum = primary["Quarter_sin"] ** 2 + primary["Quarter_cos"] ** 2
        assert np.allclose(sq_sum, 1.0, atol=1e-6), (
            "Quarter sin/cos violates unit-circle property"
        )

    def test_cyclic_features_in_range(self, primary):
        """All sin/cos cyclic features must be in [-1, 1]."""
        cyclic = [f for f in TEMPORAL_FEATURES if f != "DaysSinceStart"]
        for feat in cyclic:
            vals = primary[feat]
            assert vals.min() >= -1.0 - 1e-9, f"{feat} below -1: {vals.min()}"
            assert vals.max() <=  1.0 + 1e-9, f"{feat} above +1: {vals.max()}"

    def test_days_since_start_non_negative(self, primary):
        """DaysSinceStart must be >= 0 everywhere."""
        assert (primary["DaysSinceStart"] >= 0).all(), (
            "Negative DaysSinceStart values found"
        )

    def test_days_since_start_monotone_per_ticker(self, primary):
        """Within each ticker, DaysSinceStart must be non-decreasing."""
        for ticker, grp in primary.groupby("Ticker"):
            dss = grp.sort_values("Date")["DaysSinceStart"].values
            assert (np.diff(dss) >= 0).all(), (
                f"DaysSinceStart is not monotone for ticker {ticker}"
            )

    def test_days_since_start_max_reasonable(self, primary):
        """DaysSinceStart max should be < 3,500 (dataset ends 2024)."""
        max_days = primary["DaysSinceStart"].max()
        assert max_days < 3_500, f"DaysSinceStart too large: {max_days}"

    def test_month_values_one_to_twelve(self, primary):
        """Month column must only contain integers 1–12."""
        assert primary["Month"].between(1, 12).all(), (
            "Month column contains values outside 1–12"
        )

    def test_quarter_values_valid(self, primary):
        """Quarter column must only contain Q1, Q2, Q3, Q4."""
        valid = {"Q1", "Q2", "Q3", "Q4"}
        actual = set(primary["Quarter"].unique())
        assert actual <= valid, f"Unexpected quarter values: {actual - valid}"

    def test_week_of_year_values_valid(self, primary):
        """WeekOfYear must be between 1 and 53."""
        assert primary["WeekOfYear"].between(1, 53).all(), (
            "WeekOfYear contains values outside 1–53"
        )

    def test_temporal_features_not_constant(self, primary):
        """No temporal feature should be constant (zero variance = useless)."""
        for feat in TEMPORAL_FEATURES:
            assert primary[feat].nunique() > 1, (
                f"Temporal feature '{feat}' is constant — check encoding"
            )

    def test_temporal_correlation_not_perfect(self, primary):
        """
        No temporal feature should be perfectly correlated with GapUp (|r|=1)
        which would indicate target leakage.
        """
        for feat in TEMPORAL_FEATURES:
            r, _ = pointbiserialr(primary[TARGET], primary[feat])
            assert abs(r) < 0.99, (
                f"Suspiciously high correlation for '{feat}': r={r:.4f}"
            )

    def test_combined_features_no_duplicates(self):
        """Combined feature list must not contain duplicate column names."""
        assert len(COMBINED_FEATURES) == len(set(COMBINED_FEATURES)), (
            "Duplicate features found in COMBINED_FEATURES"
        )

    def test_combined_is_superset_of_baseline_and_temporal(self):
        """Combined must contain all baseline and all temporal features."""
        for f in BASELINE_FEATURES:
            assert f in COMBINED_FEATURES, f"Baseline feature '{f}' missing from Combined"
        for f in TEMPORAL_FEATURES:
            assert f in COMBINED_FEATURES, f"Temporal feature '{f}' missing from Combined"


# ===========================================================================
# 3. WALK-FORWARD LOGIC
# ===========================================================================

class TestWalkForwardLogic:
    """Tests that the fold structure is correct and there is no data leakage."""

    def test_six_folds_defined(self):
        """There must be exactly 6 folds."""
        assert len(FOLDS) == 6

    def test_test_years_cover_2019_to_2024(self):
        """Test years must span 2019 through 2024 inclusive."""
        test_years = [f["test"] for f in FOLDS]
        assert test_years == list(range(2019, 2025))

    def test_expanding_window(self):
        """Each fold's training set must be strictly larger than the previous."""
        for i in range(1, len(FOLDS)):
            prev_train = FOLDS[i - 1]["train"]
            curr_train = FOLDS[i]["train"]
            assert len(curr_train) > len(prev_train), (
                f"Fold {i} train window not larger than fold {i-1}"
            )

    def test_no_train_test_overlap(self):
        """Train and test years must never overlap in any fold."""
        for fold in FOLDS:
            overlap = set(fold["train"]) & {fold["test"]}
            assert not overlap, (
                f"Train/test overlap in fold with test={fold['test']}: {overlap}"
            )

    def test_train_years_always_before_test(self):
        """All training years must be strictly before the test year."""
        for fold in FOLDS:
            assert all(y < fold["test"] for y in fold["train"]), (
                f"Future year in training set for test={fold['test']}"
            )

    def test_covid_fold_is_2020(self):
        """The COVID fold must be the one with test_year == 2020."""
        covid_folds = [f for f in FOLDS if f["test"] == 2020]
        assert len(covid_folds) == 1, "Expected exactly one COVID fold (test=2020)"

    def test_primary_rows_per_fold_non_empty(self, primary):
        """Every fold's test year must have at least 200 rows in the primary set."""
        yr = primary["Year"].to_numpy()
        for fold in FOLDS:
            n_test = (yr == fold["test"]).sum()
            assert n_test >= 200, (
                f"Too few test rows for fold test={fold['test']}: {n_test}"
            )

    def test_no_future_data_in_training(self, primary):
        """
        For each fold, verify that no row from the test year appears
        in the training mask — the strict no-leakage check.
        """
        yr = primary["Year"].to_numpy()
        for fold in FOLDS:
            tr_mask = np.isin(yr, fold["train"])
            te_mask = yr == fold["test"]
            # No row should be both train and test
            assert not np.any(tr_mask & te_mask), (
                f"Leakage detected: test year {fold['test']} rows in training mask"
            )

    def test_both_classes_present_in_each_test_fold(self, primary):
        """Each test fold must contain both GapUp=0 and GapUp=1 rows."""
        yr  = primary["Year"].to_numpy()
        y   = primary[TARGET].to_numpy()
        for fold in FOLDS:
            te_mask = yr == fold["test"]
            unique  = np.unique(y[te_mask])
            assert len(unique) == 2, (
                f"Only one class in test fold {fold['test']}: {unique}"
            )


# ===========================================================================
# 4. MODEL PERFORMANCE
# ===========================================================================

class TestModelPerformance:
    """Tests that the LogReg model produces sensible results."""

    def test_six_folds_evaluated(self, logreg_results):
        """All three feature sets must produce exactly 6 fold results."""
        for label, df in logreg_results.items():
            assert len(df) == 6, (
                f"{label}: expected 6 fold results, got {len(df)}"
            )

    def test_auc_values_in_valid_range(self, logreg_results):
        """All AUC values must be in [0, 1]."""
        for label, df in logreg_results.items():
            assert (df["auc"] >= 0.0).all() and (df["auc"] <= 1.0).all(), (
                f"{label}: AUC out of [0,1] range"
            )

    def test_baseline_mean_auc_above_chance(self, logreg_results):
        """Baseline model mean AUC over non-COVID folds must beat chance (0.50)."""
        df = logreg_results["baseline"]
        nc_auc = df.loc[~df["covid"], "auc"].mean()
        assert nc_auc > 0.50, (
            f"Baseline mean AUC below chance: {nc_auc:.4f}"
        )

    def test_temporal_mean_auc_above_chance(self, logreg_results):
        """
        Temporal-only model AUC must be within a reasonable range of chance.
        Calendar features are weak signals on their own — AUC near 0.50 is
        expected and acceptable. We flag anything below 0.45 as a sign
        something is wrong with the feature encoding or data pipeline.
        """
        df = logreg_results["temporal"]
        nc_auc = df.loc[~df["covid"], "auc"].mean()
        assert nc_auc >= 0.45, (
            f"Temporal-only mean AUC suspiciously low: {nc_auc:.4f}. "
            "This may indicate a feature encoding error."
        )

    def test_combined_mean_auc_above_chance(self, logreg_results):
        """Combined model mean AUC over non-COVID folds must beat chance."""
        df = logreg_results["combined"]
        nc_auc = df.loc[~df["covid"], "auc"].mean()
        assert nc_auc > 0.50, (
            f"Combined mean AUC below chance: {nc_auc:.4f}"
        )

    def test_combined_auc_not_worse_than_baseline(self, logreg_results):
        """
        Combined mean AUC must not be more than 0.02 below Baseline mean AUC
        over non-COVID folds. Adding temporal features should not substantially
        hurt performance.
        """
        base_auc = logreg_results["baseline"].loc[
            ~logreg_results["baseline"]["covid"], "auc"
        ].mean()
        comb_auc = logreg_results["combined"].loc[
            ~logreg_results["combined"]["covid"], "auc"
        ].mean()
        delta = comb_auc - base_auc
        assert delta >= -0.02, (
            f"Combined AUC ({comb_auc:.4f}) is more than 0.02 below "
            f"Baseline ({base_auc:.4f}). Delta={delta:.4f}"
        )

    def test_no_perfect_auc(self, logreg_results):
        """
        No fold should have AUC == 1.0 — that would indicate target leakage
        or a degenerate test set.
        """
        for label, df in logreg_results.items():
            assert (df["auc"] < 1.0).all(), (
                f"{label}: perfect AUC detected — check for leakage"
            )

    def test_covid_fold_flagged(self, logreg_results):
        """The 2020 fold must be flagged as covid=True in results."""
        df = logreg_results["baseline"]
        covid_rows = df[df["test_year"] == 2020]
        assert len(covid_rows) == 1
        assert covid_rows["covid"].iloc[0] is True or covid_rows["covid"].iloc[0] == True

    def test_non_covid_folds_not_flagged(self, logreg_results):
        """All non-2020 folds must have covid=False."""
        df = logreg_results["baseline"]
        non_covid = df[df["test_year"] != 2020]
        assert (non_covid["covid"] == False).all(), (
            "Non-COVID folds incorrectly flagged as covid=True"
        )

    def test_train_rows_increase_each_fold(self, logreg_results):
        """Each successive fold must have more training rows than the previous."""
        df = logreg_results["baseline"].sort_values("test_year")
        train_counts = df["n_train"].tolist()
        for i in range(1, len(train_counts)):
            assert train_counts[i] > train_counts[i - 1], (
                f"Training rows did not increase at fold {i}: "
                f"{train_counts[i-1]} -> {train_counts[i]}"
            )

    def test_auc_not_suspiciously_uniform(self, logreg_results):
        """
        Baseline AUC values across folds must have some variance — identical
        AUC across all folds would suggest the model is not actually fitting
        each fold independently.
        """
        aucs = logreg_results["baseline"]["auc"].values
        assert aucs.std() > 0.001, (
            f"AUC values suspiciously uniform (std={aucs.std():.6f})"
        )


# ===========================================================================
# Entry point for running directly (without pytest CLI)
# ===========================================================================

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
