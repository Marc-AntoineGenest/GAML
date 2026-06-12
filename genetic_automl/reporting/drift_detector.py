"""
DriftDetector — Statistical data drift detection using KS test and PSI.

Problem
-------
A model trained on historical data degrades silently when the distribution of
incoming features changes (concept drift, covariate shift).  Without monitoring,
score degradation is only noticed after business impact has already occurred.

This module provides two complementary drift signals:

  KS test (Kolmogorov-Smirnov)
    Non-parametric two-sample test for continuous features.
    H0: both samples drawn from the same distribution.
    Small p-value (< threshold) = distributions likely differ = drift detected.
    Sensitive to location AND shape differences.

  PSI (Population Stability Index)
    Industry standard from credit scoring.  Buckets the feature into deciles,
    compares bin fractions between reference and new data.
    PSI < 0.10  — no significant drift
    PSI < 0.20  — moderate drift, monitor
    PSI >= 0.20 — significant drift, retrain

  Chi-squared test
    For categorical features (object/category dtype).
    Compares observed frequency distributions.

Design
------
- Pure numpy/scipy: no external monitoring framework dependency.
- Gracefully handles missing scipy: falls back to PSI-only detection.
- Works on any pandas DataFrame; ignores columns that exist in reference
  but not in new data (logs a warning) and vice versa.
- Returns a DriftReport dataclass — JSON-serialisable via .to_dict().
- Callable standalone or via pipeline.detect_drift(new_df).

Usage
-----
Standalone:
    from genetic_automl.reporting.drift_detector import DriftDetector
    detector = DriftDetector(pvalue_threshold=0.05, psi_threshold=0.20)
    detector.fit(X_train)          # store reference distribution
    report = detector.detect(X_new)
    if report.any_drift:
        print(report.summary())

Via pipeline (after fitting):
    report = pipeline.detect_drift(new_df)
    print(report.summary())
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)

# PSI interpretation thresholds (industry standard)
PSI_STABLE    = 0.10
PSI_MODERATE  = 0.20   # warn
PSI_CRITICAL  = 0.25   # retrain recommended

N_PSI_BINS = 10        # decile buckets for PSI



@dataclass
class FeatureDriftResult:
    """Drift statistics for a single feature."""
    feature: str
    dtype: str                      # "continuous" or "categorical"
    ks_statistic: Optional[float]   # KS test statistic (continuous only)
    ks_pvalue: Optional[float]      # KS test p-value (continuous only)
    chi2_statistic: Optional[float] # Chi-squared statistic (categorical only)
    chi2_pvalue: Optional[float]    # Chi-squared p-value (categorical only)
    psi: float                      # Population Stability Index
    drift_detected: bool            # True if any test flagged drift
    severity: str                   # "none" | "moderate" | "critical"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DriftReport:
    """Drift detection results for all features."""
    n_features_checked: int
    n_features_drifted: int
    any_drift: bool
    pvalue_threshold: float
    psi_threshold: float
    feature_results: List[FeatureDriftResult] = field(default_factory=list)
    missing_in_new: List[str] = field(default_factory=list)
    new_columns: List[str] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable one-paragraph summary."""
        if not self.any_drift:
            return (
                f"No drift detected across {self.n_features_checked} features "
                f"(p-value threshold={self.pvalue_threshold}, "
                f"PSI threshold={self.psi_threshold})."
            )
        drifted = [r for r in self.feature_results if r.drift_detected]
        critical = [r for r in drifted if r.severity == "critical"]
        moderate = [r for r in drifted if r.severity == "moderate"]
        lines = [
            f"Drift detected in {self.n_features_drifted} / "
            f"{self.n_features_checked} features.",
        ]
        if critical:
            lines.append(
                f"  CRITICAL (retrain recommended): "
                + ", ".join(r.feature for r in critical)
            )
        if moderate:
            lines.append(
                f"  MODERATE (monitor): "
                + ", ".join(r.feature for r in moderate)
            )
        if self.missing_in_new:
            lines.append(f"  Missing in new data: {self.missing_in_new}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @property
    def drifted_features(self) -> List[str]:
        return [r.feature for r in self.feature_results if r.drift_detected]

    @property
    def critical_features(self) -> List[str]:
        return [r.feature for r in self.feature_results if r.severity == "critical"]



class DriftDetector:
    """
    Statistical data drift detector using KS test and PSI.

    Parameters
    ----------
    pvalue_threshold : float
        KS / chi-squared p-value below which drift is flagged.
        Default 0.05 (standard significance level).
    psi_threshold : float
        PSI above which drift is flagged.
        Default 0.20 (industry standard for significant drift).
    n_psi_bins : int
        Number of bins for PSI calculation (default 10 = deciles).
    """

    def __init__(
        self,
        pvalue_threshold: float = 0.05,
        psi_threshold: float = PSI_MODERATE,
        n_psi_bins: int = N_PSI_BINS,
    ) -> None:
        self.pvalue_threshold = pvalue_threshold
        self.psi_threshold = psi_threshold
        self.n_psi_bins = n_psi_bins
        self._reference: Optional[pd.DataFrame] = None
        self._bin_edges: Dict[str, np.ndarray] = {}
        self._cat_categories: Dict[str, np.ndarray] = {}


    def fit(self, reference: pd.DataFrame) -> "DriftDetector":
        """
        Store the reference distribution.

        Call this once on training data immediately after fitting the model.
        The bin edges and category sets computed here are reused for all
        subsequent detect() calls.

        Parameters
        ----------
        reference : pd.DataFrame
            Feature matrix (target column already removed).
        """
        self._reference = reference.copy()
        self._bin_edges = {}
        self._cat_categories = {}

        for col in reference.columns:
            if _is_continuous(reference[col]):
                vals = reference[col].dropna().values.astype(float)
                if len(vals) > 0:
                    # Compute bin edges on reference for PSI
                    self._bin_edges[col] = np.nanpercentile(
                        vals,
                        np.linspace(0, 100, self.n_psi_bins + 1),
                    )
                    # Clamp boundaries so new data never falls outside
                    self._bin_edges[col][0]  = -np.inf
                    self._bin_edges[col][-1] =  np.inf
            else:
                self._cat_categories[col] = reference[col].dropna().unique()

        log.info(
            "DriftDetector fitted | %d features | %d continuous | %d categorical",
            len(reference.columns),
            len(self._bin_edges),
            len(self._cat_categories),
        )
        return self

    def detect(self, new_data: pd.DataFrame) -> DriftReport:
        """
        Compare *new_data* against the reference distribution.

        Parameters
        ----------
        new_data : pd.DataFrame
            Incoming feature matrix to check for drift.

        Returns
        -------
        DriftReport
            Per-feature statistics and an overall drift flag.
        """
        if self._reference is None:
            raise RuntimeError("Call fit() before detect().")

        ref_cols = set(self._reference.columns)
        new_cols = set(new_data.columns)
        missing_in_new = sorted(ref_cols - new_cols)
        extra_in_new   = sorted(new_cols - ref_cols)

        if missing_in_new:
            log.warning(
                "DriftDetector: %d reference features missing in new data: %s",
                len(missing_in_new), missing_in_new,
            )

        feature_results: List[FeatureDriftResult] = []
        cols_to_check = sorted(ref_cols & new_cols)

        for col in cols_to_check:
            result = self._check_feature(col, new_data[col])
            feature_results.append(result)

        drifted = [r for r in feature_results if r.drift_detected]

        report = DriftReport(
            n_features_checked=len(feature_results),
            n_features_drifted=len(drifted),
            any_drift=len(drifted) > 0,
            pvalue_threshold=self.pvalue_threshold,
            psi_threshold=self.psi_threshold,
            feature_results=feature_results,
            missing_in_new=missing_in_new,
            new_columns=extra_in_new,
        )

        if report.any_drift:
            log.warning(
                "Drift detected | %d / %d features | critical=%s",
                report.n_features_drifted,
                report.n_features_checked,
                report.critical_features or "none",
            )
        else:
            log.info(
                "No drift detected | %d features checked", report.n_features_checked
            )

        return report


    def _check_feature(
        self,
        col: str,
        new_series: pd.Series,
    ) -> FeatureDriftResult:
        """Run statistical tests on a single feature and return results."""
        ref_series = self._reference[col]

        if _is_continuous(ref_series):
            return self._check_continuous(col, ref_series, new_series)
        else:
            return self._check_categorical(col, ref_series, new_series)

    def _check_continuous(
        self,
        col: str,
        ref: pd.Series,
        new: pd.Series,
    ) -> FeatureDriftResult:
        ref_vals = ref.dropna().values.astype(float)
        new_vals = new.dropna().values.astype(float)

        ks_stat = ks_pvalue = None
        drift_by_ks = False

        if len(ref_vals) >= 5 and len(new_vals) >= 5:
            ks_stat, ks_pvalue = _ks_test(ref_vals, new_vals)
            drift_by_ks = ks_pvalue < self.pvalue_threshold

        psi = _compute_psi(
            ref_vals, new_vals,
            bin_edges=self._bin_edges.get(col),
            n_bins=self.n_psi_bins,
        )
        drift_by_psi = psi >= self.psi_threshold
        drift_detected = drift_by_ks or drift_by_psi
        severity = _severity(psi, drift_by_ks, self.pvalue_threshold, ks_pvalue)

        return FeatureDriftResult(
            feature=col, dtype="continuous",
            ks_statistic=round(ks_stat, 6) if ks_stat is not None else None,
            ks_pvalue=round(ks_pvalue, 6)  if ks_pvalue is not None else None,
            chi2_statistic=None, chi2_pvalue=None,
            psi=round(psi, 6),
            drift_detected=drift_detected,
            severity=severity,
        )

    def _check_categorical(
        self,
        col: str,
        ref: pd.Series,
        new: pd.Series,
    ) -> FeatureDriftResult:
        all_cats = list(self._cat_categories.get(col, ref.dropna().unique()))
        ref_counts = ref.value_counts().reindex(all_cats, fill_value=0).values
        new_counts = new.value_counts().reindex(all_cats, fill_value=0).values

        chi2_stat = chi2_pvalue = None
        drift_by_chi2 = False

        if ref_counts.sum() > 0 and new_counts.sum() > 0:
            chi2_stat, chi2_pvalue = _chi2_test(ref_counts, new_counts)
            if chi2_pvalue is not None:
                drift_by_chi2 = chi2_pvalue < self.pvalue_threshold

        psi = _compute_psi_categorical(ref_counts, new_counts)
        drift_by_psi = psi >= self.psi_threshold
        drift_detected = drift_by_chi2 or drift_by_psi
        severity = _severity(psi, drift_by_chi2, self.pvalue_threshold, chi2_pvalue)

        return FeatureDriftResult(
            feature=col, dtype="categorical",
            ks_statistic=None, ks_pvalue=None,
            chi2_statistic=round(chi2_stat, 6) if chi2_stat is not None else None,
            chi2_pvalue=round(chi2_pvalue, 6)  if chi2_pvalue is not None else None,
            psi=round(psi, 6),
            drift_detected=drift_detected,
            severity=severity,
        )



def _ks_test(ref: np.ndarray, new: np.ndarray) -> Tuple[float, float]:
    """Two-sample KS test. Returns (statistic, p-value)."""
    try:
        from scipy.stats import ks_2samp
        result = ks_2samp(ref, new)
        return float(result.statistic), float(result.pvalue)
    except ImportError:
        # scipy not available — fall back to approximate KS via max CDF difference
        stat = _ks_statistic_numpy(ref, new)
        n = (len(ref) * len(new)) / (len(ref) + len(new))
        # Kolmogorov distribution approximation
        pvalue = max(0.0, 2 * math.exp(-2 * n * stat ** 2))
        return stat, pvalue


def _ks_statistic_numpy(ref: np.ndarray, new: np.ndarray) -> float:
    """Pure numpy approximate KS statistic."""
    combined = np.sort(np.concatenate([ref, new]))
    cdf_ref = np.searchsorted(np.sort(ref), combined, side="right") / len(ref)
    cdf_new = np.searchsorted(np.sort(new), combined, side="right") / len(new)
    return float(np.max(np.abs(cdf_ref - cdf_new)))


def _chi2_test(
    ref_counts: np.ndarray,
    new_counts: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    """Chi-squared test on frequency counts. Returns (statistic, p-value)."""
    try:
        from scipy.stats import chi2_contingency
        table = np.array([ref_counts, new_counts])
        # Avoid zero-column marginals
        mask = (table.sum(axis=0) > 0)
        if mask.sum() < 2:
            return None, None
        chi2, pvalue, _, _ = chi2_contingency(table[:, mask])
        return float(chi2), float(pvalue)
    except ImportError:
        return None, None
    except Exception:
        return None, None


def _compute_psi(
    ref: np.ndarray,
    new: np.ndarray,
    bin_edges: Optional[np.ndarray],
    n_bins: int = N_PSI_BINS,
) -> float:
    """
    Compute PSI for a continuous feature.

    PSI = Σ (actual% - expected%) × ln(actual% / expected%)
    """
    if len(ref) == 0 or len(new) == 0:
        return 0.0

    if bin_edges is None:
        bin_edges = np.nanpercentile(ref, np.linspace(0, 100, n_bins + 1))
        bin_edges[0]  = -np.inf
        bin_edges[-1] =  np.inf

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    new_counts, _ = np.histogram(new, bins=bin_edges)

    return _psi_from_counts(ref_counts, new_counts)


def _compute_psi_categorical(
    ref_counts: np.ndarray,
    new_counts: np.ndarray,
) -> float:
    """PSI for categorical features using raw counts per category."""
    return _psi_from_counts(ref_counts, new_counts)


def _psi_from_counts(
    ref_counts: np.ndarray,
    new_counts: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Core PSI computation from bin/category counts."""
    ref_total = ref_counts.sum()
    new_total = new_counts.sum()
    if ref_total == 0 or new_total == 0:
        return 0.0

    ref_pct = (ref_counts / ref_total) + eps
    new_pct = (new_counts / new_total) + eps

    psi_values = (new_pct - ref_pct) * np.log(new_pct / ref_pct)
    return float(np.sum(psi_values))


def _severity(
    psi: float,
    stat_drift: bool,
    pvalue_threshold: float,
    pvalue: Optional[float],
) -> str:
    """
    Map PSI + statistical test result to a severity label.
    critical > moderate > none
    """
    if psi >= PSI_CRITICAL:
        return "critical"
    if psi >= PSI_MODERATE:
        return "moderate"
    if stat_drift and pvalue is not None and pvalue < pvalue_threshold / 10:
        return "critical"
    if stat_drift:
        return "moderate"
    return "none"


def _is_continuous(series: pd.Series) -> bool:
    """Return True if the series should be treated as continuous."""
    return pd.api.types.is_numeric_dtype(series) and series.nunique() > 10
