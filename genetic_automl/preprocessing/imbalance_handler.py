"""
ImbalanceHandler
----------------
Addresses class imbalance in classification tasks.

Applied ONLY to training data — never to validation or test sets.
For regression tasks this step is a no-op.

Methods
-------
none              : no-op
smote             : SMOTE oversampling (requires imbalanced-learn)
borderline_smote  : BorderlineSMOTE (requires imbalanced-learn)
adasyn            : ADASYN oversampling (requires imbalanced-learn)
class_weight      : does NOT resample — returns sample_weights for the model
                    (always available, no imbalanced-learn needed)

When imbalanced-learn is not installed, 'smote', 'borderline_smote', and
'adasyn' automatically fall back to 'class_weight' with a warning.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)

_IMBLEARN_METHODS = {"smote", "borderline_smote", "adasyn"}


class ImbalanceHandler:
    """
    Parameters
    ----------
    method : str
        'none' | 'smote' | 'borderline_smote' | 'adasyn' | 'class_weight'
    random_seed : int
    k_neighbors : int
        Number of nearest neighbors used by SMOTE variants.
    """

    def __init__(
        self,
        method: str = "none",
        random_seed: int = 42,
        k_neighbors: int = 5,
    ) -> None:
        self.method = method
        self.random_seed = random_seed
        self.k_neighbors = k_neighbors

        self._resampler = None
        self._class_weights: Optional[dict] = None
        self._effective_method = method  # may be changed to fallback

    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ImbalanceHandler":
        """Learn class distribution. For SMOTE methods, no fitting is needed."""
        if self.method == "none":
            return self

        self._class_weights = self._compute_class_weights(y)

        if self.method in _IMBLEARN_METHODS:
            if not self._imblearn_available():
                log.warning(
                    "imbalanced-learn not installed. "
                    "Falling back from '%s' to 'class_weight'.",
                    self.method,
                )
                self._effective_method = "class_weight"
            else:
                # Guard: k_neighbors must be < smallest minority class count.
                # imblearn uses k_neighbors+1 internally, so the effective limit
                # is minority_count - 1.
                min_class_count = int(y.value_counts().min())
                safe_k = min(self.k_neighbors, min_class_count - 1)
                if safe_k < self.k_neighbors:
                    log.warning(
                        "ImbalanceHandler('%s'): minority class has only %d sample(s); "
                        "auto-reducing k_neighbors from %d → %d to avoid crash. "
                        "Consider collecting more data for the minority class.",
                        self.method, min_class_count, self.k_neighbors, safe_k,
                    )
                    self.k_neighbors = safe_k
                if safe_k < 1:
                    log.warning(
                        "ImbalanceHandler('%s'): minority class is too small (count=%d) "
                        "for any SMOTE variant. Falling back to 'class_weight'.",
                        self.method, min_class_count,
                    )
                    self._effective_method = "class_weight"
                else:
                    self._resampler = self._build_resampler()
                    self._effective_method = self.method

        log.info(
            "ImbalanceHandler(method=%s): class distribution %s",
            self._effective_method,
            dict(y.value_counts().sort_index()),
        )
        return self

    def fit_resample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and apply resampling to TRAINING DATA ONLY.

        Returns (X_resampled, y_resampled).
        For 'class_weight' and 'none', returns the original data unchanged
        (caller should use .sample_weights to pass weights to the model).
        """
        self.fit(X, y)

        if self._effective_method in _IMBLEARN_METHODS and self._resampler is not None:
            try:
                X_res, y_res = self._resampler.fit_resample(X, y)
                X_res = pd.DataFrame(X_res, columns=X.columns)
                y_res = pd.Series(y_res, name=y.name)
                log.info(
                    "ImbalanceHandler: %d → %d samples after %s",
                    len(y),
                    len(y_res),
                    self._effective_method,
                )
                return X_res, y_res
            except Exception as e:
                log.warning("Resampling failed (%s), using original data.", e)

        return X, y

    def sample_weights(self, y: pd.Series) -> Optional[np.ndarray]:
        """
        Return per-sample weights if method='class_weight', else None.
        Pass these to the model's fit() if supported.
        """
        if self._effective_method != "class_weight" or self._class_weights is None:
            return None
        return np.array([self._class_weights.get(label, 1.0) for label in y])

    # ------------------------------------------------------------------

    def _compute_class_weights(self, y: pd.Series) -> dict:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        return dict(zip(classes, weights))

    def _build_resampler(self):
        if self.method == "smote":
            from imblearn.over_sampling import SMOTE
            return SMOTE(random_state=self.random_seed, k_neighbors=self.k_neighbors)
        if self.method == "borderline_smote":
            from imblearn.over_sampling import BorderlineSMOTE
            return BorderlineSMOTE(random_state=self.random_seed, k_neighbors=self.k_neighbors)
        if self.method == "adasyn":
            from imblearn.over_sampling import ADASYN
            return ADASYN(random_state=self.random_seed, n_neighbors=self.k_neighbors)
        raise ValueError(f"Unknown imbalance method: {self.method}")

    @staticmethod
    def _imblearn_available() -> bool:
        try:
            import imblearn  # noqa
            return True
        except ImportError:
            return False
