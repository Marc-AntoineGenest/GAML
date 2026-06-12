"""
SHAPExplainer — compute SHAP feature attributions for the final GAML model.

Why SHAP?
---------
Tree-based models (LGBM, XGBoost, GBM, RF) are not inherently interpretable at
the individual-prediction level. SHAP (SHapley Additive exPlanations) assigns
each feature a contribution score for each prediction, with a solid game-theoretic
foundation. The mean |SHAP| value per feature is the most actionable global
summary: it measures average impact on model output magnitude.

Design
------
- Uses shap.TreeExplainer for all tree models (exact, fast, no sampling needed).
- Normalises the raw SHAP output across all return shapes:
    * list of 2 arrays  — binary classification (LGBM)  → take index [1]
    * 3-D array (N,F,2) — binary classification (RF)    → take [:,:,1]
    * 2-D array (N,F)   — regression or GBM             → use as-is
- Returns a plain dict so html_reporter has no shap dependency.
- Generates a self-contained inline SVG bar chart (no JS, no CDN) for
  embedding directly in the HTML report.
- Gracefully returns None if shap is not installed or the model is unsupported.

Usage
-----
Called internally by AutoMLPipeline.fit(); not intended for direct use, but
fully standalone::

    from genetic_automl.reporting.shap_explainer import SHAPExplainer
    result = SHAPExplainer(max_samples=100).explain(model, X_preprocessed)
    if result:
        print(result["feature_names"])
        print(result["mean_abs_shap"])
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from genetic_automl.utils.logger import get_logger

log = get_logger(__name__)

# Max features shown in the bar chart (keep the SVG readable)
_MAX_CHART_FEATURES = 20


class SHAPExplainer:
    """
    Compute global SHAP importances for the fitted final model.

    Parameters
    ----------
    max_samples : int
        Cap on background rows passed to TreeExplainer. Larger values are
        more accurate but slower. 200 is a good default; use 500 for
        production runs.
    """

    def __init__(self, max_samples: int = 200) -> None:
        self.max_samples = max_samples


    def explain(
        self,
        model: Any,
        X: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Run SHAP TreeExplainer on *model* using a sample of *X*.

        Parameters
        ----------
        model :
            A fitted GAML backend model (LGBMModel, XGBModel, SklearnModel,
            RFModel) or the underlying sklearn/lgbm/xgb estimator directly.
        X : pd.DataFrame
            Preprocessed feature matrix (same space the model was trained on).
        feature_names : list[str] | None
            Column names. Inferred from X.columns if not provided.

        Returns
        -------
        dict with keys:
            feature_names   : list[str]
            mean_abs_shap   : list[float]   (same order as feature_names)
            base_value      : float
            n_samples_used  : int
            shap_svg        : str           (inline SVG bar chart)
        Returns None on any failure (shap not installed, unsupported model, …).
        """
        try:
            import shap  # noqa: F401
        except ImportError:
            log.warning(
                "shap is not installed — skipping SHAP explainability. "
                "Install it with:  pip install shap"
            )
            return None

        # Resolve the underlying estimator (GAML wrappers expose ._estimator)
        estimator = self._resolve_estimator(model)
        if estimator is None:
            log.warning("SHAPExplainer: could not resolve underlying estimator from %r.", type(model).__name__)
            return None

        names = feature_names or (list(X.columns) if hasattr(X, "columns") else [f"f{i}" for i in range(X.shape[1])])

        # Sample rows for speed
        n = min(self.max_samples, len(X))
        X_sample = X.iloc[:n].reset_index(drop=True) if hasattr(X, "iloc") else X[:n]

        try:
            mean_abs, base_value = self._compute_shap(estimator, X_sample)
        except Exception as exc:
            log.warning("SHAPExplainer: TreeExplainer failed (%s). Falling back to feature_importances_.", exc)
            mean_abs, base_value = self._fallback_importances(estimator, len(names))
            if mean_abs is None:
                return None

        if len(mean_abs) != len(names):
            log.warning(
                "SHAPExplainer: SHAP output length (%d) != n_features (%d). Skipping.",
                len(mean_abs), len(names),
            )
            return None

        # Sort descending for display
        order = np.argsort(mean_abs)[::-1]
        sorted_names = [names[i] for i in order]
        sorted_vals  = [float(mean_abs[i]) for i in order]

        result = {
            "feature_names":  sorted_names,
            "mean_abs_shap":  sorted_vals,
            "base_value":     float(base_value),
            "n_samples_used": int(n),
            "shap_svg":       _build_shap_svg(sorted_names, sorted_vals),
        }
        log.info(
            "SHAP computed | n_samples=%d | top feature=%s (%.4f) | base_value=%.4f",
            n, sorted_names[0] if sorted_names else "n/a",
            sorted_vals[0] if sorted_vals else 0.0,
            base_value,
        )
        return result


    @staticmethod
    def _resolve_estimator(model: Any) -> Any:
        """Unwrap GAML backend wrappers to get the raw sklearn/lgbm/xgb object."""
        # GAML wrappers store the raw estimator as ._estimator
        if hasattr(model, "_estimator"):
            return model._estimator
        # EnsembleModel: use its first (best) member
        if hasattr(model, "members") and model.members:
            first = model.members[0]
            return first._estimator if hasattr(first, "_estimator") else first
        # Already a raw estimator (e.g. in tests)
        return model

    @staticmethod
    def _compute_shap(estimator: Any, X_sample) -> tuple[np.ndarray, float]:
        """
        Run shap.TreeExplainer and normalise the output to (mean_abs_shap, base_value).

        Handles all return shapes produced by sklearn/lgbm/xgb tree models.
        """
        import shap

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.TreeExplainer(estimator)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sv = explainer.shap_values(X_sample)

        ev = explainer.expected_value

        # Case 1: list of arrays — lgbm binary classification → [neg_class, pos_class]
        if isinstance(sv, list):
            arr = np.array(sv[1] if len(sv) == 2 else sv[0])
            base = float(ev[1] if hasattr(ev, "__len__") and len(ev) == 2 else ev)
        else:
            arr = np.array(sv)
            # Case 2: 3-D (N, F, n_classes) — RF binary
            if arr.ndim == 3:
                arr = arr[:, :, 1]
                base = float(ev[1] if hasattr(ev, "__len__") and len(ev) > 1 else ev)
            else:
                # Case 3: plain 2-D (N, F) — regression, GBM
                base = float(ev[0] if hasattr(ev, "__len__") else ev)

        mean_abs = np.abs(arr).mean(axis=0)
        return mean_abs, base

    @staticmethod
    def _fallback_importances(estimator: Any, n_features: int) -> tuple:
        """Use raw feature_importances_ when TreeExplainer fails."""
        fi = getattr(estimator, "feature_importances_", None)
        if fi is None:
            return None, None
        fi = np.array(fi, dtype=float)
        total = fi.sum()
        if total > 0:
            fi = fi / total   # normalise to [0,1] so scale is comparable to SHAP
        return fi, 0.0



def _build_shap_svg(
    feature_names: List[str],
    mean_abs_shap: List[float],
    max_features: int = _MAX_CHART_FEATURES,
    width: int = 700,
) -> str:
    """
    Build a self-contained inline SVG horizontal bar chart.

    Shows the top-N features by mean |SHAP| value. The chart uses the same
    dark-mode colour palette as the rest of the GAML HTML report.

    Parameters
    ----------
    feature_names : list[str]   sorted descending by mean_abs_shap
    mean_abs_shap : list[float] sorted descending
    max_features  : int         cap on bars shown (default 20)
    width         : int         total SVG width in px

    Returns
    -------
    str : a complete <svg> element ready to embed in HTML.
    """
    names = feature_names[:max_features]
    vals  = mean_abs_shap[:max_features]
    n     = len(names)

    if n == 0:
        return "<svg viewBox='0 0 100 20' xmlns='http://www.w3.org/2000/svg'></svg>"

    bar_h   = 22          # height of each bar
    gap     = 6           # vertical gap between bars
    lbl_w   = 160         # left label column width
    val_w   = 55          # right value annotation column width
    pad_top = 40          # space for title
    pad_bot = 20
    bar_area_w = width - lbl_w - val_w - 20
    total_h    = pad_top + n * (bar_h + gap) + pad_bot

    max_val = max(vals) if max(vals) > 0 else 1.0

    # Colour gradient: teal (#2dd4bf) for top → blue (#3b82f6) for bottom
    def bar_colour(rank: int) -> str:
        t = rank / max(n - 1, 1)
        r = int(45  + t * (59  - 45))
        g = int(212 + t * (130 - 212))
        b = int(191 + t * (246 - 191))
        return f"rgb({r},{g},{b})"

    rows = []
    for i, (name, val) in enumerate(zip(names, vals)):
        y     = pad_top + i * (bar_h + gap)
        bw    = max(2, int(val / max_val * bar_area_w))
        col   = bar_colour(i)
        # Truncate long feature names
        display_name = (name[:22] + "…") if len(name) > 23 else name
        rows.append(f"""
  <text x="{lbl_w - 8}" y="{y + bar_h * 0.72:.0f}" text-anchor="end"
        font-family="monospace" font-size="11" fill="#94a3b8">{display_name}</text>
  <rect x="{lbl_w}" y="{y}" width="{bw}" height="{bar_h}" rx="3"
        fill="{col}" opacity="0.85"/>
  <text x="{lbl_w + bw + 6}" y="{y + bar_h * 0.72:.0f}"
        font-family="sans-serif" font-size="10" fill="#cbd5e1">{val:.4f}</text>""")

    return f"""<svg viewBox="0 0 {width} {total_h}" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:{width}px;background:transparent">
  <text x="{width // 2}" y="22" text-anchor="middle"
        font-family="sans-serif" font-size="13" font-weight="600"
        fill="#e2e8f0">Mean |SHAP| Value (feature impact on model output)</text>
  {"".join(rows)}
</svg>"""
