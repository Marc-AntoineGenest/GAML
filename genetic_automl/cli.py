"""
GAML command-line interface.

Provides the ``gaml`` command installed by setup.py::

    gaml fit data.csv --target label
    gaml fit data.csv --target label --config gaml_config.yaml --save model.joblib
    gaml predict model.joblib new_data.csv --output predictions.csv
    gaml version

Commands
--------
fit
    Load a CSV, run the full GA + final model pipeline, write an HTML report,
    and optionally save the fitted pipeline to disk.

predict
    Load a previously saved pipeline (.joblib) and run inference on a CSV,
    writing predictions to a new CSV file.

version
    Print the installed GAML version and exit.

Design decisions
----------------
- Pure stdlib argparse — no click, no typer dependency.
- Every tuneable value available in gaml_config.yaml is also a CLI flag,
  so the tool is usable without any config file.
- Explicit CLI flags always override the YAML config file when both are
  supplied, following the principle of least surprise.
- All output (progress, errors) goes to stderr; only the final score line
  goes to stdout so it can be piped / captured cleanly.
- Non-zero exit codes on error: 1 = user error, 2 = runtime error.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Top-level imports (needed here for patch()-ability in tests)
# ---------------------------------------------------------------------------
from genetic_automl import AutoMLPipeline, load_config
from genetic_automl.config import (
    AutoMLConfig, DataConfig, GeneticConfig, PipelineConfig, ReportConfig,
)
from genetic_automl.core.problem import ProblemType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _err(msg: str, code: int = 1) -> None:
    """Print an error to stderr and exit."""
    print(f"[gaml] ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def _info(msg: str) -> None:
    print(f"[gaml] {msg}", file=sys.stderr)


def _require_file(path: str, label: str) -> Path:
    p = Path(path)
    if not p.exists():
        _err(f"{label} not found: {p}")
    return p


def _load_dataframe(csv_path: Path):
    """Read a CSV into a pandas DataFrame with helpful error messages."""
    try:
        import pandas as pd
        return pd.read_csv(csv_path)
    except Exception as exc:
        _err(f"Could not read CSV '{csv_path}': {exc}", code=2)


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

def _cmd_fit(args: argparse.Namespace) -> int:
    """Run the full GAML pipeline on a CSV file."""
    import pandas as pd

    csv_path = _require_file(args.data, "Data file")
    _info(f"Loading data from '{csv_path}' ...")
    df = _load_dataframe(csv_path)
    _info(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    # -----------------------------------------------------------------------
    # Build config: start from YAML (if given), then apply CLI overrides.
    # -----------------------------------------------------------------------
    if args.config:
        config_path = _require_file(args.config, "Config file")
        _info(f"Loading config from '{config_path}' ...")
        config, gene_overrides = load_config(str(config_path))
    else:
        config = PipelineConfig()
        gene_overrides = {}

    # --- CLI overrides (always win over YAML) ---
    if args.target:
        config.target_column = args.target

    if not config.target_column or config.target_column == "target":
        if args.target is None:
            _err(
                "No target column specified. "
                "Use --target <column_name> or set target_column in your config file."
            )

    if args.target not in df.columns:
        _err(
            f"Target column '{args.target}' not found in the data. "
            f"Available columns: {list(df.columns)}"
        )

    if args.problem:
        mapping = {
            "classification": "CLASSIFICATION",
            "regression":     "REGRESSION",
        }
        key = args.problem.lower()
        if key not in mapping:
            _err(f"--problem must be 'classification' or 'regression', got '{args.problem}'")
        config.problem_type = ProblemType[mapping[key]]

    if args.backend:
        config.automl.backend = args.backend.lower()

    if args.generations is not None:
        config.genetic.generations = args.generations

    if args.population is not None:
        config.genetic.population_size = args.population

    if args.cv_folds is not None:
        config.genetic.n_cv_folds = args.cv_folds

    if args.seed is not None:
        config.genetic.random_seed = args.seed

    if args.output_dir:
        config.report.output_dir = args.output_dir

    if args.no_shap:
        config.report.shap_enabled = False

    if args.run_name:
        config.run_name = args.run_name

    # -----------------------------------------------------------------------
    # Run
    # -----------------------------------------------------------------------
    _info(
        f"Starting GA run | problem={config.problem_type.value} | "
        f"backend={config.automl.backend} | "
        f"target='{config.target_column}' | "
        f"pop={config.genetic.population_size} | "
        f"gens={config.genetic.generations}"
    )

    try:
        pipeline = AutoMLPipeline(config, gene_space_overrides=gene_overrides)
        pipeline.fit(df)
    except Exception as exc:
        _err(f"Pipeline failed: {exc}", code=2)

    metric = pipeline._metric_name
    score  = pipeline.final_score
    report = pipeline.report_path

    # Final score goes to stdout (pipeable), everything else to stderr.
    print(f"{metric}={score:.6f}")
    _info(f"Final {metric}: {score:.6f}")
    if report:
        _info(f"Report  : {report}")

    if args.save:
        save_path = pipeline.save(args.save)
        _info(f"Pipeline saved: {save_path}")

    return 0


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def _cmd_predict(args: argparse.Namespace) -> int:
    """Load a saved pipeline and run inference on a CSV."""
    import numpy as np
    import pandas as pd

    model_path = _require_file(args.model, "Model file")
    csv_path   = _require_file(args.data,  "Data file")

    _info(f"Loading pipeline from '{model_path}' ...")
    try:
        pipeline = AutoMLPipeline.load(str(model_path))
    except Exception as exc:
        _err(f"Could not load pipeline: {exc}", code=2)

    _info(f"Loading data from '{csv_path}' ...")
    df = _load_dataframe(csv_path)

    # Drop target column if it accidentally ended up in the predict CSV
    # Always read from config; pipeline._target_column is not a public attribute.
    try:
        target = pipeline.config.target_column
    except Exception:
        target = None
    if target and target in df.columns:
        _info(f"  Dropping target column '{target}' from predict input.")
        df = df.drop(columns=[target])

    _info(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")

    try:
        preds = pipeline.predict(df)
    except Exception as exc:
        _err(f"Prediction failed: {exc}", code=2)

    out_df = df.copy()
    out_df["prediction"] = preds

    # Optionally append probabilities for classification
    try:
        proba = pipeline.predict_proba(df)
        if proba is not None:
            if proba.ndim == 2:
                for i in range(proba.shape[1]):
                    out_df[f"proba_class_{i}"] = proba[:, i]
            else:
                out_df["proba"] = proba
    except Exception:
        pass  # probabilities are best-effort

    out_path = args.output or str(csv_path.with_suffix("")) + "_predictions.csv"
    out_df.to_csv(out_path, index=False)

    _info(f"Predictions written to '{out_path}' ({len(preds):,} rows).")
    print(out_path)
    return 0


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------

def _cmd_version(args: argparse.Namespace) -> int:
    from genetic_automl import __version__
    print(f"gaml {__version__}")
    return 0


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gaml",
        description=textwrap.dedent("""\
            Genetic AutoML — train and deploy ML pipelines from the command line.

            Examples:
              gaml fit data.csv --target label
              gaml fit data.csv --target label --config gaml_config.yaml --save model.joblib
              gaml predict model.joblib new_data.csv --output predictions.csv
              gaml version
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # -----------------------------------------------------------------------
    # fit
    # -----------------------------------------------------------------------
    fit_p = sub.add_parser(
        "fit",
        help="Train a GAML pipeline on a CSV file.",
        description="Run the full Genetic Algorithm + final model pipeline on a CSV dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    fit_p.add_argument(
        "data",
        metavar="DATA_CSV",
        help="Path to the input CSV file (must contain a header row).",
    )
    fit_p.add_argument(
        "--target", "-t",
        metavar="COLUMN",
        default=None,
        help="Name of the target column to predict. Required if not set in --config.",
    )
    fit_p.add_argument(
        "--config", "-c",
        metavar="YAML_PATH",
        default=None,
        help="Path to a gaml_config.yaml file. CLI flags override config values.",
    )
    fit_p.add_argument(
        "--problem", "-p",
        metavar="TYPE",
        default=None,
        choices=["classification", "regression"],
        help="Problem type: 'classification' (default) or 'regression'.",
    )
    fit_p.add_argument(
        "--backend", "-b",
        metavar="BACKEND",
        default=None,
        choices=["sklearn", "autogluon"],
        help="ML backend: 'sklearn' (default) or 'autogluon'.",
    )
    fit_p.add_argument(
        "--generations", "-g",
        metavar="N",
        type=int,
        default=None,
        help="Number of GA generations.",
    )
    fit_p.add_argument(
        "--population", "-P",
        metavar="N",
        type=int,
        default=None,
        help="GA population size per generation.",
    )
    fit_p.add_argument(
        "--cv-folds",
        metavar="N",
        type=int,
        default=None,
        dest="cv_folds",
        help="Number of cross-validation folds for fitness evaluation.",
    )
    fit_p.add_argument(
        "--seed",
        metavar="INT",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    fit_p.add_argument(
        "--output-dir", "-o",
        metavar="DIR",
        default=None,
        dest="output_dir",
        help="Directory for the HTML report and JSON run summary (default: reports/).",
    )
    fit_p.add_argument(
        "--save", "-s",
        metavar="PATH",
        default=None,
        help="Save the fitted pipeline to this .joblib path after training.",
    )
    fit_p.add_argument(
        "--run-name",
        metavar="NAME",
        default=None,
        dest="run_name",
        help="Human-readable run name shown in the HTML report.",
    )
    fit_p.add_argument(
        "--no-shap",
        action="store_true",
        dest="no_shap",
        help="Disable SHAP feature attribution in the report (faster).",
    )
    fit_p.set_defaults(func=_cmd_fit)

    # -----------------------------------------------------------------------
    # predict
    # -----------------------------------------------------------------------
    pred_p = sub.add_parser(
        "predict",
        help="Run inference with a saved GAML pipeline.",
        description="Load a saved pipeline (.joblib) and predict on a new CSV file.",
    )
    pred_p.add_argument(
        "model",
        metavar="MODEL_JOBLIB",
        help="Path to a saved pipeline file produced by 'gaml fit --save'.",
    )
    pred_p.add_argument(
        "data",
        metavar="DATA_CSV",
        help="Path to the input CSV file for prediction.",
    )
    pred_p.add_argument(
        "--output", "-o",
        metavar="PATH",
        default=None,
        help="Output CSV path. Defaults to <data>_predictions.csv.",
    )
    pred_p.set_defaults(func=_cmd_predict)

    # -----------------------------------------------------------------------
    # version
    # -----------------------------------------------------------------------
    ver_p = sub.add_parser("version", help="Print the GAML version and exit.")
    ver_p.set_defaults(func=_cmd_version)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """
    Main CLI entry point.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to sys.argv[1:] when None).
        Pass an explicit list in tests to avoid touching sys.argv.

    Returns
    -------
    int
        Exit code: 0 = success, 1 = user error, 2 = runtime error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
