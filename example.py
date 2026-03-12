"""
GAML quick-start examples.

Two usage patterns are shown side-by-side:

  A) YAML-driven (recommended)
     Edit gaml_config.yaml, then call load_config() + AutoMLPipeline.
     No Python changes needed to tune the run.

  B) Code-only
     Construct PipelineConfig directly in Python.
     Useful when config must be built programmatically.

Both patterns produce identical results -- choose whichever fits your workflow.
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

from genetic_automl import (
    AutoMLPipeline, PipelineConfig, GeneticConfig,
    AutoMLConfig, ReportConfig, load_config,
)
from genetic_automl.core.problem import ProblemType


# =============================================================================
# A) YAML-driven (recommended)
# =============================================================================

def run_yaml_driven():
    """Load all settings from gaml_config.yaml and run."""
    print("\n" + "=" * 60)
    print("YAML-DRIVEN -- Breast Cancer (edit gaml_config.yaml to tune)")
    print("=" * 60)

    data = load_breast_cancer(as_frame=True)
    df = data.frame.rename(columns={"target": "label"})

    # Single entry point: reads gaml_config.yaml from the current directory.
    # Edit that file to change backend, population size, search spaces, etc.
    config, gene_overrides = load_config("gaml_config.yaml")
    config.target_column = "label"   # match this dataset's label column

    pipeline = AutoMLPipeline(config, gene_space_overrides=gene_overrides)
    pipeline.fit(df)

    print(f"\n  Final {pipeline._metric_name}: {pipeline.final_score:.4f}")
    print(f"  Report : {pipeline.report_path}")

    # --- Save / Load demo ---
    # Persist the fitted pipeline so inference can run without re-fitting.
    save_path = pipeline.save("saved_models/breast_cancer.joblib")
    print(f"  Saved  : {save_path}")

    loaded = AutoMLPipeline.load(save_path)
    preds_original = pipeline.predict(df.drop(columns=["label"]))
    preds_loaded   = loaded.predict(df.drop(columns=["label"]))
    assert (preds_original == preds_loaded).all(), "Save/load round-trip mismatch!"
    print("  Save/load round-trip: OK")

    return pipeline


# =============================================================================
# B) Code-only (programmatic config)
# =============================================================================

def run_code_only():
    """Construct PipelineConfig entirely in Python -- no YAML file needed."""
    print("\n" + "=" * 60)
    print("CODE-ONLY  -- Diabetes Regression")
    print("=" * 60)

    data = load_diabetes(as_frame=True)
    df = data.frame   # target column is already named "target"

    config = PipelineConfig(
        problem_type=ProblemType.REGRESSION,
        target_column="target",
        run_name="diabetes_reg",
        genetic=GeneticConfig(
            population_size=6,
            generations=3,
            mutation_rate=0.3,
            early_stopping_rounds=2,
            random_seed=7,
        ),
        automl=AutoMLConfig(backend="sklearn"),
        report=ReportConfig(output_dir="reports"),
    )

    pipeline = AutoMLPipeline(config)
    pipeline.fit(df)

    print(f"\n  Final MSE : {pipeline.final_score:.4f}")
    print(f"  Report    : {pipeline.report_path}")
    return pipeline


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    yaml_pipeline = run_yaml_driven()
    code_pipeline = run_code_only()
    print("\nBoth runs complete. Check the 'reports/' directory.")
