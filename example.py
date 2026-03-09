"""
Quick-start example using the sklearn backend (no AutoGluon install needed).

To use AutoGluon instead, change:
    automl=AutoMLConfig(backend="autogluon", ...)
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes

from genetic_automl import AutoMLPipeline, PipelineConfig, GeneticConfig, AutoMLConfig, ReportConfig
from genetic_automl.core.problem import ProblemType


# ── Classification example ──────────────────────────────────────────────────

def run_classification():
    print("\n" + "=" * 60)
    print("CLASSIFICATION — Breast Cancer Dataset")
    print("=" * 60)

    data = load_breast_cancer(as_frame=True)
    df = data.frame
    df = df.rename(columns={"target": "label"})

    config = PipelineConfig(
        problem_type=ProblemType.CLASSIFICATION,
        target_column="label",
        run_name="breast_cancer_clf",
        genetic=GeneticConfig(
            population_size=6,
            generations=3,
            mutation_rate=0.3,
            early_stopping_rounds=2,
            random_seed=42,
        ),
        automl=AutoMLConfig(backend="sklearn"),
        report=ReportConfig(output_dir="reports", open_html_on_finish=False),
    )

    pipeline = AutoMLPipeline(config)
    pipeline.fit(df)
    print(f"\n✅ Final F1-macro: {pipeline.final_score:.4f}")
    print(f"📄 Report: {pipeline.report_path}")
    return pipeline


# ── Regression example ───────────────────────────────────────────────────────

def run_regression():
    print("\n" + "=" * 60)
    print("REGRESSION — Diabetes Dataset")
    print("=" * 60)

    data = load_diabetes(as_frame=True)
    df = data.frame  # target column is already named "target"

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
    print(f"\n✅ Final MSE: {pipeline.final_score:.4f}")
    print(f"📄 Report: {pipeline.report_path}")
    return pipeline


if __name__ == "__main__":
    clf_pipeline = run_classification()
    reg_pipeline = run_regression()
    print("\n🎉 Both runs complete. Check the 'reports/' directory.")
