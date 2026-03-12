# GAML — Genetic AutoML

A genetic algorithm that simultaneously evolves **preprocessing pipelines** and **model hyperparameters** for tabular data. Every combination is scored with cross-validation; the best pipeline is automatically selected and refit on your full dataset.

---

## Installation

```bash
git clone <repo-url>
cd GAML
pip install -e .
```

**Optional extras:**

```bash
pip install -e ".[autogluon]"   # AutoGluon backend (more powerful models)
pip install -e ".[imbalanced]"  # SMOTE / imbalanced-learn sampling
pip install -e ".[reporting]"   # MLflow experiment tracking
pip install pyyaml              # required to use gaml_config.yaml
```

**Run tests:**

```bash
pytest genetic_automl/tests/ -v
```

---

## Quick Start

### Option 1 — Edit the config file (recommended)

Open `gaml_config.yaml` at the project root and set your target column, problem type, and search space. Then run:

```python
import pandas as pd
from genetic_automl import load_config, AutoMLPipeline

df = pd.read_csv("your_data.csv")

config, gene_overrides = load_config("gaml_config.yaml")
pipeline = AutoMLPipeline(config, gene_space_overrides=gene_overrides)
pipeline.fit(df)

print(f"Best score: {pipeline.final_score:.4f}")
print(f"Report:     {pipeline.report_path}")

predictions = pipeline.predict(df)
```

### Option 2 — Pure Python

```python
import pandas as pd
from genetic_automl import AutoMLPipeline, PipelineConfig, GeneticConfig, AutoMLConfig
from genetic_automl.core.problem import ProblemType

config = PipelineConfig(
    problem_type=ProblemType.CLASSIFICATION,
    target_column="label",
    genetic=GeneticConfig(
        population_size=20,
        generations=15,
        n_cv_folds=3,
    ),
    automl=AutoMLConfig(backend="sklearn"),
)

pipeline = AutoMLPipeline(config)
pipeline.fit(df)
predictions = pipeline.predict(new_df)
```

---

## Configuration (`gaml_config.yaml`)

`gaml_config.yaml` at the project root is the single place to control everything. It has five sections:

### A — Run settings

```yaml
run:
  problem_type: classification   # classification | regression
  target_column: target          # column name in your DataFrame
  backend: sklearn               # sklearn | autogluon
  metric: null                   # null = default (f1_macro / mse)
```

### B — Data settings

```yaml
data:
  test_size: 0.15    # locked test set fraction (never seen by GA)
  val_size: 0.20     # validation fraction for final refit
  stratify: true     # stratified splits (classification only)
```

### C — Genetic algorithm settings

| Parameter | Default | What it controls |
|---|---|---|
| `population_size` | 20 | Pipeline configs evaluated per generation. Larger = broader search, slower. |
| `generations` | 15 | Maximum evolution cycles. |
| `n_cv_folds` | 3 | CV folds per evaluation. Higher = more reliable, slower. |
| `early_stopping_rounds` | 5 | Stop if no improvement for N generations. |
| `mutation_rate` | 0.20 | Probability a gene changes value each reproduction. |
| `crossover_rate` | 0.70 | Probability two parents recombine vs. clone. |
| `elite_ratio` | 0.10 | Fraction of top individuals kept unchanged. |
| `warm_start` | true | Seed generation-0 with known-good configs. |
| `adaptive_mutation` | true | Boost mutation when stagnating, decay on improvement. |
| `diversity_threshold` | 0.15 | Mean Hamming distance below which fresh individuals are injected. |

### D — Search space (what gets optimized)

This is where you tell GAML which parameter values to explore. Each entry is a list of candidates. **Use a single-element list to fix a parameter** and remove it from the search.

```yaml
preprocessing_search_space:
  scaler: [standard]              # fixed — always use standard scaling
  numeric_imputer: [mean, median] # search between mean and median only
  outlier_method: [none]          # disabled — skip outlier detection
```

Full preprocessing search space options:

| Gene | Options | Description |
|---|---|---|
| `numeric_imputer` | `mean` `median` `knn` `iterative` `constant` | Fill missing numeric values |
| `outlier_method` | `none` `iqr` `zscore` `isolation_forest` | Detect and handle outliers |
| `outlier_threshold` | `1.5` `2.0` `3.0` | IQR multiplier or z-score cutoff |
| `outlier_action` | `clip` `flag` | Clamp outliers or add indicator column |
| `correlation_threshold` | `null` `0.85` `0.90` `0.95` | Drop correlated features (`null` = disabled) |
| `categorical_encoder` | `onehot` `ordinal` `target` `binary` | Encode categorical columns |
| `distribution_transform` | `none` `yeo-johnson` `box-cox` `log1p` | Reduce skewness before scaling |
| `scaler` | `none` `standard` `minmax` `robust` | Normalize feature magnitudes |
| `missing_indicator` | `true` `false` | Add binary flags for originally-missing values |
| `feature_selection_method` | `none` `variance_threshold` `mutual_info` `rfe` | Drop low-value features |
| `feature_selection_k` | `0.50` `0.75` `1.0` | Fraction of features to keep |
| `imbalance_method` | `none` `smote` `borderline_smote` `adasyn` `class_weight` | Handle class imbalance |

sklearn model search space:

| Gene | Options | Description |
|---|---|---|
| `n_estimators` | `50` `100` `200` `300` `500` | Number of boosting trees |
| `max_depth` | `2` `3` `4` `5` `6` `8` | Maximum tree depth |
| `learning_rate` | `0.01` `0.05` `0.1` `0.2` | Shrinkage applied per tree |

### E — Reporting

```yaml
report:
  output_dir: reports              # HTML report and JSON export location
  mlflow_tracking_uri: mlflow_runs # MLflow tracking store (null = disabled)
  open_html_on_finish: false       # open browser automatically on completion
```

---

## Outputs

After `pipeline.fit(df)`:

| Output | How to access |
|---|---|
| Best test score | `pipeline.final_score` |
| HTML report path | `pipeline.report_path` |
| Evolution history | `pipeline.history` |
| Best preprocessor | `pipeline.best_preprocessor` |
| Best model | `pipeline.best_model` |
| Predictions | `pipeline.predict(df)` |
| Probabilities | `pipeline.predict_proba(df)` (classification) |

The HTML report includes a generation-by-generation fitness curve, diversity tracking, the best chromosome's gene values, and the final test score.

---

## Preprocessing step order

Steps always execute in this fixed order. GAML evolves **which option** to use at each step, not the order itself.

```
1. NumericImputer         — fill NaN before anything else
2. OutlierHandler         — on clean data, before scaling distorts distances
3. CorrelationFilter      — after imputation, correlation stats are reliable
4. CategoricalEncoder     — encode before scaling
5. DistributionTransform  — reduce skewness before scaling
6. Scaler                 — after all columns are numeric
7. MissingIndicator       — add binary flags for originally-missing columns
8. FeatureSelector        — on fully preprocessed data
9. ImbalanceHandler       — always last, train set only
```

---

## Development

For architecture details, internal design notes, and the improvement roadmap see [`ARCHITECTURE.md`](ARCHITECTURE.md).

```bash
# Run full test suite
pytest genetic_automl/tests/ -v

# Run extended regression tests
pytest genetic_automl/tests/test_extended.py -v
```
