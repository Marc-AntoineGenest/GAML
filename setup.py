from setuptools import setup, find_packages

setup(
    name="genetic_automl",
    version="0.3.0",
    description="Genetic Algorithm-driven AutoML pipeline for tabular data",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "numpy>=1.24",
    ],
    extras_require={
        "autogluon": ["autogluon.tabular>=1.0"],
        "imbalanced": ["imbalanced-learn>=0.12"],
        "reporting": ["mlflow>=2.10"],
        "dev": ["pytest>=8.0", "pytest-cov"],
    },
)
