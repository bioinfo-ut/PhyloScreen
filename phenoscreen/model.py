from datetime import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phenoscreen.features import FeatureExtractor, FEATURE_NAMES
from phenoscreen.utils import logger, ensure_dir


@dataclass
class TrainingResult:
    """Results from model training."""
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    feature_importances: dict[str, float]
    confusion_matrix: Optional[np.ndarray] = None


@dataclass
class PredictionResult:
    """Result for a single prediction."""

    query_id: str
    prediction: str
    probability: float
    probabilities: dict[str, float]
    top_hit_resistant: Optional[str] = None
    top_hit_susceptible: Optional[str] = None


class PhenotypeModel:
    """
    Logistic regression model for phenotype prediction.

    Uses a pipeline with StandardScaler and LogisticRegressionCV.
    LogisticRegressionCV handles regularization tuning via internal CV.
    """

    def __init__(self, best_C: Optional[float] = None, seed: Optional[int] = None):
        """
        Initialize the model.
        """
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=best_C,
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=seed
            ))
        ])
        self.classes_: Optional[np.ndarray] = None
        self.feature_names_: Optional[list[str]] = FEATURE_NAMES

    def fit(
            self,
            X: np.ndarray,
            y: np.ndarray,
    ) -> "PhenotypeModel":
        """
        Train the model.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target labels.

        Returns:
            Self for chaining.
        """
        logger.info("Training model with %d samples, %d features", X.shape[0], X.shape[1])
        self.model.fit(X, y)
        self.classes_ = self.model.named_steps["classifier"].classes_
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)

    def get_coefficients(self) -> dict[str, float]:
        """
        Get raw coefficients (with sign) for interpretation.
        """
        classifier = self.model.named_steps["classifier"]
        coefs = classifier.coef_[0]

        names = self.feature_names_ or [
            f"feature_{i}" for i in range(len(coefs))
        ]
        return dict(zip(names, coefs))


@dataclass
class Model:
    """
    Complete model for serialization.
    """

    model: PhenotypeModel
    labels: dict[str, int]

    def save(self, output_dir: Path) -> None:
        """
        Save model to directory.

        Creates:
        - model.joblib: Trained sklearn pipeline (scaler + classifier)
        - coefficients.json: Feature coefficients

        Args:
            output_dir: Directory to save bundle.
        """
        output_dir = ensure_dir(output_dir)

        # Save the full pipeline (includes scaler)
        joblib.dump(self.model.model, output_dir / "model.joblib")

        with open(output_dir / "labels.tsv", "w") as f:
            f.write("path\tphenotype\n")
            for path, phenotype in sorted(self.labels.items()):
                f.write(f"{path}\t{phenotype}\n")

        # Save coefficients for interpretability
        if self.model.classes_ is not None:
            coefs = self.model.get_coefficients()
            with open(output_dir / "coefficients.json", "w") as f:
                json.dump(coefs, f, indent=2)

        logger.info("Model saved to %s", output_dir)

    @classmethod
    def load(cls, model_dir: Path) -> "Model":
        """
        Load model from directory.

        Args:
            model_dir: Directory containing saved model.

        Returns:
            Loaded Model.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        model_dir = Path(model_dir)

        # Load pipeline
        model_path = model_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        sklearn_pipeline = joblib.load(model_path)
        model = PhenotypeModel()
        model.model = sklearn_pipeline
        model.classes_ = sklearn_pipeline.named_steps["classifier"].classes_

        labels = {}
        with open(model_dir / "labels.tsv") as f:
            next(f)  # Skip header
            for line in f:
                path, phenotype = line.strip().split("\t")
                labels[path] = int(phenotype)

        # Load metadata

        logger.info("Model loaded from %s", model_dir)

        return cls(
            model=model,
            labels=labels,
        )
