from datetime import datetime, timezone
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phyloscreen import __version__
from phyloscreen.features import FEATURE_NAMES
from phyloscreen.utils import (
    DEFAULT_PHENOTYPE, PATH_COLUMN, logger, ensure_dir, phenotype_columns,
)

FORMAT_VERSION = 2


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
    Logistic regression model for one phenotype.

    Uses a pipeline with StandardScaler and LogisticRegression; the regularisation
    strength C is chosen by the cross-validation in phyloscreen.train.
    """

    def __init__(self, best_C: Optional[float] = None, seed: Optional[int] = None):
        """
        Initialize the model.
        """
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=best_C if best_C is not None else 1.0,
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=seed
            ))
        ])
        self.classes_: Optional[np.ndarray] = None
        self.feature_names_: Optional[list[str]] = FEATURE_NAMES

    @classmethod
    def from_pipeline(cls, pipeline: Pipeline, feature_names: Optional[list[str]] = None) -> "PhenotypeModel":
        """Wrap an already-fitted sklearn pipeline."""
        model = cls()
        model.model = pipeline
        model.classes_ = pipeline.named_steps["classifier"].classes_
        model.feature_names_ = feature_names or FEATURE_NAMES
        return model

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
        return {name: float(c) for name, c in zip(names, coefs)}


@dataclass
class Model:
    """
    Complete model bundle for serialization: one classifier per phenotype plus the
    reference table (with missing labels) they were trained on.
    """

    models: dict[str, PhenotypeModel]
    references: DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def phenotypes(self) -> list[str]:
        """Phenotype names in training order."""
        return list(self.metadata.get("phenotypes") or self.models.keys())

    def save(self, output_dir: Path) -> None:
        """
        Save model to directory.

        Creates:
        - model.joblib: dict phenotype -> trained sklearn pipeline (scaler + classifier)
        - labels.tsv: reference paths and all phenotype columns (blank = missing)
        - coefficients.json: feature coefficients per phenotype
        - metadata.json: format version, phenotype names, feature names, parameters

        Args:
            output_dir: Directory to save bundle.
        """
        output_dir = ensure_dir(output_dir)

        joblib.dump({p: m.model for p, m in self.models.items()}, output_dir / "model.joblib")

        self.references.to_csv(output_dir / "labels.tsv", sep="\t", index=False, na_rep="")

        coefs = {
            p: m.get_coefficients()
            for p, m in self.models.items()
            if m.classes_ is not None
        }
        with open(output_dir / "coefficients.json", "w") as f:
            json.dump(coefs, f, indent=2)

        metadata = {
            "format_version": FORMAT_VERSION,
            "phyloscreen_version": __version__,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "phenotypes": self.phenotypes,
            "feature_names": FEATURE_NAMES,
            **{k: v for k, v in self.metadata.items() if k not in {"format_version", "phenotypes"}},
        }
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        self.metadata = metadata

        logger.info("Model saved to %s (%d phenotype(s))", output_dir, len(self.models))

    @classmethod
    def load(cls, model_dir: Path) -> "Model":
        """
        Load model from directory.

        Supports both the current bundle layout (metadata.json present) and the
        single-phenotype layout written by phyloscreen 1.0.0 (a bare pipeline in
        model.joblib and a two-column labels.tsv), which loads as one phenotype
        named ``phenotype``.

        Args:
            model_dir: Directory containing saved model.

        Returns:
            Loaded Model.

        Raises:
            FileNotFoundError: If required files are missing.
        """
        model_dir = Path(model_dir)

        model_path = model_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        labels_path = model_dir / "labels.tsv"
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        payload = joblib.load(model_path)
        references = pd.read_csv(labels_path, sep="\t")
        metadata_path = model_dir / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            if not isinstance(payload, dict):
                raise ValueError(f"Unexpected model.joblib contents in {model_dir}")
            feature_names = metadata.get("feature_names", FEATURE_NAMES)
            models = {
                p: PhenotypeModel.from_pipeline(pipeline, feature_names)
                for p, pipeline in payload.items()
            }
            phenotypes = metadata.get("phenotypes") or list(models.keys())
        else:
            # Legacy phyloscreen 1.0.0 bundle
            if isinstance(payload, dict):
                raise ValueError(f"metadata.json missing for multi-phenotype bundle in {model_dir}")
            logger.info("Loading legacy single-phenotype bundle from %s", model_dir)
            if DEFAULT_PHENOTYPE not in references.columns:
                raise ValueError(f"Legacy labels.tsv must have a {DEFAULT_PHENOTYPE!r} column")
            models = {DEFAULT_PHENOTYPE: PhenotypeModel.from_pipeline(payload)}
            phenotypes = [DEFAULT_PHENOTYPE]
            metadata = {
                "format_version": 1,
                "phenotypes": phenotypes,
                "feature_names": FEATURE_NAMES,
            }

        missing = [p for p in phenotypes if p not in models]
        if missing:
            raise ValueError(f"Phenotypes listed in metadata but missing from model.joblib: {missing}")
        missing_cols = [p for p in phenotypes if p not in references.columns]
        if missing_cols:
            raise ValueError(f"Phenotypes missing from labels.tsv: {missing_cols}")

        for col in phenotype_columns(references):
            references[col] = references[col].astype("Int64")
        metadata["phenotypes"] = phenotypes

        logger.info("Model loaded from %s (%d phenotype(s))", model_dir, len(models))

        return cls(models=models, references=references, metadata=metadata)
