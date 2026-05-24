"""
Training pipeline for phenoscreen.

Orchestrates the full training workflow:
1. Validate inputs
2. Build mash sketch from references
3. Generate training features
4. Train and cross-validate model
5. Save model bundle
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from phenoscreen.features import FeatureExtractor
from phenoscreen.mash import check_mash, sketch, screen
from phenoscreen.model import TrainingResult, PhenotypeModel, Model
from phenoscreen.utils import (
    logger, validate_references, ensure_dir,
)


@dataclass
class TrainResult:
    """Result of the training pipeline."""

    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float
    n_references: int
    n_features: int
    output_dir: Path


def train_model(
    references: Path,
    output_dir: Path,
    kmer_size: int,
    sketch_size: int,
    cv_folds: int = 3,
    threads: int = 4,
    seed: int = None,
) -> TrainResult:
    """
        Train a phenotype prediction model.

        Args:
            references: reference genome labels, paths and phenotypes.
            output_dir: Directory to save trained model bundle.
            cv_folds: Number of cross-validation folds.
            kmer_size: Size of k-mers.
            sketch_size: Size of sketch.
            threads: Number of threads to use.
            seed: Seed for random number generator.

        Returns:
            TrainResult with training metrics.

        Raises:
            FileNotFoundError: If inputs don't exist.
            ValueError: If validation fails.
        """

    mash_version = check_mash()
    logger.info("Using mash version %s", mash_version)

    references_df = validate_references(references)
    output_dir = ensure_dir(output_dir)

    logger.info("Building mash sketch from references...")
    sketch_path = sketch(
        input_files=references_df["path"].values,
        output_path=output_dir / "reference_sketch",
        threads=threads,
        kmer_size=kmer_size,
        sketch_size=sketch_size,
    )

    logger.info("Generating training features...")
    screen_results_base_path = ensure_dir(output_dir / "screen_results")
    logger.info("Running mash screen for all references...")
    all_screen_results = _screen_all_references(
        references_df=references_df,
        sketch_path=sketch_path,
        screen_results_base_path=screen_results_base_path,
        threads=threads,
    )

    logger.info("Running %d-fold cross-validation for parameter tuning...", cv_folds)
    cv_result, best_C, feature_names = _cross_validate(
        references_df=references_df,
        all_screen_results=all_screen_results,
        cv_folds=cv_folds,
        seed=seed,
    )

    logger.info("Training final model on all data...")
    final_model = _train_final_model(
        references_df=references_df,
        all_screen_results=all_screen_results,
        best_C=best_C,
        seed=seed,
    )
    final_model.feature_names_ = feature_names

    labels = dict(zip(references_df["path"], references_df["phenotype"]))
    model = Model(model=final_model, labels=labels)
    model.save(output_dir)

    return TrainResult(
        accuracy=cv_result.accuracy,
        auc=cv_result.auc,
        precision=cv_result.precision,
        recall=cv_result.recall,
        f1=cv_result.f1,
        n_references=len(references_df),
        n_features=len(feature_names),
        output_dir=output_dir,
    )

def _screen_all_references(
        references_df: DataFrame,
        sketch_path: Path,
        screen_results_base_path: Path,
        threads: int = 4,
) -> dict[str, pd.DataFrame]:
    """
    Generate training features using mash screen.

    For training, we screen each reference against the full sketch.
    The self-hit will be removed.

    Args:
        references_df: Reference genome dataframe.
        sketch_path: Path to combined reference sketch.
        screen_results_base_path: Base directory to save screen results.
        threads: Number of threads to use.

    Returns:
        Dict mapping genome path to screen results DataFrame.
    """

    results = {}
    for path, phenotype in zip(references_df["path"], references_df["phenotype"]):
        screen_results = screen(
            query_paths=[path],
            sketch_path=sketch_path,
            threads=threads
        )

        # Remove self-hit
        screen_results = screen_results[screen_results["reference_path"] != path]
        screen_results.to_csv(screen_results_base_path / f"{Path(path).stem}.tsv", sep="\t", index=False)
        results[path] = screen_results

    return results


def _cross_validate(
        references_df: DataFrame,
        all_screen_results: dict[str, DataFrame],
        cv_folds: int,
        seed: int,
) -> tuple[TrainingResult, float | None, list[str]] | None:
    """
    Perform cross-validation for hyperparameter tuning with per-fold data filtering.

    For each fold:
    - Training samples: filter screen results to only include hits to other training samples
    - Validation samples: filter screen results to only include hits to training samples

    This ensures test genomes never influence training features.

    Args:
        references_df: References genome dataframe.
        all_screen_results: Pre-computed screen results for all references.
        cv_folds: Number of CV folds.
        seed: Random seed.

    Returns:
        Tuple of (TrainingResult, best_C, feature_names)
    """
    y_all = references_df["phenotype"].values

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    C_values = np.logspace(-3, 2, 6).tolist()
    results_per_C = {C: {"y_true": [], "y_pred": [], "y_proba": []} for C in C_values}
    feature_names = None

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(references_df, y_all)):
        logger.debug("Processing fold %d/%d", fold_idx + 1, cv_folds)

        train_paths = set(references_df.iloc[train_idx]["path"])
        val_paths = set(references_df.iloc[val_idx]["path"])

        # Build labels subset for feature extractor (training genomes only)
        train_refs_df = references_df.iloc[train_idx]
        extractor = FeatureExtractor(train_refs_df)
        if feature_names is None:
            feature_names = extractor.feature_names

        # Extract training features (filter to only training set hits)
        X_train = []
        y_train = []
        for i in train_idx:
            row = references_df.iloc[i]
            results = all_screen_results[row["path"]]
            filtered = results[results["reference_path"].isin(train_paths - {row["path"]})]
            fs = extractor.extract(filtered, row["path"])
            X_train.append(fs.features)
            y_train.append(row["phenotype"])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Extract validation features (filter to only training set hits)
        X_val = []
        y_val = []
        for i in val_idx:
            row = references_df.iloc[i]
            results = all_screen_results[row["path"]]
            filtered = results[results["reference_path"].isin(train_paths)]
            fs = extractor.extract(filtered, row["path"])
            X_val.append(fs.features)
            y_val.append(row["phenotype"])

        X_val = np.array(X_val)
        y_val = np.array(y_val)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        for C in C_values:
            clf = LogisticRegression(
                C=C,
                solver="lbfgs",
                max_iter=1000,
                class_weight="balanced",
                random_state=seed
            )
            clf.fit(X_train_scaled, y_train)

            y_pred = clf.predict(X_val_scaled)
            y_proba = clf.predict_proba(X_val_scaled)[:, 1]

            results_per_C[C]["y_true"].extend(y_val)
            results_per_C[C]["y_pred"].extend(y_pred)
            results_per_C[C]["y_proba"].extend(y_proba)

    best_C = None
    best_auc = -1
    for C in C_values:
        auc = roc_auc_score(results_per_C[C]["y_true"], results_per_C[C]["y_proba"])
        if auc > best_auc:
            best_auc = auc
            best_C = C

    # Build TrainingResult from best C
    res = results_per_C[best_C]
    y_true = np.array(res["y_true"])
    y_pred = np.array(res["y_pred"])

    training_result = TrainingResult(
        accuracy=accuracy_score(y_true, y_pred),
        auc=best_auc,
        precision=precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        recall=recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        f1=f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        feature_importances={},
        confusion_matrix=confusion_matrix(y_true, y_pred),
    )

    return training_result, best_C, feature_names


def _train_final_model(
        references_df: pd.DataFrame,
        all_screen_results: dict[str, pd.DataFrame],
        best_C: float,
        seed: int,
) -> PhenotypeModel:
    """
    Train final model on all data.

    For final training, each genome's features are extracted using hits
    to all OTHER genomes (excluding self).

    Args:
        references_df: References genome dataframe.
        all_screen_results: Pre-computed screen results.
        best_C: Best regularization parameter from cross-validation.
        seed: Random seed.

    Returns:
        Trained PhenotypeModel.
    """
    all_paths = set(references_df["path"])
    extractor = FeatureExtractor(references_df)

    X = []
    y = []
    for _, row in references_df.iterrows():
        results = all_screen_results[row["path"]]
        # Exclude self (already done in _screen_all_references, but be safe)
        filtered = results[results["reference_path"].isin(all_paths - {row["path"]})]
        fs = extractor.extract(filtered, row["path"])
        X.append(fs.features)
        y.append(row["phenotype"])

    X = np.array(X)
    y = np.array(y)

    model = PhenotypeModel(best_C=best_C, seed=seed)
    model.feature_names_ = extractor.feature_names
    model.fit(X, y)

    return model