"""
Training pipeline for phyloscreen.

Orchestrates the full training workflow:
1. Validate inputs
2. Build mash sketch from references
3. Screen every reference against the sketch (shared by all phenotypes)
4. For each phenotype: cross-validate on the labelled isolates and fit a classifier
5. Save model bundle
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from phyloscreen.features import FeatureExtractor
from phyloscreen.intermediate import feature_row, write_table
from phyloscreen.mash import check_mash, sketch, screen
from phyloscreen.model import TrainingResult, PhenotypeModel, Model
from phyloscreen.utils import (
    logger, validate_references, ensure_dir, phenotype_columns, labelled_subset,
)


@dataclass
class PhenotypeTrainResult:
    """Cross-validation result for one phenotype."""

    phenotype: str
    n_labelled: int
    best_C: float
    accuracy: float
    auc: float
    precision: float
    recall: float
    f1: float


@dataclass
class TrainResult:
    """Result of the training pipeline."""

    results: dict[str, PhenotypeTrainResult]
    n_references: int
    n_features: int
    output_dir: Path

    @property
    def phenotypes(self) -> list[str]:
        return list(self.results.keys())


def train_model(
    references: Path,
    output_dir: Path,
    kmer_size: int,
    sketch_size: int,
    cv_folds: int = 3,
    threads: int = 4,
    seed: int = None,
    intermediate_dir: Optional[Path] = None,
) -> TrainResult:
    """
        Train a phenotype prediction model bundle.

        Every column of the reference table other than ``path`` is a phenotype.
        One classifier is trained per phenotype using only the isolates that have a
        value for it; all isolates are in the shared sketch.

        Args:
            references: TSV with reference genome paths and phenotype columns.
            output_dir: Directory to save trained model bundle.
            cv_folds: Number of cross-validation folds.
            kmer_size: Size of k-mers.
            sketch_size: Size of sketch.
            threads: Number of threads to use.
            seed: Seed for random number generator.
            intermediate_dir: If given, save ``features.tsv`` (the per-reference
                feature matrix the final models are fitted on) and
                ``cv_predictions.tsv`` (out-of-fold predictions for every
                reference, phenotype and C value) under this directory.

        Returns:
            TrainResult with per-phenotype training metrics.

        Raises:
            FileNotFoundError: If inputs don't exist.
            ValueError: If validation fails or a phenotype has too few labelled isolates.
        """

    mash_version = check_mash()
    logger.info("Using mash version %s", mash_version)

    references_df = validate_references(references)
    phenotypes = phenotype_columns(references_df)
    logger.info("Found %d phenotype(s): %s", len(phenotypes), phenotypes)
    _check_class_counts(references_df, phenotypes, cv_folds)
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

    models: dict[str, PhenotypeModel] = {}
    results: dict[str, PhenotypeTrainResult] = {}
    best_Cs: dict[str, float] = {}
    n_features = None
    cv_rows: list[dict] = []
    feature_rows: list[dict] = []

    for phenotype in phenotypes:
        sub_df = labelled_subset(references_df, phenotype)
        logger.info(
            "Phenotype %r: %d labelled isolates (%d positive, %d negative)",
            phenotype, len(sub_df), int((sub_df["phenotype"] == 1).sum()), int((sub_df["phenotype"] == 0).sum()),
        )

        logger.info("[%s] Running %d-fold cross-validation for parameter tuning...", phenotype, cv_folds)
        cv_result, best_C, feature_names, phenotype_cv_rows = _cross_validate(
            references_df=sub_df,
            all_screen_results=all_screen_results,
            cv_folds=cv_folds,
            seed=seed,
        )
        cv_rows.extend({"phenotype": phenotype, **row} for row in phenotype_cv_rows)

        logger.info("[%s] Training final model on all labelled isolates...", phenotype)
        final_model, phenotype_feature_rows = _train_final_model(
            references_df=sub_df,
            all_screen_results=all_screen_results,
            best_C=best_C,
            seed=seed,
        )
        feature_rows.extend({"phenotype": phenotype, **row} for row in phenotype_feature_rows)
        final_model.feature_names_ = feature_names
        n_features = len(feature_names)

        models[phenotype] = final_model
        best_Cs[phenotype] = best_C
        results[phenotype] = PhenotypeTrainResult(
            phenotype=phenotype,
            n_labelled=len(sub_df),
            best_C=best_C,
            accuracy=cv_result.accuracy,
            auc=cv_result.auc,
            precision=cv_result.precision,
            recall=cv_result.recall,
            f1=cv_result.f1,
        )

    model = Model(
        models=models,
        references=references_df,
        metadata={
            "phenotypes": phenotypes,
            "kmer_size": kmer_size,
            "sketch_size": sketch_size,
            "cv_folds": cv_folds,
            "seed": seed,
            "mash_version": str(mash_version),
            "best_C": best_Cs,
        },
    )
    model.save(output_dir)

    if intermediate_dir is not None:
        write_table(feature_rows, Path(intermediate_dir) / "features.tsv")
        write_table(cv_rows, Path(intermediate_dir) / "cv_predictions.tsv")
        logger.info("Intermediate files saved to %s", intermediate_dir)

    return TrainResult(
        results=results,
        n_references=len(references_df),
        n_features=n_features or 0,
        output_dir=output_dir,
    )


def _check_class_counts(references_df: DataFrame, phenotypes: list[str], cv_folds: int) -> None:
    """
    Fail early if any phenotype cannot be cross-validated.

    Each phenotype needs both classes present and at least ``cv_folds`` labelled
    isolates in each class, so every fold's training set contains both classes.
    """
    problems = []
    for phenotype in phenotypes:
        labels = labelled_subset(references_df, phenotype)["phenotype"]
        n0 = int((labels == 0).sum())
        n1 = int((labels == 1).sum())
        if n0 == 0 or n1 == 0 or min(n0, n1) < cv_folds:
            problems.append(f"{phenotype}: n0={n0}, n1={n1}")

    if problems:
        raise ValueError(
            f"Each phenotype needs at least {cv_folds} labelled isolates per class "
            f"(cv_folds={cv_folds}). Insufficient: " + "; ".join(problems)
        )


def _screen_all_references(
        references_df: DataFrame,
        sketch_path: Path,
        screen_results_base_path: Path,
        threads: int = 4,
) -> dict[str, pd.DataFrame]:
    """
    Screen each reference against the full sketch.

    The screen results are independent of phenotype and are shared by every
    per-phenotype model. The self-hit is removed.

    Args:
        references_df: Reference genome dataframe.
        sketch_path: Path to combined reference sketch.
        screen_results_base_path: Base directory to save screen results.
        threads: Number of threads to use.

    Returns:
        Dict mapping genome path to screen results DataFrame.
    """

    results = {}
    for path in references_df["path"]:
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
) -> tuple[TrainingResult, float | None, list[str], list[dict]]:
    """
    Perform cross-validation for hyperparameter tuning with per-fold data filtering.

    ``references_df`` holds only the isolates labelled for the phenotype being
    trained (columns ``path`` and ``phenotype``), so every neighbour considered
    below has a known label for it.

    For each fold:
    - Training samples: filter screen results to only include hits to other training samples
    - Validation samples: filter screen results to only include hits to training samples

    This ensures test genomes never influence training features.

    Args:
        references_df: Labelled references for one phenotype.
        all_screen_results: Pre-computed screen results for all references.
        cv_folds: Number of CV folds.
        seed: Random seed.

    Returns:
        Tuple of (TrainingResult, best_C, feature_names, cv_rows) where cv_rows
        holds one dict per (fold, C, validation reference) with keys
        fold, C, path, y_true, y_pred, y_proba.
    """
    y_all = references_df["phenotype"].to_numpy(dtype=int)

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    C_values = np.logspace(-3, 2, 6).tolist()
    results_per_C = {C: {"y_true": [], "y_pred": [], "y_proba": [], "path": [], "fold": []} for C in C_values}
    feature_names = None

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(references_df, y_all)):
        logger.debug("Processing fold %d/%d", fold_idx + 1, cv_folds)

        train_paths = set(references_df.iloc[train_idx]["path"])

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
        y_train = np.array(y_train, dtype=int)

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
        y_val = np.array(y_val, dtype=int)
        val_paths = [references_df.iloc[i]["path"] for i in val_idx]

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
            results_per_C[C]["path"].extend(val_paths)
            results_per_C[C]["fold"].extend([fold_idx + 1] * len(val_idx))

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

    cv_rows = [
        {
            "fold": int(fold),
            "C": C,
            "path": path,
            "y_true": int(yt),
            "y_pred": int(yp),
            "y_proba": float(pr),
        }
        for C in C_values
        for fold, path, yt, yp, pr in zip(
            results_per_C[C]["fold"], results_per_C[C]["path"],
            results_per_C[C]["y_true"], results_per_C[C]["y_pred"], results_per_C[C]["y_proba"],
        )
    ]

    return training_result, best_C, feature_names, cv_rows


def _train_final_model(
        references_df: pd.DataFrame,
        all_screen_results: dict[str, pd.DataFrame],
        best_C: float,
        seed: int,
) -> tuple[PhenotypeModel, list[dict]]:
    """
    Train final model on all labelled isolates for one phenotype.

    For final training, each genome's features are extracted using hits
    to all OTHER labelled genomes (excluding self).

    Args:
        references_df: Labelled references for one phenotype (``path``, ``phenotype``).
        all_screen_results: Pre-computed screen results.
        best_C: Best regularization parameter from cross-validation.
        seed: Random seed.

    Returns:
        Trained PhenotypeModel and the feature table rows it was fitted on
        (one dict per reference: path, label, features and hit counts).
    """
    all_paths = set(references_df["path"])
    extractor = FeatureExtractor(references_df)

    X = []
    y = []
    rows = []
    for _, row in references_df.iterrows():
        results = all_screen_results[row["path"]]
        # Exclude self (already done in _screen_all_references, but be safe)
        filtered = results[results["reference_path"].isin(all_paths - {row["path"]})]
        fs = extractor.extract(filtered, row["path"])
        X.append(fs.features)
        y.append(row["phenotype"])
        rows.append(feature_row(fs, path=row["path"], label=int(row["phenotype"])))

    X = np.array(X)
    y = np.array(y, dtype=int)

    model = PhenotypeModel(best_C=best_C, seed=seed)
    model.feature_names_ = extractor.feature_names
    model.fit(X, y)

    return model, rows
