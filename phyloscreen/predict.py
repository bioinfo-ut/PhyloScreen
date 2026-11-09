from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from phyloscreen.features import FeatureExtractor
from phyloscreen.mash import screen
from phyloscreen.model import Model
from phyloscreen.utils import find_fasta_files, logger

OUTPUT_COLUMNS = [
    "query_path",
    "phenotype",
    "prediction",
    "probability",
    "top_hit_1",
    "top_hit_1_identity",
    "top_hit_1_shared_hashes",
    "top_hit_0",
    "top_hit_0_identity",
    "top_hit_0_shared_hashes",
]


@dataclass
class QueryResult:
    """Prediction result for one query genome and one phenotype."""

    query_path: str
    phenotype: str
    prediction: Union[int, str]
    probability: float
    prob_1: float
    prob_0: float
    top_hit_1: Optional[str] = None
    top_hit_1_identity: Optional[float] = None
    top_hit_1_shared_hashes: Optional[int] = None
    top_hit_0: Optional[str] = None
    top_hit_0_identity: Optional[float] = None
    top_hit_0_shared_hashes: Optional[int] = None


def predict_phenotype(
        query_path: Path,
        model_dir: Path,
        output_path: Path,
        threads: int = 4,
) -> list[QueryResult]:
    """
    Predict every phenotype in the model bundle for a query genome.

    Mash screen runs once; each phenotype's classifier then uses only the
    references labelled for that phenotype as neighbours.

    Args:
        query_path: Path to query FASTA/FASTQ file or directory of FASTQs.
        model_dir: Path to trained model directory.
        output_path: Path to output TSV (one row per phenotype).
        threads: Number of threads for parallel processing.

    Returns:
        List of QueryResult objects, one per phenotype.
    """
    # Load model bundle
    bundle = Model.load(model_dir)
    logger.info("Loaded model from %s", model_dir)

    # Find query files
    if query_path.is_dir():
        query_files = find_fasta_files(query_path)
    else:
        query_files = [query_path]

    logger.info("Processing %d query genome(s)", len(query_files))

    # Find sketch path
    sketch_path = model_dir / "reference_sketch.msh"
    if not sketch_path.exists():
        raise FileNotFoundError(f"Reference sketch not found: {sketch_path}")

    query_id = ",".join([str(p) for p in query_files])
    logger.debug("Processing query: %s", query_id)

    # Run mash screen once for all phenotypes
    screen_results = screen(
        query_paths=query_files,
        sketch_path=sketch_path,
        threads=threads
    )

    if screen_results.empty:
        logger.warning("No hits for query %s", query_id)

    results = [
        _predict_one(bundle, phenotype, screen_results, query_path, query_id)
        for phenotype in bundle.phenotypes
    ]

    # Save results
    _save_results(results, output_path)
    logger.info("Result saved to %s", output_path)

    return results


def _predict_one(
        bundle: Model,
        phenotype: str,
        screen_results: pd.DataFrame,
        query_path: Path,
        query_id: str,
) -> QueryResult:
    """Predict a single phenotype from shared screen results."""
    if screen_results.empty:
        return QueryResult(
            query_path=str(query_path),
            phenotype=phenotype,
            prediction="unknown",
            probability=0.0,
            prob_1=0.0,
            prob_0=0.0,
        )

    extractor = FeatureExtractor(bundle.references, phenotype)
    fs = extractor.extract(screen_results, query_id)
    X = fs.features.reshape(1, -1)

    model = bundle.models[phenotype]
    prediction = model.predict(X)[0]
    probas = model.predict_proba(X)[0]

    # Map probabilities to classes
    class_idx = {int(c): i for i, c in enumerate(model.classes_)}
    prob_1 = float(probas[class_idx[1]])
    prob_0 = float(probas[class_idx[0]])

    top_hit_1 = fs.metadata.get("top_hit_1")
    top_hit_0 = fs.metadata.get("top_hit_0")

    return QueryResult(
        query_path=str(query_path),
        phenotype=phenotype,
        prediction=int(prediction),
        probability=prob_1,
        prob_1=prob_1,
        prob_0=prob_0,
        top_hit_1=top_hit_1,
        top_hit_1_identity=_get_top_identity(screen_results, top_hit_1),
        top_hit_1_shared_hashes=_get_top_shared_hashes(screen_results, top_hit_1),
        top_hit_0=top_hit_0,
        top_hit_0_identity=_get_top_identity(screen_results, top_hit_0),
        top_hit_0_shared_hashes=_get_top_shared_hashes(screen_results, top_hit_0),
    )


def _get_top_identity(screen_results: pd.DataFrame, ref_path: Optional[str]) -> Optional[float]:
    """Get identity for a specific reference from screen results."""
    if ref_path is None:
        return None
    matches = screen_results[screen_results["reference_path"] == ref_path]
    if len(matches) > 0:
        return float(matches.iloc[0]["identity"])
    return None


def _get_top_shared_hashes(screen_results: pd.DataFrame, ref_path: Optional[str]) -> Optional[int]:
    """Get shared hashes for a specific reference from screen results."""
    if ref_path is None:
        return None
    matches = screen_results[screen_results["reference_path"] == ref_path]
    if len(matches) > 0:
        return int(matches.iloc[0]["shared_hashes"])
    return None


def _save_results(results: list[QueryResult], output_path: Path) -> None:
    """Save prediction results to a TSV with one row per phenotype."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "query_path": r.query_path,
            "phenotype": r.phenotype,
            "prediction": r.prediction,
            "probability": f"{r.probability:.3f}",
            "top_hit_1": r.top_hit_1,
            "top_hit_1_identity": r.top_hit_1_identity,
            "top_hit_1_shared_hashes": r.top_hit_1_shared_hashes,
            "top_hit_0": r.top_hit_0,
            "top_hit_0_identity": r.top_hit_0_identity,
            "top_hit_0_shared_hashes": r.top_hit_0_shared_hashes,
        }
        for r in results
    ]

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    # Keep hash counts integral even when another row has no hit (None -> blank)
    for col in ("top_hit_1_shared_hashes", "top_hit_0_shared_hashes"):
        df[col] = df[col].astype("Int64")
    df.to_csv(output_path, sep="\t", index=False)
