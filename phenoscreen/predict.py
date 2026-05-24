from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from phenoscreen.features import FeatureExtractor
from phenoscreen.mash import screen
from phenoscreen.model import Model
from phenoscreen.utils import find_fasta_files, logger


@dataclass
class QueryResult:
    """Prediction result for a query genome."""

    query_path: str
    prediction: int
    probability: float
    prob_1: float
    prob_0: float
    top_hit_1: Optional[str]
    top_hit_1_identity: Optional[float]
    top_hit_1_shared_hashes: Optional[int]
    top_hit_0: Optional[str]
    top_hit_0_identity: Optional[float]
    top_hit_0_shared_hashes: Optional[int]


def predict_phenotype(
        query_path: Path,
        model_dir: Path,
        output_path: Path,
        threads: int = 4,
) -> QueryResult:
    """
    Predict phenotype for query genome.

    Args:
        query_path: Path to query FASTA/FASTQ file or directory of FASTQs.
        model_dir: Path to trained model directory.
        output_path: Path to output file (TSV or JSON).
        threads: Number of threads for parallel processing.

    Returns:
        QueryResult object.
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

    # Process queries
    extractor = FeatureExtractor(pd.DataFrame({"path": bundle.labels.keys(), "phenotype": bundle.labels.values()}))

    query_id = ",".join([str(p) for p in query_files])
    logger.debug("Processing query: %s", query_id)

    # Run mash screen
    screen_results = screen(
        query_paths=query_files,
        sketch_path=sketch_path,
        threads=threads
    )

    if screen_results.empty:
        logger.warning("No hits for query %s", query_id)
        return QueryResult(
            query_path=str(query_path),
            prediction="unknown",
            probability=0.0,
            prob_1=0.0,
            prob_0=0.0,
            top_hit_1=None,
            top_hit_1_identity=None,
            top_hit_0=None,
            top_hit_0_identity=None,
        )

    # Extract features
    fs = extractor.extract(screen_results, query_id)
    X = fs.features.reshape(1, -1)

    # Predict
    prediction = bundle.model.predict(X)[0]
    probas = bundle.model.predict_proba(X)[0]

    # Map probabilities to classes
    class_idx = {c: i for i, c in enumerate(bundle.model.classes_)}
    prob_1 = probas[class_idx[1]]
    prob_0 = probas[class_idx[0]]

    # Get top hits info from metadata
    result = QueryResult(
        query_path=str(query_path),
        prediction=prediction,
        probability=prob_1,
        prob_1=prob_1,
        prob_0=prob_0,
        top_hit_1=fs.metadata.get("top_hit_1"),
        top_hit_1_identity=_get_top_identity(
            screen_results, fs.metadata.get("top_hit_1")
        ),
        top_hit_1_shared_hashes=_get_top_shared_hashes(
            screen_results, fs.metadata.get("top_hit_1")
        ),
        top_hit_0=fs.metadata.get("top_hit_0"),
        top_hit_0_identity=_get_top_identity(
            screen_results, fs.metadata.get("top_hit_0")
        ),
        top_hit_0_shared_hashes=_get_top_shared_hashes(
            screen_results, fs.metadata.get("top_hit_0")
        ),
    )

    # Save results
    _save_results(result, output_path)
    logger.info("Result saved to %s", output_path)

    return result


def _get_top_identity(screen_results: pd.DataFrame, ref_path: Optional[str]) -> Optional[float]:
    """Get identity for a specific reference from screen results."""
    if ref_path is None:
        return None
    matches = screen_results[screen_results["reference_path"] == ref_path]
    if len(matches) > 0:
        return float(matches.iloc[0]["identity"])
    return None


def _get_top_shared_hashes(screen_results: pd.DataFrame, ref_path: Optional[str]) -> Optional[float]:
    """Get shared hashes for a specific reference from screen results."""
    if ref_path is None:
        return None
    matches = screen_results[screen_results["reference_path"] == ref_path]
    if len(matches) > 0:
        return float(matches.iloc[0]["shared_hashes"])
    return None


def _save_results(results: QueryResult, output_path: Path) -> None:
    """Save prediction results to file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "query_path": results.query_path,
        "prediction": results.prediction,
        "probability": f"{results.probability:.3f}",
        "top_hit_1": results.top_hit_1,
        "top_hit_1_identity": results.top_hit_1_identity,
        "top_hit_1_shared_hashes": results.top_hit_1_shared_hashes,
        "top_hit_0": results.top_hit_0,
        "top_hit_0_identity": results.top_hit_0_identity,
        "top_hit_0_shared_hashes": results.top_hit_0_shared_hashes,
    }

    df = pd.DataFrame([data])
    df.to_csv(output_path, sep="\t", index=False)