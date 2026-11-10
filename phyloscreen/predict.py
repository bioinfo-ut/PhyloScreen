from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd

from phyloscreen.features import FeatureExtractor, FeatureSet
from phyloscreen.intermediate import FEATURE_TABLE_COLUMNS, feature_row, write_screen_results, write_table
from phyloscreen.mash import screen
from phyloscreen.model import Model
from phyloscreen.utils import find_fasta_files, logger

OUTPUT_COLUMNS = [
    "query_id",
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
class Query:
    """One isolate to predict: an id and a FASTA/FASTQ file or a directory of read files."""

    id: str
    path: Path


@dataclass
class QueryResult:
    """Prediction result for one query genome and one phenotype."""

    query_id: str
    query_path: str
    phenotype: str
    prediction: Union[int, str]   # 0, 1, "unknown" (no labelled hits) or "error"
    # Probabilities are None when there is no prediction ("unknown" / "error")
    probability: Optional[float] = None
    prob_1: Optional[float] = None
    prob_0: Optional[float] = None
    top_hit_1: Optional[str] = None
    top_hit_1_identity: Optional[float] = None
    top_hit_1_shared_hashes: Optional[int] = None
    top_hit_0: Optional[str] = None
    top_hit_0_identity: Optional[float] = None
    top_hit_0_shared_hashes: Optional[int] = None
    error: Optional[str] = None


@dataclass
class BatchResult:
    """Results for a list of queries plus the isolates that failed (id -> message)."""

    results: list[QueryResult] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)

    @property
    def n_queries(self) -> int:
        return len({r.query_id for r in self.results})


def default_query_id(path: Path) -> str:
    """Sample id derived from a path: directory name, or file stem without .gz and one extension."""
    path = Path(path)
    if path.is_dir():
        return path.name
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    return Path(name).stem


def read_query_list(list_file: Path) -> list[Query]:
    """
    Read a TSV of queries with a required ``path`` column and an optional ``id`` column.

    Blank ids fall back to default_query_id(). Duplicate ids or paths are rejected.
    Paths are not checked for existence here; a missing path is a per-isolate failure
    reported by predict_queries().
    """
    df = pd.read_csv(list_file, sep="\t", dtype=str, keep_default_na=False)

    if "path" not in df.columns:
        raise ValueError(f"Query list {list_file} is missing a 'path' column")

    paths = df["path"].str.strip()
    if (paths == "").any():
        rows = (df.index[paths == ""] + 2).tolist()  # 1-based, after header
        raise ValueError(f"Query list has empty 'path' values on line(s): {rows}")

    ids = df["id"].str.strip() if "id" in df.columns else pd.Series([""] * len(df), index=df.index)
    queries = [
        Query(id=i if i else default_query_id(Path(p)), path=Path(p))
        for i, p in zip(ids, paths)
    ]

    dup_paths = paths[paths.duplicated()].unique().tolist()
    if dup_paths:
        raise ValueError(f"Duplicate query paths: {dup_paths}")
    seen, dup_ids = set(), []
    for q in queries:
        if q.id in seen and q.id not in dup_ids:
            dup_ids.append(q.id)
        seen.add(q.id)
    if dup_ids:
        raise ValueError(f"Duplicate query ids: {dup_ids}")

    if not queries:
        raise ValueError(f"Query list {list_file} contains no queries")

    return queries


def predict_queries(
        queries: list[Query],
        model_dir: Path,
        output_path: Path,
        threads: int = 4,
        on_progress: Optional[Callable[[Query], None]] = None,
        raise_on_error: bool = False,
        intermediate_dir: Optional[Path] = None,
) -> BatchResult:
    """
    Predict every phenotype in the bundle for a list of isolates.

    The bundle is loaded once and mash screen runs once per isolate. A failing
    isolate is logged, written as ``error`` rows (one per phenotype) and recorded
    in BatchResult.failures; the remaining isolates are still processed.

    Args:
        queries: Isolates to predict.
        model_dir: Path to trained model directory.
        output_path: Output TSV (one row per isolate and phenotype).
        threads: Threads for mash screen.
        on_progress: Called after each isolate is finished (success or failure).
        raise_on_error: Re-raise the first failure instead of recording it.
        intermediate_dir: If given, save ``screen/<query_id>.tsv`` (parsed mash
            screen table per isolate) and ``features.tsv`` (model inputs per
            isolate and phenotype) under this directory.
    """
    bundle = Model.load(model_dir)
    logger.info("Loaded model from %s", model_dir)

    sketch_path = Path(model_dir) / "reference_sketch.msh"
    if not sketch_path.exists():
        raise FileNotFoundError(f"Reference sketch not found: {sketch_path}")

    extractors = {p: FeatureExtractor(bundle.references, p) for p in bundle.phenotypes}
    logger.info("Predicting %d phenotype(s) for %d isolate(s)", len(extractors), len(queries))

    batch = BatchResult()
    feature_rows: list[dict] = []
    for query in queries:
        try:
            results, rows = _predict_query(
                query, bundle, extractors, sketch_path, threads, intermediate_dir
            )
            batch.results.extend(results)
            feature_rows.extend(rows)
        except Exception as exc:  # noqa: BLE001 - one bad isolate must not stop the batch
            if raise_on_error:
                raise
            message = f"{type(exc).__name__}: {exc}"
            logger.error("Query %s failed: %s", query.id, message)
            batch.failures[query.id] = message
            batch.results.extend(_error_results(query, bundle.phenotypes, message))
        finally:
            if on_progress is not None:
                on_progress(query)

    _save_results(batch.results, output_path)
    logger.info("Results saved to %s", output_path)
    if intermediate_dir is not None:
        write_table(
            feature_rows,
            Path(intermediate_dir) / "features.tsv",
            columns=["query_id", "query_path", "phenotype", *FEATURE_TABLE_COLUMNS],
        )
        logger.info("Intermediate files saved to %s", intermediate_dir)
    if batch.failures:
        logger.warning("%d of %d isolate(s) failed", len(batch.failures), len(queries))

    return batch


def predict_phenotype(
        query_path: Path,
        model_dir: Path,
        output_path: Path,
        threads: int = 4,
        intermediate_dir: Optional[Path] = None,
) -> list[QueryResult]:
    """
    Predict every phenotype in the model bundle for a single query genome.

    Args:
        query_path: Path to query FASTA/FASTQ file or directory of FASTQs.
        model_dir: Path to trained model directory.
        output_path: Path to output TSV (one row per phenotype).
        threads: Number of threads for parallel processing.
        intermediate_dir: See predict_queries().

    Returns:
        List of QueryResult objects, one per phenotype.

    Raises:
        Whatever the screening or prediction raised (single queries fail fast).
    """
    query = Query(id=default_query_id(query_path), path=Path(query_path))
    batch = predict_queries(
        [query], model_dir, output_path, threads=threads,
        raise_on_error=True, intermediate_dir=intermediate_dir,
    )
    return batch.results


def _predict_query(
        query: Query,
        bundle: Model,
        extractors: dict[str, FeatureExtractor],
        sketch_path: Path,
        threads: int,
        intermediate_dir: Optional[Path] = None,
) -> tuple[list[QueryResult], list[dict]]:
    """
    Screen one isolate and predict every phenotype from the shared screen results.

    Returns the per-phenotype results and, for the intermediate feature table,
    one row per predicted phenotype (none for phenotypes that came out ``unknown``).
    """
    if query.path.is_dir():
        query_files = find_fasta_files(query.path)
    elif query.path.is_file():
        query_files = [query.path]
    else:
        raise FileNotFoundError(f"Query not found: {query.path}")

    logger.debug("Query %s: %d file(s)", query.id, len(query_files))

    screen_results = screen(
        query_paths=query_files,
        sketch_path=sketch_path,
        threads=threads,
    )
    if screen_results.empty:
        logger.warning("No hits for query %s", query.id)
    if intermediate_dir is not None:
        write_screen_results(screen_results, intermediate_dir, query.id)

    results: list[QueryResult] = []
    rows: list[dict] = []
    for phenotype in bundle.phenotypes:
        result, fs = _predict_one(
            query, phenotype, extractors[phenotype], bundle.models[phenotype], screen_results
        )
        results.append(result)
        if fs is not None:
            rows.append(feature_row(fs, query_id=query.id, query_path=str(query.path), phenotype=phenotype))
    return results, rows


def _predict_one(
        query: Query,
        phenotype: str,
        extractor: FeatureExtractor,
        model,
        screen_results: pd.DataFrame,
) -> tuple[QueryResult, Optional[FeatureSet]]:
    """
    Predict a single phenotype from shared screen results.

    The result is ``unknown`` (and the FeatureSet None) when the isolate has no
    hits to references labelled for this phenotype, whether because the screen
    was empty or because every hit was to a reference with a missing label.
    """
    unknown = QueryResult(
        query_id=query.id,
        query_path=str(query.path),
        phenotype=phenotype,
        prediction="unknown",
    )
    if screen_results.empty:
        return unknown, None

    fs = extractor.extract(screen_results, query.id)
    if fs.metadata["n_hits_total"] == 0:
        logger.warning(
            "Query %s: no hits to references labelled for %r (%d hit(s) to unlabelled references)",
            query.id, phenotype, len(screen_results),
        )
        return unknown, None

    X = fs.features.reshape(1, -1)

    prediction = model.predict(X)[0]
    probas = model.predict_proba(X)[0]

    class_idx = {int(c): i for i, c in enumerate(model.classes_)}
    prob_1 = float(probas[class_idx[1]])
    prob_0 = float(probas[class_idx[0]])

    top_hit_1 = fs.metadata.get("top_hit_1")
    top_hit_0 = fs.metadata.get("top_hit_0")

    return QueryResult(
        query_id=query.id,
        query_path=str(query.path),
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
    ), fs


def _error_results(query: Query, phenotypes: list[str], message: str) -> list[QueryResult]:
    """One ``error`` row per phenotype for an isolate that could not be processed."""
    return [
        QueryResult(
            query_id=query.id,
            query_path=str(query.path),
            phenotype=phenotype,
            prediction="error",
            error=message,
        )
        for phenotype in phenotypes
    ]


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
    """Save prediction results to a TSV with one row per isolate and phenotype."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "query_id": r.query_id,
            "query_path": r.query_path,
            "phenotype": r.phenotype,
            "prediction": r.prediction,
            # Blank when there is no prediction, so it cannot be read as "confident 0"
            "probability": None if r.probability is None else f"{r.probability:.3f}",
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
