import logging
import sys
from pathlib import Path

import pandas as pd
from pandas import DataFrame

logger = logging.getLogger("phyloscreen")

PATH_COLUMN = "path"
DEFAULT_PHENOTYPE = "phenotype"


def setup_logging(verbose: bool = False) -> None:
    """
    Configure logging.

    Args:
        verbose: If True, set level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.debug("Logging initialized (verbose=%s)", verbose)


def phenotype_columns(df: DataFrame) -> list[str]:
    """Return the phenotype column names: every column except ``path``, in file order."""
    return [str(c) for c in df.columns if c != PATH_COLUMN]


def labelled_subset(df: DataFrame, phenotype: str) -> DataFrame:
    """
    Reduce a references table to the isolates labelled for one phenotype.

    Returns a two-column DataFrame (``path``, ``phenotype``) containing only rows
    whose value for ``phenotype`` is not missing, with the label cast to int.
    This is the single place that defines which isolates take part in feature
    extraction and training for a given phenotype.
    """
    if phenotype not in df.columns:
        raise KeyError(f"Phenotype column not found: {phenotype!r}")
    sub = df[[PATH_COLUMN, phenotype]].dropna(subset=[phenotype])
    sub = sub.rename(columns={phenotype: DEFAULT_PHENOTYPE})
    sub[DEFAULT_PHENOTYPE] = sub[DEFAULT_PHENOTYPE].astype(int)
    return sub.reset_index(drop=True)


def validate_references(references_file: Path) -> DataFrame:
    """
    Read and validate the reference table.

    Expected: a TSV with a ``path`` column plus one or more phenotype columns.
    Every column other than ``path`` is treated as a phenotype named by its header.
    Phenotype values must be 0, 1 or blank (missing). Phenotype columns are returned
    as pandas nullable ``Int64`` so missing values survive as ``<NA>``.
    """
    df = pd.read_csv(references_file, sep="\t")

    if PATH_COLUMN not in df.columns:
        raise ValueError(f"Input file missing column: {PATH_COLUMN!r}")

    phenotypes = phenotype_columns(df)
    if not phenotypes:
        raise ValueError(
            "Input file has no phenotype columns: expected 'path' plus at least one phenotype column"
        )

    if df[PATH_COLUMN].duplicated().any():
        duplicated = df[df[PATH_COLUMN].duplicated()][PATH_COLUMN].tolist()
        raise ValueError(f"Duplicate genome paths: {duplicated}")

    missing = [p for p in df[PATH_COLUMN] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} path(s) not found: {missing}")

    for col in phenotypes:
        values = df[col].dropna()
        if values.empty:
            raise ValueError(f"Phenotype column {col!r} has no values (all missing)")
        if not values.isin([0, 1]).all():
            invalid = values[~values.isin([0, 1])].unique()
            raise ValueError(f"Phenotype {col!r} must be 0, 1 or blank, found: {invalid.tolist()}")
        df[col] = df[col].astype("Int64")

    return df


def ensure_dir(path: Path) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure exists.

    Returns:
        The same path for chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_fasta_files(directory: Path) -> list[Path]:
    """
    Find all FASTA/FASTQ files in a directory.

    Args:
        directory: Path to search for FASTA/FASTQ files.

    Returns:
        Sorted list of FASTA/FASTQ file paths.

    Raises:
        FileNotFoundError: If directory doesn't exist.
        ValueError: If no FASTA/FASTQ files found.
    """
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    extensions = {".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz", ".fna.gz",
                  ".fastq", ".fq", ".fastq.gz", ".fq.gz"}
    fasta_files = []

    for ext in extensions:
        fasta_files.extend(directory.glob(f"*{ext}"))

    fasta_files = sorted(set(fasta_files))

    if not fasta_files:
        raise ValueError(f"No FASTA/FASTQ files found in {directory}")

    logger.debug("Found %d FASTA/FASTQ files in %s", len(fasta_files), directory)
    return fasta_files
