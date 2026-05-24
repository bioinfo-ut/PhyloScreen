import logging
import sys
from pathlib import Path

from pandas import DataFrame

logger = logging.getLogger("phenoscreen")



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


def validate_references(references_file: Path) -> DataFrame:
    import pandas as pd

    df = pd.read_csv(references_file, sep="\t")

    required_cols = {"path", "phenotype"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Input file missing columns: {missing}")

    if df["path"].duplicated().any():
        duplicated = df[df["path"].duplicated()]["path"].tolist()
        raise ValueError(f"Duplicate genome paths: {duplicated}")

    missing = [p for p in df["path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} path(s) not found: {missing}")

    if not df["phenotype"].isin([0, 1]).all():
        invalid = df[~df["phenotype"].isin([0, 1])]["phenotype"].unique()
        raise ValueError(f"Phenotype must be 0 or 1, found: {invalid}")

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