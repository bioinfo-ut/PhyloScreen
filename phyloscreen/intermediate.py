"""
Opt-in intermediate outputs (``--intermediate-dir``).

Shared by train and predict so both write the same feature table layout:
one row per (isolate, phenotype) with the model inputs and the hit counts
behind them.
"""

import re
from pathlib import Path

import pandas as pd

from phyloscreen.features import FeatureSet
from phyloscreen.utils import ensure_dir, logger

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    """Make an isolate id usable as a file name."""
    cleaned = _UNSAFE.sub("_", name).strip("._")
    return cleaned or "query"


def feature_row(fs: FeatureSet, **ids) -> dict:
    """Flatten a FeatureSet into one table row, prefixed by identifier columns."""
    meta = fs.metadata
    return {
        **ids,
        **fs.to_dict(),
        "n_hits_total": meta.get("n_hits_total"),
        "n_hits_1": meta.get("n_hits_1"),
        "n_hits_0": meta.get("n_hits_0"),
        "top_hit_1": meta.get("top_hit_1"),
        "top_hit_0": meta.get("top_hit_0"),
    }


def write_table(rows: list[dict], path: Path) -> Path:
    """Write rows as a TSV (creating parent directories); writes a header-only file for no rows."""
    ensure_dir(Path(path).parent)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    logger.debug("Wrote %d row(s) to %s", len(rows), path)
    return path


def write_screen_results(screen_results: pd.DataFrame, intermediate_dir: Path, query_id: str) -> Path:
    """Save one isolate's parsed mash screen table under ``<dir>/screen/<id>.tsv``."""
    path = ensure_dir(Path(intermediate_dir) / "screen") / f"{safe_filename(query_id)}.tsv"
    screen_results.to_csv(path, sep="\t", index=False)
    return path
