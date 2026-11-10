"""Shared fixtures: a synthetic reference panel and a fake mash screen (no mash binary needed)."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from phyloscreen import train as train_mod

N_GENOMES = 12


@dataclass
class Panel:
    """Synthetic reference panel with two phenotypes and some missing labels."""

    dir: Path
    refs_tsv: Path
    paths: list[str]
    truth: dict[str, np.ndarray]  # hidden true labels used to generate similarities
    table: pd.DataFrame           # what is written to refs_tsv (with blanks)


@pytest.fixture
def panel(tmp_path: Path) -> Panel:
    genome_dir = tmp_path / "genomes"
    genome_dir.mkdir()
    paths = []
    for i in range(N_GENOMES):
        p = genome_dir / f"g{i:02d}.fa"
        p.write_text(f">g{i:02d}\nACGT\n")
        paths.append(str(p))

    amp = np.array([1] * 6 + [0] * 6)
    cef = np.array([1 if i % 2 == 0 else 0 for i in range(N_GENOMES)])
    truth = {"amp": amp, "cef": cef}

    table = pd.DataFrame({"path": paths, "amp": amp, "cef": cef}).astype({"amp": "Int64", "cef": "Int64"})
    # Missing labels: two blanks in cef (one per class), one blank in amp
    table.loc[2, "cef"] = pd.NA
    table.loc[3, "cef"] = pd.NA
    table.loc[7, "amp"] = pd.NA

    refs_tsv = tmp_path / "refs.tsv"
    table.to_csv(refs_tsv, sep="\t", index=False, na_rep="")

    return Panel(dir=tmp_path, refs_tsv=refs_tsv, paths=paths, truth=truth, table=table)


def make_fake_screen(panel: Panel, query_truth: dict | None = None):
    """
    Build a fake mash.screen(): shared hashes are higher between genomes that
    share hidden labels, so each phenotype has learnable signal.

    A query that is not one of the panel genomes gets hidden labels from
    ``query_truth``: either one ``{"amp": .., "cef": ..}`` dict for every novel
    query, or a mapping ``{str(path): {"amp": .., "cef": ..}}`` per query
    (defaults to amp=1, cef=1).
    """
    index = {p: i for i, p in enumerate(panel.paths)}
    query_truth = query_truth or {"amp": 1, "cef": 1}

    def fake_screen(query_paths, sketch_path, threads=4, max_retries=5):
        qpath = str(query_paths[0])
        if qpath in index:
            q = index[qpath]
            q_truth = {k: v[q] for k, v in panel.truth.items()}
        else:
            q = len(panel.paths) + sum(map(ord, Path(qpath).name)) % 7
            q_truth = query_truth[qpath] if qpath in query_truth else query_truth
        rows = []
        for ref, r in index.items():
            # Equal weights keep the top-10 vote positively correlated with both labels.
            shared = 400
            shared += 250 * int(q_truth["amp"] == panel.truth["amp"][r])
            shared += 250 * int(q_truth["cef"] == panel.truth["cef"][r])
            shared += (q * 7 + r * 3) % 11  # deterministic jitter, breaks ties
            if qpath == ref:
                shared = 1000
            rows.append(
                {
                    "identity": shared / 1000,
                    "shared_hashes": shared,
                    "total_hashes": 1000,
                    "median_multiplicity": 1.0,
                    "p_value": 0.0,
                    "query_comment": "",
                    "reference_path": ref,
                }
            )
        return pd.DataFrame(rows).sort_values("shared_hashes", ascending=False).reset_index(drop=True)

    return fake_screen


def fake_sketch(input_files, output_path, kmer_size, sketch_size, threads=4):
    sketch_path = Path(str(output_path) + ".msh")
    sketch_path.write_bytes(b"")
    return sketch_path


@pytest.fixture
def trained_bundle(panel: Panel, monkeypatch) -> tuple[Path, Panel, train_mod.TrainResult]:
    """Train a two-phenotype bundle on the synthetic panel with mash calls faked."""
    monkeypatch.setattr(train_mod, "check_mash", lambda: "fake")
    monkeypatch.setattr(train_mod, "sketch", fake_sketch)
    monkeypatch.setattr(train_mod, "screen", make_fake_screen(panel))

    out = panel.dir / "model"
    result = train_mod.train_model(
        references=panel.refs_tsv,
        output_dir=out,
        kmer_size=21,
        sketch_size=1000,
        cv_folds=3,
        threads=1,
        seed=0,
    )
    return out, panel, result
