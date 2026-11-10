"""Tests for reference-table validation and the per-phenotype labelled subset."""

import pandas as pd
import pytest

from phyloscreen.utils import labelled_subset, phenotype_columns, validate_references


def _write(tmp_path, text):
    f = tmp_path / "refs.tsv"
    f.write_text(text)
    return f


def _genomes(tmp_path, n):
    paths = []
    for i in range(n):
        p = tmp_path / f"g{i}.fa"
        p.write_text(">x\nACGT\n")
        paths.append(str(p))
    return paths


def test_multi_phenotype_with_blanks(tmp_path):
    g = _genomes(tmp_path, 3)
    f = _write(tmp_path, f"path\tamp\tcef\n{g[0]}\t1\t0\n{g[1]}\t0\t\n{g[2]}\t\t1\n")

    df = validate_references(f)

    assert phenotype_columns(df) == ["amp", "cef"]
    assert str(df["amp"].dtype) == "Int64"
    assert df["amp"].isna().tolist() == [False, False, True]
    assert df["cef"].isna().tolist() == [False, True, False]
    assert df.loc[0, "amp"] == 1 and df.loc[2, "cef"] == 1


def test_single_phenotype_column_still_works(tmp_path):
    g = _genomes(tmp_path, 2)
    f = _write(tmp_path, f"path\tphenotype\n{g[0]}\t1\n{g[1]}\t0\n")

    df = validate_references(f)
    assert phenotype_columns(df) == ["phenotype"]
    assert df["phenotype"].tolist() == [1, 0]


def test_invalid_value_names_column(tmp_path):
    g = _genomes(tmp_path, 2)
    f = _write(tmp_path, f"path\tamp\tcef\n{g[0]}\t1\t0\n{g[1]}\t2\t1\n")

    with pytest.raises(ValueError, match="'amp'"):
        validate_references(f)


def test_no_phenotype_columns(tmp_path):
    g = _genomes(tmp_path, 1)
    f = _write(tmp_path, f"path\n{g[0]}\n")

    with pytest.raises(ValueError, match="no phenotype columns"):
        validate_references(f)


def test_all_missing_column(tmp_path):
    g = _genomes(tmp_path, 2)
    f = _write(tmp_path, f"path\tamp\tcef\n{g[0]}\t1\t\n{g[1]}\t0\t\n")

    with pytest.raises(ValueError, match="'cef'"):
        validate_references(f)


def test_missing_path_column(tmp_path):
    f = _write(tmp_path, "genome\tamp\nx.fa\t1\n")
    with pytest.raises(ValueError, match="'path'"):
        validate_references(f)


def test_labelled_subset_drops_na_and_casts():
    df = pd.DataFrame(
        {"path": ["a", "b", "c"], "amp": pd.array([1, None, 0], dtype="Int64"), "cef": pd.array([0, 1, None], dtype="Int64")}
    )

    sub = labelled_subset(df, "amp")
    assert list(sub.columns) == ["path", "phenotype"]
    assert sub["path"].tolist() == ["a", "c"]
    assert sub["phenotype"].tolist() == [1, 0]
    assert sub["phenotype"].dtype.kind == "i"

    with pytest.raises(KeyError):
        labelled_subset(df, "nope")
