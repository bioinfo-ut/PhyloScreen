"""Prediction tests with mash screen faked (see conftest.py)."""

import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phyloscreen import predict as predict_mod
from phyloscreen.model import Model
from phyloscreen.predict import (
    OUTPUT_COLUMNS, Query, default_query_id, predict_phenotype, predict_queries, read_query_list,
)
from tests.conftest import make_fake_screen


# --- query ids and list parsing -------------------------------------------------

def test_default_query_id(tmp_path):
    assert default_query_id(Path("/x/sample.fa")) == "sample"
    assert default_query_id(Path("/x/sample.fastq.gz")) == "sample"
    assert default_query_id(Path("/x/iso.v2.fna")) == "iso.v2"
    d = tmp_path / "reads_dir"
    d.mkdir()
    assert default_query_id(d) == "reads_dir"


def test_read_query_list_ids_and_fallbacks(tmp_path):
    d = tmp_path / "iso3"
    d.mkdir()
    f = tmp_path / "q.tsv"
    f.write_text(f"path\tid\n/a/iso1.fa\tA\n/a/iso2.fq.gz\t\n{d}\t  \n")

    qs = read_query_list(f)

    assert [q.id for q in qs] == ["A", "iso2", "iso3"]
    assert [str(q.path) for q in qs] == ["/a/iso1.fa", "/a/iso2.fq.gz", str(d)]


def test_read_query_list_path_only(tmp_path):
    f = tmp_path / "q.tsv"
    f.write_text("path\n/a/x.fa\n/a/y.fa\n")
    assert [q.id for q in read_query_list(f)] == ["x", "y"]


@pytest.mark.parametrize(
    "content, match",
    [
        ("sample\n/a/x.fa\n", "'path' column"),
        ("path\tid\n/a/x.fa\tA\n/a/y.fa\tA\n", "Duplicate query ids"),
        ("path\n/a/x.fa\n/a/x.fa\n", "Duplicate query paths"),
        ("path\tid\n/a/x.fa\tA\n\tB\n", "empty 'path'"),
        ("path\n", "no queries"),
    ],
)
def test_read_query_list_errors(tmp_path, content, match):
    f = tmp_path / "q.tsv"
    f.write_text(content)
    with pytest.raises(ValueError, match=match):
        read_query_list(f)


# --- single query ---------------------------------------------------------------

def test_predict_one_row_per_phenotype(trained_bundle, monkeypatch):
    out, panel, _ = trained_bundle
    monkeypatch.setattr(predict_mod, "screen", make_fake_screen(panel, {"amp": 1, "cef": 1}))

    query = panel.dir / "query.fa"  # novel genome, hidden truth: amp=1, cef=1
    query.write_text(">q\nACGT\n")
    result_path = panel.dir / "pred.tsv"
    results = predict_phenotype(query, out, result_path, threads=1)

    assert [r.phenotype for r in results] == ["amp", "cef"]
    for r in results:
        assert r.query_id == "query"
        assert r.prediction in (0, 1)
        assert 0.0 <= r.probability <= 1.0
        assert r.prob_1 + r.prob_0 == pytest.approx(1.0)
        assert r.top_hit_1 is not None and r.top_hit_0 is not None
        assert r.top_hit_1_shared_hashes is not None

    # Signal in the fake similarities is strong enough to recover the truth
    assert results[0].prediction == 1
    assert results[1].prediction == 1

    df = pd.read_csv(result_path, sep="\t")
    assert list(df.columns) == OUTPUT_COLUMNS
    assert df["query_id"].tolist() == ["query", "query"]
    assert df["phenotype"].tolist() == ["amp", "cef"]
    assert (df["query_path"] == str(query)).all()


def test_predict_respects_missing_labels(trained_bundle, monkeypatch):
    """g02 and g03 have no 'cef' label: they must never be a cef top hit."""
    out, panel, _ = trained_bundle
    monkeypatch.setattr(predict_mod, "screen", make_fake_screen(panel))

    # g02 is one of g03's closest neighbours (same amp cluster); both are cef-unlabelled.
    results = predict_phenotype(Path(panel.paths[3]), out, panel.dir / "p.tsv", threads=1)
    cef = next(r for r in results if r.phenotype == "cef")
    unlabelled = {panel.paths[2], panel.paths[3]}
    assert cef.top_hit_1 not in unlabelled
    assert cef.top_hit_0 not in unlabelled


def test_predict_empty_screen_gives_unknown(trained_bundle, monkeypatch):
    out, panel, _ = trained_bundle
    monkeypatch.setattr(predict_mod, "screen", lambda *a, **k: pd.DataFrame())

    results = predict_phenotype(Path(panel.paths[0]), out, panel.dir / "e.tsv", threads=1)

    assert [r.prediction for r in results] == ["unknown", "unknown"]
    df = pd.read_csv(panel.dir / "e.tsv", sep="\t")
    assert len(df) == 2
    assert df["prediction"].tolist() == ["unknown", "unknown"]


def test_predict_single_query_fails_fast(trained_bundle, monkeypatch):
    out, panel, _ = trained_bundle
    monkeypatch.setattr(predict_mod, "screen", make_fake_screen(panel))

    with pytest.raises(FileNotFoundError):
        predict_phenotype(panel.dir / "missing.fa", out, panel.dir / "x.tsv")


def test_predict_legacy_bundle(tmp_path, monkeypatch):
    refs = ["a.fa", "b.fa", "c.fa", "d.fa"]
    labels = [1, 0, 1, 0]
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)),
    ])
    X = np.array([[1, 2.0, 0.9], [0, 0.5, 0.1], [1, 1.5, 0.8], [0, 0.7, 0.2]])
    pipeline.fit(X, labels)
    joblib.dump(pipeline, tmp_path / "model.joblib")
    (tmp_path / "labels.tsv").write_text("path\tphenotype\n" + "".join(f"{r}\t{l}\n" for r, l in zip(refs, labels)))
    (tmp_path / "reference_sketch.msh").write_bytes(b"")

    screen_df = pd.DataFrame(
        {
            "identity": [0.99, 0.9, 0.8, 0.7],
            "shared_hashes": [900, 700, 500, 300],
            "total_hashes": [1000] * 4,
            "median_multiplicity": [1.0] * 4,
            "p_value": [0.0] * 4,
            "query_comment": [""] * 4,
            "reference_path": refs,
        }
    )
    monkeypatch.setattr(predict_mod, "screen", lambda *a, **k: screen_df)

    query = tmp_path / "q.fa"
    query.write_text(">q\nACGT\n")
    results = predict_phenotype(query, tmp_path, tmp_path / "out.tsv")

    assert len(results) == 1
    assert results[0].phenotype == "phenotype"
    assert results[0].prediction == 1
    assert results[0].top_hit_1 == "a.fa" and results[0].top_hit_0 == "b.fa"
    df = pd.read_csv(tmp_path / "out.tsv", sep="\t")
    assert df["phenotype"].tolist() == ["phenotype"]


# --- batch ----------------------------------------------------------------------

def _novel(panel, name):
    p = panel.dir / name
    p.write_text(">q\nACGT\n")
    return p


def test_predict_queries_batch(trained_bundle, monkeypatch):
    out, panel, _ = trained_bundle
    q1, q2, q3 = (_novel(panel, n) for n in ("s1.fa", "s2.fa", "s3.fa.gz"))
    truth = {
        str(q1): {"amp": 1, "cef": 1},
        str(q2): {"amp": 0, "cef": 0},
        str(q3): {"amp": 1, "cef": 0},
    }
    monkeypatch.setattr(predict_mod, "screen", make_fake_screen(panel, truth))

    loads = []
    real_load = Model.load
    monkeypatch.setattr(Model, "load", classmethod(lambda cls, d: loads.append(d) or real_load(d)))

    seen = []
    queries = [Query("one", q1), Query("two", q2), Query("three", q3)]
    batch = predict_queries(queries, out, panel.dir / "batch.tsv", threads=1, on_progress=seen.append)

    assert len(loads) == 1, "bundle must be loaded once for the whole batch"
    assert seen == queries
    assert batch.failures == {}
    assert batch.n_queries == 3
    assert len(batch.results) == 6

    df = pd.read_csv(panel.dir / "batch.tsv", sep="\t")
    assert list(df.columns) == OUTPUT_COLUMNS
    assert df["query_id"].tolist() == ["one", "one", "two", "two", "three", "three"]
    assert df["phenotype"].tolist() == ["amp", "cef"] * 3
    got = {(r.query_id, r.phenotype): r.prediction for r in batch.results}
    assert got == {
        ("one", "amp"): 1, ("one", "cef"): 1,
        ("two", "amp"): 0, ("two", "cef"): 0,
        ("three", "amp"): 1, ("three", "cef"): 0,
    }


def test_predict_queries_continues_after_failures(trained_bundle, monkeypatch):
    out, panel, _ = trained_bundle
    good = _novel(panel, "good.fa")
    boom = _novel(panel, "boom.fa")
    missing = panel.dir / "missing.fa"
    empty_dir = panel.dir / "empty"
    empty_dir.mkdir()

    fake = make_fake_screen(panel, {"amp": 1, "cef": 1})

    def screen_or_boom(query_paths, *a, **k):
        if str(query_paths[0]) == str(boom):
            raise subprocess.CalledProcessError(1, ["mash"], stderr="kaboom")
        return fake(query_paths, *a, **k)

    monkeypatch.setattr(predict_mod, "screen", screen_or_boom)

    queries = [Query("good", good), Query("boom", boom), Query("missing", missing), Query("empty", empty_dir)]
    batch = predict_queries(queries, out, panel.dir / "b.tsv", threads=1)

    assert set(batch.failures) == {"boom", "missing", "empty"}
    assert "CalledProcessError" in batch.failures["boom"]
    assert "FileNotFoundError" in batch.failures["missing"]
    assert "ValueError" in batch.failures["empty"]

    df = pd.read_csv(panel.dir / "b.tsv", sep="\t")
    assert len(df) == 8  # 4 isolates x 2 phenotypes, failures included
    good_rows = df[df["query_id"] == "good"]
    assert good_rows["prediction"].tolist() == ["1", "1"]
    for qid in ("boom", "missing", "empty"):
        assert df[df["query_id"] == qid]["prediction"].tolist() == ["error", "error"]
    assert df[df["query_id"] == "missing"]["top_hit_1"].isna().all()
