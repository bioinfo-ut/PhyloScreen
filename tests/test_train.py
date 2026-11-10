"""End-to-end training tests with mash faked (see conftest.py)."""

import json

import pandas as pd
import pytest

from phyloscreen.model import Model
from phyloscreen.train import _check_class_counts


def test_check_class_counts_reports_sparse_phenotypes():
    df = pd.DataFrame(
        {
            "path": [f"g{i}" for i in range(8)],
            "ok": pd.array([1, 1, 1, 1, 0, 0, 0, 0], dtype="Int64"),
            "one_pos": pd.array([1, 0, 0, 0, 0, 0, 0, 0], dtype="Int64"),
            "one_class": pd.array([1, 1, 1, None, None, 1, 1, 1], dtype="Int64"),
        }
    )

    _check_class_counts(df, ["ok"], cv_folds=3)

    with pytest.raises(ValueError) as exc:
        _check_class_counts(df, ["ok", "one_pos", "one_class"], cv_folds=3)
    msg = str(exc.value)
    assert "one_pos: n0=7, n1=1" in msg
    assert "one_class: n0=0, n1=6" in msg
    assert "ok:" not in msg


def test_train_model_multi_phenotype(trained_bundle):
    out, panel, result = trained_bundle

    assert result.phenotypes == ["amp", "cef"]
    assert result.n_references == 12
    assert result.n_features == 3
    assert result.results["amp"].n_labelled == 11
    assert result.results["cef"].n_labelled == 10
    for r in result.results.values():
        assert 0.0 <= r.auc <= 1.0
        assert r.best_C > 0

    for name in ("reference_sketch.msh", "model.joblib", "labels.tsv", "coefficients.json", "metadata.json"):
        assert (out / name).exists(), name
    assert len(list((out / "screen_results").glob("*.tsv"))) == 12

    meta = json.loads((out / "metadata.json").read_text())
    assert meta["phenotypes"] == ["amp", "cef"]
    assert set(meta["best_C"]) == {"amp", "cef"}
    assert meta["kmer_size"] == 21 and meta["sketch_size"] == 1000

    bundle = Model.load(out)
    assert bundle.phenotypes == ["amp", "cef"]
    # Missing labels survive the round trip
    assert bundle.references["cef"].isna().sum() == 2
    assert bundle.references["amp"].isna().sum() == 1


def test_train_model_intermediate_files(panel, monkeypatch):
    from phyloscreen import train as train_mod
    from tests.conftest import fake_sketch, make_fake_screen

    monkeypatch.setattr(train_mod, "check_mash", lambda: "fake")
    monkeypatch.setattr(train_mod, "sketch", fake_sketch)
    monkeypatch.setattr(train_mod, "screen", make_fake_screen(panel))

    inter = panel.dir / "inter"
    result = train_mod.train_model(
        panel.refs_tsv, panel.dir / "m", kmer_size=21, sketch_size=1000, cv_folds=3, seed=0,
        intermediate_dir=inter,
    )

    feats = pd.read_csv(inter / "features.tsv", sep="\t")
    assert list(feats.columns[:3]) == ["phenotype", "path", "label"]
    assert feats["phenotype"].value_counts().to_dict() == {"amp": 11, "cef": 10}
    assert set(feats["label"]) == {0, 1}
    for col in ("top_phenotype", "shared_hashes_ratio", "weighted_vote_top10", "n_hits_total", "top_hit_1"):
        assert col in feats.columns, col
    # self-hit excluded: every reference sees the other labelled references only
    assert feats.loc[feats["phenotype"] == "amp", "n_hits_total"].eq(10).all()
    assert feats.loc[feats["phenotype"] == "cef", "n_hits_total"].eq(9).all()

    cv = pd.read_csv(inter / "cv_predictions.tsv", sep="\t")
    assert list(cv.columns) == ["phenotype", "fold", "C", "path", "y_true", "y_pred", "y_proba"]
    assert cv["C"].nunique() == 6
    assert len(cv) == (11 + 10) * 6
    assert set(cv["fold"]) == {1, 2, 3}
    # every labelled reference is validated exactly once per C and phenotype
    per = cv.groupby(["phenotype", "C"])["path"].nunique()
    assert per.loc["amp"].eq(11).all() and per.loc["cef"].eq(10).all()
    assert cv["y_proba"].between(0, 1).all()
    assert set(cv["phenotype"]) == set(result.phenotypes)


def test_train_model_fails_on_sparse_phenotype(panel, monkeypatch):
    from phyloscreen import train as train_mod

    table = panel.table.copy()
    table["rare"] = pd.array([1] + [0] * 11, dtype="Int64")
    table.to_csv(panel.refs_tsv, sep="\t", index=False, na_rep="")

    monkeypatch.setattr(train_mod, "check_mash", lambda: "fake")
    called = []
    monkeypatch.setattr(train_mod, "sketch", lambda *a, **k: called.append(1))

    with pytest.raises(ValueError, match="rare: n0=11, n1=1"):
        train_mod.train_model(panel.refs_tsv, panel.dir / "m", kmer_size=21, sketch_size=1000)
    assert not called, "sketch must not run when validation fails"
