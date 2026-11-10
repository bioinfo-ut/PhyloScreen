"""Unit tests for feature extraction from mash screen results (no mash binary needed)."""

import numpy as np
import pandas as pd
import pytest

from phyloscreen.features import FEATURE_NAMES, FeatureExtractor


@pytest.fixture
def references():
    return pd.DataFrame(
        {
            "path": ["ref_a.fa", "ref_b.fa", "ref_c.fa"],
            "phenotype": [1, 0, 1],
        }
    )


@pytest.fixture
def screen_results():
    # Already sorted by shared_hashes descending, as mash.parse_screen_output() returns it.
    return pd.DataFrame(
        {
            "identity": [0.99, 0.97, 0.95],
            "shared_hashes": [900, 600, 300],
            "total_hashes": [1000, 1000, 1000],
            "median_multiplicity": [1.0, 1.0, 1.0],
            "p_value": [0.0, 0.0, 0.0],
            "query_comment": ["", "", ""],
            "reference_path": ["ref_a.fa", "ref_b.fa", "ref_c.fa"],
        }
    )


@pytest.fixture
def multi_references():
    # ref_a is unlabelled for 'cef'; ref_b is unlabelled for 'amp'.
    return pd.DataFrame(
        {
            "path": ["ref_a.fa", "ref_b.fa", "ref_c.fa"],
            "amp": pd.array([1, None, 0], dtype="Int64"),
            "cef": pd.array([None, 0, 1], dtype="Int64"),
        }
    )


def test_feature_values(references, screen_results):
    fs = FeatureExtractor(references).extract(screen_results, query_path="query.fa")

    assert fs.feature_names == FEATURE_NAMES
    assert fs.query_path == "query.fa"

    f = fs.to_dict()
    assert f["top_phenotype"] == 1
    assert f["shared_hashes_ratio"] == pytest.approx(900 / 600)
    assert f["weighted_vote_top10"] == pytest.approx((900 + 300) / (900 + 600 + 300))


def test_metadata(references, screen_results):
    fs = FeatureExtractor(references).extract(screen_results)

    assert fs.metadata["phenotype"] == "phenotype"
    assert fs.metadata["n_hits_total"] == 3
    assert fs.metadata["n_hits_1"] == 2
    assert fs.metadata["n_hits_0"] == 1
    assert fs.metadata["top_hit_1"] == "ref_a.fa"
    assert fs.metadata["top_hit_0"] == "ref_b.fa"


def test_unlabelled_references_are_ignored(multi_references, screen_results):
    # For 'cef', ref_a (the closest hit) has no label and must not be the top hit.
    fs = FeatureExtractor(multi_references, "cef").extract(screen_results)
    f = fs.to_dict()

    assert fs.metadata["phenotype"] == "cef"
    assert fs.metadata["n_hits_total"] == 2
    assert f["top_phenotype"] == 0  # ref_b (600) is the nearest labelled isolate
    assert f["shared_hashes_ratio"] == pytest.approx(300 / 600)
    assert f["weighted_vote_top10"] == pytest.approx(300 / (600 + 300))
    assert fs.metadata["top_hit_1"] == "ref_c.fa"
    assert fs.metadata["top_hit_0"] == "ref_b.fa"

    # For 'amp', ref_b is unlabelled; ref_a (1) and ref_c (0) remain.
    f = FeatureExtractor(multi_references, "amp").extract(screen_results).to_dict()
    assert f["top_phenotype"] == 1
    assert f["shared_hashes_ratio"] == pytest.approx(900 / 300)
    assert f["weighted_vote_top10"] == pytest.approx(900 / (900 + 300))


def test_degenerate_no_class0_hits_is_finite(screen_results):
    refs = pd.DataFrame({"path": ["ref_a.fa", "ref_b.fa", "ref_c.fa"], "phenotype": [1, 1, 1]})
    fs = FeatureExtractor(refs).extract(screen_results)

    assert np.isfinite(fs.features).all()
    assert fs.to_dict()["shared_hashes_ratio"] == pytest.approx(900 / 1)
    assert fs.metadata["top_hit_0"] is None


def test_no_hits_at_all(references):
    empty = pd.DataFrame(columns=["identity", "shared_hashes", "total_hashes", "median_multiplicity",
                                  "p_value", "query_comment", "reference_path"])
    fs = FeatureExtractor(references).extract(empty)

    assert np.isfinite(fs.features).all()
    assert fs.to_dict()["top_phenotype"] == -1
    assert fs.metadata["n_hits_total"] == 0
