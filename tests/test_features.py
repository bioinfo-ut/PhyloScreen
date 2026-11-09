"""Unit tests for feature extraction from mash screen results (no mash binary needed)."""

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

    assert fs.metadata["n_hits_total"] == 3
    assert fs.metadata["n_hits_1"] == 2
    assert fs.metadata["n_hits_0"] == 1
    assert fs.metadata["top_hit_1"] == "ref_a.fa"
    assert fs.metadata["top_hit_0"] == "ref_b.fa"
