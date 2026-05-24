"""
Feature engineering for phenotype prediction.

Extracts ML features from mash screen results based on reference phenotypes.
"""

from dataclasses import dataclass, field

from phenoscreen.utils import logger

import numpy as np
import pandas as pd
from pandas import DataFrame

FEATURE_NAMES = ['top_phenotype', 'shared_hashes_ratio', 'weighted_vote_top10']


@dataclass
class FeatureSet:
    """Container for extracted features with metadata."""

    features: np.ndarray
    feature_names: list[str]
    query_path: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary of feature name -> value."""
        return dict(zip(self.feature_names, self.features))


class FeatureExtractor:
    """
    Extract ML features from mash screen results.

    Features include:
    - Top hit phenotype
    - Shared hashes ratio top 1 phenotype hit and top 0 phenotype hit
    - Weighted vote of top 10 hits
    """

    def __init__(
            self,
            references_df: DataFrame,
    ):
        """
        Initialize feature extractor.

        Args:
            references_df: Dict mapping reference genome ID to phenotype.
        """
        self.feature_names: list[str] = FEATURE_NAMES
        self.references_df = references_df

    def extract(self, screen_results: DataFrame, query_path: str = "") -> FeatureSet:
        """
        Extract features from mash screen results.

        Args:
            screen_results: DataFrame from mash.screen() with columns:
                identity, reference_path, etc.
            query_path: Identifier for the query genome.

        Returns:
            FeatureSet with extracted features.
        """
        # Separate by phenotype
        results_with_labels = self._annotate_phenotypes(screen_results)

        phenotype_1 = results_with_labels[results_with_labels["phenotype"] == 1]
        phenotype_0 = results_with_labels[results_with_labels["phenotype"] == 0]

        features = []

        top_phenotype = results_with_labels["phenotype"].iloc[0] if len(results_with_labels) > 0 else -1
        hash_ratio = phenotype_1['shared_hashes'].max() / phenotype_0['shared_hashes'].max()
        top10 = results_with_labels.head(10)
        weighted_vote_top10 = (top10['shared_hashes'] * top10['phenotype'].astype(int)).sum() / top10['shared_hashes'].sum()

        features.extend([top_phenotype, hash_ratio, weighted_vote_top10])

        metadata = {
            "n_hits_total": len(screen_results),
            "n_hits_1": len(phenotype_1),
            "n_hits_0": len(phenotype_0),
            "top_hit_1": (
                phenotype_1.iloc[0]["reference_path"] if len(phenotype_1) > 0 else None
            ),
            "top_hit_phenotype_1": (
                phenotype_1.iloc[0]["phenotype"] if len(phenotype_1) > 0 else None
            ),
            "top_hit_0": (
                phenotype_0.iloc[0]["reference_path"] if len(phenotype_0) > 0 else None
            ),
            "top_hit_phenotype_0": (
                phenotype_0.iloc[0]["phenotype"] if len(phenotype_0) > 0 else None
            ),
        }

        return FeatureSet(
            features=np.array(features),
            feature_names=self.feature_names,
            query_path=query_path,
            metadata=metadata,
        )

    def _annotate_phenotypes(self, screen_results: pd.DataFrame) -> pd.DataFrame:
        """Add phenotype column to screen results based on reference labels."""
        df = screen_results.copy()
        df = df.merge(self.references_df[["path", "phenotype"]], left_on="reference_path", right_on="path")

        # Log any missing phenotypes
        missing = df[df["phenotype"].isna()]["reference_path"].unique()
        if len(missing) > 0:
            logger.warning(
                "References without phenotype labels: %s", missing.tolist()
            )
            df = df.dropna(subset=["phenotype"])

        return df