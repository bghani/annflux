import numpy as np

from annflux.repository.dataset import Dataset
from annflux.training.annflux.feature_extractor import BaseFeatureExtractor


class PrecomputedFeatureExtractor(BaseFeatureExtractor):
    """
    Features computed outside AnnFlux, e.g. by scripts/prepare_sound.py.

    Reads an npz with arrays `features` (N x D) and `uids` (N,) and returns the rows in dataset order.
    """

    def __init__(self, features_path: str):
        self.features_path = features_path

    def compute_features(
        self, dataset: Dataset, multi=False, batch_size=32
    ) -> np.array:
        with np.load(self.features_path) as npz:
            features, uids = npz["features"], npz["uids"].astype(str)
        row_by_uid = {uid: row for row, uid in enumerate(uids)}
        dataset_uids = dataset.as_dataframe().uid
        missing = [uid for uid in dataset_uids if uid not in row_by_uid]
        if missing:
            raise ValueError(
                f"{len(missing)} uids have no features in {self.features_path}, e.g. {missing[:3]}"
            )
        return features[[row_by_uid[uid] for uid in dataset_uids]]
