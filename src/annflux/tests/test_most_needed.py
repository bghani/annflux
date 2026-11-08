import numpy as np

from annflux.algorithms.most_needed import diversify


def species_ordered_features(num_classes=5, per_class=40, dim=8):
    """Features sorted by class, as when most needed indices come from a species-sorted dataset"""
    rng = np.random.default_rng(0)
    centers = rng.normal(scale=10.0, size=(num_classes, dim))
    features = np.concatenate(
        [center + rng.normal(scale=0.1, size=(per_class, dim)) for center in centers]
    ).astype(np.float32)
    classes = np.repeat(np.arange(num_classes), per_class)
    return features, classes


def test_diversify_spans_classes():
    np.random.seed(0)
    features, classes = species_ordered_features()
    picked = diversify(features, diversify_from=9)
    assert len(picked) == len(set(picked)) == 10
    assert set(classes[picked]) == set(range(5))


def test_diversify_fewer_items_than_requested():
    np.random.seed(0)
    features, _ = species_ordered_features(num_classes=2, per_class=3)
    picked = diversify(features, diversify_from=50)
    assert sorted(picked) == list(range(len(features)))
