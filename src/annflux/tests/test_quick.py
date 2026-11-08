import json
import os

import numpy as np
import pandas

from annflux.repo_results_to_embedding import embed_and_prepare
from annflux.shared import AnnfluxSource
from annflux.train_indeed_image import init_folder
from annflux.training.annflux.feature_extractor import make_resultset
from annflux.training.annflux.quick import make_predictions
from annflux.ui.basic import run_server


def test_make_predictions_skips_missing_neighbours_and_zero_distance():
    data = pandas.DataFrame({"uid": ["a", "b", "c"]})
    train_labels = np.array([["x"], ["y"], None], dtype="object")
    # faiss pads with index -1 / huge distance when k > number of items; a duplicate has distance 0
    indices = np.array([[1, 0, -1]])
    distances = np.array([[0.0, 1.0, 3.4e38]], dtype=np.float32)
    make_predictions({}, data, indices, distances, train_labels, [2], None, set(), None)
    assert data.at[2, "label_predicted"] == "y"
    assert np.isfinite(data.at[2, "score_predicted"])


def make_project(folder, num_items=60, num_classes=3, dim=16):
    rng = np.random.default_rng(0)
    classes = np.arange(num_items) % num_classes
    features = rng.normal(size=(num_classes, dim))[classes] + rng.normal(
        scale=0.3, size=(num_items, dim)
    )
    features[1] = features[0]  # exact duplicate
    uids = [f"item_{i:03d}" for i in range(num_items)]
    os.makedirs(folder / "images")
    pandas.DataFrame({"image_id": uids, "label": "0,1"}).to_csv(
        folder / "images.csv", index=False
    )
    source = init_folder(AnnfluxSource(folder))
    with open(source.split_path, "w") as f:
        json.dump({"test": uids[-6:]}, f)
    make_resultset(source.dataset, features.astype(np.float32), source.repository)
    embed_and_prepare(source)
    # label 3 of every 4 items, including the original of the duplicate but not the duplicate
    return {uid: f"class_{c}" for i, (uid, c) in enumerate(zip(uids, classes)) if i % 4 != 1}


def test_retraining_with_fewer_items_than_k(tmp_path):
    labels = make_project(tmp_path)
    os.environ["PROJECT_ROOT"] = str(tmp_path)
    run_server._init()
    client = run_server.get_app().test_client()

    assert client.post("/label", json=labels).get_json() == {"success": True}
    state = run_server.g_state
    assert (state.all_indices >= 0).all()
    run_server.retrain_job(state)

    data = pandas.read_csv(tmp_path / "annflux" / "annflux.csv")
    assert data.score_predicted.notna().all()
    assert data.loc[1, "label_predicted"] == "class_0"  # its duplicate's label
