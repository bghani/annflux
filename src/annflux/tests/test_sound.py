"""
End-to-end test of sound support: scripts/prepare_sound.py -> annflux go -> UI server.

The prepare script runs in a subprocess with ANNFLUX_SOUND_PYTHON (default: this interpreter),
because it is meant to run in its own environment (requirements.sound.txt).
"""

import io
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas
import pytest

soundfile = pytest.importorskip("soundfile")

import annflux.scripts  # noqa: E402
from annflux.scripts.annflux_cli import go_command  # noqa: E402
from annflux.ui.basic.run_server import _init, get_app  # noqa: E402

PREPARE_SCRIPT = Path(annflux.scripts.__file__).parent / "prepare_sound.py"
PREPARE_PYTHON = os.getenv("ANNFLUX_SOUND_PYTHON", sys.executable)
# relpath, duration (s), sample rate; "site-a/rec" and "site_a/rec" give the same slug
RECORDINGS = [
    ("site-a/rec.wav", 18.0, 48000),
    ("site_a/rec.flac", 20.0, 44100),
    ("other.wav", 15.0, 32000),
]
HOP_S = 1.0  # 3 s windows with 1 s hop: 47 segments, enough for UMAP/kNN defaults


def can_prepare(*modules) -> bool:
    imports = "; ".join(f"import {m}" for m in ("librosa", "PIL", *modules))
    return subprocess.run([PREPARE_PYTHON, "-c", imports]).returncode == 0


def make_recordings(audio_dir: Path):
    rng = np.random.default_rng(0)
    for i, (relpath, duration, sr) in enumerate(RECORDINGS):
        t = np.arange(int(duration * sr)) / sr
        tone = np.sin(2 * np.pi * (1000 + 2000 * i) * t) * (np.sin(np.pi * t) > 0)
        audio = 0.3 * tone + 0.02 * rng.standard_normal(len(t))
        (audio_dir / relpath).parent.mkdir(parents=True, exist_ok=True)
        soundfile.write(audio_dir / relpath, audio.astype(np.float32), sr)


def prepare(audio_dir: Path, project: Path, *args: str):
    subprocess.run(
        [PREPARE_PYTHON, "-I", str(PREPARE_SCRIPT), str(audio_dir), str(project), *args],
        check=True,
    )


@pytest.fixture(scope="module")
def project(tmp_path_factory) -> Path:
    if not can_prepare():
        pytest.skip(f"librosa/pillow not available for {PREPARE_PYTHON}")
    audio_dir = tmp_path_factory.mktemp("audio")
    make_recordings(audio_dir)
    project = tmp_path_factory.mktemp("project")
    prepare(audio_dir, project, "--backend", "dummy", "--hop_s", str(HOP_S))
    return project


@pytest.fixture(scope="module")
def client(project):
    go_command(str(project), ["song", "noise"])
    os.environ["PROJECT_ROOT"] = str(project)
    app = get_app()
    app.config.from_mapping(TESTING=True)
    _init()
    return app.test_client()


def test_prepare_output(project):
    segments = pandas.read_csv(project / "segments.csv", dtype={"uid": str})
    images = pandas.read_csv(project / "images.csv", dtype={"image_id": str})
    npz = np.load(project / "features.npz")
    meta = json.load(open(project / "sound_project.json"))

    assert len(segments) >= 40
    assert segments.uid.is_unique
    assert not segments.uid.str.contains(r"[^a-z0-9_]").any()
    # same slug for both site*a/rec recordings, disambiguated by a hash
    assert segments.groupby("recording_path").uid.first().str.split("__").str[0].is_unique
    assert all((project / "images" / f"{uid}.jpg").exists() for uid in segments.uid)
    # row-aligned outputs
    assert images.image_id.tolist() == segments.uid.tolist()
    assert npz["uids"].tolist() == segments.uid.tolist()
    assert npz["features"].shape == (len(segments), meta["embedding_dim"])
    assert np.isfinite(npz["features"]).all()
    assert meta["backend"] == "dummy" and meta["hop_s"] == HOP_S
    assert (segments.end_s - segments.start_s <= meta["window_s"]).all()


def test_annflux_flow(client, project):
    data = pandas.read_csv(project / "annflux" / "annflux.csv", dtype={"uid": str})
    segments = pandas.read_csv(project / "segments.csv", dtype={"uid": str})
    assert sorted(data.uid) == sorted(segments.uid)
    assert np.isfinite(data[["e_0", "e_1"]].values).all()
    assert data.label_true.isna().all()  # the "0,1" placeholder must not become labels

    response = client.get("/")
    assert b'const modality = "sound"' in response.data
    # labeling triggers quick reclassification on the precomputed features
    labels = {uid: "song" if i % 2 else "noise" for i, uid in enumerate(segments.uid[:12])}
    response = client.post("/label", json=labels)
    assert response.get_json() == {"success": True}


def _wav(response):
    assert response.status_code == 200
    assert response.mimetype == "audio/wav"
    audio, sr = soundfile.read(io.BytesIO(response.data))
    return len(audio) / sr


def test_sounds_endpoint(client, project):
    segments = pandas.read_csv(project / "segments.csv", dtype={"uid": str})
    first, middle = segments.iloc[0], segments.iloc[5]
    last = segments[segments.recording_path == first.recording_path].iloc[-1]

    assert _wav(client.get(f"/sounds/{middle.uid}")) == pytest.approx(3.0, abs=1e-3)
    assert _wav(client.get(f"/sounds/{middle.uid}?context=1")) == pytest.approx(5.0, abs=1e-3)
    # context is clipped to the recording bounds
    assert _wav(client.get(f"/sounds/{first.uid}?context=1")) == pytest.approx(4.0, abs=1e-3)
    duration = soundfile.info(
        Path(json.load(open(project / "sound_project.json"))["audio_root"])
        / last.recording_path
    ).duration
    expected = duration - (last.start_s - 2)
    assert _wav(client.get(f"/sounds/{last.uid}?context=2")) == pytest.approx(expected, abs=1e-3)

    assert client.get("/sounds/does_not_exist").status_code == 404
    assert client.get(f"/sounds/{middle.uid}?context=abc").status_code == 400
    assert client.get(f"/sounds/{middle.uid}?context=-1").status_code == 400


def test_sounds_endpoint_missing_recording(client, project, monkeypatch):
    monkeypatch.setenv("AUDIO_ROOT", str(project / "nowhere"))
    _init()  # resets the cached sound index
    uid = pandas.read_csv(project / "segments.csv", dtype={"uid": str}).uid[0]
    assert client.get(f"/sounds/{uid}").status_code == 404


def test_birdnet_backend(tmp_path):
    if not can_prepare("birdnetlib"):
        pytest.skip(f"birdnetlib not available for {PREPARE_PYTHON}")
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    make_recordings(audio_dir)
    (audio_dir / "site_a").rename(tmp_path / "unused")  # one recording is enough
    (audio_dir / "other.wav").unlink()
    prepare(audio_dir, tmp_path / "project", "--backend", "birdnet")
    meta = json.load(open(tmp_path / "project" / "sound_project.json"))
    features = np.load(tmp_path / "project" / "features.npz")["features"]
    print(f"BirdNET embedding dimension: {meta['embedding_dim']}")
    assert features.shape == (6, meta["embedding_dim"])  # 18 s in 3 s windows
