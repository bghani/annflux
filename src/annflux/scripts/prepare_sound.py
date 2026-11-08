"""
Prepare a folder of audio recordings as an AnnFlux project.

Standalone on purpose: this script does NOT import annflux, so it can run in a
separate environment (see requirements.sound.txt).

    python prepare_sound.py AUDIO_DIR PROJECT_DIR --backend birdnet

Writes into PROJECT_DIR (all row-aligned, same order):
    images.csv          image_id,label   input for `annflux go` (label is a placeholder)
    images/{uid}.jpg    spectrogram thumbnail per segment
    segments.csv        uid,recording_path,start_s,end_s,sample_rate
    features.npz        features (N x D float32), uids (N,)
    sound_project.json  parameters used, incl. audio_root that recording_path is relative to

uid = {slugified recording path relative to audio_root}__{start_ms:08d}
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import re
from pathlib import Path

import librosa
import numpy as np
import soundfile
from PIL import Image

PREPARE_VERSION = "1.0"
AUDIO_EXTENSIONS = {".wav", ".flac"}
ANALYSIS_SR = 32000  # sample rate used for thumbnails and the dummy backend
N_FFT = 1024
# Fixed dB range so thumbnails are comparable across segments and recordings
DB_MIN, DB_MAX = -40.0, 40.0
# Viridis anchor colours, interpolated to a 256 entry lookup table
_VIRIDIS = [
    (68, 1, 84),
    (71, 44, 122),
    (59, 81, 139),
    (44, 113, 142),
    (33, 144, 141),
    (39, 173, 129),
    (92, 200, 99),
    (170, 220, 50),
    (253, 231, 37),
]
_x = np.linspace(0, 1, len(_VIRIDIS))
COLORMAP = np.stack(
    [np.interp(np.linspace(0, 1, 256), _x, [c[i] for c in _VIRIDIS]) for i in range(3)],
    axis=1,
).astype(np.uint8)


# ---------------------------------------------------------------------------
# uids and windows
# ---------------------------------------------------------------------------
def slugify(text: str) -> str:
    """Lowercase, every run of non [a-z0-9] characters becomes one '_'.

    Never contains '__', so '__' is unambiguous as uid separator. Also never
    contains '-', ':' or '.', which AnnFlux init would rewrite.
    """
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "recording"


def uid_prefixes(relpaths: list[str]) -> dict[str, str]:
    """Map recording relpath -> uid prefix, disambiguating slug collisions with a short hash"""
    slugs = {p: slugify(str(Path(p).with_suffix(""))) for p in relpaths}
    counts: dict[str, int] = {}
    for slug in slugs.values():
        counts[slug] = counts.get(slug, 0) + 1
    return {
        p: s if counts[s] == 1 else f"{s}_{hashlib.sha1(p.encode()).hexdigest()[:6]}"
        for p, s in slugs.items()
    }


def make_uid(prefix: str, start_s: float) -> str:
    return f"{prefix}__{int(round(start_s * 1000)):08d}"


def window_starts(
    num_samples: int, sr: int, window_s: float, hop_s: float, min_segment_s: float
) -> list[float]:
    """Consecutive window start times; a trailing window shorter than min_segment_s is dropped.

    Same rule as birdnetlib's chunking, so both backends produce the same segments.
    """
    hop = int(hop_s * sr)
    return [
        i / sr
        for i in range(0, num_samples, hop)
        if min(num_samples - i, int(window_s * sr)) >= int(min_segment_s * sr)
    ]


def segment_audio(y: np.ndarray, sr: int, start_s: float, window_s: float) -> np.ndarray:
    """Slice of y, zero-padded to a full window so all segments have equal length"""
    n = int(window_s * sr)
    start = int(start_s * sr)
    seg = y[start : start + n]
    return np.pad(seg, (0, n - len(seg)))


# ---------------------------------------------------------------------------
# thumbnails
# ---------------------------------------------------------------------------
def spectrogram_thumbnail(
    seg: np.ndarray, sr: int, fmin: float, fmax: float, size: tuple[int, int]
) -> Image.Image:
    mel = librosa.feature.melspectrogram(
        y=seg, sr=sr, n_fft=N_FFT, hop_length=N_FFT // 4, n_mels=128, fmin=fmin, fmax=fmax
    )
    db = librosa.power_to_db(mel, ref=1.0, top_db=None)
    norm = np.clip((db - DB_MIN) / (DB_MAX - DB_MIN), 0.0, 1.0)
    rgb = COLORMAP[(norm[::-1] * 255).astype(np.uint8)]  # low frequencies at the bottom
    return Image.fromarray(rgb).resize(size, Image.BILINEAR)


# ---------------------------------------------------------------------------
# embedding backends
# ---------------------------------------------------------------------------
class Backend:
    """Embeds one recording; returns segment start times and one feature row per segment"""

    name: str
    native_window_s: float

    def __init__(self, window_s: float, hop_s: float, min_segment_s: float):
        self.window_s, self.hop_s, self.min_segment_s = window_s, hop_s, min_segment_s

    def embed(self, path: Path, y: np.ndarray, sr: int) -> tuple[list[float], np.ndarray]:
        """y is the recording at `sr` (mono), path the original file"""
        raise NotImplementedError


class DummyBackend(Backend):
    """Deterministic and cheap: mean and std of log-mel energy per band. For tests / no BirdNET."""

    name = "dummy"
    native_window_s = 3.0

    def embed(self, path, y, sr):
        starts = window_starts(len(y), sr, self.window_s, self.hop_s, self.min_segment_s)
        features = []
        for start in starts:
            mel = librosa.feature.melspectrogram(
                y=segment_audio(y, sr, start, self.window_s), sr=sr, n_fft=N_FFT, n_mels=64
            )
            log_mel = np.log(mel + 1e-6)
            features.append(np.concatenate([log_mel.mean(axis=1), log_mel.std(axis=1)]))
        return starts, np.array(features, dtype=np.float32).reshape(len(starts), -1)


class BirdnetBackend(Backend):
    """BirdNET embeddings via birdnetlib, one pass over the whole recording.

    birdnetlib always uses 3 s chunks, hop = 3 - overlap, and drops trailing chunks < 1.5 s.
    """

    name = "birdnet"
    native_window_s = 3.0
    birdnet_min_segment_s = 1.5

    def __init__(self, window_s, hop_s, min_segment_s):
        super().__init__(window_s, hop_s, min_segment_s)
        if window_s != 3.0 or not 0 < hop_s <= 3.0:
            raise ValueError("birdnet backend requires --window_s 3 and 0 < --hop_s <= 3")
        if min_segment_s != self.birdnet_min_segment_s:
            raise ValueError("birdnet backend requires --min_segment_s 1.5")
        from birdnetlib.analyzer import Analyzer

        self.analyzer = Analyzer()

    def embed(self, path, y, sr):
        from birdnetlib import Recording

        recording = Recording(self.analyzer, str(path), overlap=self.window_s - self.hop_s)
        recording.extract_embeddings()
        starts = [round(e["start_time"], 3) for e in recording.embeddings]
        features = np.array([e["embeddings"] for e in recording.embeddings], dtype=np.float32)
        return starts, features.reshape(len(starts), -1)


class BacpipeBackend(Backend):
    # TODO(bacpipe): not implemented; the API was not verified against an installed package.
    #  An adapter only needs to implement `embed` and return (starts, features).
    name = "bacpipe"
    native_window_s = 3.0

    def __init__(self, *args):
        raise NotImplementedError("bacpipe backend is not implemented yet")


BACKENDS = {b.name: b for b in [DummyBackend, BirdnetBackend, BacpipeBackend]}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def find_recordings(audio_dir: Path) -> list[Path]:
    return sorted(
        p for p in audio_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def prepare(
    audio_dir: str | Path,
    out_dir: str | Path,
    backend: str = "dummy",
    window_s: float | None = None,
    hop_s: float | None = None,
    min_segment_s: float | None = None,
    fmin: float = 150.0,
    fmax: float = 15000.0,
    thumb_size: tuple[int, int] = (224, 224),
    audio_root: str | Path | None = None,
):
    audio_dir = Path(audio_dir).resolve()
    audio_root = Path(audio_root).resolve() if audio_root is not None else audio_dir
    out_dir = Path(out_dir)
    backend_cls = BACKENDS[backend]
    window_s = window_s or backend_cls.native_window_s
    hop_s = hop_s or window_s
    min_segment_s = min_segment_s if min_segment_s is not None else window_s / 2
    embedder = backend_cls(window_s, hop_s, min_segment_s)

    recordings = find_recordings(audio_dir)
    if not recordings:
        raise FileNotFoundError(f"No {sorted(AUDIO_EXTENSIONS)} files found in {audio_dir}")
    relpaths = [p.relative_to(audio_root).as_posix() for p in recordings]
    prefixes = uid_prefixes(relpaths)

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    segments, features = [], []
    for path, relpath in zip(recordings, relpaths):
        info = soundfile.info(str(path))
        y, _ = librosa.load(str(path), sr=ANALYSIS_SR, mono=True)
        starts, feats = embedder.embed(path, y, ANALYSIS_SR)
        print(f"{relpath}: {len(starts)} segments")
        for start in starts:
            uid = make_uid(prefixes[relpath], start)
            spectrogram_thumbnail(
                segment_audio(y, ANALYSIS_SR, start, window_s), ANALYSIS_SR, fmin, fmax, thumb_size
            ).save(images_dir / f"{uid}.jpg", quality=90)
            end = min(start + window_s, info.duration)
            segments.append((uid, relpath, f"{start:.3f}", f"{end:.3f}", info.samplerate))
        features.append(feats)

    uids = [s[0] for s in segments]
    assert len(set(uids)) == len(uids), "uids are not unique"
    features = np.concatenate(features)

    with open(out_dir / "segments.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["uid", "recording_path", "start_s", "end_s", "sample_rate"])
        writer.writerows(segments)
    # "0,1" is the same placeholder `annflux init` writes for images; real labels live in annflux/labels.json
    with open(out_dir / "images.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label"])
        writer.writerows((uid, "0,1") for uid in uids)
    np.savez(out_dir / "features.npz", features=features, uids=np.array(uids))
    with open(out_dir / "sound_project.json", "w") as f:
        json.dump(
            {
                "audio_root": str(audio_root),
                "backend": backend,
                "window_s": window_s,
                "hop_s": hop_s,
                "min_segment_s": min_segment_s,
                "fmin": fmin,
                "fmax": fmax,
                "db_range": [DB_MIN, DB_MAX],
                "thumb_size": list(thumb_size),
                "embedding_dim": int(features.shape[1]),
                "prepare_version": PREPARE_VERSION,
                "prepared_at": datetime.datetime.now().isoformat(timespec="seconds"),
            },
            f,
            indent=2,
        )
    print(f"Wrote {len(uids)} segments, features {features.shape} to {out_dir}")


def main(arg_list: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("audio_dir", help="Folder with recordings (searched recursively)")
    parser.add_argument("out_dir", help="AnnFlux project folder to write")
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="birdnet")
    parser.add_argument("--window_s", type=float, help="Default: backend's native window")
    parser.add_argument("--hop_s", type=float, help="Default: window_s (no overlap)")
    parser.add_argument("--min_segment_s", type=float, help="Default: window_s / 2")
    parser.add_argument("--fmin", type=float, default=150.0)
    parser.add_argument("--fmax", type=float, default=15000.0)
    parser.add_argument("--thumb_size", type=int, nargs=2, default=[224, 224], metavar=("W", "H"))
    parser.add_argument(
        "--audio_root", help="recording_path is stored relative to this. Default: audio_dir"
    )
    args = parser.parse_args(arg_list)
    prepare(
        args.audio_dir,
        args.out_dir,
        backend=args.backend,
        window_s=args.window_s,
        hop_s=args.hop_s,
        min_segment_s=args.min_segment_s,
        fmin=args.fmin,
        fmax=args.fmax,
        thumb_size=tuple(args.thumb_size),
        audio_root=args.audio_root,
    )


if __name__ == "__main__":
    main()
