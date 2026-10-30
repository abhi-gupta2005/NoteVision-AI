"""
Shared fixtures: synthetic audio with known pitches, onsets and durations.

Every test signal is generated from a list of (midi, start_s, duration_s)
tuples so the ground truth is exact and no real recordings are needed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep the test database / uploads out of the real data folder.
os.environ.setdefault("NOTEVISION_LOG_LEVEL", "WARNING")

from config import settings  # noqa: E402
from src.pitch_detection import midi_to_frequency  # noqa: E402

SR = 22050
NoteSpec = tuple[int, float, float]  # (midi, start_s, duration_s)


def synth_notes(notes: list[NoteSpec], sr: int = SR, total: float | None = None, harmonics: int = 4,
                amp: float = 0.3, decay: float = 2.0, attack: float = 0.01, noise: float = 0.0,
                seed: int = 0) -> np.ndarray:
    """Additive-synthesis 'pluck' tones with exact onsets and durations (overlaps allowed)."""
    end = max((s + d) for _, s, d in notes) if notes else 0.0
    total = max(total or 0.0, end + 0.05)
    y = np.zeros(int(round(total * sr)), dtype=np.float64)
    for midi, start, dur in notes:
        f = midi_to_frequency(midi)
        n = int(round(dur * sr))
        t = np.arange(n) / sr
        tone = sum((amp / (k + 1)) * np.sin(2 * np.pi * f * (k + 1) * t) for k in range(harmonics))
        env = np.minimum(1.0, t / attack) * np.exp(-decay * t)
        # short linear fade-out to avoid clicks
        fade = min(n, int(0.01 * sr))
        if fade > 0:
            env[-fade:] *= np.linspace(1, 0, fade)
        i0 = int(round(start * sr))
        y[i0:i0 + n] += (tone * env)[: max(0, len(y) - i0)]
    if noise > 0:
        rng = np.random.default_rng(seed)
        y += noise * rng.standard_normal(len(y))
    peak = np.max(np.abs(y)) if y.size else 1.0
    if peak > 0.95:
        y = y / peak * 0.95
    return y.astype(np.float32)


MELODY: list[NoteSpec] = [  # C4 D4 E4 C4 C4 (repeated) G4 - each 0.5 s, back to back
    (60, 0.0, 0.5), (62, 0.5, 0.5), (64, 1.0, 0.5), (60, 1.5, 0.5), (60, 2.0, 0.5), (67, 2.5, 0.75),
]
CHORDS: list[NoteSpec] = [  # C major then G major triads, 1.5 s each, fully overlapping within each chord
    (60, 0.0, 1.5), (64, 0.0, 1.5), (67, 0.0, 1.5), (55, 1.5, 1.5), (59, 1.5, 1.5), (62, 1.5, 1.5),
]
ARPEGGIO_OVERLAP: list[NoteSpec] = [  # staggered notes that partially overlap (sustain pedal style)
    (60, 0.0, 1.0), (64, 0.4, 1.0), (67, 0.8, 1.0), (72, 1.2, 1.0),
]


@pytest.fixture(scope="session")
def tmp_audio_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("audio")


def _write(path: Path, y: np.ndarray, sr: int = SR, **kwargs) -> Path:
    sf.write(str(path), y, sr, **kwargs)
    return path


@pytest.fixture(scope="session")
def melody_wav(tmp_audio_dir: Path) -> Path:
    return _write(tmp_audio_dir / "melody.wav", synth_notes(MELODY))


@pytest.fixture(scope="session")
def chords_wav(tmp_audio_dir: Path) -> Path:
    return _write(tmp_audio_dir / "chords.wav", synth_notes(CHORDS))


@pytest.fixture(scope="session")
def arpeggio_wav(tmp_audio_dir: Path) -> Path:
    return _write(tmp_audio_dir / "arpeggio.wav", synth_notes(ARPEGGIO_OVERLAP))


@pytest.fixture(scope="session")
def stereo_44k_wav(tmp_audio_dir: Path) -> Path:
    y = synth_notes([(69, 0.0, 1.0)], sr=44100)
    stereo = np.stack([y, 0.5 * y], axis=1)
    return _write(tmp_audio_dir / "stereo44k.wav", stereo, 44100)


@pytest.fixture(scope="session")
def silent_wav(tmp_audio_dir: Path) -> Path:
    return _write(tmp_audio_dir / "silence.wav", np.zeros(SR * 2, dtype=np.float32))


@pytest.fixture(scope="session")
def leading_silence_wav(tmp_audio_dir: Path) -> Path:
    """1.0 s of silence, then the melody - used to check timeline preservation when trimming."""
    y = synth_notes([(m, s + 1.0, d) for m, s, d in MELODY])
    return _write(tmp_audio_dir / "leading_silence.wav", y)


@pytest.fixture(scope="session")
def flac_file(tmp_audio_dir: Path) -> Path:
    return _write(tmp_audio_dir / "melody.flac", synth_notes(MELODY), format="FLAC")


@pytest.fixture(scope="session")
def mp3_file(tmp_audio_dir: Path) -> Path | None:
    path = tmp_audio_dir / "melody.mp3"
    try:
        _write(path, synth_notes(MELODY), format="MP3")
    except Exception:  # noqa: BLE001 - libsndfile without MP3 encoding support
        return None
    return path


@pytest.fixture(scope="session")
def corrupted_wav(tmp_audio_dir: Path) -> Path:
    path = tmp_audio_dir / "corrupted.wav"
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEjunkjunkjunk" + os.urandom(500))
    return path


@pytest.fixture(scope="session")
def text_file_wav(tmp_audio_dir: Path) -> Path:
    path = tmp_audio_dir / "not_audio.wav"
    path.write_text("this is not audio")
    return path


@pytest.fixture(scope="session")
def empty_wav(tmp_audio_dir: Path) -> Path:
    path = tmp_audio_dir / "empty.wav"
    path.write_bytes(b"")
    return path


@pytest.fixture()
def tmp_db(tmp_path: Path):
    from database.database import Database

    return Database(tmp_path / "test.sqlite3")


@pytest.fixture(autouse=True)
def _isolated_uploads(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    """Point upload persistence at a temp folder so tests never touch data/uploads."""
    folder = tmp_path_factory.mktemp("uploads")
    monkeypatch.setattr(settings, "UPLOADS_DIR", folder)
    yield
