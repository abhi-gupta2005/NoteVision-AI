from pathlib import Path

import numpy as np
import pytest

from src.audio_loader import load_audio
from src.audio_preprocessing import (
    PreprocessingConfig,
    peak_normalize,
    preprocess,
    spectral_gate_noise_reduction,
    to_mono,
    trim_silence,
)
from tests.conftest import synth_notes


def test_to_mono_shapes():
    stereo = np.stack([np.ones(100), np.zeros(100)], axis=1).astype(np.float32)
    mono = to_mono(stereo)
    assert mono.shape == (100,)
    assert np.allclose(mono, 0.5)
    assert to_mono(np.ones(10, dtype=np.float32)).shape == (10,)


def test_stereo_44k_is_resampled_to_mono_22k(stereo_44k_wav: Path):
    audio = load_audio(stereo_44k_wav)
    out = preprocess(audio, PreprocessingConfig())
    assert out.samples.ndim == 1
    assert out.sample_rate == 22050
    assert abs(out.duration_seconds - audio.duration_seconds) < 0.002  # timeline preserved
    steps = [t["step"] for t in out.transformations]
    assert steps == ["to_mono", "resample"]
    assert out.time_offset_seconds == 0.0
    assert out.original_sample_rate == 44100 and out.original_channels == 2


def test_no_changes_by_default(melody_wav: Path):
    out = preprocess(load_audio(melody_wav))
    assert out.transformations == []
    assert out.time_offset_seconds == 0.0


def test_peak_normalize():
    y = np.array([0.1, -0.25, 0.05], dtype=np.float32)
    out, gain_db = peak_normalize(y, -1.0)
    assert np.max(np.abs(out)) == pytest.approx(10 ** (-1 / 20), abs=1e-4)
    assert gain_db > 0
    silent, g = peak_normalize(np.zeros(10, dtype=np.float32))
    assert g == 0.0 and np.all(silent == 0)


def test_noise_reduction_reduces_noise_floor():
    rng = np.random.default_rng(1)
    clean = synth_notes([(69, 0.0, 2.0)])
    noisy = clean + 0.02 * rng.standard_normal(len(clean)).astype(np.float32)
    out, floor_db = spectral_gate_noise_reduction(noisy, 22050, strength=0.8)
    assert out.shape == noisy.shape
    assert np.isfinite(floor_db)
    # energy in the final (decayed, mostly noise) 0.2 s must drop
    tail = slice(-int(0.2 * 22050), None)
    assert np.mean(out[tail] ** 2) < np.mean(noisy[tail] ** 2)


def test_trim_records_offset_and_restores_timeline(leading_silence_wav: Path):
    audio = load_audio(leading_silence_wav)
    out = preprocess(audio, PreprocessingConfig(trim_silence=True, trim_top_db=40))
    assert 0.9 <= out.time_offset_seconds <= 1.02
    assert out.duration_seconds < audio.duration_seconds
    assert any(t["step"] == "trim_silence" for t in out.transformations)


def test_trim_silence_function():
    y = np.concatenate([np.zeros(22050, dtype=np.float32), synth_notes([(60, 0.0, 0.5)])])
    trimmed, lead, trail = trim_silence(y, 22050, 40)
    assert 0.95 <= lead <= 1.01
    assert len(trimmed) < len(y)


def test_all_options_run(melody_wav: Path):
    audio = load_audio(melody_wav)
    cfg = PreprocessingConfig(normalize=True, noise_reduction=True, high_pass_hz=30.0, trim_silence=True)
    out = preprocess(audio, cfg)
    steps = [t["step"] for t in out.transformations]
    assert steps == ["high_pass", "noise_reduction", "normalize", "trim_silence"]
    assert all("description" in t for t in out.transformations)
    assert out.to_dict()["config"]["normalize"] is True


def test_silent_audio_still_valid(silent_wav: Path):
    out = preprocess(load_audio(silent_wav), PreprocessingConfig(normalize=True))
    assert out.n_samples > 0
