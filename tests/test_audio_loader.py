from pathlib import Path

import numpy as np
import pytest

from config import settings
from src.audio_loader import (
    AudioLoadError,
    AudioValidationError,
    CorruptedFileError,
    delete_uploaded_song_files,
    is_supported_extension,
    load_audio,
    probe_metadata,
    save_upload,
    validate_file,
)


def test_supported_extensions():
    assert is_supported_extension("song.wav")
    assert is_supported_extension("SONG.MP3")
    assert is_supported_extension("take.flac")
    assert not is_supported_extension("notes.txt")
    assert not is_supported_extension("song")


def test_validate_good_wav(melody_wav: Path):
    result = validate_file(melody_wav)
    assert result.ok, result.message
    m = result.metadata
    assert m is not None
    assert m.sample_rate == 22050
    assert m.channels == 1
    assert abs(m.duration_seconds - 3.3) < 0.01
    assert m.container_format == "WAV"
    assert len(m.file_hash) == 64
    assert m.file_size_bytes > 1000


def test_validate_flac(flac_file: Path):
    result = validate_file(flac_file)
    assert result.ok, result.message
    assert result.metadata.container_format == "FLAC"


def test_validate_mp3_if_encoder_available(mp3_file):
    if mp3_file is None:
        pytest.skip("libsndfile build cannot write MP3 test fixtures")
    result = validate_file(mp3_file)
    assert result.ok, result.message
    audio = load_audio(mp3_file)
    assert audio.duration_seconds > 3.0


def test_validate_unsupported_extension(tmp_path: Path):
    f = tmp_path / "x.txt"
    f.write_text("hello")
    result = validate_file(f)
    assert not result.ok
    assert "Unsupported" in result.message


def test_validate_missing_file(tmp_path: Path):
    result = validate_file(tmp_path / "missing.wav")
    assert not result.ok
    assert "not found" in result.message.lower()


def test_validate_corrupted(corrupted_wav: Path, text_file_wav: Path, empty_wav: Path):
    for f in (corrupted_wav, text_file_wav, empty_wav):
        result = validate_file(f)
        assert not result.ok, f
        assert "corrupt" in result.message.lower() or "empty" in result.message.lower(), result.message


def test_probe_corrupted_raises(text_file_wav: Path):
    with pytest.raises(CorruptedFileError):
        probe_metadata(text_file_wav)


def test_validate_too_short(tmp_path: Path):
    import soundfile as sf

    f = tmp_path / "short.wav"
    sf.write(f, np.zeros(100, dtype=np.float32), 22050)
    result = validate_file(f)
    assert not result.ok
    assert "too short" in result.message.lower()


def test_validate_size_limit(melody_wav: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "MAX_FILE_SIZE_MB", 0.0001)
    result = validate_file(melody_wav)
    assert not result.ok
    assert "MB" in result.message


def test_validate_duration_limit(melody_wav: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "MAX_DURATION_SECONDS", 1.0)
    result = validate_file(melody_wav)
    assert not result.ok
    assert "limit" in result.message.lower()


def test_load_mono(melody_wav: Path):
    audio = load_audio(melody_wav)
    assert audio.samples.ndim == 1
    assert audio.samples.dtype == np.float32
    assert audio.channels == 1
    assert audio.sample_rate == 22050
    assert abs(audio.duration_seconds - audio.metadata.duration_seconds) < 1e-3
    assert np.max(np.abs(audio.samples)) <= 1.0


def test_load_stereo_keeps_channels_and_rate(stereo_44k_wav: Path):
    audio = load_audio(stereo_44k_wav)
    assert audio.samples.shape[1] == 2
    assert audio.channels == 2
    assert audio.sample_rate == 44100
    # second channel was written at half amplitude - the original is preserved untouched
    assert np.isclose(np.max(np.abs(audio.samples[:, 1])) / np.max(np.abs(audio.samples[:, 0])), 0.5, atol=0.02)


def test_load_corrupted_raises(text_file_wav: Path):
    with pytest.raises((AudioLoadError, AudioValidationError, CorruptedFileError)):
        load_audio(text_file_wav)


def test_save_upload_and_delete(melody_wav: Path):
    data = melody_wav.read_bytes()
    path = save_upload(data, "My Song (final).wav", song_id="abc123")
    assert path.exists()
    assert path.parent.name == "abc123"
    assert path.name == "My_Song_final.wav"
    assert path.read_bytes() == data   # bytes written verbatim
    # idempotent
    path2 = save_upload(data, "My Song (final).wav", song_id="abc123")
    assert path2 == path
    assert delete_uploaded_song_files("abc123") == 1
    assert not path.exists()
