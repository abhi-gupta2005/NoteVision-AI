"""
Audio upload handling, validation, metadata extraction and loading.

Design notes
------------
* The original recording is never modified. `save_upload` copies the uploaded
  bytes to the uploads directory; all processing happens on in-memory arrays.
* `validate_file` is deliberately cheap (header probe only) so that it can run
  on every uploaded file before any heavy processing starts.
* Decoding uses libsndfile (via `soundfile`) first because it is fast and
  supports WAV/FLAC/OGG/AIFF and - with libsndfile >= 1.1 - MP3. Formats that
  libsndfile cannot decode (e.g. M4A) fall back to `audioread`/ffmpeg through
  librosa, if ffmpeg is installed on the machine.
"""
from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from config import settings
from src.utils import safe_filename, sha256_of_file, utc_now_iso

log = settings.get_logger("audio_loader")


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class AudioValidationError(ValueError):
    """Base class for all upload/validation problems (message is user-facing)."""


class UnsupportedFormatError(AudioValidationError):
    pass


class FileTooLargeError(AudioValidationError):
    pass


class CorruptedFileError(AudioValidationError):
    pass


class DurationError(AudioValidationError):
    pass


class AudioLoadError(RuntimeError):
    """Raised when a file passed validation but could not be decoded."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class AudioMetadata:
    filename: str
    file_path: str
    file_size_bytes: int
    extension: str
    container_format: str | None
    subtype: str | None
    sample_rate: int
    channels: int
    duration_seconds: float
    frames: int
    file_hash: str
    probed_with: str
    probed_at: str = field(default_factory=utc_now_iso)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def file_size_mb(self) -> float:
        return self.file_size_bytes / (1024 * 1024)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: AudioMetadata | None = None

    @property
    def message(self) -> str:
        return "; ".join(self.errors) if self.errors else "OK"


@dataclass
class LoadedAudio:
    """Decoded audio exactly as stored in the file (original rate & channels)."""

    samples: np.ndarray          # shape (n_samples,) for mono or (n_samples, n_channels)
    sample_rate: int
    channels: int
    duration_seconds: float
    metadata: AudioMetadata

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])


# --------------------------------------------------------------------------- #
# Validation & metadata
# --------------------------------------------------------------------------- #
def is_supported_extension(filename: str) -> bool:
    return Path(filename).suffix.lower() in settings.SUPPORTED_EXTENSIONS


def _probe_with_soundfile(path: Path) -> dict[str, Any]:
    info = sf.info(str(path))
    return {
        "container_format": info.format,
        "subtype": info.subtype,
        "sample_rate": int(info.samplerate),
        "channels": int(info.channels),
        "frames": int(info.frames),
        "duration_seconds": float(info.duration),
        "probed_with": f"libsndfile {sf.__libsndfile_version__}",
    }


def _probe_with_audioread(path: Path) -> dict[str, Any]:
    import audioread  # lazy: only needed for formats libsndfile cannot read

    with audioread.audio_open(str(path)) as fh:
        sr = int(fh.samplerate)
        ch = int(fh.channels)
        duration = float(fh.duration)
    return {
        "container_format": path.suffix.lower().lstrip(".").upper(),
        "subtype": None,
        "sample_rate": sr,
        "channels": ch,
        "frames": int(round(duration * sr)),
        "duration_seconds": duration,
        "probed_with": "audioread/ffmpeg",
    }


def probe_metadata(path: str | Path, compute_hash: bool = True) -> AudioMetadata:
    """Read container metadata without decoding the whole file.

    Raises CorruptedFileError if no backend can read the header.
    """
    path = Path(path)
    if not path.is_file():
        raise AudioValidationError(f"File not found: {path.name}")
    size = path.stat().st_size
    if size == 0:
        raise CorruptedFileError(f"'{path.name}' is empty (0 bytes).")

    probe: dict[str, Any] | None = None
    errors: list[str] = []
    try:
        probe = _probe_with_soundfile(path)
    except Exception as exc:  # noqa: BLE001 - libsndfile raises several types
        errors.append(f"libsndfile: {exc}")
        try:
            probe = _probe_with_audioread(path)
        except Exception as exc2:  # noqa: BLE001
            errors.append(f"audioread: {exc2}")

    if probe is None:
        raise CorruptedFileError(
            f"'{path.name}' could not be read as audio. It may be corrupted, truncated, "
            f"or in an unsupported codec. ({' | '.join(errors)})"
        )
    if probe["sample_rate"] <= 0 or probe["channels"] <= 0:
        raise CorruptedFileError(f"'{path.name}' reports an invalid sample rate or channel count.")

    return AudioMetadata(
        filename=path.name,
        file_path=str(path),
        file_size_bytes=size,
        extension=path.suffix.lower(),
        file_hash=sha256_of_file(path) if compute_hash else "",
        **probe,
    )


def validate_file(path: str | Path, compute_hash: bool = True) -> ValidationResult:
    """Validate an audio file: format, size, readability and duration limits.

    Never raises; the result carries user-facing error and warning messages.
    """
    path = Path(path)
    result = ValidationResult(ok=True)

    if not is_supported_extension(path.name):
        supported = ", ".join(sorted(settings.SUPPORTED_EXTENSIONS))
        result.errors.append(
            f"Unsupported file type '{path.suffix or '(none)'}'. Supported: {supported}."
        )
        result.ok = False
        return result

    if not path.is_file():
        result.errors.append(f"File not found: {path.name}")
        result.ok = False
        return result

    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        result.errors.append(
            f"File is {size_mb:.1f} MB, above the {settings.MAX_FILE_SIZE_MB:.0f} MB limit."
        )
        result.ok = False
        return result

    try:
        meta = probe_metadata(path, compute_hash=compute_hash)
    except AudioValidationError as exc:
        result.errors.append(str(exc))
        result.ok = False
        return result

    result.metadata = meta
    if meta.duration_seconds < settings.MIN_DURATION_SECONDS:
        result.errors.append(
            f"Recording is too short ({meta.duration_seconds:.3f} s); "
            f"minimum is {settings.MIN_DURATION_SECONDS} s."
        )
        result.ok = False
    if meta.duration_seconds > settings.MAX_DURATION_SECONDS:
        result.errors.append(
            f"Recording is {meta.duration_seconds / 60:.1f} min long; the limit is "
            f"{settings.MAX_DURATION_SECONDS / 60:.0f} min."
        )
        result.ok = False
    if meta.channels > 2:
        result.warnings.append(f"{meta.channels} channels detected; they will be mixed down to mono.")
    if meta.sample_rate < 16000:
        result.warnings.append(
            f"Low sample rate ({meta.sample_rate} Hz) limits the detectable pitch range."
        )
    if meta.extension == ".m4a" and meta.probed_with.startswith("libsndfile"):
        result.warnings.append("M4A decoded through libsndfile; if playback sounds wrong, convert to WAV.")
    return result


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_audio(path: str | Path, metadata: AudioMetadata | None = None) -> LoadedAudio:
    """Decode the complete file at its native sample rate and channel layout.

    The returned samples are float32 in [-1, 1]. Stereo files come back with
    shape (n_samples, 2); no mixing, resampling or trimming happens here.
    """
    path = Path(path)
    if metadata is None:
        metadata = probe_metadata(path)

    samples: np.ndarray | None = None
    sr: int | None = None
    try:
        samples, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001
        log.info("libsndfile could not decode %s (%s); trying librosa/audioread", path.name, exc)
        try:
            import librosa

            samples, sr = librosa.load(str(path), sr=None, mono=False)
            samples = np.asarray(samples, dtype=np.float32)
            if samples.ndim == 2:  # librosa returns (channels, n) -> (n, channels)
                samples = samples.T
        except Exception as exc2:  # noqa: BLE001
            raise AudioLoadError(
                f"'{path.name}' could not be decoded ({exc2}). Convert it to WAV or FLAC and retry."
            ) from exc2

    if samples is None or sr is None or samples.size == 0:
        raise AudioLoadError(f"'{path.name}' decoded to zero samples; the file is probably corrupted.")
    if not np.all(np.isfinite(samples)):
        n_bad = int(np.sum(~np.isfinite(samples)))
        log.warning("%s contains %d non-finite samples; they were replaced with 0", path.name, n_bad)
        samples = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)

    channels = 1 if samples.ndim == 1 else int(samples.shape[1])
    duration = samples.shape[0] / float(sr)
    return LoadedAudio(
        samples=np.ascontiguousarray(samples, dtype=np.float32),
        sample_rate=int(sr),
        channels=channels,
        duration_seconds=duration,
        metadata=metadata,
    )


# --------------------------------------------------------------------------- #
# Upload persistence
# --------------------------------------------------------------------------- #
def save_upload(data: bytes, original_filename: str, song_id: str | None = None) -> Path:
    """Copy uploaded bytes into the uploads directory, preserving the original name.

    Files are grouped per song id (hash-based) so re-uploading the same file
    does not create duplicates. The original bytes are written verbatim.
    """
    settings.ensure_directories()
    from src.utils import sha256_of_bytes

    song_id = song_id or sha256_of_bytes(data)[:16]
    folder = settings.UPLOADS_DIR / song_id
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / safe_filename(original_filename)
    if not target.exists() or target.stat().st_size != len(data):
        with open(target, "wb") as fh:
            fh.write(data)
    return target


def copy_into_uploads(source: str | Path, song_id: str) -> Path:
    """Copy an existing file (e.g. from disk) into the uploads directory."""
    settings.ensure_directories()
    source = Path(source)
    folder = settings.UPLOADS_DIR / song_id
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / safe_filename(source.name)
    if target.resolve() != source.resolve():
        shutil.copy2(source, target)
    return target


def delete_uploaded_song_files(song_id: str) -> int:
    """Remove the uploads folder for a song. Returns number of files removed."""
    folder = settings.UPLOADS_DIR / song_id
    if not folder.exists():
        return 0
    count = sum(1 for p in folder.rglob("*") if p.is_file())
    shutil.rmtree(folder, ignore_errors=True)
    return count
