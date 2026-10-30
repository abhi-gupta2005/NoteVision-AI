"""
Modular audio preprocessing pipeline.

Every transformation is recorded in `PreprocessedAudio.transformations` so
that reports can state exactly what was done to the signal before analysis.

Timeline guarantee
------------------
Resampling, mono conversion, normalisation and noise reduction do not move
events in time. The only step that can shift the timeline is optional silence
trimming; when it is enabled the removed leading offset is stored in
`time_offset_seconds` and the transcription engine adds it back, so every
timestamp reported to the user refers to the ORIGINAL recording timeline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import librosa
import numpy as np

from config import settings
from src.audio_loader import LoadedAudio

log = settings.get_logger("preprocessing")


@dataclass
class PreprocessingConfig:
    target_sample_rate: int = settings.ANALYSIS_SAMPLE_RATE
    to_mono: bool = True
    normalize: bool = False                 # peak normalisation (never on by default)
    normalize_peak_dbfs: float = -1.0
    noise_reduction: bool = False           # light spectral gating (off by default)
    noise_reduction_strength: float = 0.5   # 0..1, fraction of estimated noise floor removed
    trim_silence: bool = False              # off by default: keeps original timeline
    trim_top_db: float = 45.0
    high_pass_hz: float | None = None       # optional rumble filter (e.g. 30 Hz)
    resample_type: str = "soxr_hq"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PreprocessedAudio:
    samples: np.ndarray                     # float32 mono, shape (n,)
    sample_rate: int
    duration_seconds: float
    time_offset_seconds: float              # add to analysis times to get original times
    original_sample_rate: int
    original_channels: int
    original_duration_seconds: float
    transformations: list[dict[str, Any]] = field(default_factory=list)
    config: PreprocessingConfig = field(default_factory=PreprocessingConfig)

    @property
    def n_samples(self) -> int:
        return int(self.samples.shape[0])

    def summary(self) -> list[str]:
        return [t["description"] for t in self.transformations]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_rate": self.sample_rate,
            "duration_seconds": self.duration_seconds,
            "time_offset_seconds": self.time_offset_seconds,
            "original_sample_rate": self.original_sample_rate,
            "original_channels": self.original_channels,
            "original_duration_seconds": self.original_duration_seconds,
            "transformations": self.transformations,
            "config": self.config.to_dict(),
        }


# --------------------------------------------------------------------------- #
# Individual steps (pure functions)
# --------------------------------------------------------------------------- #
def to_mono(samples: np.ndarray) -> np.ndarray:
    """Average all channels. Shape (n,) or (n, ch) -> (n,)."""
    if samples.ndim == 1:
        return samples.astype(np.float32, copy=False)
    return np.mean(samples, axis=1, dtype=np.float32)


def resample(samples: np.ndarray, orig_sr: int, target_sr: int, res_type: str = "soxr_hq") -> np.ndarray:
    if orig_sr == target_sr:
        return samples
    try:
        out = librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr, res_type=res_type)
    except Exception:  # noqa: BLE001 - soxr may be unavailable on some builds; scipy's FFT resampler needs no extra package
        out = librosa.resample(samples, orig_sr=orig_sr, target_sr=target_sr, res_type="fft")
    return out.astype(np.float32, copy=False)


def peak_normalize(samples: np.ndarray, peak_dbfs: float = -1.0) -> tuple[np.ndarray, float]:
    """Scale so the absolute peak hits `peak_dbfs`. Returns (samples, gain_db)."""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak <= 1e-9:
        return samples, 0.0
    target = 10 ** (peak_dbfs / 20.0)
    gain = target / peak
    return (samples * gain).astype(np.float32), float(20 * np.log10(gain))


def high_pass(samples: np.ndarray, sr: int, cutoff_hz: float, order: int = 2) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt

    sos = butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, samples).astype(np.float32)


def spectral_gate_noise_reduction(
    samples: np.ndarray, sr: int, strength: float = 0.5, n_fft: int = 2048, hop_length: int = 512
) -> tuple[np.ndarray, float]:
    """Light stationary-noise reduction by spectral gating.

    The noise floor of each frequency bin is estimated as the 10th percentile
    of its magnitude over time (quiet parts of the recording). A fraction
    (`strength`) of that floor is subtracted from every frame, with the result
    floored at 10% of the original magnitude so harmonics are attenuated, not
    deleted. Returns (samples, estimated_noise_floor_db).
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength == 0.0 or samples.size < n_fft:
        return samples, float("nan")
    stft = librosa.stft(samples, n_fft=n_fft, hop_length=hop_length)
    mag, phase = np.abs(stft), np.angle(stft)
    noise_floor = np.percentile(mag, 10, axis=1, keepdims=True)
    gated = np.maximum(mag - strength * noise_floor, 0.1 * mag)
    out = librosa.istft(gated * np.exp(1j * phase), hop_length=hop_length, length=samples.shape[0])
    floor_db = float(20 * np.log10(np.mean(noise_floor) + 1e-12))
    return out.astype(np.float32), floor_db


def trim_silence(samples: np.ndarray, sr: int, top_db: float = 45.0) -> tuple[np.ndarray, float, float]:
    """Trim leading/trailing silence. Returns (samples, leading_offset_s, trailing_cut_s)."""
    trimmed, index = librosa.effects.trim(samples, top_db=top_db)
    start, end = int(index[0]), int(index[1])
    return trimmed.astype(np.float32), start / sr, (samples.shape[0] - end) / sr


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def preprocess(audio: LoadedAudio, config: PreprocessingConfig | None = None) -> PreprocessedAudio:
    """Run the configured preprocessing steps and record each transformation."""
    config = config or PreprocessingConfig()
    log_entries: list[dict[str, Any]] = []
    y = audio.samples
    sr = audio.sample_rate

    # 1. channel handling
    if y.ndim == 2 and config.to_mono:
        y = to_mono(y)
        log_entries.append(
            {"step": "to_mono", "channels_in": audio.channels, "channels_out": 1,
             "description": f"Mixed {audio.channels} channels down to mono (mean of channels)."}
        )
    elif y.ndim == 2:
        # Models need mono; a pipeline caller that disables mono conversion still
        # gets channel 0 so downstream code can rely on a 1-D array.
        y = y[:, 0].astype(np.float32)
        log_entries.append(
            {"step": "select_channel", "channel": 0,
             "description": "Kept channel 0 only (mono conversion disabled)."}
        )

    # 2. optional high-pass
    if config.high_pass_hz and config.high_pass_hz > 0:
        y = high_pass(y, sr, config.high_pass_hz)
        log_entries.append(
            {"step": "high_pass", "cutoff_hz": config.high_pass_hz,
             "description": f"Applied 2nd-order zero-phase high-pass filter at {config.high_pass_hz:.0f} Hz."}
        )

    # 3. resample
    if sr != config.target_sample_rate:
        y = resample(y, sr, config.target_sample_rate, config.resample_type)
        log_entries.append(
            {"step": "resample", "from_sr": sr, "to_sr": config.target_sample_rate,
             "method": config.resample_type,
             "description": f"Resampled {sr} Hz -> {config.target_sample_rate} Hz ({config.resample_type}). "
                            f"Timeline unchanged."}
        )
        sr = config.target_sample_rate

    # 4. optional noise reduction (before normalisation so gain does not change the estimate)
    if config.noise_reduction:
        y, floor_db = spectral_gate_noise_reduction(y, sr, config.noise_reduction_strength)
        log_entries.append(
            {"step": "noise_reduction", "method": "spectral_gating",
             "strength": config.noise_reduction_strength, "noise_floor_db": floor_db,
             "description": f"Light spectral-gating noise reduction (strength {config.noise_reduction_strength:.2f}, "
                            f"estimated noise floor {floor_db:.1f} dB)."}
        )

    # 5. optional normalisation
    if config.normalize:
        y, gain_db = peak_normalize(y, config.normalize_peak_dbfs)
        log_entries.append(
            {"step": "normalize", "peak_dbfs": config.normalize_peak_dbfs, "gain_db": gain_db,
             "description": f"Peak-normalised to {config.normalize_peak_dbfs:.1f} dBFS (gain {gain_db:+.1f} dB)."}
        )

    # 6. optional trimming (the only step that moves the timeline)
    offset = 0.0
    if config.trim_silence:
        y, offset, trailing = trim_silence(y, sr, config.trim_top_db)
        log_entries.append(
            {"step": "trim_silence", "top_db": config.trim_top_db, "leading_seconds": offset,
             "trailing_seconds": trailing,
             "description": f"Trimmed {offset:.3f} s of leading and {trailing:.3f} s of trailing silence "
                            f"(below {config.trim_top_db:.0f} dB). Timestamps are shifted back by "
                            f"{offset:.3f} s so they match the original recording."}
        )

    y = np.ascontiguousarray(y, dtype=np.float32)
    if y.size == 0:
        raise ValueError("Preprocessing produced an empty signal (is the recording silent?).")

    result = PreprocessedAudio(
        samples=y,
        sample_rate=sr,
        duration_seconds=y.shape[0] / sr,
        time_offset_seconds=float(offset),
        original_sample_rate=audio.sample_rate,
        original_channels=audio.channels,
        original_duration_seconds=audio.duration_seconds,
        transformations=log_entries,
        config=config,
    )
    log.info("Preprocessed %s: %s", audio.metadata.filename, "; ".join(result.summary()) or "no changes")
    return result


def signal_level_stats(samples: np.ndarray) -> dict[str, float]:
    """Basic level statistics used in the overview (peak / RMS in dBFS)."""
    if samples.size == 0:
        return {"peak_dbfs": float("-inf"), "rms_dbfs": float("-inf")}
    peak = float(np.max(np.abs(samples)))
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    return {
        "peak_dbfs": float(20 * np.log10(peak + 1e-12)),
        "rms_dbfs": float(20 * np.log10(rms + 1e-12)),
    }
