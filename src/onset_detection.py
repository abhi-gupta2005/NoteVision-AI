"""
Onset detection and tempo estimation (librosa).

Onsets provide the evidence used to split repeated notes of the same pitch in
monophonic mode: a steady-pitch segment is only divided where an onset with
sufficient strength is found.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import settings

log = settings.get_logger("onsets")


@dataclass
class OnsetResult:
    onset_times: np.ndarray          # seconds (analysis timeline)
    onset_strengths: np.ndarray      # normalised strength (0-1) at each onset
    envelope: np.ndarray             # onset-strength envelope per frame (normalised 0-1)
    envelope_times: np.ndarray       # frame centre times (seconds)
    hop_length: int
    sample_rate: int
    rms_envelope: np.ndarray | None = None   # frame-wise RMS energy aligned with envelope_times

    @property
    def n_onsets(self) -> int:
        return int(self.onset_times.shape[0])

    def strength_at(self, time_s: float, window_s: float = 0.03) -> float:
        """Peak normalised onset strength within +/- window of a time."""
        if self.envelope.size == 0:
            return 0.0
        mask = np.abs(self.envelope_times - time_s) <= window_s
        if not np.any(mask):
            idx = int(np.argmin(np.abs(self.envelope_times - time_s)))
            return float(self.envelope[idx])
        return float(np.max(self.envelope[mask]))

    def energy_rise_at(self, time_s: float, before_s: float = 0.06, after_s: float = 0.06, guard_s: float = 0.005) -> float:
        """Ratio of mean RMS just after an onset to mean RMS just before it.

        A real (re-)attack raises the energy (ratio > 1); a fade-out, a decay
        artefact or the end of the signal does not. Returns 0 when there is no
        audio after the time, and inf when the "before" window is silent.
        """
        if self.rms_envelope is None or self.rms_envelope.size == 0:
            return float("inf")  # no energy information: do not veto the onset
        t = self.envelope_times
        before = self.rms_envelope[(t >= time_s - before_s) & (t < time_s - guard_s)]
        after = self.rms_envelope[(t > time_s + guard_s) & (t <= time_s + after_s)]
        if after.size == 0:
            return 0.0
        if before.size == 0:
            return float("inf")
        b = float(np.mean(before))
        a = float(np.mean(after))
        return float("inf") if b <= 1e-9 else a / b

    def to_dict(self) -> dict[str, Any]:
        return {"onset_times": self.onset_times.tolist(), "onset_strengths": self.onset_strengths.tolist(),
                "hop_length": self.hop_length, "sample_rate": self.sample_rate}


def detect_onsets(
    samples: np.ndarray,
    sample_rate: int,
    hop_length: int = settings.PYIN_HOP_LENGTH,
    backtrack: bool = True,
    delta: float = 0.07,
    wait_seconds: float = 0.03,
) -> OnsetResult:
    """Spectral-flux onset detection with peak picking.

    `delta` is the minimum height above the local mean for a peak to count and
    `wait_seconds` the minimum spacing between onsets. With `backtrack=True`
    each onset is moved back to the preceding energy minimum, which lines up
    better with the perceived note start.
    """
    import librosa

    if samples.ndim != 1:
        raise ValueError("detect_onsets expects a mono signal")
    n_fft = 2048
    if samples.shape[0] < n_fft:
        samples = np.pad(samples, (0, n_fft - samples.shape[0]))
    envelope = librosa.onset.onset_strength(y=samples, sr=sample_rate, hop_length=hop_length, n_fft=n_fft)
    peak = float(np.max(envelope)) if envelope.size else 0.0
    norm_env = envelope / peak if peak > 0 else envelope
    env_times = librosa.frames_to_time(np.arange(envelope.shape[0]), sr=sample_rate, hop_length=hop_length)
    wait = max(1, int(round(wait_seconds * sample_rate / hop_length)))
    frames = librosa.onset.onset_detect(
        onset_envelope=envelope, sr=sample_rate, hop_length=hop_length, backtrack=backtrack,
        units="frames", delta=delta, wait=wait, normalize=True,
    )
    frames = np.asarray(frames, dtype=int)
    times = librosa.frames_to_time(frames, sr=sample_rate, hop_length=hop_length) if frames.size else np.array([])
    # strength: look slightly ahead of a back-tracked onset for the actual peak
    strengths = np.array([
        float(np.max(norm_env[f: min(f + wait + 2, norm_env.shape[0])])) if norm_env.size else 0.0 for f in frames
    ]) if frames.size else np.array([])
    rms = librosa.feature.rms(y=samples, frame_length=n_fft, hop_length=hop_length)[0]
    rms = rms[: envelope.shape[0]] if rms.shape[0] >= envelope.shape[0] else np.pad(rms, (0, envelope.shape[0] - rms.shape[0]))
    log.debug("Detected %d onsets", frames.size)
    return OnsetResult(onset_times=times, onset_strengths=strengths, envelope=norm_env,
                       envelope_times=env_times, hop_length=hop_length, sample_rate=sample_rate, rms_envelope=rms)


def estimate_tempo(samples: np.ndarray, sample_rate: int, hop_length: int = 512) -> float | None:
    """Global tempo estimate in BPM, or None if it cannot be estimated.

    This is an *estimate* from the onset autocorrelation and is unreliable for
    rubato, very short clips or recordings without a clear pulse.
    """
    import librosa

    try:
        if samples.shape[0] < sample_rate * 2:   # < 2 s: no meaningful pulse
            return None
        env = librosa.onset.onset_strength(y=samples, sr=sample_rate, hop_length=hop_length)
        tempo = librosa.feature.tempo(onset_envelope=env, sr=sample_rate, hop_length=hop_length)
        value = float(np.atleast_1d(tempo)[0])
        return value if np.isfinite(value) and value > 0 else None
    except Exception as exc:  # noqa: BLE001 - tempo is optional information
        log.warning("Tempo estimation failed: %s", exc)
        return None
