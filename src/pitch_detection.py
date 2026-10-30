"""
Frequency <-> MIDI <-> note-name mapping and monophonic pitch tracking (pYIN).

Conversions use 12-tone equal temperament with A4 = 440 Hz:

    midi_note = round(69 + 12 * log2(frequency_hz / 440))

Invalid inputs (None, NaN, inf, zero, negative) never raise - they map to None
so that callers can exclude them explicitly and report how many were excluded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from config import settings

A4_FREQUENCY_HZ = 440.0
A4_MIDI = 69
MIDI_MIN, MIDI_MAX = 0, 127
PITCH_CLASS_NAMES: list[str] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_TO_SHARP = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E", "E#": "F", "B#": "C"}


# --------------------------------------------------------------------------- #
# Scalar conversions
# --------------------------------------------------------------------------- #
def is_valid_frequency(frequency_hz: Any) -> bool:
    try:
        f = float(frequency_hz)
    except (TypeError, ValueError):
        return False
    return math.isfinite(f) and f > 0.0


def frequency_to_midi_float(frequency_hz: Any) -> float | None:
    """Unrounded MIDI pitch (e.g. 60.23) or None for invalid input."""
    if not is_valid_frequency(frequency_hz):
        return None
    return A4_MIDI + 12.0 * math.log2(float(frequency_hz) / A4_FREQUENCY_HZ)


def frequency_to_midi(frequency_hz: Any) -> int | None:
    """Nearest equal-tempered MIDI note number, or None if invalid / out of 0-127."""
    value = frequency_to_midi_float(frequency_hz)
    if value is None:
        return None
    midi = int(round(value))
    if midi < MIDI_MIN or midi > MIDI_MAX:
        return None
    return midi


def midi_to_frequency(midi_note: int | float) -> float:
    return A4_FREQUENCY_HZ * 2.0 ** ((float(midi_note) - A4_MIDI) / 12.0)


def midi_to_pitch_class(midi_note: int) -> str:
    return PITCH_CLASS_NAMES[int(midi_note) % 12]


def midi_to_octave(midi_note: int) -> int:
    """Scientific pitch notation: MIDI 60 = C4, MIDI 21 = A0."""
    return int(midi_note) // 12 - 1


def midi_to_note_name(midi_note: int) -> str:
    return f"{midi_to_pitch_class(midi_note)}{midi_to_octave(midi_note)}"


def note_name_to_midi(name: str) -> int:
    """Parse names such as 'C4', 'D#5', 'Bb3', 'A-1'."""
    text = name.strip()
    i = 1
    while i < len(text) and text[i] in "#b":
        i += 1
    pitch, octave_text = text[:i], text[i:]
    pitch = pitch[0].upper() + pitch[1:]
    pitch = _FLAT_TO_SHARP.get(pitch, pitch)
    if pitch not in PITCH_CLASS_NAMES:
        raise ValueError(f"Unknown pitch name: {name!r}")
    try:
        octave = int(octave_text)
    except ValueError as exc:
        raise ValueError(f"Missing or invalid octave in note name: {name!r}") from exc
    return (octave + 1) * 12 + PITCH_CLASS_NAMES.index(pitch)


def cents_deviation(frequency_hz: Any, midi_note: int | None = None) -> float | None:
    """Deviation (in cents) of a frequency from its nearest equal-tempered note.

    Positive = sharp, negative = flat. None for invalid frequencies.
    """
    value = frequency_to_midi_float(frequency_hz)
    if value is None:
        return None
    if midi_note is None:
        midi_note = int(round(value))
    return float((value - midi_note) * 100.0)


# --------------------------------------------------------------------------- #
# Vectorised conversions
# --------------------------------------------------------------------------- #
def frequencies_to_midi_float_array(frequencies: np.ndarray) -> np.ndarray:
    """Element-wise unrounded MIDI pitch; NaN where the frequency is invalid."""
    f = np.asarray(frequencies, dtype=np.float64)
    out = np.full(f.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(f) & (f > 0)
    out[valid] = A4_MIDI + 12.0 * np.log2(f[valid] / A4_FREQUENCY_HZ)
    return out


# --------------------------------------------------------------------------- #
# Monophonic pitch tracking (pYIN)
# --------------------------------------------------------------------------- #
@dataclass
class PitchTrack:
    times: np.ndarray            # frame centre times in seconds (analysis timeline)
    f0_hz: np.ndarray            # estimated f0 per frame, NaN when unvoiced
    voiced_flag: np.ndarray      # bool per frame
    voiced_prob: np.ndarray      # probability of voicing per frame
    midi_float: np.ndarray       # unrounded MIDI pitch, NaN when unvoiced
    sample_rate: int
    hop_length: int
    frame_length: int
    method: str = "pyin"

    @property
    def n_frames(self) -> int:
        return int(self.times.shape[0])

    @property
    def frame_period_seconds(self) -> float:
        return self.hop_length / self.sample_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "times": self.times.tolist(), "f0_hz": np.nan_to_num(self.f0_hz, nan=0.0).tolist(),
            "voiced_prob": self.voiced_prob.tolist(), "sample_rate": self.sample_rate,
            "hop_length": self.hop_length, "frame_length": self.frame_length, "method": self.method,
        }


def track_pitch_pyin(
    samples: np.ndarray,
    sample_rate: int,
    fmin_note: str = settings.PYIN_FMIN_NOTE,
    fmax_note: str = settings.PYIN_FMAX_NOTE,
    frame_length: int = settings.PYIN_FRAME_LENGTH,
    hop_length: int = settings.PYIN_HOP_LENGTH,
) -> PitchTrack:
    """Probabilistic YIN fundamental-frequency tracking (librosa.pyin).

    pYIN assumes a single dominant pitch per frame; on polyphonic material it
    follows whichever partial dominates, which is why it is reserved for the
    monophonic analysis mode.
    """
    import librosa

    if samples.ndim != 1:
        raise ValueError("track_pitch_pyin expects a mono signal")
    fmin = librosa.note_to_hz(fmin_note)
    fmax = min(librosa.note_to_hz(fmax_note), sample_rate / 2.0 - 1.0)
    if samples.shape[0] < frame_length:
        samples = np.pad(samples, (0, frame_length - samples.shape[0]))
    f0, voiced_flag, voiced_prob = librosa.pyin(
        samples.astype(np.float32), fmin=fmin, fmax=fmax, sr=sample_rate,
        frame_length=frame_length, hop_length=hop_length, fill_na=np.nan,
    )
    times = librosa.frames_to_time(np.arange(f0.shape[0]), sr=sample_rate, hop_length=hop_length)
    return PitchTrack(
        times=times.astype(np.float64), f0_hz=f0.astype(np.float64), voiced_flag=voiced_flag.astype(bool),
        voiced_prob=voiced_prob.astype(np.float64), midi_float=frequencies_to_midi_float_array(f0),
        sample_rate=sample_rate, hop_length=hop_length, frame_length=frame_length, method="pyin",
    )
