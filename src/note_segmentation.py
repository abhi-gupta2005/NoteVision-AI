"""
Turn a frame-level pitch track (monophonic mode) into discrete note events.

Algorithm
---------
1. Each frame is assigned its nearest MIDI pitch, or "unvoiced" when pYIN's
   voicing probability is below `min_voiced_probability`.
2. Very short runs (<= `glitch_run_frames`) are absorbed by their neighbours:
   a one- or two-frame pitch jump is almost always an octave error or a
   transition artefact, not a note.
3. Short unvoiced gaps (<= `gap_tolerance_frames`) between two runs of the
   same pitch are bridged (a note does not end because pYIN lost one frame).
4. Every remaining run of constant pitch is a candidate note. A run is split
   into separate events ONLY where the onset detector found an onset of
   sufficient strength inside it - this is how repeated notes at the same
   pitch stay separate without inventing boundaries where there is no
   attack.
5. Candidates shorter than `min_note_duration_seconds` are dropped.

Each resulting note carries the median measured f0 of its frames (not the
equal-tempered frequency) and the mean voicing probability as confidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from config import settings
from src.onset_detection import OnsetResult
from src.pitch_detection import PitchTrack

log = settings.get_logger("segmentation")


@dataclass
class RawNote:
    """Model-agnostic note candidate on the ANALYSIS timeline (before offset)."""

    start: float
    end: float
    midi: int
    frequency_hz: float | None
    confidence: float | None
    onset_strength: float | None = None
    frequency_source: str = "equal_tempered"
    source_track: str = "mix"
    n_frames: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class SegmentationConfig:
    min_note_duration_seconds: float = settings.MIN_NOTE_DURATION_SECONDS
    min_voiced_probability: float = settings.MIN_VOICED_PROBABILITY
    gap_tolerance_frames: int = settings.GAP_TOLERANCE_FRAMES
    glitch_run_frames: int = settings.GLITCH_RUN_FRAMES
    split_on_onsets: bool = True
    onset_split_relative_threshold: float = settings.ONSET_SPLIT_RELATIVE_THRESHOLD
    onset_split_min_separation_seconds: float = settings.ONSET_SPLIT_MIN_SEPARATION_SECONDS
    onset_energy_rise_ratio: float = settings.ONSET_ENERGY_RISE_RATIO

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def run_length_encode(values: np.ndarray) -> list[list[int]]:
    """[[value, start_index, end_index_exclusive], ...] for constant runs."""
    if values.size == 0:
        return []
    change = np.flatnonzero(np.diff(values) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [values.size]])
    return [[int(values[s]), int(s), int(e)] for s, e in zip(starts, ends)]


def _absorb_glitches(labels: np.ndarray, max_run: int) -> np.ndarray:
    """Re-label voiced runs of length <= max_run with a neighbouring pitch."""
    labels = labels.copy()
    for _ in range(3):  # a few passes are enough to settle
        runs = run_length_encode(labels)
        changed = False
        for i, (value, s, e) in enumerate(runs):
            if value < 0 or (e - s) > max_run:
                continue
            prev_val = runs[i - 1][0] if i > 0 else None
            next_val = runs[i + 1][0] if i + 1 < len(runs) else None
            if prev_val is not None and prev_val >= 0 and prev_val == next_val:
                labels[s:e] = prev_val
            elif prev_val is not None and prev_val >= 0 and (runs[i - 1][2] - runs[i - 1][1]) > max_run:
                labels[s:e] = prev_val
            elif next_val is not None and next_val >= 0 and (runs[i + 1][2] - runs[i + 1][1]) > max_run:
                labels[s:e] = next_val
            else:
                labels[s:e] = -1  # isolated glitch between unvoiced frames
            changed = True
        if not changed:
            break
    return labels


def _bridge_gaps(labels: np.ndarray, max_gap: int) -> np.ndarray:
    labels = labels.copy()
    runs = run_length_encode(labels)
    for i in range(1, len(runs) - 1):
        value, s, e = runs[i]
        if value == -1 and (e - s) <= max_gap and runs[i - 1][0] >= 0 and runs[i - 1][0] == runs[i + 1][0]:
            labels[s:e] = runs[i - 1][0]
    return labels


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def segment_pitch_track(
    track: PitchTrack,
    onsets: OnsetResult | None = None,
    config: SegmentationConfig | None = None,
) -> list[RawNote]:
    config = config or SegmentationConfig()
    n = track.n_frames
    if n == 0:
        return []
    period = track.frame_period_seconds
    half = period / 2.0

    voiced = track.voiced_flag & (track.voiced_prob >= config.min_voiced_probability) & np.isfinite(track.midi_float)
    labels = np.full(n, -1, dtype=int)
    labels[voiced] = np.round(track.midi_float[voiced]).astype(int)
    labels[(labels < 0) & voiced] = -1
    labels[labels > 127] = -1

    labels = _absorb_glitches(labels, config.glitch_run_frames)
    labels = _bridge_gaps(labels, config.gap_tolerance_frames)

    notes: list[RawNote] = []
    for value, s, e in run_length_encode(labels):
        if value < 0:
            continue
        start = max(0.0, float(track.times[s]) - half)
        end = float(track.times[e - 1]) + half
        boundaries = [(s, e)]

        if config.split_on_onsets and onsets is not None and onsets.n_onsets:
            cut_frames: list[int] = []
            min_sep = config.onset_split_min_separation_seconds
            inside = (onsets.onset_times > start + min_sep) & (onsets.onset_times < end - min_sep)
            for t_on, strength in zip(onsets.onset_times[inside], onsets.onset_strengths[inside]):
                if strength < config.onset_split_relative_threshold:
                    continue  # no convincing attack: do not split
                if onsets.energy_rise_at(t_on) < config.onset_energy_rise_ratio:
                    continue  # spectral flux without an energy rise (fade-out, decay artefact): not a new note
                frame = int(np.clip(np.searchsorted(track.times, t_on), s + 1, e - 1))
                if cut_frames and (frame - cut_frames[-1]) * period < min_sep:
                    continue
                cut_frames.append(frame)
            if cut_frames:
                edges = [s] + cut_frames + [e]
                boundaries = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
                # A piece shorter than the minimum duration is merged back into its neighbour
                # rather than dropped, so the note keeps its full extent.
                min_frames = max(1, int(np.ceil(config.min_note_duration_seconds / period)))
                merged: list[tuple[int, int]] = []
                for bs, be in boundaries:
                    if merged and (be - bs) < min_frames:
                        merged[-1] = (merged[-1][0], be)
                    else:
                        merged.append((bs, be))
                if len(merged) > 1 and (merged[0][1] - merged[0][0]) < min_frames:
                    merged[1] = (merged[0][0], merged[1][1])
                    merged.pop(0)
                boundaries = merged

        for bs, be in boundaries:
            if be <= bs:
                continue
            seg_start = max(0.0, float(track.times[bs]) - half)
            seg_end = float(track.times[be - 1]) + half
            if seg_end - seg_start < config.min_note_duration_seconds:
                continue
            f0 = track.f0_hz[bs:be]
            f0 = f0[np.isfinite(f0) & (f0 > 0)]
            freq = float(np.median(f0)) if f0.size else None
            probs = track.voiced_prob[bs:be]
            conf = float(np.mean(probs)) if probs.size else None
            strength = onsets.strength_at(seg_start) if onsets is not None else None
            notes.append(RawNote(
                start=seg_start, end=seg_end, midi=int(value), frequency_hz=freq, confidence=conf,
                onset_strength=strength, frequency_source="measured_f0_median", n_frames=int(be - bs),
            ))

    notes.sort(key=lambda r: (r.start, r.midi))
    log.debug("Segmented %d notes from %d frames", len(notes), n)
    return notes
