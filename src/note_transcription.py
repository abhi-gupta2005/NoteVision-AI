"""
Note transcription engine: audio -> timestamped note events.

Three analysis modes, each invoking only the components it needs:

* monophonic  - pYIN pitch tracking + onset-aware segmentation (one note at a time)
* polyphonic  - Spotify Basic Pitch (ICASSP 2022 model) via onnxruntime; keeps
                simultaneous and overlapping notes
* full_mix    - Basic Pitch on the whole mix, optionally per Demucs stem
                (vocals / bass / other) when source separation is installed

All events share the `NoteEvent` schema. Timestamps are on the ORIGINAL
recording timeline (the preprocessing offset is added back here).
"""
from __future__ import annotations

import heapq
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from src.audio_preprocessing import PreprocessedAudio
from src.note_segmentation import RawNote, SegmentationConfig, segment_pitch_track
from src.onset_detection import OnsetResult, detect_onsets
from src.pitch_detection import (
    PitchTrack,
    cents_deviation,
    is_valid_frequency,
    midi_to_frequency,
    midi_to_note_name,
    midi_to_octave,
    midi_to_pitch_class,
    track_pitch_pyin,
)
from src.progress import AnalysisCancelled, CancelToken, ProgressCallback, no_progress, scaled
from src.utils import clean_float, format_timestamp

log = settings.get_logger("transcription")


# --------------------------------------------------------------------------- #
# Modes & configuration
# --------------------------------------------------------------------------- #
class TranscriptionMode(str, Enum):
    MONOPHONIC = "monophonic"
    POLYPHONIC = "polyphonic"
    FULL_MIX = "full_mix"

    @property
    def label(self) -> str:
        return MODE_LABELS[self]

    @property
    def description(self) -> str:
        return MODE_DESCRIPTIONS[self]


MODE_LABELS = {
    TranscriptionMode.MONOPHONIC: "A - Monophonic (single voice / instrument)",
    TranscriptionMode.POLYPHONIC: "B - Polyphonic (piano, guitar, chords)",
    TranscriptionMode.FULL_MIX: "C - Full music recording (mixed song)",
}
MODE_DESCRIPTIONS = {
    TranscriptionMode.MONOPHONIC: (
        "pYIN fundamental-frequency tracking with onset-aware segmentation. Best for a single "
        "melody line (voice, flute, one guitar string). Only one note at a time is reported."
    ),
    TranscriptionMode.POLYPHONIC: (
        "Spotify Basic Pitch neural transcription. Detects multiple simultaneous notes and keeps "
        "overlapping events. Best for solo piano, guitar and other harmonic instruments."
    ),
    TranscriptionMode.FULL_MIX: (
        "Basic Pitch on a complete mix, optionally per separated stem (vocals / bass / other) when "
        "Demucs is installed. Quality depends strongly on the recording: percussion, dense "
        "arrangements and effects reduce accuracy."
    ),
}


@dataclass
class TranscriptionConfig:
    mode: str = TranscriptionMode.POLYPHONIC.value
    # monophonic (pYIN)
    pyin_fmin_note: str = settings.PYIN_FMIN_NOTE
    pyin_fmax_note: str = settings.PYIN_FMAX_NOTE
    pyin_frame_length: int = settings.PYIN_FRAME_LENGTH
    pyin_hop_length: int = settings.PYIN_HOP_LENGTH
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    # polyphonic / full mix (Basic Pitch)
    onset_threshold: float = settings.BASIC_PITCH_ONSET_THRESHOLD
    frame_threshold: float = settings.BASIC_PITCH_FRAME_THRESHOLD
    minimum_note_length_ms: float = settings.BASIC_PITCH_MIN_NOTE_LENGTH_MS
    minimum_frequency_hz: float | None = settings.BASIC_PITCH_MIN_FREQUENCY_HZ
    maximum_frequency_hz: float | None = settings.BASIC_PITCH_MAX_FREQUENCY_HZ
    melodia_trick: bool = settings.BASIC_PITCH_MELODIA_TRICK
    # full mix
    use_source_separation: bool = False
    stems: tuple[str, ...] = ("vocals", "bass", "other")
    # post-processing
    min_confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stems"] = list(self.stems)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TranscriptionConfig":
        data = dict(data)
        seg = data.pop("segmentation", None) or {}
        data["stems"] = tuple(data.get("stems", ("vocals", "bass", "other")))
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(segmentation=SegmentationConfig(**seg), **known)

    @property
    def mode_enum(self) -> TranscriptionMode:
        return TranscriptionMode(self.mode)


class TranscriptionError(RuntimeError):
    """A mode could not process the recording. `suggested_mode` offers a fallback."""

    def __init__(self, message: str, suggested_mode: TranscriptionMode | None = None) -> None:
        super().__init__(message)
        self.suggested_mode = suggested_mode


# --------------------------------------------------------------------------- #
# Event schema
# --------------------------------------------------------------------------- #
EVENT_COLUMNS: list[str] = [
    "event_id", "song_id", "note_name", "midi_note", "pitch_class", "octave", "frequency_hz",
    "start_time_seconds", "end_time_seconds", "duration_seconds", "confidence", "instrument_label",
    "source_track", "onset_strength", "detection_method", "cents_deviation", "frequency_source",
    "overlaps_other_events", "overlap_count", "is_valid", "validation_error",
]
DISPLAY_COLUMNS: dict[str, str] = {
    "event_id": "Event ID", "note_name": "Note", "midi_note": "MIDI Number", "frequency_hz": "Frequency (Hz)",
    "start_time_formatted": "Start Time", "end_time_formatted": "End Time", "duration_seconds": "Duration (s)",
    "confidence": "Confidence",
}


@dataclass
class NoteEvent:
    event_id: str
    song_id: str
    note_name: str
    midi_note: int
    pitch_class: str
    octave: int
    frequency_hz: float
    start_time_seconds: float
    end_time_seconds: float
    duration_seconds: float
    confidence: float | None = None
    instrument_label: str | None = None
    source_track: str | None = "mix"
    onset_strength: float | None = None
    detection_method: str = ""
    cents_deviation: float | None = None
    frequency_source: str = "equal_tempered"
    overlaps_other_events: bool = False
    overlap_count: int = 0
    is_valid: bool = True
    validation_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NoteEvent":
        known = {k: data.get(k) for k in cls.__dataclass_fields__}
        known["overlaps_other_events"] = bool(known.get("overlaps_other_events") or False)
        known["overlap_count"] = int(known.get("overlap_count") or 0)
        known["is_valid"] = bool(known.get("is_valid", True))
        return cls(**known)

    @property
    def start_time_formatted(self) -> str:
        return format_timestamp(self.start_time_seconds)

    @property
    def end_time_formatted(self) -> str:
        return format_timestamp(self.end_time_seconds)


def validate_event(event: NoteEvent) -> list[str]:
    """Return a list of problems (empty = valid)."""
    problems: list[str] = []
    for name in ("start_time_seconds", "end_time_seconds", "duration_seconds"):
        value = getattr(event, name)
        if value is None or not np.isfinite(value):
            problems.append(f"{name} is not finite")
    if not problems:
        if event.start_time_seconds < 0:
            problems.append("start_time_seconds is negative")
        if event.end_time_seconds < event.start_time_seconds:
            problems.append("end_time_seconds is before start_time_seconds")
        if abs((event.end_time_seconds - event.start_time_seconds) - event.duration_seconds) > 1e-3:
            problems.append("duration_seconds does not match end - start")
    if event.midi_note is None or not (0 <= int(event.midi_note) <= 127):
        problems.append("midi_note outside 0-127")
    if not is_valid_frequency(event.frequency_hz):
        problems.append("frequency_hz is not a positive finite number")
    if event.confidence is not None and (not np.isfinite(event.confidence) or not (0.0 <= event.confidence <= 1.0)):
        problems.append("confidence outside 0-1")
    return problems


def compute_overlaps(events: list[NoteEvent]) -> None:
    """Flag events that overlap in time with any other event (in place).

    Sweep-line over events sorted by start; two events overlap when
    start_j < end_i and start_i < end_j (touching boundaries do not count).
    """
    for ev in events:
        ev.overlap_count = 0
        ev.overlaps_other_events = False
    order = sorted(range(len(events)), key=lambda i: (events[i].start_time_seconds, events[i].end_time_seconds))
    active: list[tuple[float, int]] = []  # heap of (end, index)
    for i in order:
        start = events[i].start_time_seconds
        while active and active[0][0] <= start:
            heapq.heappop(active)
        for _end, j in active:
            events[j].overlap_count += 1
        events[i].overlap_count += len(active)
        heapq.heappush(active, (events[i].end_time_seconds, i))
    for ev in events:
        ev.overlaps_other_events = ev.overlap_count > 0


def build_note_events(
    raw_notes: list[RawNote],
    song_id: str,
    detection_method: str,
    time_offset: float = 0.0,
    instrument_label: str | None = None,
    min_confidence: float = 0.0,
) -> tuple[list[NoteEvent], list[NoteEvent]]:
    """Convert model-level notes into validated `NoteEvent`s on the original timeline.

    Returns (valid_events, excluded_events). Invalid events are never dropped
    silently; they are returned separately with the reason attached.
    """
    decimals = settings.TIMESTAMP_DECIMALS
    prefix = song_id[:8]
    raw_sorted = sorted(raw_notes, key=lambda r: (r.start, r.midi))
    valid: list[NoteEvent] = []
    excluded: list[NoteEvent] = []
    for index, raw in enumerate(raw_sorted):
        midi = int(raw.midi)
        start = round(float(raw.start) + time_offset, decimals)
        end = round(float(raw.end) + time_offset, decimals)
        duration = round(end - start, decimals)
        freq = clean_float(raw.frequency_hz)
        freq_source = raw.frequency_source
        if freq is None or freq <= 0:
            freq = midi_to_frequency(midi) if 0 <= midi <= 127 else float("nan")
            freq_source = "equal_tempered"
        confidence = clean_float(raw.confidence)
        if confidence is not None:
            confidence = float(min(max(confidence, 0.0), 1.0))
        event = NoteEvent(
            event_id=f"{prefix}-{index + 1:05d}",
            song_id=song_id,
            note_name=midi_to_note_name(midi) if 0 <= midi <= 127 else "?",
            midi_note=midi,
            pitch_class=midi_to_pitch_class(midi) if 0 <= midi <= 127 else "?",
            octave=midi_to_octave(midi) if 0 <= midi <= 127 else 0,
            frequency_hz=float(freq),
            start_time_seconds=start,
            end_time_seconds=end,
            duration_seconds=duration,
            confidence=confidence,
            instrument_label=instrument_label,
            source_track=raw.source_track,
            onset_strength=clean_float(raw.onset_strength),
            detection_method=detection_method,
            cents_deviation=(round(cents_deviation(freq, midi), 2) if is_valid_frequency(freq) else None),
            frequency_source=freq_source,
        )
        problems = validate_event(event)
        if problems:
            event.is_valid = False
            event.validation_error = "; ".join(problems)
            excluded.append(event)
            continue
        if confidence is not None and min_confidence > 0 and confidence < min_confidence:
            event.is_valid = False
            event.validation_error = f"confidence below {min_confidence:.2f} (filtered)"
            excluded.append(event)
            continue
        valid.append(event)
    compute_overlaps(valid)
    return valid, excluded


def events_to_dataframe(events: list[NoteEvent]) -> pd.DataFrame:
    """Tabular view with formatted time columns; stable column order."""
    rows = [e.to_dict() for e in events]
    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    if df.empty:
        df["start_time_formatted"] = pd.Series(dtype=str)
        df["end_time_formatted"] = pd.Series(dtype=str)
        return df
    df["start_time_formatted"] = df["start_time_seconds"].map(format_timestamp)
    df["end_time_formatted"] = df["end_time_seconds"].map(format_timestamp)
    df["midi_note"] = df["midi_note"].astype(int)
    df["octave"] = df["octave"].astype(int)
    df["overlap_count"] = df["overlap_count"].astype(int)
    df["overlaps_other_events"] = df["overlaps_other_events"].astype(bool)
    df["is_valid"] = df["is_valid"].astype(bool)
    return df.sort_values(["start_time_seconds", "midi_note"]).reset_index(drop=True)


def dataframe_to_events(df: pd.DataFrame) -> list[NoteEvent]:
    events: list[NoteEvent] = []
    for row in df.to_dict(orient="records"):
        cleaned = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()}
        events.append(NoteEvent.from_dict(cleaned))
    return events


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class TranscriptionResult:
    events: list[NoteEvent]
    excluded_events: list[NoteEvent]
    mode: str
    detection_method: str
    model_name: str
    model_version: str
    elapsed_seconds: float
    warnings: list[str] = field(default_factory=list)
    pitch_track: PitchTrack | None = None
    onsets: OnsetResult | None = None
    stems_used: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        return len(self.events)

    def model_info(self) -> dict[str, Any]:
        return {"mode": self.mode, "detection_method": self.detection_method, "model_name": self.model_name,
                "model_version": self.model_version, "elapsed_seconds": round(self.elapsed_seconds, 3),
                "stems_used": self.stems_used, **self.details}


# --------------------------------------------------------------------------- #
# Basic Pitch wrapper (bundled ONNX model, see src/basic_pitch_onnx.py)
# --------------------------------------------------------------------------- #
def basic_pitch_available() -> tuple[bool, str]:
    """(available, detail) without loading the model."""
    from src import basic_pitch_onnx

    return basic_pitch_onnx.is_available()


def basic_pitch_version() -> str:
    from src import basic_pitch_onnx

    return basic_pitch_onnx.MODEL_VERSION


def get_basic_pitch_model() -> tuple[Any, str]:
    """Load the bundled ICASSP-2022 ONNX model once and reuse it (thread-safe)."""
    from src import basic_pitch_onnx

    ok, detail = basic_pitch_onnx.is_available()
    if not ok:
        raise TranscriptionError(detail + " Switch to the monophonic mode (pYIN) meanwhile.",
                                 suggested_mode=TranscriptionMode.MONOPHONIC)
    model = basic_pitch_onnx.get_model()
    return model, model.backend


# Basic Pitch's contour output has 3 bins per semitone from A0 (27.5 Hz). Measured
# on in-tune test tones, the energy peak of a note at MIDI m sits at bin
# 3*(m-21) + 1 (one third of a semitone above the nominal formula), so the bin
# centre frequency is 27.5 * 2**((bin - CONTOUR_BIN_OFFSET) / 36).
CONTOUR_BINS_PER_SEMITONE = 3
CONTOUR_BIN_OFFSET = 1
CONTOUR_BASE_FREQUENCY_HZ = 27.5
CONTOUR_LOWEST_MIDI = 21


def contour_bin_to_frequency(bin_index: float) -> float:
    return CONTOUR_BASE_FREQUENCY_HZ * 2.0 ** ((bin_index - CONTOUR_BIN_OFFSET) / (12 * CONTOUR_BINS_PER_SEMITONE))


def refine_frequency_from_contour(contour_block: np.ndarray, midi: int, search_bins: int = 2) -> tuple[float, str]:
    """Estimate a note's f0 from the model's pitch-contour activations.

    The mean activation profile over the note's frames is searched within
    +/-`search_bins` of the note's nominal bin; the peak is refined with
    parabolic interpolation (resolution ~1/3 semitone before interpolation).
    Falls back to the equal-tempered frequency when the contour is empty.
    """
    nominal = midi_to_frequency(midi)
    if contour_block is None or contour_block.size == 0:
        return nominal, "equal_tempered"
    n_bins = contour_block.shape[1]
    centre = CONTOUR_BINS_PER_SEMITONE * (midi - CONTOUR_LOWEST_MIDI) + CONTOUR_BIN_OFFSET
    if not (0 <= centre < n_bins):
        return nominal, "equal_tempered"
    lo, hi = max(0, centre - search_bins), min(n_bins, centre + search_bins + 1)
    profile = contour_block[:, lo:hi].mean(axis=0)
    if not np.all(np.isfinite(profile)) or float(profile.max()) <= 1e-6:
        return nominal, "equal_tempered"
    k = int(np.argmax(profile)) + lo
    refined = float(k)
    if 1 <= k < n_bins - 1:
        a, b, c = float(contour_block[:, k - 1].mean()), float(contour_block[:, k].mean()), float(contour_block[:, k + 1].mean())
        denom = a - 2 * b + c
        if denom < 0:  # proper peak
            delta = 0.5 * (a - c) / denom
            refined = k + float(np.clip(delta, -0.5, 0.5))
    return contour_bin_to_frequency(refined), "basic_pitch_contour"


def run_basic_pitch(
    samples: np.ndarray,
    sample_rate: int,
    config: TranscriptionConfig,
    progress: ProgressCallback = no_progress,
    cancel: CancelToken | None = None,
    source_track: str = "mix",
) -> tuple[list[RawNote], dict[str, Any]]:
    """Run Basic Pitch on an in-memory mono signal with progress and cancellation.

    Uses the bundled ONNX model (src/basic_pitch_onnx.py): windowed inference
    with per-window progress, the original note-decoding algorithm, and a
    contour-based frequency refinement for every note.
    """
    from src import basic_pitch_onnx as bp

    if sample_rate != bp.AUDIO_SAMPLE_RATE:
        raise TranscriptionError(f"Basic Pitch requires {bp.AUDIO_SAMPLE_RATE} Hz audio; got {sample_rate} Hz.")
    model, backend = get_basic_pitch_model()
    try:
        unwrapped, info = model.run(samples, progress, cancel, label=source_track)
    except AnalysisCancelled:
        raise
    except RuntimeError as exc:
        raise TranscriptionError(str(exc), suggested_mode=TranscriptionMode.MONOPHONIC) from exc

    min_note_len_frames = int(np.round(config.minimum_note_length_ms / 1000 * (bp.AUDIO_SAMPLE_RATE / bp.FFT_HOP)))
    notes = bp.output_to_notes_polyphonic(
        unwrapped["note"], unwrapped["onset"], onset_thresh=config.onset_threshold, frame_thresh=config.frame_threshold,
        min_note_len=max(1, min_note_len_frames), infer_onsets=True, max_freq=config.maximum_frequency_hz,
        min_freq=config.minimum_frequency_hz, melodia_trick=config.melodia_trick,
    )
    contours = unwrapped["contour"]
    times = bp.model_frames_to_time(contours.shape[0])
    n_frames = contours.shape[0]

    raw: list[RawNote] = []
    for start_frame, end_frame, midi, amplitude in notes:
        start_frame = int(np.clip(start_frame, 0, n_frames - 1))
        end_frame = int(np.clip(end_frame, start_frame + 1, n_frames))
        start_s = float(times[start_frame])
        end_s = float(times[end_frame - 1] + bp.FRAME_PERIOD_SECONDS)
        midi = int(midi)
        freq, freq_source = refine_frequency_from_contour(contours[start_frame:end_frame], midi)
        raw.append(RawNote(start=start_s, end=end_s, midi=midi, frequency_hz=freq, confidence=float(amplitude),
                           frequency_source=freq_source, source_track=source_track, n_frames=end_frame - start_frame,
                           extra={"model_amplitude": float(amplitude)}))
    info = dict(info)
    info["backend"] = backend
    return raw, info


# --------------------------------------------------------------------------- #
# Mode implementations
# --------------------------------------------------------------------------- #
def transcribe_monophonic(
    audio: PreprocessedAudio, config: TranscriptionConfig, song_id: str,
    progress: ProgressCallback = no_progress, cancel: CancelToken | None = None,
) -> TranscriptionResult:
    t0 = time.time()
    progress(0.05, "Tracking fundamental frequency (pYIN)")
    track = track_pitch_pyin(audio.samples, audio.sample_rate, config.pyin_fmin_note, config.pyin_fmax_note,
                             config.pyin_frame_length, config.pyin_hop_length)
    if cancel is not None:
        cancel.check()
    progress(0.70, "Detecting onsets")
    onsets = detect_onsets(audio.samples, audio.sample_rate, hop_length=config.pyin_hop_length)
    progress(0.85, "Segmenting notes")
    raw = segment_pitch_track(track, onsets, config.segmentation)
    events, excluded = build_note_events(raw, song_id, "pyin", audio.time_offset_seconds,
                                         min_confidence=config.min_confidence)
    warnings: list[str] = [
        "Monophonic mode reports one pitch at a time. Chords and accompaniment are not transcribed; "
        "if the recording contains several simultaneous notes use the polyphonic mode."
    ]
    voiced_ratio = float(np.mean(track.voiced_flag)) if track.n_frames else 0.0
    if track.n_frames and voiced_ratio < 0.2:
        warnings.append(f"Only {voiced_ratio:.0%} of frames were judged voiced; the recording may be noisy, "
                        f"percussive or polyphonic.")
    if not events:
        warnings.append("No note events were found. The signal may be silent, noisy, or outside the C1-C8 range.")
    import librosa

    progress(1.0, f"Monophonic transcription finished: {len(events)} events")
    return TranscriptionResult(
        events=events, excluded_events=excluded, mode=TranscriptionMode.MONOPHONIC.value, detection_method="pyin",
        model_name="pYIN (librosa.pyin) + onset segmentation", model_version=f"librosa {librosa.__version__}",
        elapsed_seconds=time.time() - t0, warnings=warnings, pitch_track=track, onsets=onsets,
        details={"n_frames": track.n_frames, "voiced_ratio": round(voiced_ratio, 4),
                 "frame_period_seconds": track.frame_period_seconds, "n_onsets": onsets.n_onsets},
    )


def transcribe_polyphonic(
    audio: PreprocessedAudio, config: TranscriptionConfig, song_id: str,
    progress: ProgressCallback = no_progress, cancel: CancelToken | None = None,
) -> TranscriptionResult:
    t0 = time.time()
    ok, detail = basic_pitch_available()
    if not ok:
        raise TranscriptionError(detail + " Switch to the monophonic mode (pYIN) or install Basic Pitch.",
                                 suggested_mode=TranscriptionMode.MONOPHONIC)
    progress(0.02, "Loading Basic Pitch model")
    raw, info = run_basic_pitch(audio.samples, audio.sample_rate, config, scaled(progress, 0.05, 0.9), cancel)
    progress(0.92, "Building note events")
    events, excluded = build_note_events(raw, song_id, "basic_pitch", audio.time_offset_seconds,
                                         min_confidence=config.min_confidence)
    warnings = [
        "Basic Pitch is a neural network estimate: octave errors, missed soft notes and spurious harmonics "
        "are possible. Confidence is the model's note activation (0-1)."
    ]
    if not events:
        warnings.append("No note events passed the thresholds. Try lowering the onset/frame thresholds.")
    progress(1.0, f"Polyphonic transcription finished: {len(events)} events")
    return TranscriptionResult(
        events=events, excluded_events=excluded, mode=TranscriptionMode.POLYPHONIC.value, detection_method="basic_pitch",
        model_name=settings.BASIC_PITCH_MODEL_NAME, model_version=f"{basic_pitch_version()} via {info['backend']}",
        elapsed_seconds=time.time() - t0, warnings=warnings, details=info,
    )


def transcribe_full_mix(
    audio: PreprocessedAudio, config: TranscriptionConfig, song_id: str,
    progress: ProgressCallback = no_progress, cancel: CancelToken | None = None,
) -> TranscriptionResult:
    from src import source_separation

    t0 = time.time()
    ok, detail = basic_pitch_available()
    if not ok:
        raise TranscriptionError(detail + " Switch to the monophonic mode (pYIN) or install Basic Pitch.",
                                 suggested_mode=TranscriptionMode.MONOPHONIC)
    warnings = [
        "Full-mix transcription is the hardest case: vocals, percussion, effects and overlapping instruments "
        "reduce accuracy. Treat results as an approximation and check them against the spectrogram."
    ]
    stems_used: list[str] = []
    raw_all: list[RawNote] = []
    details: dict[str, Any] = {}

    if config.use_source_separation:
        info = source_separation.is_available()
        if info.available:
            try:
                stems = source_separation.separate(audio.samples, audio.sample_rate, tuple(config.stems),
                                                   progress=scaled(progress, 0.02, 0.45), cancel=cancel)
            except Exception as exc:  # noqa: BLE001 - fall back to the mix, but tell the user
                log.exception("Source separation failed")
                warnings.append(f"Source separation failed ({exc}); analysed the full mix instead.")
                stems = {}
            n = max(len(stems), 1)
            for i, (name, stem_audio) in enumerate(stems.items()):
                if cancel is not None:
                    cancel.check()
                sub = scaled(progress, 0.45 + 0.45 * i / n, 0.45 + 0.45 * (i + 1) / n)
                raw, info_i = run_basic_pitch(stem_audio, audio.sample_rate, config, sub, cancel, source_track=name)
                raw_all.extend(raw)
                stems_used.append(name)
                details[f"stem_{name}"] = info_i
            if stems:
                details["separation"] = source_separation.describe(info)
        else:
            warnings.append(info.reason + " The full mix was analysed without separation.")

    if not stems_used:
        raw, info = run_basic_pitch(audio.samples, audio.sample_rate, config, scaled(progress, 0.05, 0.9), cancel,
                                    source_track="mix")
        raw_all.extend(raw)
        details["mix"] = info
        stems_used = ["mix"]

    progress(0.92, "Building note events")
    events, excluded = build_note_events(raw_all, song_id, "basic_pitch", audio.time_offset_seconds,
                                         min_confidence=config.min_confidence)
    if not events:
        warnings.append("No note events passed the thresholds. Try lowering the onset/frame thresholds.")
    progress(1.0, f"Full-mix transcription finished: {len(events)} events")
    backend = next((v.get("backend") for v in details.values() if isinstance(v, dict) and "backend" in v), "unknown")
    return TranscriptionResult(
        events=events, excluded_events=excluded, mode=TranscriptionMode.FULL_MIX.value, detection_method="basic_pitch",
        model_name=settings.BASIC_PITCH_MODEL_NAME + (" + Demucs stems" if "separation" in details else ""),
        model_version=f"{basic_pitch_version()} via {backend}", elapsed_seconds=time.time() - t0,
        warnings=warnings, stems_used=stems_used, details=details,
    )


def transcribe(
    audio: PreprocessedAudio, config: TranscriptionConfig, song_id: str,
    progress: ProgressCallback = no_progress, cancel: CancelToken | None = None,
) -> TranscriptionResult:
    """Dispatch to the implementation for `config.mode`."""
    mode = TranscriptionMode(config.mode)
    log.info("Transcribing %s in %s mode (%.1f s of audio)", song_id, mode.value, audio.duration_seconds)
    if mode is TranscriptionMode.MONOPHONIC:
        return transcribe_monophonic(audio, config, song_id, progress, cancel)
    if mode is TranscriptionMode.POLYPHONIC:
        return transcribe_polyphonic(audio, config, song_id, progress, cancel)
    return transcribe_full_mix(audio, config, song_id, progress, cancel)
