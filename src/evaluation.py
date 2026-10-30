"""
Transcription quality evaluation against ground-truth annotations.

Ground truth is a list of (start_s, end_s, midi) tuples, e.g. from a MIDI file
or a CSV with columns start_time_seconds, end_time_seconds, midi_note. Metrics
follow the standard MIREX note-transcription definitions as implemented in
`mir_eval.transcription` (a Basic Pitch dependency, so always available here):

* precision / recall / F1 with an onset tolerance (default 50 ms) and pitch
  tolerance (default 50 cents); offsets are evaluated separately so that
  offset errors (which are hard to annotate) do not dominate the score.
* mean absolute onset and offset error over the matched note pairs.

Nothing here claims perfection: these numbers describe one recording under
one tolerance and are only as good as the annotation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.note_transcription import NoteEvent
from src.pitch_detection import midi_to_frequency

GroundTruth = Sequence[tuple[float, float, int]]


@dataclass
class EvaluationResult:
    n_reference: int
    n_estimated: int
    n_matched: int
    precision: float
    recall: float
    f1: float
    precision_with_offset: float
    recall_with_offset: float
    f1_with_offset: float
    mean_onset_error_ms: float | None
    mean_offset_error_ms: float | None
    median_onset_error_ms: float | None
    onset_tolerance_ms: float
    pitch_tolerance_cents: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_arrays(items: GroundTruth) -> tuple[np.ndarray, np.ndarray]:
    if not items:
        return np.zeros((0, 2)), np.zeros(0)
    intervals = np.array([[float(s), float(e)] for s, e, _ in items], dtype=float)
    pitches = np.array([midi_to_frequency(int(m)) for _, _, m in items], dtype=float)
    return intervals, pitches


def evaluate_transcription(
    events: Sequence[NoteEvent],
    ground_truth: GroundTruth,
    onset_tolerance_s: float = 0.05,
    pitch_tolerance_cents: float = 50.0,
    offset_ratio: float = 0.2,
) -> EvaluationResult:
    import mir_eval.transcription as mt

    ref_int, ref_pitch = _to_arrays(ground_truth)
    est_int, est_pitch = _to_arrays([(e.start_time_seconds, e.end_time_seconds, e.midi_note) for e in events])
    kwargs = dict(onset_tolerance=onset_tolerance_s, pitch_tolerance=pitch_tolerance_cents)

    if len(ref_int) == 0 or len(est_int) == 0:
        return EvaluationResult(len(ref_int), len(est_int), 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, None,
                                onset_tolerance_s * 1000, pitch_tolerance_cents)

    p, r, f, _ = mt.precision_recall_f1_overlap(ref_int, ref_pitch, est_int, est_pitch, offset_ratio=None, **kwargs)
    p_off, r_off, f_off, _ = mt.precision_recall_f1_overlap(ref_int, ref_pitch, est_int, est_pitch,
                                                            offset_ratio=offset_ratio, **kwargs)
    matches = mt.match_notes(ref_int, ref_pitch, est_int, est_pitch, offset_ratio=None, **kwargs)
    if matches:
        onset_err = np.array([abs(est_int[j, 0] - ref_int[i, 0]) for i, j in matches]) * 1000
        offset_err = np.array([abs(est_int[j, 1] - ref_int[i, 1]) for i, j in matches]) * 1000
        mean_on, mean_off, med_on = float(onset_err.mean()), float(offset_err.mean()), float(np.median(onset_err))
    else:
        mean_on = mean_off = med_on = None
    return EvaluationResult(
        n_reference=len(ref_int), n_estimated=len(est_int), n_matched=len(matches), precision=float(p), recall=float(r),
        f1=float(f), precision_with_offset=float(p_off), recall_with_offset=float(r_off), f1_with_offset=float(f_off),
        mean_onset_error_ms=mean_on, mean_offset_error_ms=mean_off, median_onset_error_ms=med_on,
        onset_tolerance_ms=onset_tolerance_s * 1000, pitch_tolerance_cents=pitch_tolerance_cents,
    )


def load_ground_truth_csv(path: str | Path) -> list[tuple[float, float, int]]:
    """Read annotations from CSV with start_time_seconds, end_time_seconds and midi_note (or note_name) columns."""
    from src.pitch_detection import note_name_to_midi

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    start = cols.get("start_time_seconds") or cols.get("start") or cols.get("onset")
    end = cols.get("end_time_seconds") or cols.get("end") or cols.get("offset")
    if start is None or end is None:
        raise ValueError("Ground-truth CSV needs start/end time columns (start_time_seconds, end_time_seconds).")
    if "midi_note" in cols:
        midis = df[cols["midi_note"]].astype(int)
    elif "note_name" in cols:
        midis = df[cols["note_name"]].map(note_name_to_midi)
    else:
        raise ValueError("Ground-truth CSV needs a midi_note or note_name column.")
    return [(float(s), float(e), int(m)) for s, e, m in zip(df[start], df[end], midis)]
