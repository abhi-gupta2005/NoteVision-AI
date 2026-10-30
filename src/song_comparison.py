"""
Cross-song comparison of note statistics.

Distributions are normalised to proportions before comparing so that songs of
different lengths can be set side by side. Similarity of note distributions
says NOTHING about melodic identity: two songs in the same key will have
similar pitch-class histograms even if their melodies share nothing. The
optional sequence comparison is a separate, equally approximate indicator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd

from src.pitch_detection import PITCH_CLASS_NAMES

DISTRIBUTION_DISCLAIMER = (
    "Similar note distributions do not mean two recordings share a melody: songs in the same key "
    "naturally use the same pitch classes. Use the similarity values only as a rough indicator of "
    "tonal material, and the sequence similarity only as an approximate indicator of shared note order."
)


@dataclass
class SongSummary:
    song_id: str
    name: str
    duration_seconds: float
    total_events: int
    unique_midi_pitches: int
    unique_pitch_classes: int
    mean_duration_seconds: float | None
    median_duration_seconds: float | None
    most_frequent_note: str | None
    tempo_bpm: float | None
    mode: str
    pitch_class_counts: dict[str, int] = field(default_factory=dict)
    octave_counts: dict[int, int] = field(default_factory=dict)
    note_sequence: list[int] = field(default_factory=list)   # MIDI numbers ordered by start time

    @classmethod
    def from_analysis(cls, result: Any) -> "SongSummary":
        """Build from an `AnalysisResult` (duck-typed to avoid a circular import)."""
        stats = result.statistics
        pc = {row["pitch_class"]: int(row["count"]) for row in stats.pitch_class_table.to_dict(orient="records")}
        octs = {int(row["octave"]): int(row["count"]) for row in stats.octave_table.to_dict(orient="records")}
        df = result.events_df
        seq = df.sort_values(["start_time_seconds", "midi_note"])["midi_note"].astype(int).tolist() if not df.empty else []
        return cls(
            song_id=result.song_id, name=result.metadata.filename, duration_seconds=result.metadata.duration_seconds,
            total_events=stats.valid_events, unique_midi_pitches=stats.unique_midi_pitches,
            unique_pitch_classes=stats.unique_pitch_classes, mean_duration_seconds=stats.mean_event_duration_seconds,
            median_duration_seconds=stats.median_event_duration_seconds, most_frequent_note=stats.most_frequent_note,
            tempo_bpm=result.tempo_bpm, mode=result.transcription_config.mode, pitch_class_counts=pc,
            octave_counts=octs, note_sequence=seq,
        )


@dataclass
class ComparisonResult:
    summary_table: pd.DataFrame
    pitch_class_distribution: pd.DataFrame      # rows = songs, columns = 12 pitch classes (proportions)
    octave_distribution: pd.DataFrame           # rows = songs, columns = octaves (proportions)
    pitch_class_counts: pd.DataFrame            # raw counts
    cosine_similarity: pd.DataFrame
    jensen_shannon_distance: pd.DataFrame
    sequence_similarity: pd.DataFrame | None
    notes: list[str]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def normalise_rows(df: pd.DataFrame) -> pd.DataFrame:
    totals = df.sum(axis=1).replace(0, np.nan)
    return df.div(totals, axis=0).fillna(0.0)


def cosine_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    m = df.to_numpy(dtype=float)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    unit = np.divide(m, norms, out=np.zeros_like(m), where=norms > 0)
    sim = unit @ unit.T
    return pd.DataFrame(np.clip(sim, 0.0, 1.0), index=df.index, columns=df.index)


def jensen_shannon_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Jensen-Shannon distance (0 = identical, 1 = disjoint) between row distributions."""
    p = normalise_rows(df).to_numpy(dtype=float)
    n = p.shape[0]
    out = np.zeros((n, n))

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    for i in range(n):
        for j in range(i + 1, n):
            m = 0.5 * (p[i] + p[j])
            js = 0.5 * _kl(p[i], m) + 0.5 * _kl(p[j], m)
            out[i, j] = out[j, i] = float(np.sqrt(max(js, 0.0)))
    return pd.DataFrame(out, index=df.index, columns=df.index)


def sequence_similarity(seq_a: list[int], seq_b: list[int], use_pitch_class: bool = True) -> float:
    """Ratio of matching ordered notes (difflib), 0..1. Transposition-invariant when comparing pitch classes
    relative to the first note is NOT applied; this is a plain order comparison."""
    if not seq_a or not seq_b:
        return 0.0
    a = [m % 12 for m in seq_a] if use_pitch_class else list(seq_a)
    b = [m % 12 for m in seq_b] if use_pitch_class else list(seq_b)
    return float(SequenceMatcher(None, a, b, autojunk=False).ratio())


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def compare_songs(summaries: list[SongSummary], include_sequences: bool = True) -> ComparisonResult:
    if not summaries:
        raise ValueError("At least one song is needed for a comparison")
    names = _unique_names([s.name for s in summaries])

    summary_table = pd.DataFrame([{
        "song": name, "song_id": s.song_id, "mode": s.mode, "duration_seconds": round(s.duration_seconds, 3),
        "total_events": s.total_events, "events_per_minute": round(60.0 * s.total_events / s.duration_seconds, 2) if s.duration_seconds else 0.0,
        "unique_pitches": s.unique_midi_pitches, "unique_pitch_classes": s.unique_pitch_classes,
        "mean_duration_seconds": None if s.mean_duration_seconds is None else round(s.mean_duration_seconds, 4),
        "median_duration_seconds": None if s.median_duration_seconds is None else round(s.median_duration_seconds, 4),
        "most_frequent_note": s.most_frequent_note, "tempo_bpm": None if s.tempo_bpm is None else round(s.tempo_bpm, 1),
    } for name, s in zip(names, summaries)])

    pc_counts = pd.DataFrame([{pc: s.pitch_class_counts.get(pc, 0) for pc in PITCH_CLASS_NAMES} for s in summaries], index=names)
    all_octaves = sorted({o for s in summaries for o in s.octave_counts})
    oct_counts = pd.DataFrame([{o: s.octave_counts.get(o, 0) for o in all_octaves} for s in summaries], index=names)

    pc_dist = normalise_rows(pc_counts)
    oct_dist = normalise_rows(oct_counts) if all_octaves else oct_counts
    cos = cosine_similarity_matrix(pc_dist)
    js = jensen_shannon_matrix(pc_counts)

    seq_df = None
    if include_sequences and len(summaries) > 1:
        n = len(summaries)
        mat = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                mat[i, j] = mat[j, i] = sequence_similarity(summaries[i].note_sequence, summaries[j].note_sequence)
        seq_df = pd.DataFrame(mat, index=names, columns=names)

    notes = [DISTRIBUTION_DISCLAIMER]
    if any(s.mode != summaries[0].mode for s in summaries):
        notes.append("The songs were analysed with different modes/models; their event counts are not directly comparable.")
    if any(s.total_events == 0 for s in summaries):
        notes.append("At least one song has no detected events; its distribution is empty.")
    return ComparisonResult(summary_table, pc_dist, oct_dist, pc_counts, cos, js, seq_df, notes)


def _unique_names(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n} ({seen[n]})")
        else:
            seen[n] = 1
            out.append(n)
    return out
