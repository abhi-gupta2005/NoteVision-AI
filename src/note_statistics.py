"""
Statistical analysis of note events for one song.

Terminology (kept strictly apart everywhere in the app):
* occurrence count  - how many separate events of a note were detected
* total duration    - the sum of those events' durations in seconds

Invalid events never enter the statistics; their number is reported in
`excluded_invalid_events`.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from config import settings
from src.note_transcription import NoteEvent, events_to_dataframe
from src.pitch_detection import PITCH_CLASS_NAMES

NOTE_TABLE_COLUMNS = [
    "note_name", "midi_note", "pitch_class", "octave", "count", "percent_of_events", "total_duration_seconds",
    "percent_of_total_duration", "mean_duration_seconds", "median_duration_seconds", "min_duration_seconds",
    "max_duration_seconds", "first_occurrence_seconds", "last_occurrence_seconds", "mean_confidence",
]


@dataclass
class SongStatistics:
    total_events: int
    valid_events: int
    excluded_invalid_events: int
    unique_midi_pitches: int
    unique_pitch_classes: int
    total_note_duration_seconds: float
    min_event_duration_seconds: float | None
    max_event_duration_seconds: float | None
    mean_event_duration_seconds: float | None
    median_event_duration_seconds: float | None
    most_frequent_note: str | None
    most_frequent_note_count: int
    least_frequent_note: str | None
    least_frequent_note_count: int
    longest_total_duration_note: str | None
    pitch_range_low: str | None
    pitch_range_high: str | None
    max_polyphony: int
    mean_polyphony: float
    overlapping_events: int
    short_outliers: int
    long_outliers: int
    outlier_bounds: tuple[float, float] | None
    note_table: pd.DataFrame = field(repr=False)
    pitch_class_table: pd.DataFrame = field(repr=False)
    octave_table: pd.DataFrame = field(repr=False)
    duration_histogram: dict[str, list[float]] = field(default_factory=dict, repr=False)
    timeline: dict[str, Any] = field(default_factory=dict, repr=False)
    polyphony: dict[str, list[float]] = field(default_factory=dict, repr=False)
    duration_percentiles: dict[str, float] = field(default_factory=dict)

    # -- serialisation ------------------------------------------------------
    def summary_dict(self) -> dict[str, Any]:
        """Scalar fields only (for overview cards, DB and Excel summary sheets)."""
        d = {k: v for k, v in asdict(self).items()
             if k not in ("note_table", "pitch_class_table", "octave_table", "duration_histogram", "timeline", "polyphony")}
        d["outlier_bounds"] = list(self.outlier_bounds) if self.outlier_bounds else None
        return d

    def to_dict(self) -> dict[str, Any]:
        d = self.summary_dict()
        d["note_table"] = self.note_table.to_dict(orient="records")
        d["pitch_class_table"] = self.pitch_class_table.to_dict(orient="records")
        d["octave_table"] = self.octave_table.to_dict(orient="records")
        d["duration_histogram"] = self.duration_histogram
        d["timeline"] = self.timeline
        d["polyphony"] = self.polyphony
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SongStatistics":
        data = dict(data)
        ob = data.get("outlier_bounds")
        data["outlier_bounds"] = tuple(ob) if ob else None
        data["note_table"] = pd.DataFrame(data.get("note_table") or [], columns=NOTE_TABLE_COLUMNS)
        data["pitch_class_table"] = pd.DataFrame(data.get("pitch_class_table") or [],
                                                 columns=["pitch_class", "count", "percent_of_events",
                                                          "total_duration_seconds", "mean_duration_seconds"])
        data["octave_table"] = pd.DataFrame(data.get("octave_table") or [],
                                            columns=["octave", "count", "percent_of_events", "total_duration_seconds",
                                                     "mean_duration_seconds", "notes"])
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


# --------------------------------------------------------------------------- #
# Component calculations (each usable on its own)
# --------------------------------------------------------------------------- #
def note_occurrence_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-note (MIDI pitch) occurrence and duration statistics."""
    if df.empty:
        return pd.DataFrame(columns=NOTE_TABLE_COLUMNS)
    total_events = len(df)
    total_duration = float(df["duration_seconds"].sum())
    g = df.groupby(["midi_note", "note_name", "pitch_class", "octave"], sort=True)
    table = g.agg(
        count=("event_id", "count"),
        total_duration_seconds=("duration_seconds", "sum"),
        mean_duration_seconds=("duration_seconds", "mean"),
        median_duration_seconds=("duration_seconds", "median"),
        min_duration_seconds=("duration_seconds", "min"),
        max_duration_seconds=("duration_seconds", "max"),
        first_occurrence_seconds=("start_time_seconds", "min"),
        last_occurrence_seconds=("start_time_seconds", "max"),
        mean_confidence=("confidence", "mean"),
    ).reset_index()
    table["percent_of_events"] = 100.0 * table["count"] / total_events
    table["percent_of_total_duration"] = 100.0 * table["total_duration_seconds"] / total_duration if total_duration > 0 else 0.0
    table = table[NOTE_TABLE_COLUMNS].sort_values("midi_note").reset_index(drop=True)
    return table


def pitch_class_table(df: pd.DataFrame) -> pd.DataFrame:
    """Occurrences per pitch class (C, C#, ... B) regardless of octave; all 12 rows present."""
    cols = ["pitch_class", "count", "percent_of_events", "total_duration_seconds", "mean_duration_seconds"]
    if df.empty:
        return pd.DataFrame({"pitch_class": PITCH_CLASS_NAMES, "count": 0, "percent_of_events": 0.0,
                             "total_duration_seconds": 0.0, "mean_duration_seconds": 0.0})[cols]
    g = df.groupby("pitch_class").agg(count=("event_id", "count"), total_duration_seconds=("duration_seconds", "sum"),
                                      mean_duration_seconds=("duration_seconds", "mean"))
    g = g.reindex(PITCH_CLASS_NAMES).fillna(0.0).reset_index().rename(columns={"index": "pitch_class"})
    g["count"] = g["count"].astype(int)
    g["percent_of_events"] = 100.0 * g["count"] / len(df)
    return g[cols]


def octave_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["octave", "count", "percent_of_events", "total_duration_seconds", "mean_duration_seconds", "notes"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    g = df.groupby("octave").agg(count=("event_id", "count"), total_duration_seconds=("duration_seconds", "sum"),
                                 mean_duration_seconds=("duration_seconds", "mean"),
                                 notes=("note_name", lambda s: ", ".join(sorted(set(s), key=lambda n: df.loc[df["note_name"] == n, "midi_note"].iloc[0]))))
    g = g.reset_index()
    g["percent_of_events"] = 100.0 * g["count"] / len(df)
    return g[cols].sort_values("octave").reset_index(drop=True)


def polyphony_timeline(df: pd.DataFrame, duration_seconds: float, grid_seconds: float = settings.POLYPHONY_GRID_SECONDS,
                       max_points: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Number of simultaneously sounding events sampled on a regular time grid."""
    if df.empty or duration_seconds <= 0:
        return np.array([]), np.array([])
    end = max(duration_seconds, float(df["end_time_seconds"].max()))
    n_points = int(np.ceil(end / grid_seconds)) + 1
    if n_points > max_points:
        grid_seconds = end / max_points
        n_points = max_points + 1
    grid = np.arange(n_points) * grid_seconds
    starts = df["start_time_seconds"].to_numpy(dtype=float)
    ends = df["end_time_seconds"].to_numpy(dtype=float)
    # Difference array: +1 at the first grid index >= start, -1 at the first index >= end.
    counts = np.zeros(n_points + 1, dtype=int)
    si = np.clip(np.searchsorted(grid, starts, side="left"), 0, n_points)
    ei = np.clip(np.searchsorted(grid, ends, side="left"), 0, n_points)
    np.add.at(counts, si, 1)
    np.add.at(counts, ei, -1)
    active = np.cumsum(counts)[:n_points]
    return grid, active


def timeline_distribution(df: pd.DataFrame, duration_seconds: float, n_bins: int = settings.TIMELINE_BINS) -> dict[str, Any]:
    """Events starting in each of `n_bins` equal time bins across the song."""
    if df.empty or duration_seconds <= 0:
        return {"bin_starts": [], "bin_width": 0.0, "counts": [], "duration_sum": []}
    end = max(duration_seconds, float(df["end_time_seconds"].max()))
    n_bins = max(1, min(n_bins, int(np.ceil(end)) or 1))
    edges = np.linspace(0.0, end, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, df["start_time_seconds"].to_numpy(dtype=float), side="right") - 1, 0, n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    dur = np.bincount(idx, weights=df["duration_seconds"].to_numpy(dtype=float), minlength=n_bins)
    return {"bin_starts": edges[:-1].tolist(), "bin_width": float(edges[1] - edges[0]), "counts": counts.tolist(),
            "duration_sum": dur.tolist()}


def duration_outliers(df: pd.DataFrame) -> tuple[int, int, tuple[float, float] | None]:
    """Unusually short / long events by the 1.5 x IQR rule. Returns (n_short, n_long, (low, high))."""
    if len(df) < 4:
        return 0, 0, None
    d = df["duration_seconds"].to_numpy(dtype=float)
    q1, q3 = np.percentile(d, [25, 75])
    iqr = q3 - q1
    low, high = max(0.0, q1 - 1.5 * iqr), q3 + 1.5 * iqr
    return int(np.sum(d < low)), int(np.sum(d > high)), (float(low), float(high))


def duration_histogram(df: pd.DataFrame, bins: int = 30) -> dict[str, list[float]]:
    if df.empty:
        return {"edges": [], "counts": []}
    counts, edges = np.histogram(df["duration_seconds"].to_numpy(dtype=float), bins=bins)
    return {"edges": edges.tolist(), "counts": counts.tolist()}


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def compute_statistics(events: list[NoteEvent], duration_seconds: float, excluded_events: list[NoteEvent] | None = None,
                       events_df: pd.DataFrame | None = None) -> SongStatistics:
    excluded_events = excluded_events or []
    df = events_df if events_df is not None else events_to_dataframe(events)
    if "is_valid" in df and not df.empty:
        df = df[df["is_valid"]].copy()
    n_excluded = len(excluded_events)
    n_valid = len(df)

    note_tbl = note_occurrence_table(df)
    pc_tbl = pitch_class_table(df)
    oct_tbl = octave_table(df)
    grid, active = polyphony_timeline(df, duration_seconds)
    n_short, n_long, bounds = duration_outliers(df)

    if n_valid:
        d = df["duration_seconds"].to_numpy(dtype=float)
        by_count = note_tbl.sort_values(["count", "midi_note"], ascending=[False, True])
        least = note_tbl.sort_values(["count", "midi_note"], ascending=[True, True]).iloc[0]
        most = by_count.iloc[0]
        longest = note_tbl.sort_values("total_duration_seconds", ascending=False).iloc[0]
        lo_row = df.loc[df["midi_note"].idxmin()]
        hi_row = df.loc[df["midi_note"].idxmax()]
        percentiles = {f"p{p}": float(np.percentile(d, p)) for p in (5, 25, 50, 75, 95)}
        stats = SongStatistics(
            total_events=n_valid + n_excluded, valid_events=n_valid, excluded_invalid_events=n_excluded,
            unique_midi_pitches=int(df["midi_note"].nunique()), unique_pitch_classes=int(df["pitch_class"].nunique()),
            total_note_duration_seconds=float(d.sum()),
            min_event_duration_seconds=float(d.min()), max_event_duration_seconds=float(d.max()),
            mean_event_duration_seconds=float(d.mean()), median_event_duration_seconds=float(np.median(d)),
            most_frequent_note=str(most["note_name"]), most_frequent_note_count=int(most["count"]),
            least_frequent_note=str(least["note_name"]), least_frequent_note_count=int(least["count"]),
            longest_total_duration_note=str(longest["note_name"]),
            pitch_range_low=str(lo_row["note_name"]), pitch_range_high=str(hi_row["note_name"]),
            max_polyphony=int(active.max()) if active.size else 0,
            mean_polyphony=float(active[active > 0].mean()) if active.size and np.any(active > 0) else 0.0,
            overlapping_events=int(df["overlaps_other_events"].sum()),
            short_outliers=n_short, long_outliers=n_long, outlier_bounds=bounds,
            note_table=note_tbl, pitch_class_table=pc_tbl, octave_table=oct_tbl,
            duration_histogram=duration_histogram(df), timeline=timeline_distribution(df, duration_seconds),
            polyphony={"times": grid.tolist(), "counts": active.tolist()}, duration_percentiles=percentiles,
        )
    else:
        stats = SongStatistics(
            total_events=n_excluded, valid_events=0, excluded_invalid_events=n_excluded, unique_midi_pitches=0,
            unique_pitch_classes=0, total_note_duration_seconds=0.0, min_event_duration_seconds=None,
            max_event_duration_seconds=None, mean_event_duration_seconds=None, median_event_duration_seconds=None,
            most_frequent_note=None, most_frequent_note_count=0, least_frequent_note=None, least_frequent_note_count=0,
            longest_total_duration_note=None, pitch_range_low=None, pitch_range_high=None, max_polyphony=0,
            mean_polyphony=0.0, overlapping_events=0, short_outliers=0, long_outliers=0, outlier_bounds=None,
            note_table=note_tbl, pitch_class_table=pc_tbl, octave_table=oct_tbl,
            duration_histogram={"edges": [], "counts": []}, timeline=timeline_distribution(df, duration_seconds),
            polyphony={"times": [], "counts": []},
        )
    return stats


def repeated_notes(df: pd.DataFrame, min_repeats: int = 2) -> pd.DataFrame:
    """Notes that occur at least `min_repeats` times, with every occurrence's start time."""
    if df.empty:
        return pd.DataFrame(columns=["note_name", "midi_note", "count", "start_times"])
    g = df.groupby(["note_name", "midi_note"]).agg(count=("event_id", "count"),
                                                   start_times=("start_time_seconds", lambda s: sorted(s.tolist())))
    g = g.reset_index()
    g = g[g["count"] >= min_repeats].sort_values(["count", "midi_note"], ascending=[False, True])
    return g.reset_index(drop=True)


def consecutive_repeats(df: pd.DataFrame, max_gap_seconds: float = 0.5) -> pd.DataFrame:
    """Pairs of consecutive events with the same pitch separated by a short gap."""
    if len(df) < 2:
        return pd.DataFrame(columns=["note_name", "first_event_id", "second_event_id", "first_start", "second_start", "gap_seconds"])
    rows = []
    d = df.sort_values("start_time_seconds").reset_index(drop=True)
    by_pitch = d.groupby("midi_note")
    for _midi, grp in by_pitch:
        grp = grp.sort_values("start_time_seconds")
        prev = None
        for _, row in grp.iterrows():
            if prev is not None:
                gap = row["start_time_seconds"] - prev["end_time_seconds"]
                if -0.01 <= gap <= max_gap_seconds:
                    rows.append({"note_name": row["note_name"], "first_event_id": prev["event_id"],
                                 "second_event_id": row["event_id"], "first_start": prev["start_time_seconds"],
                                 "second_start": row["start_time_seconds"], "gap_seconds": float(gap)})
            prev = row
    out = pd.DataFrame(rows, columns=["note_name", "first_event_id", "second_event_id", "first_start", "second_start", "gap_seconds"])
    return out.sort_values("first_start").reset_index(drop=True)
