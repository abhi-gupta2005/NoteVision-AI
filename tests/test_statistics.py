import numpy as np
import pytest

from src.note_segmentation import RawNote
from src.note_statistics import (
    SongStatistics,
    compute_statistics,
    consecutive_repeats,
    duration_outliers,
    note_occurrence_table,
    octave_table,
    pitch_class_table,
    polyphony_timeline,
    repeated_notes,
    timeline_distribution,
)
from src.note_transcription import build_note_events, events_to_dataframe


def events_from(specs, song="stat0001", **kw):
    raw = [RawNote(start=s, end=s + d, midi=m, frequency_hz=None, confidence=c) for m, s, d, c in specs]
    return build_note_events(raw, song, "test", **kw)


SPECS = [  # (midi, start, duration, confidence)
    (60, 0.0, 0.5, 0.9), (60, 1.0, 0.25, 0.8), (60, 2.0, 1.0, 0.7),   # C4 x3: total 1.75 s
    (72, 0.5, 2.0, 0.9),                                               # C5 x1 but long: 2.0 s
    (64, 3.0, 0.1, 0.6), (67, 3.0, 0.1, 0.6),                          # E4 + G4 simultaneous
]


def test_counts_are_separate_from_durations():
    events, _ = events_from(SPECS)
    stats = compute_statistics(events, duration_seconds=4.0)
    assert stats.valid_events == 6 and stats.total_events == 6 and stats.excluded_invalid_events == 0
    assert stats.most_frequent_note == "C4" and stats.most_frequent_note_count == 3
    assert stats.longest_total_duration_note == "C5"          # one long event beats three short ones in duration
    row = stats.note_table.set_index("note_name").loc["C4"]
    assert row["count"] == 3
    assert row["total_duration_seconds"] == pytest.approx(1.75)
    assert row["mean_duration_seconds"] == pytest.approx(1.75 / 3)
    assert row["median_duration_seconds"] == pytest.approx(0.5)
    assert row["min_duration_seconds"] == pytest.approx(0.25) and row["max_duration_seconds"] == pytest.approx(1.0)
    assert row["percent_of_events"] == pytest.approx(50.0)
    assert stats.unique_midi_pitches == 4 and stats.unique_pitch_classes == 3   # C4 and C5 share pitch class C
    assert stats.least_frequent_note_count == 1
    assert stats.min_event_duration_seconds == pytest.approx(0.1)
    assert stats.max_event_duration_seconds == pytest.approx(2.0)
    assert stats.pitch_range_low == "C4" and stats.pitch_range_high == "C5"


def test_pitch_class_groups_octaves_together():
    events, _ = events_from(SPECS)
    df = events_to_dataframe(events)
    pc = pitch_class_table(df).set_index("pitch_class")
    assert len(pc) == 12
    assert pc.loc["C", "count"] == 4            # C4 x3 + C5 x1
    assert pc.loc["E", "count"] == 1 and pc.loc["F", "count"] == 0
    assert pc["percent_of_events"].sum() == pytest.approx(100.0)
    oc = octave_table(df).set_index("octave")
    assert oc.loc[4, "count"] == 5 and oc.loc[5, "count"] == 1
    assert "C4" in oc.loc[4, "notes"] and "G4" in oc.loc[4, "notes"]


def test_polyphony_timeline():
    events, _ = events_from(SPECS)
    df = events_to_dataframe(events)
    grid, active = polyphony_timeline(df, 4.0, grid_seconds=0.05)
    assert active.max() == 2
    at = lambda t: active[np.argmin(np.abs(grid - t))]  # noqa: E731
    assert at(0.25) == 1      # only C4
    assert at(0.75) == 1      # C5 only (C4 ended at 0.5)
    assert at(1.1) == 2       # C4 + C5
    assert at(3.05) == 2      # E4 + G4
    assert at(3.5) == 0
    stats = compute_statistics(events, 4.0)
    assert stats.max_polyphony == 2
    assert stats.overlapping_events == 5   # all except the first C4


def test_timeline_distribution_bins():
    events, _ = events_from(SPECS)
    df = events_to_dataframe(events)
    tl = timeline_distribution(df, 4.0, n_bins=4)
    assert tl["counts"] == [2, 1, 1, 2]
    assert sum(tl["counts"]) == 6
    assert tl["bin_width"] == pytest.approx(1.0)


def test_outliers_and_percentiles():
    specs = [(60, i * 1.0, 0.5, 0.9) for i in range(20)] + [(60, 25.0, 5.0, 0.9), (60, 31.0, 0.01, 0.9)]
    events, _ = events_from(specs)
    stats = compute_statistics(events, 40.0)
    assert stats.long_outliers == 1 and stats.short_outliers == 1
    assert stats.outlier_bounds[0] <= 0.5 <= stats.outlier_bounds[1]
    assert stats.duration_percentiles["p50"] == pytest.approx(0.5)
    n_short, n_long, bounds = duration_outliers(events_to_dataframe(events[:3]))
    assert (n_short, n_long, bounds) == (0, 0, None)


def test_invalid_events_excluded_and_counted():
    events, excluded = events_from(SPECS + [(60, 5.0, -1.0, 0.9)])
    assert len(excluded) == 1
    stats = compute_statistics(events, 6.0, excluded)
    assert stats.excluded_invalid_events == 1
    assert stats.valid_events == 6 and stats.total_events == 7


def test_empty_statistics():
    stats = compute_statistics([], 10.0)
    assert stats.valid_events == 0 and stats.most_frequent_note is None
    assert stats.note_table.empty and len(stats.pitch_class_table) == 12
    assert stats.polyphony == {"times": [], "counts": []}
    d = stats.to_dict()
    back = SongStatistics.from_dict(d)
    assert back.valid_events == 0


def test_statistics_serialisation_roundtrip():
    events, _ = events_from(SPECS)
    stats = compute_statistics(events, 4.0)
    back = SongStatistics.from_dict(stats.to_dict())
    assert back.most_frequent_note == "C4"
    assert back.note_table.shape == stats.note_table.shape
    assert back.polyphony["counts"] == stats.polyphony["counts"]
    assert back.outlier_bounds == stats.outlier_bounds


def test_repeated_notes_helpers():
    events, _ = events_from(SPECS)
    df = events_to_dataframe(events)
    rep = repeated_notes(df)
    assert list(rep["note_name"]) == ["C4"] and rep.iloc[0]["count"] == 3
    assert rep.iloc[0]["start_times"] == [0.0, 1.0, 2.0]
    cons = consecutive_repeats(df, max_gap_seconds=1.0)
    assert len(cons) == 2 and set(cons["note_name"]) == {"C4"}


def test_note_table_empty():
    assert note_occurrence_table(events_to_dataframe([])).empty
