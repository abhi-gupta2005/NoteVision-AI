"""Song comparison, transcription evaluation metrics, visualisation smoke tests and utilities."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import visualization as viz
from src.evaluation import evaluate_transcription, load_ground_truth_csv
from src.note_segmentation import RawNote
from src.note_transcription import build_note_events, events_to_dataframe
from src.song_comparison import DISTRIBUTION_DISCLAIMER, SongSummary, compare_songs, sequence_similarity
from src.utils import format_timestamp, make_export_filename, parse_timestamp, safe_filename, stable_hash, to_json
from tests.conftest import MELODY


def summary(name, pc_counts, seq, song_id=None, events=10):
    return SongSummary(song_id=song_id or name, name=name, duration_seconds=10.0, total_events=events,
                       unique_midi_pitches=3, unique_pitch_classes=3, mean_duration_seconds=0.5, median_duration_seconds=0.4,
                       most_frequent_note="C4", tempo_bpm=120.0, mode="monophonic", pitch_class_counts=pc_counts,
                       octave_counts={4: events}, note_sequence=seq)


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def test_identical_distributions_are_maximally_similar():
    a = summary("a", {"C": 5, "E": 3, "G": 2}, [60, 64, 67, 60])
    b = summary("b", {"C": 10, "E": 6, "G": 4}, [60, 64, 67, 60], events=20)   # same proportions, twice as many events
    cmp = compare_songs([a, b])
    assert cmp.cosine_similarity.loc["a", "b"] == pytest.approx(1.0)
    assert cmp.jensen_shannon_distance.loc["a", "b"] == pytest.approx(0.0, abs=1e-9)
    assert cmp.sequence_similarity.loc["a", "b"] == pytest.approx(1.0)
    assert cmp.pitch_class_distribution.loc["a", "C"] == pytest.approx(0.5)
    assert DISTRIBUTION_DISCLAIMER in cmp.notes
    assert list(cmp.summary_table["song"]) == ["a", "b"]
    assert cmp.summary_table.iloc[1]["events_per_minute"] == pytest.approx(120.0)


def test_different_distributions_are_less_similar():
    a = summary("a", {"C": 10}, [60] * 5)
    b = summary("b", {"F#": 10}, [66] * 5)
    cmp = compare_songs([a, b])
    assert cmp.cosine_similarity.loc["a", "b"] == pytest.approx(0.0)
    assert cmp.jensen_shannon_distance.loc["a", "b"] == pytest.approx(1.0)
    assert cmp.sequence_similarity.loc["a", "b"] == 0.0


def test_duplicate_names_and_mode_warning():
    a = summary("song", {"C": 1}, [60])
    b = summary("song", {"D": 1}, [62], song_id="other")
    b.mode = "polyphonic"
    cmp = compare_songs([a, b])
    assert list(cmp.pitch_class_distribution.index) == ["song", "song (2)"]
    assert any("different modes" in n for n in cmp.notes)
    with pytest.raises(ValueError):
        compare_songs([])


def test_sequence_similarity_edge_cases():
    assert sequence_similarity([], [60]) == 0.0
    assert sequence_similarity([60, 72], [60, 72], use_pitch_class=True) == 1.0
    assert 0.0 < sequence_similarity([60, 62, 64], [60, 62, 65]) < 1.0


# --------------------------------------------------------------------------- #
# Evaluation metrics
# --------------------------------------------------------------------------- #
def test_evaluation_perfect_and_imperfect():
    truth = [(s, s + d, m) for m, s, d in MELODY]
    raw = [RawNote(start=s, end=e, midi=m, frequency_hz=None, confidence=0.9) for s, e, m in truth]
    events, _ = build_note_events(raw, "eval0001", "test")
    res = evaluate_transcription(events, truth)
    assert res.precision == 1.0 and res.recall == 1.0 and res.f1 == 1.0
    assert res.mean_onset_error_ms == pytest.approx(0.0) and res.n_matched == 6
    # drop one note and shift another by 20 ms -> recall drops, onset error > 0
    shifted = [RawNote(start=s + (0.02 if i == 0 else 0), end=e, midi=m, frequency_hz=None, confidence=0.9)
               for i, (s, e, m) in enumerate(truth[:-1])]
    events2, _ = build_note_events(shifted, "eval0002", "test")
    res2 = evaluate_transcription(events2, truth)
    assert res2.recall == pytest.approx(5 / 6)
    assert res2.precision == 1.0
    assert 0 < res2.mean_onset_error_ms < 10
    assert res2.to_dict()["onset_tolerance_ms"] == 50.0
    empty = evaluate_transcription([], truth)
    assert empty.f1 == 0.0 and empty.mean_onset_error_ms is None


def test_evaluation_on_real_monophonic_transcription(melody_wav: Path):
    from src.analysis_pipeline import run_analysis
    from src.note_transcription import TranscriptionConfig

    result = run_analysis(melody_wav, TranscriptionConfig(mode="monophonic"))
    truth = [(s, s + d, m) for m, s, d in MELODY]
    res = evaluate_transcription(result.events, truth)
    assert res.precision >= 0.8 and res.recall >= 0.8, res
    assert res.mean_onset_error_ms < 40
    assert res.mean_offset_error_ms < 80


def test_ground_truth_csv(tmp_path: Path):
    p = tmp_path / "gt.csv"
    pd.DataFrame({"start_time_seconds": [0.0, 1.0], "end_time_seconds": [0.5, 1.5], "note_name": ["C4", "Bb3"]}).to_csv(p, index=False)
    assert load_ground_truth_csv(p) == [(0.0, 0.5, 60), (1.0, 1.5, 58)]
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_ground_truth_csv(p)


# --------------------------------------------------------------------------- #
# Visualisation smoke tests (figures build from real event tables)
# --------------------------------------------------------------------------- #
def test_figures_build_from_events():
    from src.note_statistics import compute_statistics

    raw = [RawNote(start=s, end=s + d, midi=m, frequency_hz=None, confidence=0.8) for m, s, d in MELODY]
    events, _ = build_note_events(raw, "viz00001", "test")
    df = events_to_dataframe(events)
    stats = compute_statistics(events, 3.3)
    y = np.random.default_rng(0).standard_normal(22050 * 2).astype(np.float32) * 0.1
    figs = [
        viz.waveform_figure(y, 22050, onset_times=[0.5, 1.0]),
        viz.spectrogram_figure(y, 22050, fmax=4000),
        viz.spectrogram_figure(y, 22050, fmax=None, log_frequency=True),
        viz.piano_roll_figure(df, color_by="confidence", selected_event_id="viz00001-00002"),
        viz.piano_roll_figure(df, color_by="source_track"),
        viz.note_frequency_figure(stats.note_table),
        viz.pitch_class_figure(stats.pitch_class_table),
        viz.octave_figure(stats.octave_table),
        viz.duration_distribution_figure(df, 0.5, 0.5, (0.1, 1.0)),
        viz.duration_by_note_figure(stats.note_table),
        viz.timeline_figure(df, selected_event_id="viz00001-00001"),
        viz.polyphony_figure(stats.polyphony["times"], stats.polyphony["counts"]),
        viz.timeline_distribution_figure(stats.timeline["bin_starts"], stats.timeline["bin_width"], stats.timeline["counts"]),
        viz.pitch_track_figure(np.linspace(0, 1, 50), np.full(50, 440.0), np.full(50, 0.9)),
        viz.empty_figure("nothing"),
    ]
    for fig in figs:
        assert fig.to_dict()["layout"]
    assert viz.piano_roll_figure(pd.DataFrame()).layout.annotations
    for png in (viz.static_piano_roll_png(df, 3.3), viz.static_note_frequency_png(stats.note_table),
                viz.static_pitch_class_png(stats.pitch_class_table), viz.static_duration_histogram_png(df, 0.5, 0.5),
                viz.static_polyphony_png(stats.polyphony["times"], stats.polyphony["counts"])):
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_comparison_figures():
    dist = pd.DataFrame([[0.5, 0.5], [0.2, 0.8]], index=["a", "b"], columns=["C", "D"])
    assert viz.comparison_distribution_figure(dist, "t", "x").to_dict()["data"]
    sim = pd.DataFrame([[1, 0.5], [0.5, 1]], index=["a", "b"], columns=["a", "b"])
    assert viz.similarity_heatmap_figure(sim).to_dict()["data"]
    summ = pd.DataFrame({"song": ["a", "b"], "total_events": [3, 5]})
    assert viz.comparison_metric_figure(summ, "total_events", "t", "y").to_dict()["data"]


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seconds, text", [(0, "00:00.000"), (1.5, "00:01.500"), (65.123456, "01:05.123"), (3600, "60:00.000"),
                                           (59.9999, "01:00.000"), (float("nan"), "--:--.---"), (-1, "--:--.---"), (None, "--:--.---")])
def test_format_timestamp(seconds, text):
    assert format_timestamp(seconds) == text


def test_parse_timestamp():
    assert parse_timestamp("01:05.5") == pytest.approx(65.5)
    assert parse_timestamp("12.25") == 12.25
    assert parse_timestamp("1:00:01") == 3601.0


def test_safe_filename():
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("गाना.wav") == "file.wav"            # non-ASCII stem keeps its extension
    assert safe_filename("My Song (final mix).WAV") == "My_Song_final_mix.wav"   # extension lower-cased
    assert safe_filename("   ") == "file"
    assert safe_filename("x" * 300 + ".wav").endswith(".wav") and len(safe_filename("x" * 300 + ".wav")) <= 120
    name = make_export_filename("Take 1.wav", "note_events", "csv")
    assert name.startswith("Take_1_note_events_") and name.endswith(".csv")


def test_json_and_hash_helpers():
    text = to_json({"a": np.float32(1.5), "b": np.int64(3), "c": np.array([1, 2]), "d": float("nan"), "e": Path("x")})
    assert '"a": 1.5' in text and '"d": null' in text and '"c": [1, 2]' in text
    assert stable_hash({"x": 1, "y": [1, 2]}) == stable_hash({"y": [1, 2], "x": 1})
    assert stable_hash({"x": 1}) != stable_hash({"x": 2})
