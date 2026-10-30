"""Note-event schema, validation, overlaps and real model transcription on synthetic audio."""

import numpy as np
import pytest

from src.audio_loader import load_audio
from src.audio_preprocessing import PreprocessingConfig, preprocess
from src.note_segmentation import RawNote
from src.note_transcription import (
    NoteEvent,
    TranscriptionConfig,
    TranscriptionError,
    TranscriptionMode,
    basic_pitch_available,
    build_note_events,
    compute_overlaps,
    dataframe_to_events,
    events_to_dataframe,
    refine_frequency_from_contour,
    run_basic_pitch,
    transcribe,
    validate_event,
)

from src.progress import AnalysisCancelled, CancelToken
from tests.conftest import ARPEGGIO_OVERLAP, MELODY

BP_AVAILABLE, BP_DETAIL = basic_pitch_available()
needs_basic_pitch = pytest.mark.skipif(not BP_AVAILABLE, reason=BP_DETAIL)


def raw(start, end, midi, freq=None, conf=0.9, **kw) -> RawNote:
    return RawNote(start=start, end=end, midi=midi, frequency_hz=freq, confidence=conf, **kw)


# --------------------------------------------------------------------------- #
# Schema / validation
# --------------------------------------------------------------------------- #
def test_build_note_events_fields_and_ids():
    events, excluded = build_note_events([raw(0.5, 1.0, 60, 262.0), raw(0.0, 0.25, 69)], "abcdef0123456789", "test")
    assert excluded == []
    assert [e.event_id for e in events] == ["abcdef01-00001", "abcdef01-00002"]   # sorted by start time
    first = events[0]
    assert (first.note_name, first.midi_note, first.pitch_class, first.octave) == ("A4", 69, "A", 4)
    assert first.frequency_hz == pytest.approx(440.0)
    assert first.frequency_source == "equal_tempered"          # no measured frequency supplied
    assert first.duration_seconds == pytest.approx(0.25)
    second = events[1]
    assert second.frequency_hz == 262.0 and second.cents_deviation == pytest.approx(2.48, abs=0.05)
    assert second.start_time_formatted == "00:00.500"
    assert second.detection_method == "test"


def test_time_offset_restores_original_timeline():
    events, _ = build_note_events([raw(0.0, 0.5, 60)], "s", "m", time_offset=1.25)
    assert events[0].start_time_seconds == 1.25 and events[0].end_time_seconds == 1.75


def test_invalid_events_are_excluded_not_dropped():
    bad = [raw(1.0, 0.5, 60), raw(float("nan"), 1.0, 60), raw(0.0, float("inf"), 60), raw(0.0, 1.0, 200), raw(-1.0, 0.5, 60)]
    events, excluded = build_note_events(bad + [raw(2.0, 2.5, 62)], "s", "m")
    assert len(events) == 1 and events[0].midi_note == 62
    assert len(excluded) == 5
    assert all(not e.is_valid and e.validation_error for e in excluded)
    assert "before start" in excluded[1].validation_error or "before start" in " ".join(e.validation_error for e in excluded)


def test_confidence_filter_moves_events_to_excluded():
    events, excluded = build_note_events([raw(0, 1, 60, conf=0.2), raw(1, 2, 62, conf=0.8)], "s", "m", min_confidence=0.5)
    assert [e.midi_note for e in events] == [62]
    assert len(excluded) == 1 and "confidence" in excluded[0].validation_error


def test_validate_event_direct():
    ev = NoteEvent("x", "s", "C4", 60, "C", 4, 261.6, 0.0, 1.0, 1.0)
    assert validate_event(ev) == []
    ev.duration_seconds = 0.5
    assert any("duration" in p for p in validate_event(ev))
    ev.frequency_hz = -3
    assert any("frequency" in p for p in validate_event(ev))


def test_zero_duration_event_is_valid():
    events, excluded = build_note_events([raw(1.0, 1.0, 60)], "s", "m")
    assert len(events) == 1 and excluded == []


def test_overlaps_match_brute_force():
    rng = np.random.default_rng(0)
    events = []
    for i in range(120):
        s = float(rng.uniform(0, 20))
        events.append(NoteEvent(f"e{i}", "s", "C4", 60, "C", 4, 261.6, s, s + float(rng.uniform(0.05, 2.0)), 0.0))
    for e in events:
        e.duration_seconds = e.end_time_seconds - e.start_time_seconds
    compute_overlaps(events)
    for a in events:
        brute = sum(1 for b in events if b is not a and b.start_time_seconds < a.end_time_seconds and a.start_time_seconds < b.end_time_seconds)
        assert a.overlap_count == brute
        assert a.overlaps_other_events == (brute > 0)


def test_touching_events_do_not_overlap():
    events, _ = build_note_events([raw(0.0, 1.0, 60), raw(1.0, 2.0, 60), raw(0.5, 1.5, 64)], "s", "m")
    by_id = {e.event_id: e for e in events}          # ids follow start order: C4@0, E4@0.5, C4@1.0
    assert by_id["s-00001"].overlap_count == 1       # first C4 overlaps only with the E4
    assert by_id["s-00002"].overlap_count == 2       # E4 overlaps both C4s
    assert by_id["s-00003"].overlap_count == 1       # second C4 touches the first (no overlap) and overlaps E4
    assert all(e.overlaps_other_events for e in events)


def test_dataframe_roundtrip():
    events, _ = build_note_events([raw(0.0, 1.0, 60, 262.0), raw(0.5, 1.5, 64)], "s", "m")
    df = events_to_dataframe(events)
    assert list(df["event_id"]) == ["s-00001", "s-00002"]
    assert "start_time_formatted" in df and df["start_time_formatted"].iloc[1] == "00:00.500"
    back = dataframe_to_events(df)
    assert [e.to_dict() for e in back] == [e.to_dict() for e in events]
    empty = events_to_dataframe([])
    assert empty.empty and "start_time_formatted" in empty


# --------------------------------------------------------------------------- #
# Monophonic mode (pYIN)
# --------------------------------------------------------------------------- #
def test_monophonic_transcription_matches_ground_truth(melody_wav):
    audio = preprocess(load_audio(melody_wav))
    result = transcribe(audio, TranscriptionConfig(mode="monophonic"), "song0001")
    assert result.detection_method == "pyin"
    midis = [e.midi_note for e in result.events]
    assert midis == [m for m, _, _ in MELODY]
    for ev, (midi, start, dur) in zip(result.events, MELODY):
        assert abs(ev.start_time_seconds - start) < 0.04
        assert abs(ev.duration_seconds - dur) < 0.08
        assert ev.frequency_source == "measured_f0_median"
    assert all(not e.overlaps_other_events for e in result.events)  # monophonic: no overlaps
    assert result.pitch_track is not None and result.onsets is not None
    assert any("one pitch at a time" in w for w in result.warnings)


def test_monophonic_silent_audio_gives_no_events(silent_wav):
    audio = preprocess(load_audio(silent_wav))
    result = transcribe(audio, TranscriptionConfig(mode="monophonic"), "silent01")
    assert result.events == []
    assert any("No note events" in w for w in result.warnings)


def test_trim_offset_is_added_back(leading_silence_wav):
    audio = preprocess(load_audio(leading_silence_wav), PreprocessingConfig(trim_silence=True))
    result = transcribe(audio, TranscriptionConfig(mode="monophonic"), "offset01")
    assert result.events, "expected events"
    assert abs(result.events[0].start_time_seconds - 1.0) < 0.05   # first note really starts at 1.0 s in the file


# --------------------------------------------------------------------------- #
# Polyphonic mode (Basic Pitch)
# --------------------------------------------------------------------------- #
@needs_basic_pitch
def test_polyphonic_chords_detects_simultaneous_notes(chords_wav):
    audio = preprocess(load_audio(chords_wav))
    result = transcribe(audio, TranscriptionConfig(mode="polyphonic"), "chord001")
    assert result.detection_method == "basic_pitch"
    strong = [e for e in result.events if e.confidence >= 0.5]
    first = {e.midi_note for e in strong if e.start_time_seconds < 0.5}
    second = {e.midi_note for e in strong if 1.4 < e.start_time_seconds < 2.0}
    assert {60, 64, 67} <= first, first
    assert {55, 59, 62} <= second, second
    chord_events = [e for e in strong if e.midi_note in (60, 64, 67) and e.start_time_seconds < 0.5]
    assert all(e.overlaps_other_events and e.overlap_count >= 2 for e in chord_events)
    for e in chord_events:
        assert abs(e.start_time_seconds - 0.0) < 0.05
        assert abs(e.end_time_seconds - 1.5) < 0.15
        assert e.frequency_source == "basic_pitch_contour"
        assert abs(e.cents_deviation) < 15


@needs_basic_pitch
def test_polyphonic_partial_overlaps_preserved(arpeggio_wav):
    audio = preprocess(load_audio(arpeggio_wav))
    result = transcribe(audio, TranscriptionConfig(mode="polyphonic"), "arp00001")
    strong: dict[int, NoteEvent] = {}
    for e in sorted(result.events, key=lambda e: e.start_time_seconds):   # earliest confident event per pitch
        if e.confidence >= 0.5:
            strong.setdefault(e.midi_note, e)
    assert {60, 64, 67, 72} <= set(strong), set(strong)
    for midi, start, dur in ARPEGGIO_OVERLAP:
        assert abs(strong[midi].start_time_seconds - start) < 0.05, (midi, strong[midi].start_time_seconds)
    assert strong[64].overlap_count >= 2   # E4 overlaps C4 and G4


@needs_basic_pitch
def test_polyphonic_repeated_note_is_split(melody_wav):
    audio = preprocess(load_audio(melody_wav))
    result = transcribe(audio, TranscriptionConfig(mode="polyphonic"), "mel00001")
    c4 = sorted([e for e in result.events if e.midi_note == 60 and e.confidence >= 0.5], key=lambda e: e.start_time_seconds)
    starts = [round(e.start_time_seconds, 2) for e in c4]
    assert len(c4) == 3, starts
    assert abs(c4[1].start_time_seconds - 1.5) < 0.05 and abs(c4[2].start_time_seconds - 2.0) < 0.05


@needs_basic_pitch
@pytest.mark.parametrize("cents", [0, 33, -25])
def test_contour_frequency_refinement(cents):
    f = 440.0 * 2 ** (cents / 1200)
    t = np.arange(22050) / 22050
    y = (0.3 * np.sin(2 * np.pi * f * t)).astype(np.float32)
    notes, info = run_basic_pitch(y, 22050, TranscriptionConfig())
    a4 = [n for n in notes if n.midi == 69]
    assert a4, notes
    err = 1200 * np.log2(a4[0].frequency_hz / f)
    assert abs(err) < 10, err
    assert info["backend"] == "onnx" or info["backend"]


def test_refine_frequency_fallbacks():
    assert refine_frequency_from_contour(np.zeros((0, 264)), 69) == (pytest.approx(440.0), "equal_tempered")
    assert refine_frequency_from_contour(np.zeros((10, 264)), 69)[1] == "equal_tempered"
    block = np.zeros((10, 264))
    block[:, 3 * (69 - 21) + 1] = 1.0
    freq, src = refine_frequency_from_contour(block, 69)
    assert src == "basic_pitch_contour" and freq == pytest.approx(440.0, rel=1e-6)


@needs_basic_pitch
def test_cancellation(chords_wav):
    audio = preprocess(load_audio(chords_wav))
    token = CancelToken()
    token.cancel()
    with pytest.raises(AnalysisCancelled):
        transcribe(audio, TranscriptionConfig(mode="polyphonic"), "cancel01", cancel=token)


@needs_basic_pitch
def test_progress_callback_reaches_completion(chords_wav):
    audio = preprocess(load_audio(chords_wav))
    seen = []
    transcribe(audio, TranscriptionConfig(mode="polyphonic"), "prog0001", progress=lambda p, m: seen.append(p))
    assert seen and seen[-1] == 1.0 and all(0 <= p <= 1 for p in seen)


def test_wrong_sample_rate_rejected():
    if not BP_AVAILABLE:
        pytest.skip(BP_DETAIL)
    with pytest.raises(TranscriptionError):
        run_basic_pitch(np.zeros(44100, dtype=np.float32), 44100, TranscriptionConfig())


def test_unavailable_model_suggests_fallback(monkeypatch, chords_wav):
    import src.note_transcription as nt

    monkeypatch.setattr(nt, "basic_pitch_available", lambda: (False, "Basic Pitch is not installed."))
    audio = preprocess(load_audio(chords_wav))
    with pytest.raises(TranscriptionError) as exc:
        transcribe(audio, TranscriptionConfig(mode="polyphonic"), "nobp0001")
    assert exc.value.suggested_mode is TranscriptionMode.MONOPHONIC


@needs_basic_pitch
def test_full_mix_without_separation_falls_back_to_mix(chords_wav):
    audio = preprocess(load_audio(chords_wav))
    result = transcribe(audio, TranscriptionConfig(mode="full_mix", use_source_separation=True), "mix00001")
    assert result.stems_used == ["mix"]
    assert result.events
    assert any("separation" in w.lower() for w in result.warnings)


def test_config_roundtrip():
    cfg = TranscriptionConfig(mode="monophonic", onset_threshold=0.4, stems=("vocals",))
    back = TranscriptionConfig.from_dict(cfg.to_dict())
    assert back == cfg
