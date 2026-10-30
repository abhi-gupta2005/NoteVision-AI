import numpy as np
import pytest

from src.note_segmentation import SegmentationConfig, run_length_encode, segment_pitch_track
from src.onset_detection import OnsetResult, detect_onsets
from src.pitch_detection import PitchTrack, frequencies_to_midi_float_array, midi_to_frequency, track_pitch_pyin
from tests.conftest import MELODY, synth_notes

SR, HOP = 22050, 256
PERIOD = HOP / SR


def make_track(midis: list[float | None], prob: float = 0.9) -> PitchTrack:
    """Build a synthetic frame-level track from per-frame MIDI values (None = unvoiced)."""
    f0 = np.array([midi_to_frequency(m) if m is not None else np.nan for m in midis], dtype=float)
    voiced = np.array([m is not None for m in midis])
    probs = np.where(voiced, prob, 0.05)
    times = np.arange(len(midis)) * PERIOD
    return PitchTrack(times=times, f0_hz=f0, voiced_flag=voiced, voiced_prob=probs,
                      midi_float=frequencies_to_midi_float_array(f0), sample_rate=SR, hop_length=HOP, frame_length=2048)


def make_onsets(times: list[float], strengths: list[float] | None = None) -> OnsetResult:
    strengths = strengths or [1.0] * len(times)
    env_t = np.arange(0, (max(times) if times else 0) + 1.0, PERIOD)
    env = np.zeros_like(env_t)
    for t, s in zip(times, strengths):
        env[np.argmin(np.abs(env_t - t))] = s
    return OnsetResult(np.array(times), np.array(strengths), env, env_t, HOP, SR)


def test_rle():
    assert run_length_encode(np.array([1, 1, 2, 2, 2, -1, 1])) == [[1, 0, 2], [2, 2, 5], [-1, 5, 6], [1, 6, 7]]
    assert run_length_encode(np.array([])) == []


def test_pitch_change_creates_separate_notes():
    track = make_track([60] * 40 + [62] * 40 + [64] * 40)
    notes = segment_pitch_track(track, None, SegmentationConfig(min_note_duration_seconds=0.05))
    assert [n.midi for n in notes] == [60, 62, 64]
    assert notes[0].start == 0.0
    assert abs(notes[0].end - 40 * PERIOD) < PERIOD
    assert abs(notes[1].start - 40 * PERIOD) < PERIOD
    assert notes[0].frequency_hz == pytest.approx(midi_to_frequency(60), rel=1e-6)
    assert notes[0].frequency_source == "measured_f0_median"


def test_repeated_note_without_onset_evidence_stays_one_event():
    track = make_track([60] * 100)
    notes = segment_pitch_track(track, make_onsets([]), SegmentationConfig())
    assert len(notes) == 1


def test_repeated_note_split_only_on_strong_onset():
    track = make_track([60] * 100)                       # 100 frames = 1.16 s of steady C4
    weak = make_onsets([0.58], [0.1])                     # below threshold -> no split
    strong = make_onsets([0.58], [0.9])
    assert len(segment_pitch_track(track, weak, SegmentationConfig())) == 1
    notes = segment_pitch_track(track, strong, SegmentationConfig())
    assert len(notes) == 2
    assert all(n.midi == 60 for n in notes)
    assert abs(notes[1].start - 0.58) < 2 * PERIOD
    assert notes[0].end <= notes[1].start + 1e-9


def test_onset_without_energy_rise_does_not_split():
    """Spectral flux at a fade-out must not create a new note: the RMS has to rise after the onset."""
    track = make_track([60] * 100)
    onsets = make_onsets([0.58], [0.9])
    t = onsets.envelope_times
    onsets.rms_envelope = np.where(t < 0.58, 0.5, 0.1)          # energy drops after the "onset"
    assert len(segment_pitch_track(track, onsets, SegmentationConfig())) == 1
    onsets.rms_envelope = np.where(t < 0.58, 0.2, 0.5)          # energy rises: a real re-attack
    assert len(segment_pitch_track(track, onsets, SegmentationConfig())) == 2
    assert onsets.energy_rise_at(0.58) > 2.0
    assert onsets.energy_rise_at(t[-1] + 1.0) == 0.0             # nothing after the end of the signal


def test_short_trailing_piece_is_merged_not_dropped():
    track = make_track([60] * 100)                               # 1.16 s
    onsets = make_onsets([1.10], [0.9])                          # onset 60 ms before the end
    notes = segment_pitch_track(track, onsets, SegmentationConfig(min_note_duration_seconds=0.1))
    assert len(notes) == 1
    assert notes[0].end == pytest.approx(100 * PERIOD, abs=PERIOD)


def test_onset_split_disabled():
    track = make_track([60] * 100)
    notes = segment_pitch_track(track, make_onsets([0.58], [0.9]), SegmentationConfig(split_on_onsets=False))
    assert len(notes) == 1


def test_glitch_frames_are_absorbed():
    track = make_track([60] * 30 + [72] * 1 + [60] * 30)   # single octave-error frame
    notes = segment_pitch_track(track, None, SegmentationConfig(glitch_run_frames=2))
    assert [n.midi for n in notes] == [60]
    assert notes[0].n_frames == 61


def test_short_unvoiced_gap_is_bridged_but_long_gap_is_not():
    bridged = make_track([60] * 30 + [None] * 2 + [60] * 30)
    notes = segment_pitch_track(bridged, None, SegmentationConfig(gap_tolerance_frames=3))
    assert len(notes) == 1
    split = make_track([60] * 30 + [None] * 10 + [60] * 30)
    notes = segment_pitch_track(split, None, SegmentationConfig(gap_tolerance_frames=3))
    assert len(notes) == 2


def test_minimum_duration_filter():
    track = make_track([60] * 2 + [None] * 5 + [62] * 40)
    notes = segment_pitch_track(track, None, SegmentationConfig(min_note_duration_seconds=0.1, glitch_run_frames=0))
    assert [n.midi for n in notes] == [62]


def test_low_voicing_probability_is_unvoiced():
    track = make_track([60] * 40, prob=0.2)
    assert segment_pitch_track(track, None, SegmentationConfig(min_voiced_probability=0.35)) == []


def test_end_to_end_melody_with_repeated_note():
    """Real pYIN + onsets on synthetic audio: C4 D4 E4 C4 C4 G4 -> six events, repeated C4 kept separate."""
    y = synth_notes(MELODY)
    track = track_pitch_pyin(y, SR)
    onsets = detect_onsets(y, SR, hop_length=HOP)
    notes = segment_pitch_track(track, onsets, SegmentationConfig())
    midis = [n.midi for n in notes]
    assert midis == [m for m, _, _ in MELODY], midis
    for note, (midi, start, dur) in zip(notes, MELODY):
        assert abs(note.start - start) < 0.04, (note.start, start)
        assert abs(note.end - (start + dur)) < 0.06, (note.end, start + dur)
        assert abs(1200 * np.log2(note.frequency_hz / midi_to_frequency(midi))) < 15  # measured f0 within 15 cents
        assert 0.0 <= note.confidence <= 1.0
