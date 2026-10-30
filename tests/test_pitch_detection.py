import math

import numpy as np
import pytest

from src.pitch_detection import (
    PITCH_CLASS_NAMES,
    cents_deviation,
    frequencies_to_midi_float_array,
    frequency_to_midi,
    frequency_to_midi_float,
    is_valid_frequency,
    midi_to_frequency,
    midi_to_note_name,
    midi_to_octave,
    midi_to_pitch_class,
    note_name_to_midi,
    track_pitch_pyin,
)
from tests.conftest import synth_notes


@pytest.mark.parametrize("freq, midi", [(440.0, 69), (261.6256, 60), (27.5, 21), (4186.01, 108), (8.1758, 0), (12543.85, 127),
                                        (442.0, 69), (255.0, 60), (16.0, 12)])
def test_frequency_to_midi(freq, midi):
    assert frequency_to_midi(freq) == midi


@pytest.mark.parametrize("bad", [0, 0.0, -1, -440.0, float("nan"), float("inf"), -float("inf"), None, "abc", ""])
def test_invalid_frequencies(bad):
    assert not is_valid_frequency(bad)
    assert frequency_to_midi(bad) is None
    assert frequency_to_midi_float(bad) is None
    assert cents_deviation(bad) is None


def test_out_of_midi_range_is_none():
    assert frequency_to_midi(5.0) is None          # below MIDI 0
    assert frequency_to_midi(20000.0) is None      # above MIDI 127


def test_midi_to_frequency_roundtrip():
    for midi in range(0, 128):
        assert frequency_to_midi(midi_to_frequency(midi)) == midi


@pytest.mark.parametrize("midi, name", [(60, "C4"), (61, "C#4"), (69, "A4"), (21, "A0"), (108, "C8"), (0, "C-1"), (59, "B3"), (127, "G9")])
def test_note_names(midi, name):
    assert midi_to_note_name(midi) == name
    assert note_name_to_midi(name) == midi


def test_pitch_class_and_octave():
    assert midi_to_pitch_class(60) == "C"
    assert midi_to_pitch_class(70) == "A#"
    assert midi_to_octave(60) == 4
    assert midi_to_octave(48) == 3
    assert midi_to_octave(72) == 5
    # C3, C4, C5 share a pitch class but differ in octave
    assert len({midi_to_pitch_class(m) for m in (48, 60, 72)}) == 1
    assert len({midi_to_octave(m) for m in (48, 60, 72)}) == 3
    assert len(PITCH_CLASS_NAMES) == 12


def test_flat_names():
    assert note_name_to_midi("Bb3") == 58
    assert note_name_to_midi("Db4") == 61
    with pytest.raises(ValueError):
        note_name_to_midi("H4")
    with pytest.raises(ValueError):
        note_name_to_midi("C")


def test_cents_deviation():
    assert abs(cents_deviation(440.0)) < 1e-9
    assert abs(cents_deviation(440.0 * 2 ** (25 / 1200)) - 25.0) < 1e-6
    assert abs(cents_deviation(440.0 * 2 ** (-30 / 1200)) + 30.0) < 1e-6
    # explicit reference note
    assert abs(cents_deviation(466.16, 69) - 100.0) < 0.1


def test_vectorised_conversion():
    arr = np.array([440.0, 0.0, np.nan, -5.0, 880.0, np.inf])
    out = frequencies_to_midi_float_array(arr)
    assert out[0] == pytest.approx(69.0)
    assert out[4] == pytest.approx(81.0)
    assert all(math.isnan(v) for v in out[[1, 2, 3, 5]])


def test_pyin_tracks_known_pitch():
    y = synth_notes([(69, 0.0, 1.0)])
    track = track_pitch_pyin(y, 22050)
    assert track.n_frames > 50
    mid = slice(track.n_frames // 4, 3 * track.n_frames // 4)
    f0 = track.f0_hz[mid]
    f0 = f0[np.isfinite(f0)]
    assert len(f0) > 10
    assert abs(np.median(f0) - 440.0) < 2.0
    assert np.mean(track.voiced_flag[mid]) > 0.9
