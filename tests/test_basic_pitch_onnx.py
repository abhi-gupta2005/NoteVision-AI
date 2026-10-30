"""The bundled Basic Pitch ONNX runner: model file integrity, windowing maths and note decoding."""
import hashlib

import numpy as np
import pytest

from src import basic_pitch_onnx as bp


def test_model_file_present_and_unmodified():
    ok, detail = bp.is_available()
    assert ok, detail
    digest = hashlib.sha256(bp.MODEL_PATH.read_bytes()).hexdigest()
    assert digest == bp.MODEL_SHA256
    assert (bp.MODEL_PATH.parent / "LICENSE").exists()
    assert (bp.MODEL_PATH.parent / "NOTICE.md").exists()


def test_is_available_reports_missing_file(tmp_path):
    ok, detail = bp.is_available(tmp_path / "missing.onnx")
    assert not ok and "not found" in detail


def test_windowing_and_unwrapping_shapes():
    n = 5 * bp.AUDIO_SAMPLE_RATE  # 5 s
    overlap = bp.N_OVERLAPPING_FRAMES * bp.FFT_HOP
    hop = bp.AUDIO_N_SAMPLES - overlap
    padded = np.concatenate([np.zeros(overlap // 2, dtype=np.float32), np.zeros(n, dtype=np.float32)])
    windows = list(bp.window_audio(padded, hop))
    assert all(w.shape == (bp.AUDIO_N_SAMPLES, 1) for w in windows)
    assert len(windows) == int(np.ceil(padded.shape[0] / hop))
    fake = np.zeros((len(windows), bp.ANNOT_N_FRAMES, 88), dtype=np.float32)
    out = bp.unwrap_output(fake, n, bp.N_OVERLAPPING_FRAMES)
    assert out.shape == (int(np.floor(n * bp.ANNOTATIONS_FPS / bp.AUDIO_SAMPLE_RATE)), 88)
    with pytest.raises(ValueError):
        bp.unwrap_output(np.zeros((3, 4)), n, 2)


def test_frames_to_time_is_monotonic_and_near_hop_grid():
    t = bp.model_frames_to_time(600)
    assert np.all(np.diff(t) > 0)
    assert abs(t[1] - t[0] - bp.FRAME_PERIOD_SECONDS) < 1e-9
    assert abs(t[599] - 599 * bp.FRAME_PERIOD_SECONDS) < 0.05   # window offset compensation stays small


def test_note_decoding_on_synthetic_activations():
    n_frames, n_bins = 200, 88
    frames = np.zeros((n_frames, n_bins))
    onsets = np.zeros((n_frames, n_bins))
    # note A: bin 39 (MIDI 60) frames 10-60, note B: same bin frames 100-150, note C: bin 46 frames 10-80
    for b, s, e in ((39, 10, 60), (39, 100, 150), (46, 10, 80)):
        frames[s:e, b] = 0.9
        onsets[s, b] = 0.9
    notes = bp.output_to_notes_polyphonic(frames, onsets, onset_thresh=0.5, frame_thresh=0.3, min_note_len=5,
                                          infer_onsets=True, max_freq=None, min_freq=None, melodia_trick=True)
    found = sorted((m, s) for s, e, m, a in notes)
    assert found == [(60, 10), (60, 100), (67, 10)], notes
    by_start = {(m, s): (e, a) for s, e, m, a in notes}
    assert 55 <= by_start[(60, 10)][0] <= 62 and by_start[(60, 10)][1] == pytest.approx(0.9)
    assert 75 <= by_start[(67, 10)][0] <= 82
    # frequency constraint removes the high note
    notes_lim = bp.output_to_notes_polyphonic(frames, onsets, 0.5, 0.3, 5, True, max_freq=300.0, min_freq=None)
    assert all(m == 60 for _, _, m, _ in notes_lim)
    # without the melodia trick, an un-onset note (no onset peak) is not recovered
    frames2 = np.zeros((n_frames, n_bins)); frames2[20:70, 50] = 0.8
    assert bp.output_to_notes_polyphonic(frames2, np.zeros_like(frames2), 0.5, 0.3, 5, False, None, None, melodia_trick=False) == []
    recovered = bp.output_to_notes_polyphonic(frames2, np.zeros_like(frames2), 0.5, 0.3, 5, False, None, None, melodia_trick=True)
    assert [m for _, _, m, _ in recovered] == [71]


def test_silent_input_gives_no_notes_and_finite_outputs():
    model = bp.get_model()
    out, info = model.run(np.zeros(bp.AUDIO_SAMPLE_RATE, dtype=np.float32))
    assert set(out) == {"note", "onset", "contour"}
    assert out["note"].shape[1] == 88 and out["contour"].shape[1] == 264
    assert all(np.all(np.isfinite(v)) for v in out.values())
    assert info["n_windows"] == 1 and info["backend"].startswith("onnxruntime")
    notes = bp.output_to_notes_polyphonic(out["note"], out["onset"], 0.5, 0.3, 11, True, None, None)
    assert notes == []
