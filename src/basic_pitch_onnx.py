"""
Self-contained runner for Spotify's Basic Pitch model (ICASSP 2022 "nmp") via onnxruntime.

Why this module exists
----------------------
The `basic-pitch` pip package pulls in TensorFlow on Python >= 3.11 unless it is
installed with `--no-deps`, which cannot be expressed in a requirements.txt (and
therefore not on Streamlit Community Cloud). The package itself ships an ONNX
export of the model that only needs `onnxruntime`. NoteVision AI bundles that
ONNX file (models/basic_pitch/nmp.onnx, 230 KB) and re-implements the few
pieces of the inference pipeline it uses:

* 2-second windowing with 30-frame overlap and output un-wrapping
* the note-decoding algorithm (onset peaks -> note extents, "melodia trick")
* frame -> time conversion

The model weights and the decoding algorithm are
Copyright 2022 Spotify AB, Apache License 2.0 (see models/basic_pitch/LICENSE
and models/basic_pitch/NOTICE.md). The code below is a faithful port of
basic_pitch 0.4.0 (`inference.py`, `note_creation.py`); outputs are identical.
"""
from __future__ import annotations

import importlib.util
import threading
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import scipy.signal

from config import settings
from src.progress import CancelToken, ProgressCallback, no_progress

log = settings.get_logger("basic_pitch_onnx")

# --------------------------------------------------------------------------- #
# Model constants (basic_pitch.constants)
# --------------------------------------------------------------------------- #
FFT_HOP = 256
AUDIO_SAMPLE_RATE = 22050
AUDIO_WINDOW_LENGTH = 2                                   # seconds per model window
ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP            # 86 frames / s
ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH    # 172 frames per window
AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP   # 43844 samples per window
N_OVERLAPPING_FRAMES = 30
ANNOTATIONS_BASE_FREQUENCY = 27.5                         # A0, lowest piano key
ANNOTATIONS_N_SEMITONES = 88
CONTOURS_BINS_PER_SEMITONE = 3
MIDI_OFFSET = 21                                          # note bin 0 == MIDI 21 (A0)
MAX_FREQ_IDX = 87
FRAME_PERIOD_SECONDS = FFT_HOP / AUDIO_SAMPLE_RATE        # ~11.6 ms

MODEL_NAME = "Spotify Basic Pitch (ICASSP 2022, nmp)"
MODEL_VERSION = "nmp.onnx from basic-pitch 0.4.0"
MODEL_PATH = settings.MODELS_DIR / "basic_pitch" / "nmp.onnx"
MODEL_SHA256 = "2c3c1d144bfa61ad236e92e169c13535c880469a12a047d4e73451f2c059a0ec"

INPUT_NAME = "serving_default_input_2:0"
OUTPUT_NAMES = {"note": "StatefulPartitionedCall:1", "onset": "StatefulPartitionedCall:2", "contour": "StatefulPartitionedCall:0"}

_LOCK = threading.Lock()
_MODEL: "BasicPitchOnnx | None" = None


# --------------------------------------------------------------------------- #
# Availability & loading
# --------------------------------------------------------------------------- #
def is_available(model_path: Path | None = None) -> tuple[bool, str]:
    """(available, human-readable detail) without loading the model."""
    model_path = model_path or MODEL_PATH
    if importlib.util.find_spec("onnxruntime") is None:
        return False, "onnxruntime is not installed (pip install onnxruntime)."
    if not model_path.is_file():
        return False, f"Model file not found: {model_path} (it ships with the repository under models/basic_pitch/)."
    try:
        from importlib.metadata import version

        ort_version = version("onnxruntime")
    except Exception:  # noqa: BLE001
        ort_version = "unknown"
    return True, f"{MODEL_NAME} via onnxruntime {ort_version} (CPU)"


def backend_version() -> str:
    try:
        from importlib.metadata import version

        return f"onnxruntime {version('onnxruntime')}"
    except Exception:  # noqa: BLE001
        return "onnxruntime"


class BasicPitchOnnx:
    """Thin wrapper around an onnxruntime session for the Basic Pitch model."""

    def __init__(self, model_path: Path | None = None) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path or MODEL_PATH)
        t0 = time.time()
        options = ort.SessionOptions()
        options.log_severity_level = 3  # errors only
        self.session = ort.InferenceSession(str(self.model_path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs = [i.name for i in self.session.get_inputs()]
        outputs = [o.name for o in self.session.get_outputs()]
        if INPUT_NAME not in inputs or any(name not in outputs for name in OUTPUT_NAMES.values()):
            raise RuntimeError(f"{self.model_path.name} does not look like the Basic Pitch model "
                               f"(inputs {inputs}, outputs {outputs}).")
        self.load_seconds = time.time() - t0
        self.backend = backend_version()

    def predict_window(self, window: np.ndarray) -> dict[str, np.ndarray]:
        """Run one batch of windows of shape (batch, AUDIO_N_SAMPLES, 1)."""
        x = np.ascontiguousarray(window, dtype=np.float32)
        raw = self.session.run([OUTPUT_NAMES["note"], OUTPUT_NAMES["onset"], OUTPUT_NAMES["contour"]], {INPUT_NAME: x})
        return {"note": raw[0], "onset": raw[1], "contour": raw[2]}

    def run(self, samples: np.ndarray, progress: ProgressCallback = no_progress, cancel: CancelToken | None = None,
            label: str = "mix") -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Full-length inference -> {"note", "onset", "contour"} matrices of shape (n_frames, bins)."""
        y = np.ascontiguousarray(samples, dtype=np.float32)
        if y.ndim != 1:
            raise ValueError("Basic Pitch expects a mono signal")
        original_length = int(y.shape[0])
        overlap_len = N_OVERLAPPING_FRAMES * FFT_HOP
        hop_size = AUDIO_N_SAMPLES - overlap_len
        padded = np.concatenate([np.zeros(overlap_len // 2, dtype=np.float32), y])
        n_windows = int(np.ceil(padded.shape[0] / hop_size))
        collected: dict[str, list[np.ndarray]] = {"note": [], "onset": [], "contour": []}
        t0 = time.time()
        for i, window in enumerate(window_audio(padded, hop_size)):
            if cancel is not None:
                cancel.check()
            out = self.predict_window(window[None, :, :])
            for key, value in out.items():
                collected[key].append(value)
            if i % 5 == 0 or i == n_windows - 1:
                progress((i + 1) / max(n_windows, 1), f"Basic Pitch inference ({label}): window {i + 1}/{n_windows}")
        unwrapped = {k: unwrap_output(np.concatenate(v), original_length, N_OVERLAPPING_FRAMES) for k, v in collected.items()}
        for key, arr in unwrapped.items():
            if not np.all(np.isfinite(arr)):
                raise RuntimeError(f"Basic Pitch produced non-finite values in its '{key}' output.")
        info = {"backend": self.backend, "n_windows": n_windows, "n_frames": int(unwrapped["contour"].shape[0]),
                "inference_seconds": round(time.time() - t0, 3), "frame_period_seconds": FRAME_PERIOD_SECONDS,
                "model_file": self.model_path.name}
        return unwrapped, info


def get_model() -> BasicPitchOnnx:
    """Load the model once per process and reuse it (thread-safe)."""
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            ok, detail = is_available()
            if not ok:
                raise RuntimeError(detail)
            _MODEL = BasicPitchOnnx()
            log.info("Loaded %s (%s) in %.2f s", MODEL_NAME, _MODEL.backend, _MODEL.load_seconds)
        return _MODEL


# --------------------------------------------------------------------------- #
# Windowing (basic_pitch.inference)
# --------------------------------------------------------------------------- #
def window_audio(audio: np.ndarray, hop_size: int) -> Iterable[np.ndarray]:
    """Yield fixed-length windows of shape (AUDIO_N_SAMPLES, 1), zero-padding the last one."""
    for i in range(0, audio.shape[0], hop_size):
        window = audio[i: i + AUDIO_N_SAMPLES]
        if len(window) < AUDIO_N_SAMPLES:
            window = np.pad(window, pad_width=[[0, AUDIO_N_SAMPLES - len(window)]])
        yield np.expand_dims(window, axis=-1)


def unwrap_output(output: np.ndarray, audio_original_length: int, n_overlapping_frames: int) -> np.ndarray:
    """Stitch batched window predictions (n_windows, n_frames, bins) into one (n_times, bins) matrix."""
    if output.ndim != 3:
        raise ValueError("expected a 3-D batched output")
    n_olap = int(0.5 * n_overlapping_frames)
    if n_olap > 0:
        output = output[:, n_olap:-n_olap, :]
    n_output_frames_original = int(np.floor(audio_original_length * (ANNOTATIONS_FPS / AUDIO_SAMPLE_RATE)))
    unwrapped = output.reshape(output.shape[0] * output.shape[1], output.shape[2])
    return unwrapped[:n_output_frames_original, :]


def model_frames_to_time(n_frames: int) -> np.ndarray:
    """Frame index -> seconds, compensating for the per-window offset of the stitched output."""
    original_times = np.arange(n_frames) * FFT_HOP / AUDIO_SAMPLE_RATE
    window_numbers = np.floor(np.arange(n_frames) / ANNOT_N_FRAMES)
    window_offset = (FFT_HOP / AUDIO_SAMPLE_RATE) * (ANNOT_N_FRAMES - (AUDIO_N_SAMPLES / FFT_HOP)) + 0.0018
    return original_times - (window_offset * window_numbers)


# --------------------------------------------------------------------------- #
# Note decoding (basic_pitch.note_creation)
# --------------------------------------------------------------------------- #
def _hz_to_midi(freq: float) -> float:
    return 69.0 + 12.0 * np.log2(freq / 440.0)


def get_inferred_onsets(onsets: np.ndarray, frames: np.ndarray, n_diff: int = 2) -> np.ndarray:
    """Add onsets inferred from sharp rises in frame activation to the predicted onsets."""
    diffs = []
    for n in range(1, n_diff + 1):
        frames_appended = np.concatenate([np.zeros((n, frames.shape[1])), frames])
        diffs.append(frames_appended[n:, :] - frames_appended[:-n, :])
    frame_diff = np.min(diffs, axis=0)
    frame_diff[frame_diff < 0] = 0
    frame_diff[:n_diff, :] = 0
    peak = np.max(frame_diff)
    if peak > 0:
        frame_diff = np.max(onsets) * frame_diff / peak      # rescale to the onsets' maximum
    return np.max([onsets, frame_diff], axis=0)


def constrain_frequency(onsets: np.ndarray, frames: np.ndarray, max_freq: float | None, min_freq: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Zero activations outside [min_freq, max_freq] (Hz)."""
    if max_freq is not None:
        max_freq_idx = int(np.round(_hz_to_midi(max_freq) - MIDI_OFFSET))
        onsets[:, max_freq_idx:] = 0
        frames[:, max_freq_idx:] = 0
    if min_freq is not None:
        min_freq_idx = int(np.round(_hz_to_midi(min_freq) - MIDI_OFFSET))
        onsets[:, :min_freq_idx] = 0
        frames[:, :min_freq_idx] = 0
    return onsets, frames


def output_to_notes_polyphonic(
    frames: np.ndarray,
    onsets: np.ndarray,
    onset_thresh: float,
    frame_thresh: float,
    min_note_len: int,
    infer_onsets: bool,
    max_freq: float | None,
    min_freq: float | None,
    melodia_trick: bool = True,
    energy_tol: int = 11,
) -> list[tuple[int, int, int, float]]:
    """Decode raw model output into note events (start_frame, end_frame, midi, amplitude).

    Direct port of basic_pitch.note_creation.output_to_notes_polyphonic:
    1. local maxima of the onset activations above `onset_thresh` start notes
       (walking backwards in time); each note extends while the frame
       activation stays above `frame_thresh` (with `energy_tol` frames of
       tolerance) and claims the energy of its bin and the two neighbours;
    2. with `melodia_trick`, remaining energy above `frame_thresh` that no
       onset explained is grown into notes forwards and backwards in time.
    Notes shorter than `min_note_len` frames are dropped.
    """
    frames = np.array(frames, dtype=np.float64, copy=True)
    onsets = np.array(onsets, dtype=np.float64, copy=True)
    n_frames = frames.shape[0]

    onsets, frames = constrain_frequency(onsets, frames, max_freq, min_freq)
    if infer_onsets:
        onsets = get_inferred_onsets(onsets, frames)

    peak_thresh_mat = np.zeros(onsets.shape)
    peaks = scipy.signal.argrelmax(onsets, axis=0)
    peak_thresh_mat[peaks] = onsets[peaks]

    onset_idx = np.where(peak_thresh_mat >= onset_thresh)
    onset_time_idx = onset_idx[0][::-1]  # backwards in time
    onset_freq_idx = onset_idx[1][::-1]

    remaining_energy = np.zeros(frames.shape)
    remaining_energy[:, :] = frames[:, :]

    note_events: list[tuple[int, int, int, float]] = []
    for note_start_idx, freq_idx in zip(onset_time_idx, onset_freq_idx):
        if note_start_idx >= n_frames - 1:
            continue
        i = note_start_idx + 1
        k = 0  # frames since the energy dropped below the threshold
        while i < n_frames - 1 and k < energy_tol:
            if remaining_energy[i, freq_idx] < frame_thresh:
                k += 1
            else:
                k = 0
            i += 1
        i -= k  # back to the last frame above threshold
        if i - note_start_idx <= min_note_len:
            continue
        remaining_energy[note_start_idx:i, freq_idx] = 0
        if freq_idx < MAX_FREQ_IDX:
            remaining_energy[note_start_idx:i, freq_idx + 1] = 0
        if freq_idx > 0:
            remaining_energy[note_start_idx:i, freq_idx - 1] = 0
        amplitude = float(np.mean(frames[note_start_idx:i, freq_idx]))
        note_events.append((int(note_start_idx), int(i), int(freq_idx + MIDI_OFFSET), amplitude))

    if melodia_trick:
        energy_shape = remaining_energy.shape
        while np.max(remaining_energy) > frame_thresh:
            i_mid, freq_idx = np.unravel_index(np.argmax(remaining_energy), energy_shape)
            remaining_energy[i_mid, freq_idx] = 0

            i = i_mid + 1  # forward pass
            k = 0
            while i < n_frames - 1 and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0
                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0
                i += 1
            i_end = i - 1 - k

            i = i_mid - 1  # backward pass
            k = 0
            while i > 0 and k < energy_tol:
                if remaining_energy[i, freq_idx] < frame_thresh:
                    k += 1
                else:
                    k = 0
                remaining_energy[i, freq_idx] = 0
                if freq_idx < MAX_FREQ_IDX:
                    remaining_energy[i, freq_idx + 1] = 0
                if freq_idx > 0:
                    remaining_energy[i, freq_idx - 1] = 0
                i -= 1
            i_start = i + 1 + k
            if i_start < 0 or i_end >= n_frames or i_end - i_start <= min_note_len:
                continue
            amplitude = float(np.mean(frames[i_start:i_end, freq_idx]))
            note_events.append((int(i_start), int(i_end), int(freq_idx + MIDI_OFFSET), amplitude))

    return note_events


def contour_bin_frequencies() -> np.ndarray:
    """Nominal centre frequency of each of the 264 contour bins (3 per semitone from A0)."""
    return ANNOTATIONS_BASE_FREQUENCY * 2.0 ** (np.arange(ANNOTATIONS_N_SEMITONES * CONTOURS_BINS_PER_SEMITONE)
                                                / (12 * CONTOURS_BINS_PER_SEMITONE))
