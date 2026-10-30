"""
Optional source separation (Demucs) for the "full music recording" mode.

Demucs and PyTorch are NOT core dependencies. When they are missing,
`is_available()` returns False with an explanation and the full-mix mode
transcribes the mixed signal directly (with a quality warning).

GPU use is never assumed: the device is CUDA only if `torch.cuda.is_available()`
returns True at runtime; otherwise the CPU is used (slow but correct).
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np

from config import settings
from src.progress import CancelToken, ProgressCallback, no_progress

log = settings.get_logger("separation")

DEFAULT_MODEL = "htdemucs"
STEM_NAMES = ("vocals", "drums", "bass", "other")


@dataclass
class SeparationInfo:
    available: bool
    reason: str
    model_name: str | None = None
    device: str | None = None
    version: str | None = None


def is_available() -> SeparationInfo:
    """Report whether Demucs can run here (without importing torch eagerly)."""
    if importlib.util.find_spec("torch") is None or importlib.util.find_spec("demucs") is None:
        return SeparationInfo(
            available=False,
            reason="Optional source separation is not installed. Install it with "
                   "`pip install -r requirements-optional.txt` (PyTorch + Demucs, large download).",
        )
    try:
        import torch  # noqa: WPS433
        import demucs  # noqa: F401

        device = "cuda" if torch.cuda.is_available() else "cpu"
        version = getattr(demucs, "__version__", "unknown")
        return SeparationInfo(True, "Demucs available", DEFAULT_MODEL, device, str(version))
    except Exception as exc:  # noqa: BLE001
        return SeparationInfo(False, f"Demucs is installed but failed to import: {exc}")


def separate(
    samples: np.ndarray,
    sample_rate: int,
    stems: tuple[str, ...] = ("vocals", "bass", "other"),
    model_name: str = DEFAULT_MODEL,
    progress: ProgressCallback = no_progress,
    cancel: CancelToken | None = None,
) -> dict[str, np.ndarray]:
    """Separate a mono signal into stems (each returned mono at `sample_rate`).

    Raises RuntimeError with a readable message if Demucs is unavailable.
    """
    info = is_available()
    if not info.available:
        raise RuntimeError(info.reason)

    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    progress(0.02, f"Loading separation model '{model_name}' ({info.device})")
    model = get_model(model_name)
    model.to(info.device)
    model.eval()
    if cancel is not None:
        cancel.check()

    # Demucs expects (batch, channels, samples) at the model's own sample rate.
    import librosa

    model_sr = int(model.samplerate)
    y = samples.astype(np.float32)
    if sample_rate != model_sr:
        y = librosa.resample(y, orig_sr=sample_rate, target_sr=model_sr)
    stereo = np.stack([y, y], axis=0)
    wav = torch.from_numpy(stereo)[None]
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)

    progress(0.10, "Separating sources (this can take a while on CPU)")
    with torch.no_grad():
        sources = apply_model(model, wav, device=info.device, shifts=0, split=True, overlap=0.25, progress=False)[0]
    sources = sources * ref.std() + ref.mean()
    if cancel is not None:
        cancel.check()

    out: dict[str, np.ndarray] = {}
    for name, tensor in zip(model.sources, sources):
        if name not in stems:
            continue
        mono = tensor.mean(0).cpu().numpy().astype(np.float32)
        if model_sr != sample_rate:
            mono = librosa.resample(mono, orig_sr=model_sr, target_sr=sample_rate)
        # Match the input length exactly so timestamps line up with the mix.
        if mono.shape[0] >= samples.shape[0]:
            mono = mono[: samples.shape[0]]
        else:
            mono = np.pad(mono, (0, samples.shape[0] - mono.shape[0]))
        out[name] = mono
    progress(1.0, f"Separated into {', '.join(out)}")
    log.info("Demucs separation produced stems: %s", ", ".join(out))
    return out


def describe(info: SeparationInfo | None = None) -> dict[str, Any]:
    info = info or is_available()
    return {"available": info.available, "reason": info.reason, "model": info.model_name,
            "device": info.device, "version": info.version}
