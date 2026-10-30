"""
Small shared helpers: time formatting, hashing, safe filenames, JSON support.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------------------------- #
# Time formatting
# --------------------------------------------------------------------------- #
def format_timestamp(seconds: float | None, decimals: int = 3) -> str:
    """Format seconds as MM:SS.mmm (hours are folded into minutes: 61:05.000).

    Non-finite or negative inputs return "--:--.---" so that a single bad value
    never breaks a table render.
    """
    if seconds is None or not np.isfinite(seconds) or seconds < 0:
        return "--:--." + "-" * decimals
    minutes = int(seconds // 60)
    secs = seconds - minutes * 60
    width = 3 + decimals if decimals > 0 else 2
    text = f"{minutes:02d}:{secs:0{width}.{decimals}f}"
    # Guard against rounding 59.9995 -> "60.000"
    if secs >= 59.9995 and decimals <= 3:
        return f"{minutes + 1:02d}:{0:0{width}.{decimals}f}"
    return text


def parse_timestamp(text: str) -> float:
    """Parse 'MM:SS.mmm', 'SS.mmm' or plain seconds into float seconds."""
    text = text.strip()
    if ":" in text:
        parts = text.split(":")
        total = 0.0
        for part in parts:
            total = total * 60 + float(part)
        return total
    return float(text)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
def sha256_of_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(obj: Any) -> str:
    """Deterministic hash of a JSON-serialisable object (used for config keys)."""
    payload = json.dumps(obj, sort_keys=True, default=_json_default).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str, max_length: int = 120) -> str:
    """Return a filesystem-safe ASCII version of an arbitrary (user-supplied) name.

    The extension is preserved even when the stem contains only non-ASCII
    characters (e.g. a Hindi or Chinese title becomes "file.wav").
    """
    base = Path(name.replace("\\", "/")).name  # strip any directory components (Windows or POSIX)
    stem, suffix = Path(base).stem, Path(base).suffix

    def _clean(text: str) -> str:
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        text = _SAFE_CHARS.sub("_", text)
        return re.sub(r"_{2,}", "_", text).strip("._")

    stem = _clean(stem) or "file"
    suffix = _clean(suffix.lstrip(".")).lower()
    suffix = f".{suffix}" if suffix and len(suffix) <= 8 else ""
    if len(stem) + len(suffix) > max_length:
        stem = stem[: max_length - len(suffix)]
    return stem + suffix


def make_export_filename(song_name: str, kind: str, extension: str) -> str:
    """Meaningful, timestamped export filename, e.g. song_note_events_20260101-120000.csv"""
    stem = Path(safe_filename(song_name)).stem or "song"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    extension = extension.lstrip(".")
    return f"{stem}_{kind}_{stamp}.{extension}"


# --------------------------------------------------------------------------- #
# JSON helpers (numpy-aware)
# --------------------------------------------------------------------------- #
def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


def _sanitize(obj: Any) -> Any:
    """Recursively replace NaN/inf floats with None (json.dumps would emit invalid 'NaN')."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.floating):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    return obj


def to_json(obj: Any, **kwargs: Any) -> str:
    return json.dumps(_sanitize(obj), default=_json_default, **kwargs)


def from_json(text: str | None) -> Any:
    if text is None or text == "":
        return None
    return json.loads(text)


def clean_float(value: Any, default: float | None = None) -> float | None:
    """Convert to float, returning `default` for None/NaN/inf/non-numeric."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default
