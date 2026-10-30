"""
Central configuration for NoteVision AI.

Everything that a deployment might want to tune lives here: paths, upload
limits, model defaults and logging. No machine-specific paths are hard-coded;
all paths are derived from the project root.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Application identity
# --------------------------------------------------------------------------- #
APP_NAME = "NoteVision AI"
APP_VERSION = "1.0.0"
APP_TAGLINE = "AI-powered musical note detection, timestamping, frequency analysis and reporting"
REPOSITORY_URL = "https://github.com/abhi-gupta2005/NoteVision-AI"

# --------------------------------------------------------------------------- #
# Paths (all relative to the project root)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
EXPORTS_DIR = PROJECT_ROOT / "exports"
MODELS_DIR = PROJECT_ROOT / "models"
LOG_DIR = PROJECT_ROOT / "logs"
DB_PATH = DATA_DIR / "notevision.sqlite3"

# Uploaded recordings are copied into UPLOADS_DIR/<song_id>/<original name>.
# They are kept persistently (so previous analyses can be re-opened and
# re-exported) until the song is deleted from the History page.
# Set NOTEVISION_PERSIST_UPLOADS=0 to keep uploads only for the running session.
PERSIST_UPLOADED_AUDIO = os.environ.get("NOTEVISION_PERSIST_UPLOADS", "1") != "0"

# --------------------------------------------------------------------------- #
# Upload validation
# --------------------------------------------------------------------------- #
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".wav": "Waveform Audio (PCM/float)",
    ".flac": "Free Lossless Audio Codec",
    ".mp3": "MPEG Layer-3 (decoded by libsndfile >= 1.1 or ffmpeg)",
    ".ogg": "Ogg Vorbis",
    ".aiff": "Audio Interchange File Format",
    ".aif": "Audio Interchange File Format",
    ".m4a": "MPEG-4 Audio (requires ffmpeg on the system PATH)",
}
MAX_FILE_SIZE_MB = float(os.environ.get("NOTEVISION_MAX_FILE_MB", 200))
MAX_DURATION_SECONDS = float(os.environ.get("NOTEVISION_MAX_DURATION_S", 15 * 60))
MIN_DURATION_SECONDS = 0.1

# --------------------------------------------------------------------------- #
# Analysis defaults
# --------------------------------------------------------------------------- #
# Both pYIN and Basic Pitch are run at 22.05 kHz mono. Basic Pitch *requires*
# this rate; pYIN works at any rate and 22.05 kHz keeps it fast.
ANALYSIS_SAMPLE_RATE = 22050

# Timestamps are stored as floating-point seconds rounded to this many decimals
# (4 decimals = 0.1 ms). Native model resolution is coarser: Basic Pitch
# ~11.6 ms per frame (hop 256 @ 22.05 kHz) and pYIN hop_length / sample_rate.
TIMESTAMP_DECIMALS = 4

# pYIN (monophonic mode)
PYIN_FMIN_NOTE = "C1"      # 32.7 Hz
PYIN_FMAX_NOTE = "C8"      # 4186 Hz
PYIN_FRAME_LENGTH = 2048
PYIN_HOP_LENGTH = 256

# Monophonic segmentation
MIN_NOTE_DURATION_SECONDS = 0.05
MIN_VOICED_PROBABILITY = 0.35
GAP_TOLERANCE_FRAMES = 3
GLITCH_RUN_FRAMES = 2
ONSET_SPLIT_RELATIVE_THRESHOLD = 0.25   # onset strength relative to envelope max
ONSET_SPLIT_MIN_SEPARATION_SECONDS = 0.06
ONSET_ENERGY_RISE_RATIO = 1.15          # RMS after / before an onset must rise by >= 15 % to split a held pitch

# Basic Pitch (polyphonic / full-mix modes) - model defaults
BASIC_PITCH_ONSET_THRESHOLD = 0.5
BASIC_PITCH_FRAME_THRESHOLD = 0.3
BASIC_PITCH_MIN_NOTE_LENGTH_MS = 127.7
BASIC_PITCH_MELODIA_TRICK = True
BASIC_PITCH_MIN_FREQUENCY_HZ: float | None = None
BASIC_PITCH_MAX_FREQUENCY_HZ: float | None = None
BASIC_PITCH_MODEL_NAME = "Spotify Basic Pitch (ICASSP 2022, nmp)"

# Statistics
TIMELINE_BINS = 40
POLYPHONY_GRID_SECONDS = 0.05

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_LEVEL = os.environ.get("NOTEVISION_LOG_LEVEL", "INFO").upper()
LOG_FILE = LOG_DIR / "notevision.log"
_LOGGING_CONFIGURED = False


def ensure_directories() -> None:
    """Create all runtime directories (idempotent)."""
    for d in (DATA_DIR, UPLOADS_DIR, PROCESSED_DIR, EXPORTS_DIR, MODELS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def configure_logging(level: str | None = None) -> logging.Logger:
    """Configure application-wide logging once (console + rotating file)."""
    global _LOGGING_CONFIGURED
    logger = logging.getLogger("notevision")
    if _LOGGING_CONFIGURED:
        return logger
    ensure_directories()
    logger.setLevel(getattr(logging, (level or LOG_LEVEL), logging.INFO))
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    try:
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:  # read-only file system etc. - console logging still works
        logger.warning("Could not open log file %s; logging to console only", LOG_FILE)

    # Third-party libraries are chatty at import time (Basic Pitch warns about
    # every backend that is *not* installed). Keep them at WARNING or above.
    for noisy in ("numba", "matplotlib", "PIL", "basic_pitch"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    logger.propagate = False
    _LOGGING_CONFIGURED = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the application logger."""
    configure_logging()
    return logging.getLogger(f"notevision.{name}")
