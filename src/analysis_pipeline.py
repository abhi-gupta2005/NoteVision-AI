"""
End-to-end analysis orchestration:

    validate -> load -> preprocess -> transcribe -> tempo -> statistics -> persist

`run_analysis` is synchronous and UI-agnostic; it reports progress through a
callback and honours a `CancelToken`, so the Streamlit app can run it in a
background thread and the tests can call it directly.
"""
from __future__ import annotations

import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from config import settings
from src.audio_loader import AudioMetadata, AudioValidationError, load_audio, validate_file
from src.audio_preprocessing import PreprocessedAudio, PreprocessingConfig, preprocess, signal_level_stats
from src.note_statistics import SongStatistics, compute_statistics
from src.note_transcription import (
    NoteEvent,
    TranscriptionConfig,
    TranscriptionResult,
    events_to_dataframe,
    transcribe,
)
from src.onset_detection import estimate_tempo
from src.progress import AnalysisCancelled, CancelToken, ProgressCallback, no_progress, scaled
from src.utils import stable_hash, utc_now_iso

if TYPE_CHECKING:  # pragma: no cover
    from database.database import Database

log = settings.get_logger("pipeline")


# --------------------------------------------------------------------------- #
# Result container
# --------------------------------------------------------------------------- #
@dataclass
class AnalysisResult:
    song_id: str
    analysis_id: str
    metadata: AudioMetadata
    audio: PreprocessedAudio | None
    transcription: TranscriptionResult | None
    events: list[NoteEvent]
    excluded_events: list[NoteEvent]
    events_df: pd.DataFrame
    statistics: SongStatistics
    tempo_bpm: float | None
    transcription_config: TranscriptionConfig
    preprocessing_config: PreprocessingConfig
    config_hash: str
    status: str = "completed"
    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)
    software_versions: dict[str, str] = field(default_factory=dict)
    model_info: dict[str, Any] = field(default_factory=dict)
    level_stats: dict[str, float] = field(default_factory=dict)
    from_cache: bool = False
    error_message: str | None = None

    @property
    def name(self) -> str:
        return self.metadata.filename

    @property
    def n_events(self) -> int:
        return len(self.events)

    def processing_info(self) -> dict[str, Any]:
        """Flat key/value description of how the result was produced."""
        info: dict[str, Any] = {
            "application": f"{settings.APP_NAME} {settings.APP_VERSION}",
            "analysis_id": self.analysis_id,
            "song_id": self.song_id,
            "status": self.status,
            "created_at_utc": self.created_at,
            "loaded_from_cache": self.from_cache,
            "analysis_mode": self.transcription_config.mode,
            "detection_method": self.model_info.get("detection_method"),
            "model_name": self.model_info.get("model_name"),
            "model_version": self.model_info.get("model_version"),
            "inference_backend": self.model_info.get("backend"),
            "stems_used": ", ".join(self.model_info.get("stems_used") or []) or "mix",
            "analysis_sample_rate_hz": self.audio.sample_rate if self.audio else settings.ANALYSIS_SAMPLE_RATE,
            "time_offset_seconds": self.audio.time_offset_seconds if self.audio else 0.0,
            "timestamp_precision": f"{settings.TIMESTAMP_DECIMALS} decimals (seconds); model frame "
                                   f"{self.model_info.get('frame_period_seconds', 'n/a')} s",
            "preprocessing_steps": "; ".join(self.audio.summary()) if self.audio and self.audio.summary() else "none",
            "excluded_invalid_events": self.statistics.excluded_invalid_events,
            "gpu_used": False,
        }
        for key, value in self.transcription_config.to_dict().items():
            if key == "segmentation":
                for k2, v2 in value.items():
                    info[f"segmentation.{k2}"] = v2
            else:
                info[f"transcription.{key}"] = value
        for key, value in self.preprocessing_config.to_dict().items():
            info[f"preprocessing.{key}"] = value
        for key, value in self.timings.items():
            info[f"timing.{key}_seconds"] = round(value, 3)
        for key, value in self.software_versions.items():
            info[f"version.{key}"] = value
        return info


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def software_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform(), "app": settings.APP_VERSION}
    for name in ("numpy", "scipy", "librosa", "soundfile", "pandas", "onnxruntime", "streamlit", "plotly"):
        try:
            from importlib.metadata import version

            versions[name] = version(name)
        except Exception:  # noqa: BLE001
            versions[name] = "not installed"
    from src.basic_pitch_onnx import MODEL_VERSION

    versions["basic_pitch_model"] = MODEL_VERSION
    return versions


def make_config_hash(transcription_config: TranscriptionConfig, preprocessing_config: PreprocessingConfig) -> str:
    return stable_hash({"transcription": transcription_config.to_dict(), "preprocessing": preprocessing_config.to_dict(),
                        "schema": 1})


def make_analysis_id(song_id: str, config_hash: str) -> str:
    return f"{song_id}-{config_hash}"


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def run_analysis(
    file_path: str | Path,
    transcription_config: TranscriptionConfig | None = None,
    preprocessing_config: PreprocessingConfig | None = None,
    db: "Database | None" = None,
    progress: ProgressCallback = no_progress,
    cancel: CancelToken | None = None,
    force: bool = False,
    persist: bool = True,
) -> AnalysisResult:
    """Analyse one recording. Raises AudioValidationError / TranscriptionError / AnalysisCancelled."""
    transcription_config = transcription_config or TranscriptionConfig()
    preprocessing_config = preprocessing_config or PreprocessingConfig()
    timings: dict[str, float] = {}
    t_total = time.time()
    file_path = Path(file_path)

    # 1. validation + metadata
    progress(0.01, f"Validating {file_path.name}")
    t0 = time.time()
    validation = validate_file(file_path)
    if not validation.ok or validation.metadata is None:
        raise AudioValidationError(validation.message)
    metadata = validation.metadata
    timings["validation"] = time.time() - t0
    song_id = metadata.file_hash[:16]
    config_hash = make_config_hash(transcription_config, preprocessing_config)
    analysis_id = make_analysis_id(song_id, config_hash)
    warnings: list[str] = list(validation.warnings)
    if cancel is not None:
        cancel.check()

    # 2. cache lookup (same file content + same configuration)
    cached = None
    if db is not None and not force:
        try:
            cached = db.load_analysis(analysis_id)
        except Exception as exc:  # noqa: BLE001 - a cache problem must not block a fresh analysis
            log.warning("Could not read cached analysis %s: %s", analysis_id, exc)
    if cached is not None and cached.get("status") == "completed":
        progress(0.1, "Found a previous analysis with identical settings; loading it")
        audio = _load_and_preprocess(file_path, metadata, preprocessing_config, progress, timings)
        result = _result_from_record(cached, metadata, audio, transcription_config, preprocessing_config, config_hash)
        result.from_cache = True
        result.timings.update({"total": time.time() - t_total, **timings})
        progress(1.0, "Loaded from cache")
        return result

    if db is not None and persist:
        db.upsert_song(metadata, song_id)
        db.start_analysis(analysis_id, song_id, transcription_config, preprocessing_config, config_hash)

    try:
        # 3. load + preprocess
        audio = _load_and_preprocess(file_path, metadata, preprocessing_config, progress, timings)
        if cancel is not None:
            cancel.check()

        # 4. transcription
        t0 = time.time()
        transcription = transcribe(audio, transcription_config, song_id, scaled(progress, 0.25, 0.85), cancel)
        timings["transcription"] = time.time() - t0
        warnings.extend(transcription.warnings)

        # 5. tempo (estimate - optional)
        progress(0.87, "Estimating tempo")
        t0 = time.time()
        tempo = estimate_tempo(audio.samples, audio.sample_rate)
        timings["tempo"] = time.time() - t0

        # 6. statistics
        progress(0.9, "Computing statistics")
        t0 = time.time()
        events_df = events_to_dataframe(transcription.events)
        stats = compute_statistics(transcription.events, metadata.duration_seconds, transcription.excluded_events, events_df)
        timings["statistics"] = time.time() - t0
        if stats.excluded_invalid_events:
            warnings.append(f"{stats.excluded_invalid_events} event(s) failed validation or the confidence filter "
                            f"and were excluded from statistics (listed separately).")

        result = AnalysisResult(
            song_id=song_id, analysis_id=analysis_id, metadata=metadata, audio=audio, transcription=transcription,
            events=transcription.events, excluded_events=transcription.excluded_events, events_df=events_df,
            statistics=stats, tempo_bpm=tempo, transcription_config=transcription_config,
            preprocessing_config=preprocessing_config, config_hash=config_hash, warnings=warnings,
            timings=timings, software_versions=software_versions(), model_info=transcription.model_info(),
            level_stats=signal_level_stats(audio.samples),
        )
        result.timings["total"] = time.time() - t_total

        # 7. persist
        if db is not None and persist:
            progress(0.95, "Saving to database")
            t0 = time.time()
            db.save_result(result)
            result.timings["database"] = time.time() - t0
        progress(1.0, f"Done: {result.n_events} note events")
        log.info("Analysis %s completed with %d events in %.1f s", analysis_id, result.n_events, result.timings["total"])
        return result

    except AnalysisCancelled:
        if db is not None and persist:
            db.set_analysis_status(analysis_id, "cancelled", "Cancelled by user")
        log.info("Analysis %s cancelled", analysis_id)
        raise
    except Exception as exc:
        if db is not None and persist:
            db.set_analysis_status(analysis_id, "failed", str(exc))
        log.exception("Analysis %s failed", analysis_id)
        raise


def _load_and_preprocess(file_path: Path, metadata: AudioMetadata, preprocessing_config: PreprocessingConfig,
                         progress: ProgressCallback, timings: dict[str, float]) -> PreprocessedAudio:
    progress(0.05, "Decoding audio")
    t0 = time.time()
    loaded = load_audio(file_path, metadata)
    timings["load"] = time.time() - t0
    progress(0.15, "Preprocessing (mono / resample)")
    t0 = time.time()
    audio = preprocess(loaded, preprocessing_config)
    timings["preprocess"] = time.time() - t0
    return audio


def _result_from_record(record: dict[str, Any], metadata: AudioMetadata, audio: PreprocessedAudio | None,
                        transcription_config: TranscriptionConfig, preprocessing_config: PreprocessingConfig,
                        config_hash: str) -> AnalysisResult:
    """Rebuild an AnalysisResult from a database record (events + stats)."""
    events: list[NoteEvent] = record["events"]
    excluded: list[NoteEvent] = record.get("excluded_events", [])
    events_df = events_to_dataframe(events)
    stats = record.get("statistics")
    if stats is None:
        stats = compute_statistics(events, metadata.duration_seconds, excluded, events_df)
    return AnalysisResult(
        song_id=record["song_id"], analysis_id=record["analysis_id"], metadata=metadata, audio=audio, transcription=None,
        events=events, excluded_events=excluded, events_df=events_df, statistics=stats, tempo_bpm=record.get("tempo_bpm"),
        transcription_config=transcription_config, preprocessing_config=preprocessing_config, config_hash=config_hash,
        status=record.get("status", "completed"), warnings=record.get("warnings") or [], timings={},
        created_at=record.get("finished_at") or record.get("started_at") or utc_now_iso(),
        software_versions=record.get("software_versions") or {}, model_info=record.get("model_info") or {},
        level_stats=signal_level_stats(audio.samples) if audio is not None else {},
    )


def result_from_database(db: "Database", analysis_id: str, load_audio_file: bool = True) -> AnalysisResult | None:
    """Re-open a stored analysis (History page). Audio is re-read for plots when the file still exists."""
    record = db.load_analysis(analysis_id)
    if record is None:
        return None
    meta_dict = record["song"]
    metadata = AudioMetadata(**{k: meta_dict.get(k) for k in AudioMetadata.__dataclass_fields__ if k in meta_dict})
    tcfg = TranscriptionConfig.from_dict(record["transcription_config"])
    pcfg = PreprocessingConfig(**{k: v for k, v in record["preprocessing_config"].items()
                                  if k in PreprocessingConfig.__dataclass_fields__})
    audio = None
    path = Path(metadata.file_path) if metadata.file_path else None
    if load_audio_file and path is not None and path.exists():
        try:
            audio = preprocess(load_audio(path, metadata), pcfg)
        except Exception as exc:  # noqa: BLE001 - the stored events are still useful without the audio
            log.warning("Could not reload audio for %s: %s", analysis_id, exc)
    result = _result_from_record(record, metadata, audio, tcfg, pcfg, record["config_hash"])
    result.from_cache = True
    return result
