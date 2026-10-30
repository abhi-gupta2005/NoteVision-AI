"""
SQLite persistence for songs, analyses, note events and statistics.

Identity & de-duplication
-------------------------
* song_id      = first 16 hex chars of the SHA-256 of the file content, so the
                 same recording uploaded twice is one song.
* analysis_id  = song_id + "-" + hash(configuration). Saving the same result
                 again replaces its events inside one transaction, so repeated
                 saves never create duplicate events.

Original audio files are stored under data/uploads/<song_id>/ while the song
exists (see settings.PERSIST_UPLOADED_AUDIO); `delete_song` removes them too.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import pandas as pd

from config import settings
from src.audio_loader import AudioMetadata, delete_uploaded_song_files
from src.audio_preprocessing import PreprocessingConfig
from src.note_statistics import SongStatistics
from src.note_transcription import EVENT_COLUMNS, NoteEvent, TranscriptionConfig
from src.utils import from_json, to_json, utc_now_iso

if TYPE_CHECKING:  # pragma: no cover
    from src.analysis_pipeline import AnalysisResult

log = settings.get_logger("database")

SCHEMA_VERSION = 1
_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS songs (
    song_id            TEXT PRIMARY KEY,
    filename           TEXT NOT NULL,
    file_path          TEXT,
    file_hash          TEXT NOT NULL,
    file_size_bytes    INTEGER,
    extension          TEXT,
    container_format   TEXT,
    subtype            TEXT,
    sample_rate        INTEGER,
    channels           INTEGER,
    duration_seconds   REAL,
    frames             INTEGER,
    probed_with        TEXT,
    metadata_json      TEXT,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS analyses (
    analysis_id               TEXT PRIMARY KEY,
    song_id                   TEXT NOT NULL REFERENCES songs(song_id) ON DELETE CASCADE,
    mode                      TEXT NOT NULL,
    config_hash               TEXT NOT NULL,
    transcription_config_json TEXT,
    preprocessing_config_json TEXT,
    status                    TEXT NOT NULL,
    error_message             TEXT,
    model_name                TEXT,
    model_version             TEXT,
    model_info_json           TEXT,
    software_versions_json    TEXT,
    tempo_bpm                 REAL,
    statistics_json           TEXT,
    warnings_json             TEXT,
    timings_json              TEXT,
    n_events                  INTEGER NOT NULL DEFAULT 0,
    n_excluded                INTEGER NOT NULL DEFAULT 0,
    started_at                TEXT NOT NULL,
    finished_at               TEXT,
    UNIQUE (song_id, config_hash)
);

CREATE TABLE IF NOT EXISTS note_events (
    analysis_id           TEXT NOT NULL REFERENCES analyses(analysis_id) ON DELETE CASCADE,
    event_id              TEXT NOT NULL,
    song_id               TEXT NOT NULL,
    note_name             TEXT,
    midi_note             INTEGER,
    pitch_class           TEXT,
    octave                INTEGER,
    frequency_hz          REAL,
    start_time_seconds    REAL,
    end_time_seconds      REAL,
    duration_seconds      REAL,
    confidence            REAL,
    instrument_label      TEXT,
    source_track          TEXT,
    onset_strength        REAL,
    detection_method      TEXT,
    cents_deviation       REAL,
    frequency_source      TEXT,
    overlaps_other_events INTEGER,
    overlap_count         INTEGER,
    is_valid              INTEGER NOT NULL DEFAULT 1,
    validation_error      TEXT,
    PRIMARY KEY (analysis_id, event_id)
);
CREATE INDEX IF NOT EXISTS idx_events_analysis ON note_events(analysis_id);
CREATE INDEX IF NOT EXISTS idx_events_song ON note_events(song_id);
CREATE INDEX IF NOT EXISTS idx_analyses_song ON analyses(song_id);
"""


class Database:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else settings.DB_PATH
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------ #
    # connection handling
    # ------------------------------------------------------------------ #
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            try:
                conn.execute("PRAGMA journal_mode = WAL")
            except sqlite3.DatabaseError:
                pass
            conn.executescript(_SCHEMA)
            conn.execute("INSERT OR REPLACE INTO schema_info(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

    # ------------------------------------------------------------------ #
    # songs
    # ------------------------------------------------------------------ #
    def upsert_song(self, metadata: AudioMetadata, song_id: str) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO songs (song_id, filename, file_path, file_hash, file_size_bytes, extension, container_format,
                                   subtype, sample_rate, channels, duration_seconds, frames, probed_with, metadata_json,
                                   created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(song_id) DO UPDATE SET
                    filename = excluded.filename, file_path = excluded.file_path, file_size_bytes = excluded.file_size_bytes,
                    sample_rate = excluded.sample_rate, channels = excluded.channels, duration_seconds = excluded.duration_seconds,
                    frames = excluded.frames, probed_with = excluded.probed_with, metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (song_id, metadata.filename, metadata.file_path, metadata.file_hash, metadata.file_size_bytes,
                 metadata.extension, metadata.container_format, metadata.subtype, metadata.sample_rate, metadata.channels,
                 metadata.duration_seconds, metadata.frames, metadata.probed_with, to_json(metadata.to_dict()), now, now),
            )

    def get_song(self, song_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM songs WHERE song_id = ?", (song_id,)).fetchone()
        return dict(row) if row else None

    def list_songs(self) -> pd.DataFrame:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.song_id, s.filename, s.duration_seconds, s.sample_rate, s.channels, s.file_size_bytes,
                       s.file_path, s.created_at,
                       COUNT(a.analysis_id) AS n_analyses,
                       SUM(CASE WHEN a.status = 'completed' THEN 1 ELSE 0 END) AS n_completed,
                       MAX(a.finished_at) AS last_analysis_at
                FROM songs s LEFT JOIN analyses a ON a.song_id = s.song_id
                GROUP BY s.song_id ORDER BY s.created_at DESC
                """
            ).fetchall()
        return pd.DataFrame([dict(r) for r in rows], columns=[
            "song_id", "filename", "duration_seconds", "sample_rate", "channels", "file_size_bytes", "file_path",
            "created_at", "n_analyses", "n_completed", "last_analysis_at"])

    def delete_song(self, song_id: str, delete_files: bool = True) -> dict[str, int]:
        """Delete a song, all its analyses and events (cascade) and optionally its uploaded audio."""
        with self._connect() as conn:
            n_events = conn.execute("SELECT COUNT(*) FROM note_events WHERE song_id = ?", (song_id,)).fetchone()[0]
            n_analyses = conn.execute("SELECT COUNT(*) FROM analyses WHERE song_id = ?", (song_id,)).fetchone()[0]
            conn.execute("DELETE FROM note_events WHERE song_id = ?", (song_id,))
            conn.execute("DELETE FROM analyses WHERE song_id = ?", (song_id,))
            deleted = conn.execute("DELETE FROM songs WHERE song_id = ?", (song_id,)).rowcount
        n_files = delete_uploaded_song_files(song_id) if delete_files else 0
        log.info("Deleted song %s (%d analyses, %d events, %d files)", song_id, n_analyses, n_events, n_files)
        return {"songs": int(deleted), "analyses": int(n_analyses), "events": int(n_events), "files": int(n_files)}

    # ------------------------------------------------------------------ #
    # analyses
    # ------------------------------------------------------------------ #
    def start_analysis(self, analysis_id: str, song_id: str, transcription_config: TranscriptionConfig,
                       preprocessing_config: PreprocessingConfig, config_hash: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO analyses (analysis_id, song_id, mode, config_hash, transcription_config_json,
                                      preprocessing_config_json, status, started_at)
                VALUES (?, ?, ?, ?, ?, ?, 'running', ?)
                ON CONFLICT(analysis_id) DO UPDATE SET status = 'running', error_message = NULL,
                    started_at = excluded.started_at, finished_at = NULL
                """,
                (analysis_id, song_id, transcription_config.mode, config_hash, to_json(transcription_config.to_dict()),
                 to_json(preprocessing_config.to_dict()), utc_now_iso()),
            )

    def set_analysis_status(self, analysis_id: str, status: str, error_message: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE analyses SET status = ?, error_message = ?, finished_at = ? WHERE analysis_id = ?",
                         (status, error_message, utc_now_iso(), analysis_id))

    def save_result(self, result: "AnalysisResult") -> None:
        """Persist a completed analysis (song row, analysis row, events) atomically."""
        self.upsert_song(result.metadata, result.song_id)
        all_events = list(result.events) + list(result.excluded_events)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO analyses (analysis_id, song_id, mode, config_hash, transcription_config_json,
                                      preprocessing_config_json, status, error_message, model_name, model_version,
                                      model_info_json, software_versions_json, tempo_bpm, statistics_json, warnings_json,
                                      timings_json, n_events, n_excluded, started_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(analysis_id) DO UPDATE SET
                    status = excluded.status, error_message = NULL, model_name = excluded.model_name,
                    model_version = excluded.model_version, model_info_json = excluded.model_info_json,
                    software_versions_json = excluded.software_versions_json, tempo_bpm = excluded.tempo_bpm,
                    statistics_json = excluded.statistics_json, warnings_json = excluded.warnings_json,
                    timings_json = excluded.timings_json, n_events = excluded.n_events, n_excluded = excluded.n_excluded,
                    finished_at = excluded.finished_at
                """,
                (result.analysis_id, result.song_id, result.transcription_config.mode, result.config_hash,
                 to_json(result.transcription_config.to_dict()), to_json(result.preprocessing_config.to_dict()),
                 result.status, result.model_info.get("model_name"), result.model_info.get("model_version"),
                 to_json(result.model_info), to_json(result.software_versions), result.tempo_bpm,
                 to_json(result.statistics.to_dict()), to_json(result.warnings), to_json(result.timings),
                 len(result.events), len(result.excluded_events), result.created_at, utc_now_iso()),
            )
            # Replace events in the same transaction -> no duplicates on repeated saves.
            conn.execute("DELETE FROM note_events WHERE analysis_id = ?", (result.analysis_id,))
            conn.executemany(
                f"INSERT INTO note_events (analysis_id, {', '.join(EVENT_COLUMNS)}) VALUES ({', '.join('?' * (len(EVENT_COLUMNS) + 1))})",
                [(result.analysis_id, *[_sql_value(getattr(e, c)) for c in EVENT_COLUMNS]) for e in all_events],
            )
        log.info("Saved analysis %s (%d events)", result.analysis_id, len(all_events))

    def list_analyses(self, song_id: str | None = None) -> pd.DataFrame:
        query = """
            SELECT a.analysis_id, a.song_id, s.filename, a.mode, a.status, a.n_events, a.n_excluded, a.tempo_bpm,
                   a.model_name, a.model_version, a.started_at, a.finished_at, a.error_message, a.config_hash
            FROM analyses a JOIN songs s ON s.song_id = a.song_id
        """
        params: tuple = ()
        if song_id:
            query += " WHERE a.song_id = ?"
            params = (song_id,)
        query += " ORDER BY a.started_at DESC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return pd.DataFrame([dict(r) for r in rows], columns=[
            "analysis_id", "song_id", "filename", "mode", "status", "n_events", "n_excluded", "tempo_bpm", "model_name",
            "model_version", "started_at", "finished_at", "error_message", "config_hash"])

    def load_events_df(self, analysis_id: str, include_invalid: bool = False) -> pd.DataFrame:
        query = f"SELECT {', '.join(EVENT_COLUMNS)} FROM note_events WHERE analysis_id = ?"
        if not include_invalid:
            query += " AND is_valid = 1"
        query += " ORDER BY start_time_seconds, midi_note"
        with self._connect() as conn:
            rows = conn.execute(query, (analysis_id,)).fetchall()
        df = pd.DataFrame([dict(r) for r in rows], columns=EVENT_COLUMNS)
        if not df.empty:
            df["overlaps_other_events"] = df["overlaps_other_events"].astype(bool)
            df["is_valid"] = df["is_valid"].astype(bool)
        return df

    def load_analysis(self, analysis_id: str) -> dict[str, Any] | None:
        """Full record: analysis row + song + events (valid / excluded) + statistics."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM analyses WHERE analysis_id = ?", (analysis_id,)).fetchone()
            if row is None:
                return None
            song = conn.execute("SELECT * FROM songs WHERE song_id = ?", (row["song_id"],)).fetchone()
            event_rows = conn.execute(
                f"SELECT {', '.join(EVENT_COLUMNS)} FROM note_events WHERE analysis_id = ? ORDER BY start_time_seconds, midi_note",
                (analysis_id,),
            ).fetchall()
        events, excluded = [], []
        for r in event_rows:
            d = dict(r)
            d["overlaps_other_events"] = bool(d.get("overlaps_other_events"))
            d["is_valid"] = bool(d.get("is_valid", 1))
            ev = NoteEvent.from_dict(d)
            (events if ev.is_valid else excluded).append(ev)
        stats_dict = from_json(row["statistics_json"])
        song_dict = dict(song) if song else {}
        meta_json = from_json(song_dict.get("metadata_json")) or {}
        song_dict.update(meta_json)
        return {
            "analysis_id": row["analysis_id"], "song_id": row["song_id"], "status": row["status"], "mode": row["mode"],
            "config_hash": row["config_hash"], "error_message": row["error_message"],
            "transcription_config": from_json(row["transcription_config_json"]) or {},
            "preprocessing_config": from_json(row["preprocessing_config_json"]) or {},
            "model_info": from_json(row["model_info_json"]) or {}, "software_versions": from_json(row["software_versions_json"]) or {},
            "tempo_bpm": row["tempo_bpm"], "warnings": from_json(row["warnings_json"]) or [],
            "timings": from_json(row["timings_json"]) or {}, "started_at": row["started_at"], "finished_at": row["finished_at"],
            "statistics": SongStatistics.from_dict(stats_dict) if stats_dict else None,
            "events": events, "excluded_events": excluded, "song": song_dict,
        }

    def delete_analysis(self, analysis_id: str) -> int:
        with self._connect() as conn:
            conn.execute("DELETE FROM note_events WHERE analysis_id = ?", (analysis_id,))
            return int(conn.execute("DELETE FROM analyses WHERE analysis_id = ?", (analysis_id,)).rowcount)

    # ------------------------------------------------------------------ #
    # misc
    # ------------------------------------------------------------------ #
    def counts(self) -> dict[str, int]:
        with self._connect() as conn:
            return {
                "songs": conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0],
                "analyses": conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0],
                "events": conn.execute("SELECT COUNT(*) FROM note_events").fetchone()[0],
            }

    def vacuum(self) -> None:
        conn = sqlite3.connect(str(self.path))
        try:
            conn.execute("VACUUM")
        finally:
            conn.close()


def _sql_value(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    if value is None:
        return None
    try:
        import numpy as np

        if isinstance(value, np.generic):
            value = value.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, float) and value != value:  # NaN -> NULL
        return None
    return value
