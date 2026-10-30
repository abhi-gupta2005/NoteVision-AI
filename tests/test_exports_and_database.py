"""CSV / Excel / PDF exports, SQLite persistence and the analysis pipeline cache."""
import io
from pathlib import Path

import pandas as pd
import pytest

from src.analysis_pipeline import make_analysis_id, make_config_hash, result_from_database, run_analysis
from src.audio_loader import AudioValidationError
from src.audio_preprocessing import PreprocessingConfig
from src.note_transcription import EVENT_COLUMNS, TranscriptionConfig
from src.report_generator import ExportError, events_csv_bytes, excel_bytes, export_all, pdf_bytes


@pytest.fixture(scope="module")
def melody_result(melody_wav: Path):
    return run_analysis(melody_wav, TranscriptionConfig(mode="monophonic"))


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def test_pipeline_result_contents(melody_result):
    r = melody_result
    assert r.status == "completed" and not r.from_cache
    assert r.n_events == 6
    assert r.statistics.valid_events == 6
    assert r.events_df.shape[0] == 6
    assert r.tempo_bpm is None or r.tempo_bpm > 0
    assert "transcription" in r.timings and r.timings["total"] > 0
    assert r.software_versions["librosa"]
    info = r.processing_info()
    assert info["analysis_mode"] == "monophonic" and info["gpu_used"] is False
    assert info["excluded_invalid_events"] == 0


def test_pipeline_rejects_invalid_file(text_file_wav: Path):
    with pytest.raises(AudioValidationError):
        run_analysis(text_file_wav, TranscriptionConfig(mode="monophonic"))


def test_config_hash_is_stable_and_sensitive():
    a = make_config_hash(TranscriptionConfig(), PreprocessingConfig())
    b = make_config_hash(TranscriptionConfig(), PreprocessingConfig())
    c = make_config_hash(TranscriptionConfig(onset_threshold=0.4), PreprocessingConfig())
    assert a == b and a != c
    assert make_analysis_id("song", a) == f"song-{a}"


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def test_csv_export(melody_result):
    data = events_csv_bytes(melody_result.events_df)
    df = pd.read_csv(io.BytesIO(data))
    assert len(df) == 6
    for col in EVENT_COLUMNS:
        assert col in df.columns
    assert list(df["note_name"]) == ["C4", "D4", "E4", "C4", "C4", "G4"]
    assert df["start_time_seconds"].is_monotonic_increasing
    assert "start_time_formatted" in df.columns


def test_csv_export_empty():
    data = events_csv_bytes(pd.DataFrame(columns=EVENT_COLUMNS))
    assert data.decode().startswith("event_id,song_id")


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #
def test_excel_export_sheets(melody_result):
    from openpyxl import load_workbook

    data = excel_bytes(melody_result)
    wb = load_workbook(io.BytesIO(data))
    required = ["Song Metadata", "Note Events", "Note Statistics", "Pitch-Class Statistics", "Octave Statistics",
                "Processing Information"]
    for name in required:
        assert name in wb.sheetnames, name
    events = wb["Note Events"]
    headers = [c.value for c in events[1]]
    assert "event_id" in headers and "note_name" in headers
    assert events.max_row == 7                       # header + 6 events
    pc = wb["Pitch-Class Statistics"]
    assert pc.max_row == 13                          # header + 12 pitch classes
    meta = {row[0].value: row[1].value for row in wb["Song Metadata"].iter_rows(min_row=2)}
    assert meta["Filename"] == "melody.wav"
    assert meta["Analysis mode"] == "monophonic"


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def test_pdf_export(melody_result):
    data = pdf_bytes(melody_result)
    assert data[:5] == b"%PDF-"
    assert len(data) > 20_000                        # contains embedded charts
    assert b"/Page" in data


def test_pdf_export_without_charts(melody_result):
    data = pdf_bytes(melody_result, include_charts=False)
    assert data[:5] == b"%PDF-"


def test_export_all_writes_files(melody_result, tmp_path: Path):
    paths = export_all(melody_result, tmp_path)
    assert set(paths) == {"csv", "xlsx", "pdf"}
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
        assert p.name.startswith("melody_")


def test_export_failure_is_reported(melody_result, monkeypatch):
    import src.report_generator as rg

    monkeypatch.setattr(rg, "excel_bytes", lambda r: (_ for _ in ()).throw(ExportError("boom")))
    paths = export_all(melody_result, Path(melody_result.metadata.file_path).parent / "exports_partial")
    assert "xlsx" not in paths and "csv" in paths


# --------------------------------------------------------------------------- #
# Database
# --------------------------------------------------------------------------- #
def test_database_roundtrip_and_dedupe(tmp_db, melody_result):
    tmp_db.save_result(melody_result)
    tmp_db.save_result(melody_result)                # saving twice must not duplicate events
    counts = tmp_db.counts()
    assert counts == {"songs": 1, "analyses": 1, "events": 6}

    record = tmp_db.load_analysis(melody_result.analysis_id)
    assert record["status"] == "completed"
    assert [e.to_dict() for e in record["events"]] == [e.to_dict() for e in melody_result.events]
    assert record["statistics"].most_frequent_note == "C4"
    assert record["transcription_config"]["mode"] == "monophonic"
    assert record["song"]["filename"] == "melody.wav"

    songs = tmp_db.list_songs()
    assert len(songs) == 1 and songs.iloc[0]["n_completed"] == 1
    analyses = tmp_db.list_analyses(melody_result.song_id)
    assert len(analyses) == 1 and analyses.iloc[0]["n_events"] == 6
    df = tmp_db.load_events_df(melody_result.analysis_id)
    assert len(df) == 6 and list(df.columns) == EVENT_COLUMNS


def test_database_status_updates(tmp_db, melody_result):
    tmp_db.upsert_song(melody_result.metadata, melody_result.song_id)
    tmp_db.start_analysis("a1", melody_result.song_id, TranscriptionConfig(), PreprocessingConfig(), "h")
    assert tmp_db.load_analysis("a1")["status"] == "running"
    tmp_db.set_analysis_status("a1", "failed", "model exploded")
    rec = tmp_db.load_analysis("a1")
    assert rec["status"] == "failed" and rec["error_message"] == "model exploded"
    assert tmp_db.delete_analysis("a1") == 1
    assert tmp_db.load_analysis("a1") is None


def test_database_delete_song_cascades(tmp_db, melody_result):
    tmp_db.save_result(melody_result)
    summary = tmp_db.delete_song(melody_result.song_id, delete_files=False)
    assert summary["analyses"] == 1 and summary["events"] == 6
    assert tmp_db.counts() == {"songs": 0, "analyses": 0, "events": 0}


def test_pipeline_cache_and_reopen(tmp_db, melody_wav: Path):
    cfg = TranscriptionConfig(mode="monophonic")
    first = run_analysis(melody_wav, cfg, db=tmp_db)
    assert not first.from_cache
    second = run_analysis(melody_wav, cfg, db=tmp_db)
    assert second.from_cache
    assert second.n_events == first.n_events
    assert [e.to_dict() for e in second.events] == [e.to_dict() for e in first.events]
    assert second.statistics.most_frequent_note == first.statistics.most_frequent_note
    third = run_analysis(melody_wav, cfg, db=tmp_db, force=True)
    assert not third.from_cache
    assert tmp_db.counts()["events"] == 6            # still one copy of the events

    reopened = result_from_database(tmp_db, first.analysis_id)
    assert reopened is not None and reopened.n_events == 6 and reopened.audio is not None
    assert result_from_database(tmp_db, "does-not-exist") is None


def test_pipeline_records_failure(tmp_db, melody_wav: Path, monkeypatch):
    import src.analysis_pipeline as ap

    def boom(*args, **kwargs):
        raise RuntimeError("model crashed")

    monkeypatch.setattr(ap, "transcribe", boom)
    with pytest.raises(RuntimeError):
        run_analysis(melody_wav, TranscriptionConfig(mode="monophonic"), db=tmp_db)
    analyses = tmp_db.list_analyses()
    assert analyses.iloc[0]["status"] == "failed"
    assert "model crashed" in analyses.iloc[0]["error_message"]
