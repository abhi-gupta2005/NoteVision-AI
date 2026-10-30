"""
NoteVision AI - Streamlit application entry point.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from config import settings
from database.database import Database
from src import visualization as viz
from src.analysis_pipeline import AnalysisResult, result_from_database, run_analysis, software_versions
from src.audio_loader import AudioValidationError, save_upload, validate_file
from src.audio_preprocessing import PreprocessingConfig
from src.note_segmentation import SegmentationConfig
from src.note_statistics import consecutive_repeats, repeated_notes
from src.note_transcription import (
    MODE_DESCRIPTIONS,
    MODE_LABELS,
    TranscriptionConfig,
    TranscriptionError,
    TranscriptionMode,
    basic_pitch_available,
)
from src.progress import AnalysisCancelled, CancelToken, ProgressState, scaled
from src.report_generator import ExportError, events_csv_bytes, excel_bytes, pdf_bytes
from src.song_comparison import SongSummary, compare_songs
from src.source_separation import is_available as separation_available
from src.utils import format_timestamp, make_export_filename

log = settings.get_logger("app")
settings.ensure_directories()

DISCLAIMER = (
    "Automatic note detection may be imperfect, especially in full songs with vocals, percussion and "
    "overlapping instruments. Treat every result as an estimate and verify important findings by ear."
)
PAGES = ["Upload & Analyze", "Results Dashboard", "Compare Songs", "History", "Help & About"]


def _streamlit_version() -> tuple[int, int]:
    try:
        major, minor = st.__version__.split(".")[:2]
        return int(major), int(minor)
    except Exception:  # noqa: BLE001
        return (1, 40)


# Streamlit >= 1.49 replaced `use_container_width=True` with `width="stretch"`.
FULL_WIDTH: dict[str, Any] = {"width": "stretch"} if _streamlit_version() >= (1, 49) else {"use_container_width": True}


# --------------------------------------------------------------------------- #
# Services & state
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_db() -> Database:
    return Database()


@st.cache_resource(show_spinner=False)
def capability_report() -> dict[str, Any]:
    ok, detail = basic_pitch_available()
    sep = separation_available()
    return {"basic_pitch": ok, "basic_pitch_detail": detail, "separation": sep.available, "separation_detail": sep.reason,
            "versions": software_versions()}


def init_state() -> None:
    st.session_state.setdefault("results", {})          # song_id -> AnalysisResult
    st.session_state.setdefault("uploads", {})          # key -> {path, name, validation}
    st.session_state.setdefault("job", None)
    st.session_state.setdefault("job_report", None)
    st.session_state.setdefault("selected_song", None)
    st.session_state.setdefault("nav_page", PAGES[0])


@dataclass
class AnalysisJob:
    """Runs a batch of analyses in a background thread; the UI polls `progress`."""

    items: list[tuple[Path, str]]
    transcription_config: TranscriptionConfig
    preprocessing_config: PreprocessingConfig
    force: bool
    progress: ProgressState = field(default_factory=ProgressState)
    cancel: CancelToken = field(default_factory=CancelToken)
    results: dict[str, AnalysisResult] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    current: str = ""
    done: bool = False
    started_at: float = field(default_factory=time.time)

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True, name="notevision-analysis").start()

    def _run(self) -> None:
        db = get_db()
        n = len(self.items)
        for i, (path, name) in enumerate(self.items):
            self.current = name
            if self.cancel.is_cancelled:
                self.errors[name] = "Cancelled before it started."
                continue
            sub = scaled(self.progress.update, i / n, (i + 1) / n)
            try:
                result = run_analysis(path, self.transcription_config, self.preprocessing_config, db=db,
                                      progress=sub, cancel=self.cancel, force=self.force)
                self.results[result.song_id] = result
            except AnalysisCancelled:
                self.errors[name] = "Cancelled by user."
            except TranscriptionError as exc:
                hint = f" Suggested mode: {MODE_LABELS[exc.suggested_mode]}." if exc.suggested_mode else ""
                self.errors[name] = f"{exc}{hint}"
            except AudioValidationError as exc:
                self.errors[name] = f"Invalid file: {exc}"
            except Exception as exc:  # noqa: BLE001 - surfaced to the user, logged with traceback
                log.error("Unexpected failure for %s:\n%s", name, traceback.format_exc())
                self.errors[name] = f"Unexpected error: {exc}"
        self.progress.update(1.0, "Finished")
        self.done = True


# --------------------------------------------------------------------------- #
# Small UI helpers
# --------------------------------------------------------------------------- #
def results_dict() -> dict[str, AnalysisResult]:
    return st.session_state["results"]


def song_label(result: AnalysisResult) -> str:
    return f"{result.metadata.filename}  ({result.transcription_config.mode}, {result.n_events} events)"


def select_result(key: str = "dashboard_song") -> AnalysisResult | None:
    results = results_dict()
    if not results:
        return None
    ids = list(results)
    default = st.session_state.get("selected_song") or ids[-1]
    index = ids.index(default) if default in ids else len(ids) - 1
    chosen = st.selectbox("Analysed recording", ids, index=index, format_func=lambda i: song_label(results[i]), key=key)
    st.session_state["selected_song"] = chosen
    return results[chosen]


def valid_events(result: AnalysisResult) -> pd.DataFrame:
    df = result.events_df
    if df.empty or "is_valid" not in df:
        return df
    return df[df["is_valid"]].reset_index(drop=True)


def display_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "event_id": "Event ID", "note_name": "Note", "midi_note": "MIDI", "frequency_hz": "Frequency (Hz)",
        "start_time_formatted": "Start", "end_time_formatted": "End", "start_time_seconds": "Start (s)",
        "end_time_seconds": "End (s)", "duration_seconds": "Duration (s)", "confidence": "Confidence",
        "cents_deviation": "Cents", "overlap_count": "Overlaps", "source_track": "Track", "octave": "Octave",
        "pitch_class": "Pitch class",
    }
    present = [c for c in cols if c in df]
    out = df[present].rename(columns=cols)
    if "Frequency (Hz)" in out:
        out["Frequency (Hz)"] = out["Frequency (Hz)"].round(2)
    return out


def add_local_files(path_text: str, uploads: dict[str, dict[str, Any]], max_files: int = 50) -> int:
    """Register audio files from a local path (file or folder) without copying them."""
    path = Path(path_text).expanduser()
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in settings.SUPPORTED_EXTENSIONS)
    else:
        return 0
    added = 0
    for p in candidates[:max_files]:
        key = f"local:{p.resolve()}"
        if key in uploads:
            continue
        uploads[key] = {"path": p, "name": p.name, "validation": validate_file(p), "source": "local"}
        added += 1
    return added


# --------------------------------------------------------------------------- #
# Page: Upload & Analyze
# --------------------------------------------------------------------------- #
def page_upload() -> None:
    st.header("Upload & Analyze")
    st.caption(DISCLAIMER)
    caps = capability_report()
    if not caps["basic_pitch"]:
        st.warning(f"Polyphonic / full-mix modes are unavailable: {caps['basic_pitch_detail']} "
                   f"Monophonic mode (pYIN) still works.")

    uploaded = st.file_uploader(
        "Drop one or more recordings here", type=[e.lstrip(".") for e in settings.SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        help=f"Up to {settings.MAX_FILE_SIZE_MB:.0f} MB and {settings.MAX_DURATION_SECONDS / 60:.0f} minutes per file.",
    )
    uploads: dict[str, dict[str, Any]] = st.session_state["uploads"]
    if uploaded:
        for f in uploaded:
            key = f"{f.name}:{f.size}"
            if key in uploads:
                continue
            with st.spinner(f"Checking {f.name}"):
                data = f.getvalue()
                path = save_upload(data, f.name)
                uploads[key] = {"path": path, "name": f.name, "validation": validate_file(path), "source": "upload"}
    if uploaded is not None:  # keep the list in sync with the uploader widget (files removed by the user)
        current_keys = {f"{f.name}:{f.size}" for f in uploaded}
        for key in [k for k in uploads if k not in current_keys and uploads[k].get("source", "upload") == "upload"]:
            uploads.pop(key)

    with st.expander("Or analyse files that are already on this computer (no upload needed)"):
        st.caption("Enter the path of an audio file or of a folder; the files are read in place and never modified.")
        c1, c2 = st.columns([4, 1])
        local_path = c1.text_input("File or folder path", key="local_path_input", label_visibility="collapsed",
                                   placeholder=r"C:\Users\you\Music\song.wav  or  C:\Users\you\Music")
        if c2.button("Add", key="add_local") and local_path.strip():
            added = add_local_files(local_path.strip(), uploads)
            st.toast(f"Added {added} file(s)" if added else "No supported audio files found at that path")
        if any(it.get("source") == "local" for it in uploads.values()) and st.button("Clear local files", key="clear_local"):
            for k in [k for k, it in uploads.items() if it.get("source") == "local"]:
                uploads.pop(k)
            st.rerun()

    ready: list[tuple[Path, str]] = []
    if uploads:
        rows = []
        for item in uploads.values():
            v = item["validation"]
            m = v.metadata
            rows.append({
                "File": item["name"], "Source": item.get("source", "upload"), "Status": "ready" if v.ok else "rejected",
                "Duration": format_timestamp(m.duration_seconds) if m else "-",
                "Sample rate": f"{m.sample_rate} Hz" if m else "-", "Channels": str(m.channels) if m else "-",
                "Size (MB)": f"{m.file_size_mb:.2f}" if m else "-",
                "Message": v.message if not v.ok else ("; ".join(v.warnings) if v.warnings else "OK"),
            })
            if v.ok:
                ready.append((item["path"], item["name"]))
        st.dataframe(pd.DataFrame(rows), **FULL_WIDTH, hide_index=True)
    else:
        st.info("No files yet. Upload WAV, MP3, FLAC, OGG or AIFF recordings to begin.")

    st.subheader("Analysis mode")
    mode_values = [m.value for m in TranscriptionMode]
    mode = st.radio("Choose how the recording should be interpreted", mode_values, index=1,
                    format_func=lambda v: MODE_LABELS[TranscriptionMode(v)], horizontal=False)
    st.caption(MODE_DESCRIPTIONS[TranscriptionMode(mode)])

    with st.expander("Advanced settings", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Transcription**")
            onset_t = st.slider("Basic Pitch onset threshold", 0.1, 0.9, settings.BASIC_PITCH_ONSET_THRESHOLD, 0.05,
                                help="Lower = more (and more spurious) note starts.")
            frame_t = st.slider("Basic Pitch frame threshold", 0.1, 0.9, settings.BASIC_PITCH_FRAME_THRESHOLD, 0.05,
                                help="Lower = notes are held longer / softer notes kept.")
            min_len = st.slider("Minimum note length (ms)", 20, 400, int(settings.BASIC_PITCH_MIN_NOTE_LENGTH_MS), 5)
            min_conf = st.slider("Discard events below confidence", 0.0, 0.9, 0.0, 0.05,
                                 help="0 keeps everything the model reports; discarded events are listed separately.")
            fmin = st.text_input("pYIN lowest note", settings.PYIN_FMIN_NOTE)
            fmax = st.text_input("pYIN highest note", settings.PYIN_FMAX_NOTE)
            split_onsets = st.checkbox("Split repeated notes on onset evidence (monophonic)", value=True)
            use_sep = st.checkbox("Use source separation for full-mix mode (requires Demucs)", value=False,
                                  disabled=not caps["separation"], help=caps["separation_detail"])
        with c2:
            st.markdown("**Preprocessing** (timeline is preserved; nothing is applied by default)")
            normalize = st.checkbox("Peak-normalise to -1 dBFS", value=False)
            noise = st.checkbox("Light noise reduction (spectral gating)", value=False)
            noise_strength = st.slider("Noise reduction strength", 0.1, 1.0, 0.5, 0.1, disabled=not noise)
            highpass = st.checkbox("High-pass filter (remove rumble below 30 Hz)", value=False)
            trim = st.checkbox("Trim leading/trailing silence (timestamps are shifted back automatically)", value=False)
            force = st.checkbox("Re-analyse even if an identical analysis is stored", value=False)

    tcfg = TranscriptionConfig(
        mode=mode, onset_threshold=onset_t, frame_threshold=frame_t, minimum_note_length_ms=float(min_len),
        min_confidence=min_conf, pyin_fmin_note=fmin.strip() or settings.PYIN_FMIN_NOTE,
        pyin_fmax_note=fmax.strip() or settings.PYIN_FMAX_NOTE, use_source_separation=use_sep,
        segmentation=SegmentationConfig(split_on_onsets=split_onsets),
    )
    pcfg = PreprocessingConfig(normalize=normalize, noise_reduction=noise, noise_reduction_strength=noise_strength,
                               high_pass_hz=30.0 if highpass else None, trim_silence=trim)

    job: AnalysisJob | None = st.session_state["job"]
    disabled = not ready or (job is not None and not job.done)
    if st.button(f"Analyze {len(ready)} file(s)", type="primary", disabled=disabled):
        new_job = AnalysisJob(items=ready, transcription_config=tcfg, preprocessing_config=pcfg, force=force)
        st.session_state["job"] = new_job
        st.session_state["job_report"] = None
        new_job.start()
        st.rerun()

    if st.session_state["job"] is not None:
        job_progress_fragment()
    report = st.session_state.get("job_report")
    if report:
        if report["ok"]:
            st.success(f"Analysed {len(report['ok'])} recording(s): " + ", ".join(report["ok"]) +
                       ". Open the Results Dashboard to explore them.")
        for name, err in report["errors"].items():
            st.error(f"{name}: {err}")


@st.fragment(run_every=0.5)
def job_progress_fragment() -> None:
    job: AnalysisJob | None = st.session_state.get("job")
    if job is None:
        return
    fraction, message = job.progress.snapshot()
    st.progress(fraction, text=f"{job.current} - {message}  ({job.progress.elapsed_seconds:.0f} s)")
    if not job.done:
        if st.button("Cancel", key="cancel_job"):
            job.cancel.cancel()
            st.toast("Cancelling after the current step...")
        return
    for song_id, result in job.results.items():
        st.session_state["results"][song_id] = result
        st.session_state["selected_song"] = song_id
    st.session_state["job_report"] = {"ok": [r.metadata.filename for r in job.results.values()], "errors": dict(job.errors)}
    st.session_state["job"] = None
    st.rerun()


# --------------------------------------------------------------------------- #
# Page: Results Dashboard
# --------------------------------------------------------------------------- #
def page_dashboard() -> None:
    st.header("Results Dashboard")
    result = select_result()
    if result is None:
        st.info("No analysed recordings in this session yet. Analyse a file first, or open one from History.")
        return
    tabs = st.tabs(["Overview", "Waveform", "Spectrogram", "Piano Roll", "Note Frequency", "Duration Analysis",
                    "Timeline", "Data Table", "Export"])
    with tabs[0]:
        tab_overview(result)
    with tabs[1]:
        tab_waveform(result)
    with tabs[2]:
        tab_spectrogram(result)
    with tabs[3]:
        tab_piano_roll(result)
    with tabs[4]:
        tab_note_frequency(result)
    with tabs[5]:
        tab_durations(result)
    with tabs[6]:
        tab_timeline(result)
    with tabs[7]:
        tab_table(result)
    with tabs[8]:
        tab_export(result)


def tab_overview(result: AnalysisResult) -> None:
    s = result.statistics
    m = result.metadata
    c = st.columns(4)
    c[0].metric("File", m.filename if len(m.filename) < 22 else m.filename[:19] + "...", help=m.filename)
    c[1].metric("Duration", format_timestamp(m.duration_seconds))
    c[2].metric("Status", result.status + (" (cached)" if result.from_cache else ""))
    c[3].metric("Detected note events", s.valid_events, help="Valid events used in statistics")
    c = st.columns(4)
    c[0].metric("Unique pitches", s.unique_midi_pitches, help=f"{s.unique_pitch_classes} pitch classes")
    c[1].metric("Most common note", f"{s.most_frequent_note or '-'}", help=f"{s.most_frequent_note_count} occurrences")
    c[2].metric("Estimated tempo", f"{result.tempo_bpm:.0f} BPM" if result.tempo_bpm else "n/a",
                help="Global estimate from onset periodicity; unreliable for rubato or short clips.")
    c[3].metric("Max simultaneous notes", s.max_polyphony)
    c = st.columns(4)
    c[0].metric("Pitch range", f"{s.pitch_range_low or '-'} - {s.pitch_range_high or '-'}")
    c[1].metric("Mean / median duration", f"{(s.mean_event_duration_seconds or 0):.3f} / {(s.median_event_duration_seconds or 0):.3f} s")
    c[2].metric("Overlapping events", s.overlapping_events)
    c[3].metric("Excluded events", s.excluded_invalid_events, help="Failed validation or the confidence filter")

    if result.warnings:
        with st.expander("Notes and warnings from this analysis", expanded=True):
            for w in result.warnings:
                st.markdown(f"- {w}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Recording**")
        st.table(pd.DataFrame({
            "Field": ["Format", "Sample rate", "Channels", "Size", "SHA-256", "Peak / RMS level"],
            "Value": [f"{m.container_format} / {m.subtype}", f"{m.sample_rate} Hz", m.channels, f"{m.file_size_mb:.2f} MB",
                      m.file_hash[:16] + "...", f"{result.level_stats.get('peak_dbfs', float('nan')):.1f} / "
                                              f"{result.level_stats.get('rms_dbfs', float('nan')):.1f} dBFS"],
        }).astype(str))
    with c2:
        st.markdown("**Processing**")
        mi = result.model_info
        steps = result.audio.summary() if result.audio else []
        st.table(pd.DataFrame({
            "Field": ["Mode", "Model", "Version", "Preprocessing", "Transcription time", "Analysed at (UTC)"],
            "Value": [result.transcription_config.mode, mi.get("model_name", "-"), mi.get("model_version", "-"),
                      "; ".join(steps) if steps else "none (original timeline, mono @ 22.05 kHz)",
                      f"{result.timings.get('transcription', 0):.1f} s" if result.timings else "-", result.created_at],
        }).astype(str))


def tab_waveform(result: AnalysisResult) -> None:
    if result.audio is None:
        st.info("Audio is not loaded for this result (the original file is no longer available).")
        return
    show_onsets = False
    if result.transcription is not None and result.transcription.onsets is not None:
        show_onsets = st.checkbox("Show detected onsets", value=False)
    onsets = (result.transcription.onsets.onset_times + result.audio.time_offset_seconds).tolist() \
        if show_onsets and result.transcription and result.transcription.onsets else None
    st.plotly_chart(viz.waveform_figure(result.audio.samples, result.audio.sample_rate, onset_times=onsets,
                                        time_offset=result.audio.time_offset_seconds), **FULL_WIDTH)
    st.caption("Drag the range slider below the plot to inspect a portion; double-click to reset. "
               "The waveform is the preprocessed mono signal at the analysis sample rate.")
    st.audio(str(result.metadata.file_path)) if Path(result.metadata.file_path).exists() else None


def tab_spectrogram(result: AnalysisResult) -> None:
    if result.audio is None:
        st.info("Audio is not loaded for this result.")
        return
    c1, c2 = st.columns([1, 3])
    fmax = c1.select_slider("Maximum frequency (Hz)", [1000, 2000, 4000, 8000, 11025], value=4000)
    logf = c1.checkbox("Logarithmic frequency axis", value=False)
    fig = viz.spectrogram_figure(result.audio.samples, result.audio.sample_rate, fmax=float(fmax), log_frequency=logf,
                                 time_offset=result.audio.time_offset_seconds)
    c2.plotly_chart(fig, **FULL_WIDTH)
    st.caption("STFT with 2048-sample window and 512-sample hop; colour is magnitude in dB relative to the peak. "
               "Use the toolbar to zoom; hover for time, frequency and level.")
    if result.transcription is not None and result.transcription.pitch_track is not None:
        pt = result.transcription.pitch_track
        st.plotly_chart(viz.pitch_track_figure(pt.times, pt.f0_hz, pt.voiced_prob, result.audio.time_offset_seconds),
                        **FULL_WIDTH)


def tab_piano_roll(result: AnalysisResult) -> None:
    df = valid_events(result)
    if df.empty:
        st.info("No valid note events to draw.")
        return
    c1, c2, c3 = st.columns(3)
    color_by = c1.selectbox("Colour by", ["none", "confidence", "source_track"], index=0)
    has_conf = df["confidence"].notna().any()
    min_conf = c2.slider("Hide events below confidence", 0.0, 1.0, 0.0, 0.05, disabled=not has_conf, key="pr_conf")
    selected = c3.selectbox("Highlight event", ["(none)"] + df["event_id"].tolist(), index=0, key="pr_sel")
    shown = df[df["confidence"].fillna(1.0) >= min_conf] if has_conf else df
    st.plotly_chart(viz.piano_roll_figure(shown, color_by=color_by, duration_seconds=result.metadata.duration_seconds,
                                          selected_event_id=None if selected == "(none)" else selected),
                    **FULL_WIDTH)
    st.caption(f"{len(shown)} of {len(df)} events shown. Width = estimated duration; overlapping events are drawn on top "
               f"of each other with a thin white outline. Hover any bar for the note name and timing.")


def tab_note_frequency(result: AnalysisResult) -> None:
    s = result.statistics
    if s.note_table.empty:
        st.info("No valid note events.")
        return
    sub = st.tabs(["By note (pitch + octave)", "By pitch class", "By octave"])
    with sub[0]:
        metric = st.radio("Show", ["Occurrence count", "Total duration (s)"], horizontal=True, key="nf_metric")
        if metric == "Occurrence count":
            st.plotly_chart(viz.note_frequency_figure(s.note_table), **FULL_WIDTH)
        else:
            st.plotly_chart(viz.note_frequency_figure(s.note_table, title="Total duration per note",
                                                      value_column="total_duration_seconds", label="Total duration (s)"),
                            **FULL_WIDTH)
        st.dataframe(s.note_table.round(4), **FULL_WIDTH, hide_index=True)
        st.caption("Occurrence count = number of separate events; total duration = sum of their durations. "
                   "A note played ten times counts ten occurrences regardless of how long each one lasted.")
    with sub[1]:
        st.plotly_chart(viz.pitch_class_figure(s.pitch_class_table), **FULL_WIDTH)
        st.dataframe(s.pitch_class_table.round(4), **FULL_WIDTH, hide_index=True)
    with sub[2]:
        st.plotly_chart(viz.octave_figure(s.octave_table), **FULL_WIDTH)
        st.dataframe(s.octave_table.round(4), **FULL_WIDTH, hide_index=True)


def tab_durations(result: AnalysisResult) -> None:
    s = result.statistics
    df = valid_events(result)
    if df.empty:
        st.info("No valid note events.")
        return
    c = st.columns(5)
    c[0].metric("Shortest", f"{s.min_event_duration_seconds:.3f} s")
    c[1].metric("Longest", f"{s.max_event_duration_seconds:.3f} s")
    c[2].metric("Mean", f"{s.mean_event_duration_seconds:.3f} s")
    c[3].metric("Median", f"{s.median_event_duration_seconds:.3f} s")
    c[4].metric("Outliers (short / long)", f"{s.short_outliers} / {s.long_outliers}")
    st.plotly_chart(viz.duration_distribution_figure(df, s.mean_event_duration_seconds, s.median_event_duration_seconds,
                                                     s.outlier_bounds), **FULL_WIDTH)
    st.plotly_chart(viz.duration_by_note_figure(s.note_table), **FULL_WIDTH)
    if s.outlier_bounds:
        lo, hi = s.outlier_bounds
        out = df[(df["duration_seconds"] < lo) | (df["duration_seconds"] > hi)]
        if not out.empty:
            st.markdown(f"**Unusually short or long events** (outside {lo:.3f}-{hi:.3f} s, 1.5 x IQR rule)")
            st.dataframe(display_table(out), **FULL_WIDTH, hide_index=True)


def tab_timeline(result: AnalysisResult) -> None:
    s = result.statistics
    df = valid_events(result)
    if df.empty:
        st.info("No valid note events.")
        return
    chrono = df.sort_values("start_time_seconds")
    options = ["(none)"] + chrono["event_id"].tolist()
    selected = st.selectbox("Inspect an event", options, index=0,
                            format_func=lambda i: i if i == "(none)" else
                            f"{i}  {chrono.loc[chrono['event_id'] == i, 'note_name'].iloc[0]} @ "
                            f"{format_timestamp(chrono.loc[chrono['event_id'] == i, 'start_time_seconds'].iloc[0])}")
    sel_id = None if selected == "(none)" else selected
    st.plotly_chart(viz.timeline_figure(df, selected_event_id=sel_id), **FULL_WIDTH)
    if sel_id:
        row = df[df["event_id"] == sel_id].iloc[0]
        st.markdown(f"**Event {sel_id}**")
        info = {k: row[k] for k in ["note_name", "midi_note", "pitch_class", "octave", "frequency_hz", "cents_deviation",
                                    "start_time_formatted", "end_time_formatted", "duration_seconds", "confidence",
                                    "onset_strength", "overlap_count", "source_track", "detection_method", "frequency_source"]}
        st.json({k: (None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v))
                 for k, v in info.items()})
    st.plotly_chart(viz.polyphony_figure(s.polyphony.get("times", []), s.polyphony.get("counts", [])), **FULL_WIDTH)
    tl = s.timeline
    st.plotly_chart(viz.timeline_distribution_figure(tl.get("bin_starts", []), tl.get("bin_width", 0.0), tl.get("counts", [])),
                    **FULL_WIDTH)


def tab_table(result: AnalysisResult) -> None:
    df = valid_events(result)
    if df.empty:
        st.info("No valid note events.")
        if result.excluded_events:
            st.markdown("**Excluded events**")
            st.dataframe(pd.DataFrame([e.to_dict() for e in result.excluded_events]), **FULL_WIDTH)
        return
    duration = float(max(result.metadata.duration_seconds, df["end_time_seconds"].max()))
    c1, c2, c3 = st.columns(3)
    notes = c1.multiselect("Note", sorted(df["note_name"].unique(), key=lambda n: df.loc[df["note_name"] == n, "midi_note"].iloc[0]))
    octaves = c2.multiselect("Octave", sorted(df["octave"].unique()))
    has_conf = df["confidence"].notna().any()
    min_conf = c3.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05, disabled=not has_conf, key="tbl_conf")
    c4, c5, c6 = st.columns(3)
    interval = c4.slider("Time interval (s)", 0.0, float(np.ceil(duration)), (0.0, float(np.ceil(duration))), 0.1)
    sort_by = c5.selectbox("Sort by", ["Start time", "Duration (longest first)", "Duration (shortest first)", "Pitch", "Confidence"])
    repeats_only = c6.checkbox("Only notes that repeat (>= 2 occurrences)")

    f = df
    if notes:
        f = f[f["note_name"].isin(notes)]
    if octaves:
        f = f[f["octave"].isin(octaves)]
    if has_conf and min_conf > 0:
        f = f[f["confidence"].fillna(0) >= min_conf]
    f = f[(f["end_time_seconds"] >= interval[0]) & (f["start_time_seconds"] <= interval[1])]
    if repeats_only:
        counts = f["midi_note"].value_counts()
        f = f[f["midi_note"].isin(counts[counts >= 2].index)]
    if sort_by == "Start time":
        f = f.sort_values(["start_time_seconds", "midi_note"])
    elif sort_by == "Duration (longest first)":
        f = f.sort_values("duration_seconds", ascending=False)
    elif sort_by == "Duration (shortest first)":
        f = f.sort_values("duration_seconds", ascending=True)
    elif sort_by == "Pitch":
        f = f.sort_values(["midi_note", "start_time_seconds"])
    else:
        f = f.sort_values("confidence", ascending=False, na_position="last")

    st.markdown(f"**{len(f)} of {len(df)} events**")
    st.dataframe(display_table(f), **FULL_WIDTH, hide_index=True, height=420)
    st.download_button("Download filtered events (CSV)", events_csv_bytes(f),
                       file_name=make_export_filename(result.metadata.filename, "filtered_events", "csv"), mime="text/csv")

    with st.expander("Repeated notes and every occurrence of a selected note"):
        rep = repeated_notes(df)
        if rep.empty:
            st.write("No note occurs more than once.")
        else:
            rep_disp = rep.copy()
            rep_disp["start_times"] = rep_disp["start_times"].map(lambda ts: ", ".join(format_timestamp(t) for t in ts))
            st.dataframe(rep_disp.rename(columns={"note_name": "Note", "midi_note": "MIDI", "count": "Occurrences",
                                                  "start_times": "Starts at"}), **FULL_WIDTH, hide_index=True)
        pick = st.selectbox("Show every occurrence of", ["(choose a note)"] + sorted(
            df["note_name"].unique(), key=lambda n: df.loc[df["note_name"] == n, "midi_note"].iloc[0]))
        if pick != "(choose a note)":
            occ = df[df["note_name"] == pick].sort_values("start_time_seconds")
            st.dataframe(display_table(occ), **FULL_WIDTH, hide_index=True)
        cons = consecutive_repeats(df)
        if not cons.empty:
            st.markdown("**Consecutive repeats of the same pitch (gap <= 0.5 s)**")
            cons_disp = cons.copy()
            for col in ("first_start", "second_start"):
                cons_disp[col] = cons_disp[col].map(format_timestamp)
            st.dataframe(cons_disp, **FULL_WIDTH, hide_index=True)

    if result.excluded_events:
        with st.expander(f"Excluded events ({len(result.excluded_events)})"):
            st.dataframe(pd.DataFrame([e.to_dict() for e in result.excluded_events]), **FULL_WIDTH, hide_index=True)


def tab_export(result: AnalysisResult) -> None:
    st.markdown("Downloads use meaningful, timestamped filenames. Exports are generated when you click.")
    name = result.metadata.filename
    c1, c2, c3 = st.columns(3)
    with c1:
        try:
            st.download_button("Download CSV (all events)", events_csv_bytes(result.events_df),
                               file_name=make_export_filename(name, "note_events", "csv"), mime="text/csv", **FULL_WIDTH)
        except ExportError as exc:
            st.error(str(exc))
    with c2:
        if st.button("Prepare Excel workbook", **FULL_WIDTH):
            try:
                with st.spinner("Building workbook"):
                    st.session_state[f"xlsx_{result.analysis_id}"] = excel_bytes(result)
            except ExportError as exc:
                st.error(str(exc))
        data = st.session_state.get(f"xlsx_{result.analysis_id}")
        if data:
            st.download_button("Download Excel", data, file_name=make_export_filename(name, "report", "xlsx"),
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", **FULL_WIDTH)
    with c3:
        if st.button("Prepare PDF report", **FULL_WIDTH):
            try:
                with st.spinner("Rendering PDF"):
                    st.session_state[f"pdf_{result.analysis_id}"] = pdf_bytes(result)
            except ExportError as exc:
                st.error(str(exc))
        data = st.session_state.get(f"pdf_{result.analysis_id}")
        if data:
            st.download_button("Download PDF", data, file_name=make_export_filename(name, "report", "pdf"),
                               mime="application/pdf", **FULL_WIDTH)
    with st.expander("Processing information included in the reports"):
        st.dataframe(pd.DataFrame([{"Setting": k, "Value": str(v)} for k, v in result.processing_info().items()]),
                     **FULL_WIDTH, hide_index=True)


# --------------------------------------------------------------------------- #
# Page: Compare Songs
# --------------------------------------------------------------------------- #
def page_compare() -> None:
    st.header("Compare Songs")
    results = results_dict()
    if len(results) < 2:
        st.info("Analyse at least two recordings (or open them from History) to compare them.")
        return
    ids = st.multiselect("Recordings to compare", list(results), default=list(results)[-min(len(results), 8):],
                         format_func=lambda i: song_label(results[i]))
    if len(ids) < 2:
        st.warning("Select at least two recordings.")
        return
    if len(ids) > 8:
        st.warning("Charts show the first 8 selected recordings; tables include all of them.")
    cmp = compare_songs([SongSummary.from_analysis(results[i]) for i in ids])
    for note in cmp.notes:
        st.caption(note)
    st.subheader("Summary")
    st.dataframe(cmp.summary_table.drop(columns=["song_id"]), **FULL_WIDTH, hide_index=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.comparison_metric_figure(cmp.summary_table, "total_events", "Total note events", "Events"), **FULL_WIDTH)
    c2.plotly_chart(viz.comparison_metric_figure(cmp.summary_table, "unique_pitches", "Unique pitches", "Pitches"), **FULL_WIDTH)
    c1.plotly_chart(viz.comparison_metric_figure(cmp.summary_table, "mean_duration_seconds", "Average note duration", "Seconds"), **FULL_WIDTH)
    c2.plotly_chart(viz.comparison_metric_figure(cmp.summary_table, "events_per_minute", "Note density", "Events per minute"), **FULL_WIDTH)
    st.subheader("Distributions (normalised by each song's event count)")
    st.plotly_chart(viz.comparison_distribution_figure(cmp.pitch_class_distribution, "Pitch-class distribution", "Pitch class"), **FULL_WIDTH)
    st.plotly_chart(viz.comparison_distribution_figure(cmp.octave_distribution, "Octave distribution", "Octave"), **FULL_WIDTH)
    st.subheader("Similarity of note distributions")
    c1, c2 = st.columns(2)
    c1.plotly_chart(viz.similarity_heatmap_figure(cmp.cosine_similarity), **FULL_WIDTH)
    c2.markdown("**Jensen-Shannon distance** (0 = identical pitch-class distribution, 1 = nothing in common)")
    c2.dataframe(cmp.jensen_shannon_distance.round(3), **FULL_WIDTH)
    if cmp.sequence_similarity is not None:
        c2.markdown("**Note-sequence similarity** (order of pitch classes, 0-1; approximate)")
        c2.dataframe(cmp.sequence_similarity.round(3), **FULL_WIDTH)
    st.download_button("Download comparison summary (CSV)", cmp.summary_table.to_csv(index=False).encode("utf-8"),
                       file_name=make_export_filename("comparison", "summary", "csv"), mime="text/csv")


# --------------------------------------------------------------------------- #
# Page: History
# --------------------------------------------------------------------------- #
def page_history() -> None:
    st.header("History (SQLite database)")
    db = get_db()
    counts = db.counts()
    st.caption(f"Database: {settings.DB_PATH}  -  {counts['songs']} songs, {counts['analyses']} analyses, "
               f"{counts['events']} stored events. Uploaded audio is kept under data/uploads/<song_id>/ until the song is deleted. "
               f"On a hosted deployment (e.g. Streamlit Community Cloud) this storage is temporary and is cleared when the app restarts.")
    songs = db.list_songs()
    if songs.empty:
        st.info("No stored analyses yet.")
        return
    st.subheader("Songs")
    st.dataframe(songs.drop(columns=["file_path"]), **FULL_WIDTH, hide_index=True)
    st.subheader("Analyses")
    analyses = db.list_analyses()
    st.dataframe(analyses.drop(columns=["config_hash"]), **FULL_WIDTH, hide_index=True)
    completed = analyses[analyses["status"] == "completed"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Open a stored analysis**")
        if completed.empty:
            st.write("No completed analyses.")
        else:
            pick = st.selectbox("Analysis", completed["analysis_id"].tolist(),
                                format_func=lambda a: f"{completed.loc[completed['analysis_id'] == a, 'filename'].iloc[0]} - "
                                                      f"{completed.loc[completed['analysis_id'] == a, 'mode'].iloc[0]} "
                                                      f"({completed.loc[completed['analysis_id'] == a, 'n_events'].iloc[0]} events)")
            if st.button("Load into dashboard"):
                with st.spinner("Loading"):
                    result = result_from_database(db, pick)
                if result is None:
                    st.error("Analysis not found.")
                else:
                    st.session_state["results"][result.song_id] = result
                    st.session_state["selected_song"] = result.song_id
                    st.success(f"Loaded {result.metadata.filename}. Open the Results Dashboard or Export tab.")
    with c2:
        st.markdown("**Delete a recording and all its analysis data**")
        target = st.selectbox("Song", songs["song_id"].tolist(),
                              format_func=lambda sid: songs.loc[songs["song_id"] == sid, "filename"].iloc[0])
        confirm = st.checkbox("I understand this permanently removes the stored events, statistics and the uploaded audio copy.")
        if st.button("Delete", type="secondary", disabled=not confirm):
            summary = db.delete_song(target, delete_files=True)
            st.session_state["results"].pop(target, None)
            st.success(f"Deleted {summary['analyses']} analyses, {summary['events']} events and {summary['files']} file(s).")
            st.rerun()


# --------------------------------------------------------------------------- #
# Page: Help & About
# --------------------------------------------------------------------------- #
def page_help() -> None:
    st.header("Help & About")
    st.markdown(f"**{settings.APP_NAME} {settings.APP_VERSION}** - {settings.APP_TAGLINE}.  \n"
                f"Source code, documentation and issue tracker: [{settings.REPOSITORY_URL}]({settings.REPOSITORY_URL})")
    st.warning(DISCLAIMER)
    st.subheader("Analysis modes")
    for mode in TranscriptionMode:
        st.markdown(f"**{MODE_LABELS[mode]}** - {MODE_DESCRIPTIONS[mode]}")
    st.subheader("How to read the results")
    st.markdown(
        "- **Occurrence count** is the number of separate events of a note; **total duration** is the sum of their lengths.\n"
        "- **Confidence** is the model's note activation (Basic Pitch) or the mean voicing probability (pYIN). "
        "It is not a calibrated probability.\n"
        "- **Frequency** is measured from the pitch track (pYIN, median f0) or refined from the model's pitch contour "
        "(Basic Pitch, ~1/3-semitone bins with interpolation). The `frequency_source` column tells you which.\n"
        "- **Timestamps** are seconds on the original recording timeline, rounded to 4 decimals; the models' native "
        "frame resolution is ~11.6 ms.\n"
        "- **Overlaps** counts how many other events sound at the same time as an event.\n"
        "- Statistics exclude events that failed validation or the confidence filter; the number excluded is always shown."
    )
    caps = capability_report()
    st.subheader("System")
    st.markdown(f"- Basic Pitch: {'available' if caps['basic_pitch'] else 'NOT available'} - {caps['basic_pitch_detail']}\n"
                f"- Source separation (Demucs): {'available' if caps['separation'] else 'not installed'} - {caps['separation_detail']}\n"
                f"- GPU acceleration: not used (all inference runs on the CPU through onnxruntime).")
    st.dataframe(pd.DataFrame([{"Component": k, "Version": v} for k, v in caps["versions"].items()]),
                 **FULL_WIDTH, hide_index=True)
    st.subheader("Known limitations")
    st.markdown(
        "- Chords and accompaniment are invisible to the monophonic mode.\n"
        "- Neural transcription can report harmonics as extra notes (usually with low confidence) and make octave errors.\n"
        "- Full songs with drums, vocals and effects are transcribed approximately at best.\n"
        "- Tempo is a single global estimate.\n"
        "- Similar note distributions between songs do not imply a shared melody."
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title=settings.APP_NAME, page_icon="🎵", layout="wide")
    init_state()
    with st.sidebar:
        st.title(settings.APP_NAME)
        st.caption(settings.APP_TAGLINE)
        page = st.radio("Navigate", PAGES, key="nav_page")
        job: AnalysisJob | None = st.session_state.get("job")
        if job is not None and not job.done:
            st.info("Analysis running...")
        n = len(results_dict())
        st.caption(f"{n} recording(s) analysed in this session")
        caps = capability_report()
        st.caption(("Basic Pitch ready (ONNX, CPU)" if caps["basic_pitch"] else "Basic Pitch unavailable - monophonic mode only"))
        st.divider()
        st.caption(DISCLAIMER)
    if page == PAGES[0]:
        page_upload()
    elif page == PAGES[1]:
        page_dashboard()
    elif page == PAGES[2]:
        page_compare()
    elif page == PAGES[3]:
        page_history()
    else:
        page_help()


if __name__ == "__main__":
    main()
