"""
CSV, Excel and PDF report generation.

Every exporter returns bytes (for Streamlit download buttons) and can also
write to the exports directory. Failures raise `ExportError` with a readable
message; nothing is swallowed.
"""
from __future__ import annotations

import io
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from config import settings
from src.note_transcription import EVENT_COLUMNS
from src.utils import format_timestamp, make_export_filename

if TYPE_CHECKING:  # pragma: no cover
    from src.analysis_pipeline import AnalysisResult

log = settings.get_logger("reports")


class ExportError(RuntimeError):
    pass


CSV_COLUMNS = EVENT_COLUMNS + ["start_time_formatted", "end_time_formatted"]

LIMITATIONS_TEXT = (
    "Automatic note detection is an estimate. Monophonic mode (pYIN) reports one pitch at a time and cannot "
    "describe chords. Polyphonic and full-mix modes (Spotify Basic Pitch) may miss soft notes, add spurious "
    "harmonics, and make octave errors; accuracy drops for vocals, percussion, dense arrangements and heavy "
    "effects. Confidence values are model activations (Basic Pitch) or mean voicing probabilities (pYIN), not "
    "calibrated probabilities of correctness. Timestamps refer to the original recording timeline; the model's "
    "frame resolution is about 11.6 ms (Basic Pitch) or hop/sample-rate (pYIN). Tempo is a global estimate."
)


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def events_csv_bytes(events_df: pd.DataFrame, include_invalid: bool = False) -> bytes:
    try:
        df = events_df.copy()
        if not include_invalid and "is_valid" in df and not df.empty:
            df = df[df["is_valid"]]
        for col in CSV_COLUMNS:
            if col not in df:
                df[col] = None
        df = df[CSV_COLUMNS]
        return df.to_csv(index=False, lineterminator="\n").encode("utf-8")
    except Exception as exc:  # noqa: BLE001
        raise ExportError(f"CSV export failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Excel
# --------------------------------------------------------------------------- #
def _metadata_rows(result: "AnalysisResult") -> list[dict[str, Any]]:
    m = result.metadata
    rows = [
        ("Filename", m.filename), ("Song ID", result.song_id), ("Analysis ID", result.analysis_id),
        ("File size (MB)", round(m.file_size_mb, 3)), ("Container format", m.container_format), ("Subtype / codec", m.subtype),
        ("Sample rate (Hz)", m.sample_rate), ("Channels", m.channels), ("Duration (s)", round(m.duration_seconds, 4)),
        ("Duration (MM:SS.mmm)", format_timestamp(m.duration_seconds)), ("Frames", m.frames), ("SHA-256", m.file_hash),
        ("Probed with", m.probed_with), ("Analysis mode", result.transcription_config.mode),
        ("Model", result.model_info.get("model_name")), ("Model version", result.model_info.get("model_version")),
        ("Estimated tempo (BPM)", None if result.tempo_bpm is None else round(result.tempo_bpm, 1)),
        ("Analysed at (UTC)", result.created_at), ("Status", result.status),
    ]
    if result.level_stats:
        rows.append(("Peak level (dBFS)", round(result.level_stats.get("peak_dbfs", float("nan")), 2)))
        rows.append(("RMS level (dBFS)", round(result.level_stats.get("rms_dbfs", float("nan")), 2)))
    return [{"Field": k, "Value": v} for k, v in rows]


def _summary_rows(result: "AnalysisResult") -> list[dict[str, Any]]:
    s = result.statistics
    pairs = [
        ("Total detected note events", s.total_events), ("Valid events used in statistics", s.valid_events),
        ("Excluded invalid / filtered events", s.excluded_invalid_events), ("Unique MIDI pitches", s.unique_midi_pitches),
        ("Unique pitch classes", s.unique_pitch_classes), ("Pitch range", f"{s.pitch_range_low} - {s.pitch_range_high}" if s.pitch_range_low else None),
        ("Most frequent note", f"{s.most_frequent_note} ({s.most_frequent_note_count} events)" if s.most_frequent_note else None),
        ("Least frequent note", f"{s.least_frequent_note} ({s.least_frequent_note_count} events)" if s.least_frequent_note else None),
        ("Note with the longest total duration", s.longest_total_duration_note),
        ("Total note duration (s)", round(s.total_note_duration_seconds, 4)),
        ("Shortest event (s)", s.min_event_duration_seconds), ("Longest event (s)", s.max_event_duration_seconds),
        ("Mean event duration (s)", s.mean_event_duration_seconds), ("Median event duration (s)", s.median_event_duration_seconds),
        ("Maximum simultaneous notes", s.max_polyphony), ("Mean simultaneous notes (while sounding)", round(s.mean_polyphony, 3)),
        ("Events overlapping another event", s.overlapping_events),
        ("Unusually short events (1.5 IQR)", s.short_outliers), ("Unusually long events (1.5 IQR)", s.long_outliers),
    ]
    return [{"Statistic": k, "Value": (round(v, 4) if isinstance(v, float) else v)} for k, v in pairs]


def excel_bytes(result: "AnalysisResult") -> bytes:
    """Workbook with: Song Metadata, Note Events, Note Statistics, Pitch-Class Statistics, Octave Statistics,
    Processing Information (+ Analysis Summary and Excluded Events)."""
    try:
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter

        buf = io.BytesIO()
        events = result.events_df.copy()
        if "is_valid" in events and not events.empty:
            valid = events[events["is_valid"]][CSV_COLUMNS]
        else:
            valid = events.reindex(columns=CSV_COLUMNS)
        excluded = pd.DataFrame([e.to_dict() for e in result.excluded_events], columns=EVENT_COLUMNS)
        proc = pd.DataFrame([{"Setting": k, "Value": _excel_safe(v)} for k, v in result.processing_info().items()])

        sheets: list[tuple[str, pd.DataFrame]] = [
            ("Song Metadata", pd.DataFrame(_metadata_rows(result))),
            ("Analysis Summary", pd.DataFrame(_summary_rows(result))),
            ("Note Events", valid),
            ("Note Statistics", result.statistics.note_table),
            ("Pitch-Class Statistics", result.statistics.pitch_class_table),
            ("Octave Statistics", result.statistics.octave_table),
            ("Processing Information", proc),
            ("Excluded Events", excluded),
        ]
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            for name, df in sheets:
                df = df.copy()
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].map(_excel_safe)
                df.to_excel(writer, sheet_name=name, index=False)
                ws = writer.sheets[name]
                header_fill = PatternFill("solid", fgColor="2A78D6")
                for cell in ws[1]:
                    cell.font = Font(bold=True, color="FFFFFF")
                    cell.fill = header_fill
                    cell.alignment = Alignment(vertical="center")
                ws.freeze_panes = "A2"
                for i, col in enumerate(df.columns, start=1):
                    width = max([len(str(col))] + [len(str(v)) for v in df[col].head(200).tolist()]) if len(df) else len(str(col))
                    ws.column_dimensions[get_column_letter(i)].width = min(max(10, width + 2), 60)
            # Limitations note on the summary sheet
            ws = writer.sheets["Analysis Summary"]
            ws.cell(row=ws.max_row + 2, column=1, value="Limitations").font = Font(bold=True)
            ws.cell(row=ws.max_row + 1, column=1, value=LIMITATIONS_TEXT).alignment = Alignment(wrap_text=True, vertical="top")
            ws.column_dimensions["A"].width = 48
            ws.column_dimensions["B"].width = 40
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        raise ExportError(f"Excel export failed: {exc}") from exc


def _excel_safe(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return str(value)
    if isinstance(value, bool):
        return "yes" if value else "no"
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, float) and value != value:
        return None
    return value


# --------------------------------------------------------------------------- #
# PDF
# --------------------------------------------------------------------------- #
def pdf_bytes(result: "AnalysisResult", max_event_rows: int = 600, include_charts: bool = True) -> bytes:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (
            Image,
            KeepTogether,
            PageBreak,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        from src import visualization as viz

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=16 * mm,
                                bottomMargin=16 * mm, title=f"NoteVision AI report - {result.metadata.filename}",
                                author=settings.APP_NAME)
        styles = getSampleStyleSheet()
        h1 = ParagraphStyle("h1", parent=styles["Title"], alignment=TA_LEFT, fontSize=18, spaceAfter=4)
        h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#0b0b0b"))
        body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9, leading=12)
        small = ParagraphStyle("small", parent=body, fontSize=7.5, leading=9.5, textColor=colors.HexColor("#52514e"))
        cell = ParagraphStyle("cell", parent=body, fontSize=7.5, leading=9)

        def table(rows: list[list[Any]], col_widths: list[float] | None = None, header: bool = True, font_size: float = 7.5) -> Table:
            data = [[Paragraph(str("" if v is None else v), cell) if isinstance(v, str) and len(v) > 40 else ("" if v is None else v)
                     for v in row] for row in rows]
            t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
            style = [
                ("FONTSIZE", (0, 0), (-1, -1), font_size), ("LEADING", (0, 0), (-1, -1), font_size + 2),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#e8e7e3")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f5")]),
            ]
            if header:
                style += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2a78d6")),
                          ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
            t.setStyle(TableStyle(style))
            return t

        s = result.statistics
        m = result.metadata
        story: list[Any] = [
            Paragraph(f"{settings.APP_NAME} - Note Analysis Report", h1),
            Paragraph(f"<b>{m.filename}</b> &nbsp;|&nbsp; generated {datetime.now().strftime('%Y-%m-%d %H:%M')} "
                      f"&nbsp;|&nbsp; {settings.APP_NAME} {settings.APP_VERSION}", small),
            Spacer(1, 6),
            Paragraph("1. Song metadata", h2),
            table([["Field", "Value"]] + [[r["Field"], r["Value"]] for r in _metadata_rows(result)], col_widths=[55 * mm, 120 * mm]),
            Paragraph("2. Analysis summary", h2),
            table([["Statistic", "Value"]] + [[r["Statistic"], r["Value"]] for r in _summary_rows(result)], col_widths=[85 * mm, 90 * mm]),
        ]
        if result.warnings:
            story.append(Paragraph("Notes from the analysis", h2))
            for w in result.warnings:
                story.append(Paragraph(f"&bull; {w}", body))

        if include_charts and not result.events_df.empty:
            story.append(PageBreak())
            story.append(Paragraph("3. Charts", h2))
            valid_df = result.events_df[result.events_df["is_valid"]] if "is_valid" in result.events_df else result.events_df
            charts = [
                viz.static_piano_roll_png(valid_df, m.duration_seconds),
                viz.static_note_frequency_png(s.note_table),
                viz.static_pitch_class_png(s.pitch_class_table),
                viz.static_duration_histogram_png(valid_df, s.mean_event_duration_seconds, s.median_event_duration_seconds),
                viz.static_polyphony_png(s.polyphony.get("times", []), s.polyphony.get("counts", [])),
            ]
            for png in charts:
                img = Image(io.BytesIO(png))
                ratio = img.imageHeight / float(img.imageWidth)
                img.drawWidth = 175 * mm
                img.drawHeight = 175 * mm * ratio
                story.append(KeepTogether([img, Spacer(1, 4)]))

        story.append(PageBreak())
        story.append(Paragraph("4. Note occurrence statistics", h2))
        nt = s.note_table
        if nt.empty:
            story.append(Paragraph("No valid note events were detected.", body))
        else:
            rows = [["Note", "MIDI", "Count", "% events", "Total dur (s)", "Mean (s)", "Median (s)", "Min (s)", "Max (s)", "First at"]]
            for r in nt.sort_values("midi_note").itertuples():
                rows.append([r.note_name, r.midi_note, r.count, f"{r.percent_of_events:.1f}", f"{r.total_duration_seconds:.3f}",
                             f"{r.mean_duration_seconds:.3f}", f"{r.median_duration_seconds:.3f}", f"{r.min_duration_seconds:.3f}",
                             f"{r.max_duration_seconds:.3f}", format_timestamp(r.first_occurrence_seconds)])
            story.append(table(rows))

        story.append(Paragraph("5. Pitch-class and octave statistics", h2))
        pc_rows = [["Pitch class", "Count", "% events", "Total dur (s)"]] + [
            [r.pitch_class, int(r.count), f"{r.percent_of_events:.1f}", f"{r.total_duration_seconds:.3f}"]
            for r in s.pitch_class_table.itertuples()]
        oc_rows = [["Octave", "Count", "% events", "Total dur (s)", "Notes"]] + [
            [int(r.octave), int(r.count), f"{r.percent_of_events:.1f}", f"{r.total_duration_seconds:.3f}", r.notes]
            for r in s.octave_table.itertuples()]
        story.append(table(pc_rows, col_widths=[30 * mm, 25 * mm, 25 * mm, 30 * mm]))
        story.append(Spacer(1, 6))
        story.append(table(oc_rows, col_widths=[20 * mm, 20 * mm, 22 * mm, 28 * mm, 85 * mm]))

        story.append(PageBreak())
        story.append(Paragraph("6. Detailed event table", h2))
        df = result.events_df
        df = df[df["is_valid"]] if "is_valid" in df and not df.empty else df
        if df.empty:
            story.append(Paragraph("No events.", body))
        else:
            if len(df) > max_event_rows:
                story.append(Paragraph(f"Showing the first {max_event_rows} of {len(df)} events (sorted by start time). "
                                       f"The CSV / Excel exports contain every event.", small))
            rows = [["Event ID", "Note", "MIDI", "Freq (Hz)", "Start", "End", "Dur (s)", "Conf.", "Overlaps"]]
            for r in df.head(max_event_rows).itertuples():
                rows.append([r.event_id, r.note_name, int(r.midi_note), f"{r.frequency_hz:.2f}", format_timestamp(r.start_time_seconds),
                             format_timestamp(r.end_time_seconds), f"{r.duration_seconds:.3f}",
                             "n/a" if pd.isna(r.confidence) else f"{r.confidence:.2f}", int(r.overlap_count)])
            story.append(table(rows, col_widths=[26 * mm, 13 * mm, 12 * mm, 20 * mm, 22 * mm, 22 * mm, 16 * mm, 14 * mm, 17 * mm]))

        story.append(Paragraph("7. Processing information and limitations", h2))
        proc_rows = [["Setting", "Value"]] + [[k, str(v)] for k, v in result.processing_info().items()]
        story.append(table(proc_rows, col_widths=[70 * mm, 105 * mm], font_size=7))
        story.append(Spacer(1, 6))
        story.append(Paragraph(LIMITATIONS_TEXT, small))

        def _footer(canvas, doc_):  # noqa: ANN001
            canvas.saveState()
            canvas.setFont("Helvetica", 7)
            canvas.setFillColor(colors.HexColor("#52514e"))
            canvas.drawString(16 * mm, 10 * mm, f"{settings.APP_NAME} {settings.APP_VERSION} - {m.filename}")
            canvas.drawRightString(A4[0] - 16 * mm, 10 * mm, f"Page {doc_.page}")
            canvas.restoreState()

        doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
        return buf.getvalue()
    except Exception as exc:  # noqa: BLE001
        raise ExportError(f"PDF export failed: {exc}") from exc


# --------------------------------------------------------------------------- #
# Convenience: write to the exports folder
# --------------------------------------------------------------------------- #
def export_all(result: "AnalysisResult", directory: str | Path | None = None) -> dict[str, Path]:
    """Write CSV, Excel and PDF for a result; returns the paths. Partial failures are reported, not hidden."""
    directory = Path(directory) if directory else settings.EXPORTS_DIR
    directory.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    errors: list[str] = []
    for kind, ext, fn in (("note_events", "csv", lambda: events_csv_bytes(result.events_df)),
                          ("report", "xlsx", lambda: excel_bytes(result)),
                          ("report", "pdf", lambda: pdf_bytes(result))):
        try:
            path = directory / make_export_filename(result.metadata.filename, kind, ext)
            path.write_bytes(fn())
            outputs[ext] = path
        except ExportError as exc:
            errors.append(str(exc))
            log.error("%s", exc)
    if errors and not outputs:
        raise ExportError("; ".join(errors))
    if errors:
        log.warning("Some exports failed: %s", "; ".join(errors))
    return outputs
