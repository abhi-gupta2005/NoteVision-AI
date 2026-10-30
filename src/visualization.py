"""
Interactive (Plotly) and static (Matplotlib) visualisations.

All figures are built from real analysis results passed in as arrays or
DataFrames; nothing here fabricates data. The look follows a small, fixed
design system: one accent hue for single-series charts, a fixed categorical
order for multi-series charts, a single-hue sequential ramp for magnitude
(spectrogram, confidence, similarity), thin marks, recessive gridlines and
hover tooltips everywhere.
"""
from __future__ import annotations

import io
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.pitch_detection import PITCH_CLASS_NAMES, midi_to_note_name
from src.utils import format_timestamp

# --------------------------------------------------------------------------- #
# Palette (validated categorical order; see dataviz reference palette)
# --------------------------------------------------------------------------- #
SERIES_COLORS: list[str] = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#6250d6",  # 7 violet
    "#e34948",  # 8 red
]
ACCENT = SERIES_COLORS[0]
MUTED = "#9c9b94"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#ffffff"
GRID = "#e8e7e3"
SEQUENTIAL_BLUE = [
    "#f4f8fe", "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQUENTIAL_SCALE = [[i / (len(SEQUENTIAL_BLUE) - 1), c] for i, c in enumerate(SEQUENTIAL_BLUE)]
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"

_BASE_LAYOUT: dict[str, Any] = dict(
    template="plotly_white",
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    font=dict(family="Inter, Segoe UI, Helvetica, Arial, sans-serif", size=13, color=TEXT_PRIMARY),
    margin=dict(l=60, r=24, t=56, b=52),
    hoverlabel=dict(bgcolor=SURFACE, font=dict(color=TEXT_PRIMARY, size=12), bordercolor=GRID),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=TEXT_SECONDARY)),
)
_AXIS = dict(
    showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=GRID,
    tickfont=dict(color=TEXT_SECONDARY), title_font=dict(color=TEXT_SECONDARY),
)


def _layout(fig: go.Figure, title: str, xaxis_title: str, yaxis_title: str, **kwargs: Any) -> go.Figure:
    fig.update_layout(title=dict(text=title, font=dict(size=16, color=TEXT_PRIMARY), x=0), **_BASE_LAYOUT, **kwargs)
    fig.update_xaxes(title_text=xaxis_title, **_AXIS)
    fig.update_yaxes(title_text=yaxis_title, **_AXIS)
    return fig


def empty_figure(message: str = "No data to display") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=TEXT_SECONDARY, size=14))
    fig.update_layout(**_BASE_LAYOUT, xaxis=dict(visible=False), yaxis=dict(visible=False), height=260)
    return fig


# --------------------------------------------------------------------------- #
# Waveform
# --------------------------------------------------------------------------- #
def _minmax_envelope(y: np.ndarray, sr: int, max_points: int = 4000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = y.shape[0]
    if n <= max_points:
        t = np.arange(n) / sr
        return t, y, y
    block = int(np.ceil(n / max_points))
    n_blocks = int(np.ceil(n / block))
    padded = np.full(n_blocks * block, np.nan, dtype=np.float32)
    padded[:n] = y
    blocks = padded.reshape(n_blocks, block)
    lo = np.nanmin(blocks, axis=1)
    hi = np.nanmax(blocks, axis=1)
    t = (np.arange(n_blocks) * block + block / 2) / sr
    return t, lo, hi


def waveform_figure(
    y: np.ndarray,
    sr: int,
    title: str = "Waveform",
    onset_times: Sequence[float] | None = None,
    time_offset: float = 0.0,
    max_points: int = 4000,
) -> go.Figure:
    """Amplitude vs. time with a range slider for inspecting any portion."""
    if y.size == 0:
        return empty_figure("Empty signal")
    t, lo, hi = _minmax_envelope(y, sr, max_points)
    t = t + time_offset
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=hi, mode="lines", line=dict(width=1, color=ACCENT), name="peak +",
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=t, y=lo, mode="lines", line=dict(width=1, color=ACCENT), fill="tonexty",
                             fillcolor="rgba(42,120,214,0.35)", name="amplitude",
                             customdata=np.stack([hi, lo], axis=1),
                             hovertemplate="t = %{x:.3f} s<br>peak +%{customdata[0]:.3f} / %{customdata[1]:.3f}<extra></extra>",
                             showlegend=False))
    if onset_times:
        for ot in onset_times[:400]:  # keep the figure light
            fig.add_vline(x=ot, line=dict(color=MUTED, width=1, dash="dot"), opacity=0.5)
    _layout(fig, title, "Time (s)", "Amplitude")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.08), range=[t[0], t[-1]])
    fig.update_yaxes(range=[-1.05, 1.05])
    fig.update_layout(height=340, hovermode="x")
    return fig


# --------------------------------------------------------------------------- #
# Spectrogram
# --------------------------------------------------------------------------- #
def spectrogram_figure(
    y: np.ndarray,
    sr: int,
    n_fft: int = 2048,
    hop_length: int = 512,
    fmax: float | None = 8000.0,
    log_frequency: bool = False,
    time_offset: float = 0.0,
    max_time_bins: int = 1200,
    max_freq_bins: int = 400,
    title: str = "Spectrogram (STFT magnitude, dB)",
) -> go.Figure:
    """Time-frequency heatmap in dB relative to the peak magnitude."""
    import librosa

    if y.size < n_fft:
        return empty_figure("Signal too short for a spectrogram")
    stft = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    db = librosa.amplitude_to_db(stft, ref=np.max, top_db=80.0)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(db.shape[1]), sr=sr, hop_length=hop_length) + time_offset

    if fmax is not None:
        keep = freqs <= fmax
        db, freqs = db[keep], freqs[keep]
    if log_frequency:
        keep = freqs >= 20.0
        db, freqs = db[keep], freqs[keep]

    # Down-sample for the browser: average over blocks (dB domain is fine for display).
    if db.shape[1] > max_time_bins:
        step = int(np.ceil(db.shape[1] / max_time_bins))
        n = (db.shape[1] // step) * step
        db = db[:, :n].reshape(db.shape[0], -1, step).mean(axis=2)
        times = times[:n:step]
    if db.shape[0] > max_freq_bins and not log_frequency:
        step = int(np.ceil(db.shape[0] / max_freq_bins))
        n = (db.shape[0] // step) * step
        db = db[:n].reshape(-1, step, db.shape[1]).mean(axis=1)
        freqs = freqs[:n:step]

    fig = go.Figure(
        go.Heatmap(
            z=db, x=times, y=freqs, colorscale=SEQUENTIAL_SCALE, zmin=-80, zmax=0,
            colorbar=dict(title="dB", thickness=12, tickfont=dict(color=TEXT_SECONDARY)),
            hovertemplate="t = %{x:.3f} s<br>f = %{y:.0f} Hz<br>%{z:.1f} dB<extra></extra>",
        )
    )
    _layout(fig, title, "Time (s)", "Frequency (Hz)")
    if log_frequency:
        fig.update_yaxes(type="log")
    fig.update_layout(height=420)
    return fig


# --------------------------------------------------------------------------- #
# Piano roll
# --------------------------------------------------------------------------- #
def _hover_customdata(df: pd.DataFrame) -> np.ndarray:
    conf = df["confidence"].map(lambda c: f"{c:.2f}" if pd.notna(c) else "n/a") if "confidence" in df else "n/a"
    return np.stack(
        [
            df["event_id"].astype(str).values,
            df["note_name"].astype(str).values,
            df["midi_note"].astype(int).values,
            df["frequency_hz"].astype(float).round(2).values,
            df["start_time_seconds"].map(format_timestamp).values,
            df["end_time_seconds"].map(format_timestamp).values,
            df["duration_seconds"].astype(float).round(3).values,
            np.asarray(conf),
            df.get("source_track", pd.Series(["mix"] * len(df))).astype(str).values,
        ],
        axis=1,
    )


_HOVER_TEMPLATE = (
    "<b>%{customdata[1]}</b> (MIDI %{customdata[2]}, %{customdata[3]} Hz)<br>"
    "start %{customdata[4]} - end %{customdata[5]}<br>"
    "duration %{customdata[6]} s - confidence %{customdata[7]}<br>"
    "track %{customdata[8]} - id %{customdata[0]}<extra></extra>"
)


def piano_roll_figure(
    events: pd.DataFrame,
    title: str = "Piano roll",
    color_by: str = "none",
    duration_seconds: float | None = None,
    selected_event_id: str | None = None,
) -> go.Figure:
    """Each event is a rectangle: x from start to end, y at its MIDI pitch.

    Overlapping events are drawn as separate overlaid bars with a thin surface
    ring so simultaneous notes remain visible. `color_by` is "none" (single
    hue), "confidence" (sequential ramp) or "source_track" (categorical).
    """
    if events is None or events.empty:
        return empty_figure("No note events detected")
    df = events.sort_values("start_time_seconds").reset_index(drop=True)
    fig = go.Figure()
    base_marker = dict(line=dict(color=SURFACE, width=1))

    if color_by == "source_track" and "source_track" in df and df["source_track"].nunique() > 1:
        for i, (track, part) in enumerate(df.groupby("source_track", sort=True)):
            fig.add_trace(go.Bar(
                x=part["duration_seconds"], base=part["start_time_seconds"], y=part["midi_note"],
                orientation="h", width=0.8, name=str(track),
                marker=dict(color=SERIES_COLORS[i % len(SERIES_COLORS)], **base_marker),
                customdata=_hover_customdata(part), hovertemplate=_HOVER_TEMPLATE,
            ))
        fig.update_layout(showlegend=True)
    elif color_by == "confidence" and "confidence" in df and df["confidence"].notna().any():
        conf = df["confidence"].fillna(0.0).clip(0, 1)
        fig.add_trace(go.Bar(
            x=df["duration_seconds"], base=df["start_time_seconds"], y=df["midi_note"], orientation="h",
            width=0.8, name="events",
            marker=dict(color=conf, colorscale=SEQUENTIAL_SCALE, cmin=0, cmax=1,
                        colorbar=dict(title="confidence", thickness=12), **base_marker),
            customdata=_hover_customdata(df), hovertemplate=_HOVER_TEMPLATE, showlegend=False,
        ))
    else:
        fig.add_trace(go.Bar(
            x=df["duration_seconds"], base=df["start_time_seconds"], y=df["midi_note"], orientation="h",
            width=0.8, name="events", marker=dict(color=ACCENT, **base_marker),
            customdata=_hover_customdata(df), hovertemplate=_HOVER_TEMPLATE, showlegend=False,
        ))

    if selected_event_id is not None and (df["event_id"] == selected_event_id).any():
        sel = df[df["event_id"] == selected_event_id]
        fig.add_trace(go.Bar(
            x=sel["duration_seconds"], base=sel["start_time_seconds"], y=sel["midi_note"], orientation="h",
            width=0.95, name="selected", marker=dict(color="rgba(0,0,0,0)", line=dict(color=TEXT_PRIMARY, width=2)),
            hoverinfo="skip", showlegend=False,
        ))

    lo, hi = int(df["midi_note"].min()), int(df["midi_note"].max())
    lo, hi = max(0, lo - 1), min(127, hi + 1)
    span = hi - lo
    if span <= 40:
        tick_vals = list(range(lo, hi + 1))                       # every semitone
    elif span <= 72:
        tick_vals = [m for m in range(lo, hi + 1) if m % 12 in (0, 4, 7)]   # C, E, G of every octave
    else:
        tick_vals = [m for m in range(lo, hi + 1) if m % 12 == 0]
    tick_text = [midi_to_note_name(m) for m in tick_vals]
    _layout(fig, title, "Time (s)", "Pitch")
    fig.update_yaxes(tickvals=tick_vals, ticktext=tick_text, range=[lo - 0.5, hi + 0.5])
    x_end = duration_seconds if duration_seconds else float(df["end_time_seconds"].max())
    fig.update_xaxes(range=[0, x_end * 1.01])
    fig.update_layout(barmode="overlay", bargap=0.0, height=max(360, min(760, 18 * (span + 4))))
    return fig


# --------------------------------------------------------------------------- #
# Note frequency / distribution charts
# --------------------------------------------------------------------------- #
def note_frequency_figure(note_table: pd.DataFrame, title: str = "Note occurrence counts",
                          value_column: str = "count", label: str = "Occurrences") -> go.Figure:
    """Bar chart of occurrences per note (ordered by pitch), exact counts on each bar."""
    if note_table is None or note_table.empty:
        return empty_figure("No notes")
    df = note_table.sort_values("midi_note")
    hover = [
        f"<b>{n}</b> (MIDI {m})<br>{label}: {v}<br>total duration {d:.3f} s<br>{p:.1f}% of events"
        for n, m, v, d, p in zip(df["note_name"], df["midi_note"], df[value_column],
                                 df.get("total_duration_seconds", pd.Series([0] * len(df))),
                                 df.get("percent_of_events", pd.Series([0] * len(df))))
    ]
    fig = go.Figure(go.Bar(
        x=df["note_name"], y=df[value_column], marker=dict(color=ACCENT, line=dict(color=SURFACE, width=1)),
        text=df[value_column].map(lambda v: f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}"),
        textposition="outside", textfont=dict(color=TEXT_SECONDARY, size=11),
        hovertext=hover, hoverinfo="text", name=label,
    ))
    _layout(fig, title, "Note", label)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=list(df["note_name"]))
    fig.update_layout(height=380, bargap=0.35, showlegend=False)
    return fig


def pitch_class_figure(pc_table: pd.DataFrame, title: str = "Events per pitch class") -> go.Figure:
    if pc_table is None or pc_table.empty:
        return empty_figure("No notes")
    df = pc_table.set_index("pitch_class").reindex(PITCH_CLASS_NAMES).fillna(0).reset_index()
    fig = go.Figure(go.Bar(
        x=df["pitch_class"], y=df["count"], marker=dict(color=ACCENT, line=dict(color=SURFACE, width=1)),
        text=df["count"].astype(int), textposition="outside", textfont=dict(color=TEXT_SECONDARY, size=11),
        customdata=np.stack([df["total_duration_seconds"], df["percent_of_events"]], axis=1),
        hovertemplate="<b>%{x}</b><br>%{y} events<br>%{customdata[0]:.3f} s total<br>%{customdata[1]:.1f}%<extra></extra>",
    ))
    _layout(fig, title, "Pitch class", "Occurrences")
    fig.update_layout(height=340, bargap=0.35, showlegend=False)
    return fig


def octave_figure(octave_table: pd.DataFrame, title: str = "Events per octave") -> go.Figure:
    if octave_table is None or octave_table.empty:
        return empty_figure("No notes")
    df = octave_table.sort_values("octave")
    fig = go.Figure(go.Bar(
        x=df["octave"].astype(str), y=df["count"], marker=dict(color=ACCENT, line=dict(color=SURFACE, width=1)),
        text=df["count"].astype(int), textposition="outside", textfont=dict(color=TEXT_SECONDARY, size=11),
        customdata=np.stack([df["total_duration_seconds"], df["percent_of_events"]], axis=1),
        hovertemplate="Octave <b>%{x}</b><br>%{y} events<br>%{customdata[0]:.3f} s total<br>%{customdata[1]:.1f}%<extra></extra>",
    ))
    _layout(fig, title, "Octave", "Occurrences")
    fig.update_xaxes(type="category")
    fig.update_layout(height=340, bargap=0.35, showlegend=False)
    return fig


# --------------------------------------------------------------------------- #
# Duration analysis
# --------------------------------------------------------------------------- #
def duration_distribution_figure(events: pd.DataFrame, mean: float | None = None, median: float | None = None,
                                 outlier_bounds: tuple[float, float] | None = None,
                                 title: str = "Note duration distribution") -> go.Figure:
    if events is None or events.empty:
        return empty_figure("No note events")
    d = events["duration_seconds"].astype(float)
    fig = go.Figure(go.Histogram(
        x=d, nbinsx=min(60, max(10, int(np.sqrt(len(d)) * 2))), marker=dict(color=ACCENT, line=dict(color=SURFACE, width=1)),
        hovertemplate="%{x} s<br>%{y} events<extra></extra>", name="events", showlegend=False,
    ))
    if mean is not None and np.isfinite(mean):
        fig.add_vline(x=mean, line=dict(color=SERIES_COLORS[1], width=2),
                      annotation_text=f"mean {mean:.3f} s", annotation_position="top right",
                      annotation_font=dict(color=TEXT_SECONDARY))
    if median is not None and np.isfinite(median):
        fig.add_vline(x=median, line=dict(color=SERIES_COLORS[2], width=2, dash="dash"),
                      annotation_text=f"median {median:.3f} s", annotation_position="top left",
                      annotation_font=dict(color=TEXT_SECONDARY))
    if outlier_bounds is not None:
        lo, hi = outlier_bounds
        fig.add_vrect(x0=lo, x1=hi, fillcolor="rgba(42,120,214,0.06)", line_width=0,
                      annotation_text="typical range", annotation_position="bottom left",
                      annotation_font=dict(color=MUTED, size=11))
    _layout(fig, title, "Duration (s)", "Number of events")
    fig.update_layout(height=360, bargap=0.05)
    return fig


def duration_by_note_figure(note_table: pd.DataFrame, title: str = "Average vs. median duration per note") -> go.Figure:
    if note_table is None or note_table.empty:
        return empty_figure("No notes")
    df = note_table.sort_values("midi_note")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["note_name"], y=df["mean_duration_seconds"], name="Average",
                         marker=dict(color=SERIES_COLORS[0], line=dict(color=SURFACE, width=1)),
                         hovertemplate="<b>%{x}</b><br>average %{y:.3f} s<extra></extra>"))
    fig.add_trace(go.Bar(x=df["note_name"], y=df["median_duration_seconds"], name="Median",
                         marker=dict(color=SERIES_COLORS[1], line=dict(color=SURFACE, width=1)),
                         hovertemplate="<b>%{x}</b><br>median %{y:.3f} s<extra></extra>"))
    _layout(fig, title, "Note", "Duration (s)")
    fig.update_xaxes(type="category")
    fig.update_layout(barmode="group", bargap=0.3, height=380)
    return fig


# --------------------------------------------------------------------------- #
# Timeline & polyphony
# --------------------------------------------------------------------------- #
def timeline_figure(events: pd.DataFrame, title: str = "Event timeline", selected_event_id: str | None = None) -> go.Figure:
    """Chronological scatter of events (x=start, y=pitch); marker size ~ duration."""
    if events is None or events.empty:
        return empty_figure("No note events")
    df = events.sort_values("start_time_seconds")
    size = 8 + 24 * (df["duration_seconds"] / max(float(df["duration_seconds"].max()), 1e-6))
    fig = go.Figure(go.Scatter(
        x=df["start_time_seconds"], y=df["midi_note"], mode="markers",
        marker=dict(size=size, color=ACCENT, opacity=0.8, line=dict(color=SURFACE, width=2)),
        customdata=_hover_customdata(df), hovertemplate=_HOVER_TEMPLATE, name="events", showlegend=False,
    ))
    if selected_event_id is not None and (df["event_id"] == selected_event_id).any():
        sel = df[df["event_id"] == selected_event_id]
        fig.add_trace(go.Scatter(x=sel["start_time_seconds"], y=sel["midi_note"], mode="markers",
                                 marker=dict(size=22, color="rgba(0,0,0,0)", line=dict(color=TEXT_PRIMARY, width=2)),
                                 hoverinfo="skip", showlegend=False))
    lo, hi = int(df["midi_note"].min()) - 1, int(df["midi_note"].max()) + 1
    span = hi - lo
    tick_vals = list(range(lo, hi + 1)) if span <= 30 else [m for m in range(lo, hi + 1) if m % 12 == 0]
    _layout(fig, title, "Start time (s)", "Pitch")
    fig.update_yaxes(tickvals=tick_vals, ticktext=[midi_to_note_name(m) for m in tick_vals])
    fig.update_layout(height=380)
    return fig


def polyphony_figure(times: Sequence[float], counts: Sequence[int], title: str = "Simultaneous notes over time") -> go.Figure:
    if len(times) == 0:
        return empty_figure("No note events")
    fig = go.Figure(go.Scatter(
        x=list(times), y=list(counts), mode="lines", line=dict(color=ACCENT, width=2, shape="hv"),
        fill="tozeroy", fillcolor="rgba(42,120,214,0.10)",
        hovertemplate="t = %{x:.2f} s<br>%{y} simultaneous notes<extra></extra>", name="polyphony", showlegend=False,
    ))
    _layout(fig, title, "Time (s)", "Simultaneous notes")
    fig.update_yaxes(rangemode="tozero", dtick=1)
    fig.update_layout(height=280, hovermode="x")
    return fig


def timeline_distribution_figure(bin_starts: Sequence[float], bin_width: float, counts: Sequence[int],
                                 title: str = "Note events across the song timeline") -> go.Figure:
    if len(bin_starts) == 0:
        return empty_figure("No note events")
    centers = np.asarray(bin_starts) + bin_width / 2
    fig = go.Figure(go.Bar(
        x=centers, y=list(counts), width=bin_width * 0.9, marker=dict(color=ACCENT, line=dict(color=SURFACE, width=1)),
        customdata=np.stack([np.asarray(bin_starts), np.asarray(bin_starts) + bin_width], axis=1),
        hovertemplate="%{customdata[0]:.2f}-%{customdata[1]:.2f} s<br>%{y} events start here<extra></extra>",
        name="events", showlegend=False,
    ))
    _layout(fig, title, "Time (s)", "Events starting in bin")
    fig.update_layout(height=280, bargap=0.1)
    return fig


# --------------------------------------------------------------------------- #
# Pitch track (monophonic mode diagnostics)
# --------------------------------------------------------------------------- #
def pitch_track_figure(times: np.ndarray, f0_hz: np.ndarray, voiced_prob: np.ndarray | None = None,
                       time_offset: float = 0.0, title: str = "Estimated fundamental frequency (pYIN)") -> go.Figure:
    if times is None or len(times) == 0:
        return empty_figure("No pitch track")
    midi = 69 + 12 * np.log2(np.where(f0_hz > 0, f0_hz, np.nan) / 440.0)
    custom = np.stack([np.nan_to_num(f0_hz), voiced_prob if voiced_prob is not None else np.zeros_like(f0_hz)], axis=1)
    fig = go.Figure(go.Scatter(
        x=times + time_offset, y=midi, mode="lines", line=dict(color=ACCENT, width=2), connectgaps=False,
        customdata=custom, hovertemplate="t = %{x:.3f} s<br>%{customdata[0]:.1f} Hz<br>voiced p = %{customdata[1]:.2f}<extra></extra>",
        name="f0", showlegend=False,
    ))
    finite = midi[np.isfinite(midi)]
    if finite.size:
        lo, hi = int(np.floor(finite.min())) - 1, int(np.ceil(finite.max())) + 1
        tick_vals = list(range(lo, hi + 1)) if hi - lo <= 30 else [m for m in range(lo, hi + 1) if m % 12 == 0]
        fig.update_yaxes(tickvals=tick_vals, ticktext=[midi_to_note_name(m) for m in tick_vals])
    _layout(fig, title, "Time (s)", "Pitch")
    fig.update_layout(height=320, hovermode="x")
    return fig


# --------------------------------------------------------------------------- #
# Comparison charts (multi-song)
# --------------------------------------------------------------------------- #
def comparison_distribution_figure(distributions: pd.DataFrame, title: str, xaxis_title: str,
                                   normalized: bool = True) -> go.Figure:
    """Grouped bars: one series per song (rows = songs, columns = categories)."""
    if distributions is None or distributions.empty:
        return empty_figure("Nothing to compare")
    fig = go.Figure()
    for i, (song, row) in enumerate(distributions.iterrows()):
        if i >= len(SERIES_COLORS):
            break  # never generate a 9th hue; the table view carries the rest
        fig.add_trace(go.Bar(
            x=[str(c) for c in distributions.columns], y=row.values, name=str(song)[:40],
            marker=dict(color=SERIES_COLORS[i], line=dict(color=SURFACE, width=1)),
            hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.1%}<extra></extra>" if normalized
            else "<b>%{fullData.name}</b><br>%{x}: %{y}<extra></extra>",
        ))
    _layout(fig, title, xaxis_title, "Share of events" if normalized else "Events")
    fig.update_xaxes(type="category")
    if normalized:
        fig.update_yaxes(tickformat=".0%")
    fig.update_layout(barmode="group", height=400, showlegend=True)
    return fig


def similarity_heatmap_figure(matrix: pd.DataFrame, title: str = "Pitch-class distribution similarity (cosine)") -> go.Figure:
    if matrix is None or matrix.empty:
        return empty_figure("Nothing to compare")
    labels = [str(c)[:28] for c in matrix.columns]
    fig = go.Figure(go.Heatmap(
        z=matrix.values, x=labels, y=labels, colorscale=SEQUENTIAL_SCALE, zmin=0, zmax=1,
        text=np.round(matrix.values, 2), texttemplate="%{text}", textfont=dict(size=12),
        colorbar=dict(title="similarity", thickness=12),
        hovertemplate="%{y} vs %{x}<br>similarity %{z:.3f}<extra></extra>",
    ))
    _layout(fig, title, "", "")
    fig.update_layout(height=max(320, 60 * len(labels) + 120))
    return fig


def comparison_metric_figure(summary: pd.DataFrame, metric: str, title: str, yaxis_title: str) -> go.Figure:
    """One bar per song for a single metric (single series -> one hue)."""
    if summary is None or summary.empty or metric not in summary:
        return empty_figure("Nothing to compare")
    fig = go.Figure(go.Bar(
        x=summary["song"].astype(str).str.slice(0, 40), y=summary[metric],
        marker=dict(color=ACCENT, line=dict(color=SURFACE, width=1)),
        text=summary[metric].map(lambda v: f"{v:.3g}"), textposition="outside", textfont=dict(color=TEXT_SECONDARY),
        hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>", showlegend=False,
    ))
    _layout(fig, title, "Song", yaxis_title)
    fig.update_xaxes(type="category")
    fig.update_layout(height=340, bargap=0.4)
    return fig


# --------------------------------------------------------------------------- #
# Static (Matplotlib) renderers for PDF reports
# --------------------------------------------------------------------------- #
def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY, "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
    })
    return plt


def _integer_yaxis(ax) -> None:
    from matplotlib.ticker import MaxNLocator

    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def _fig_to_png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=SURFACE)
    buf.seek(0)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return buf.getvalue()


def static_piano_roll_png(events: pd.DataFrame, duration_seconds: float | None = None) -> bytes:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    if events is not None and not events.empty:
        ax.barh(events["midi_note"], events["duration_seconds"], left=events["start_time_seconds"], height=0.8,
                color=ACCENT, edgecolor=SURFACE, linewidth=0.5)
        lo, hi = int(events["midi_note"].min()) - 1, int(events["midi_note"].max()) + 1
        ticks = list(range(lo, hi + 1)) if hi - lo <= 24 else [m for m in range(lo, hi + 1) if m % 12 == 0]
        ax.set_yticks(ticks)
        ax.set_yticklabels([midi_to_note_name(m) for m in ticks])
        ax.set_ylim(lo - 0.5, hi + 0.5)
        ax.set_xlim(0, duration_seconds or float(events["end_time_seconds"].max()))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pitch")
    ax.set_title("Piano roll", loc="left", color=TEXT_PRIMARY)
    return _fig_to_png(fig)


def static_note_frequency_png(note_table: pd.DataFrame) -> bytes:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    if note_table is not None and not note_table.empty:
        df = note_table.sort_values("midi_note")
        bars = ax.bar(df["note_name"], df["count"], color=ACCENT, edgecolor=SURFACE, linewidth=0.5)
        ax.bar_label(bars, fmt="%d", fontsize=7, color=TEXT_SECONDARY, padding=2)
        ax.tick_params(axis="x", labelrotation=90 if len(df) > 20 else 0)
    ax.set_xlabel("Note")
    ax.set_ylabel("Occurrences")
    ax.set_title("Note occurrence counts", loc="left", color=TEXT_PRIMARY)
    ax.grid(axis="x", visible=False)
    _integer_yaxis(ax)
    return _fig_to_png(fig)


def static_pitch_class_png(pc_table: pd.DataFrame) -> bytes:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    if pc_table is not None and not pc_table.empty:
        df = pc_table.set_index("pitch_class").reindex(PITCH_CLASS_NAMES).fillna(0)
        bars = ax.bar(df.index, df["count"], color=ACCENT, edgecolor=SURFACE, linewidth=0.5)
        ax.bar_label(bars, fmt="%d", fontsize=7, color=TEXT_SECONDARY, padding=2)
    ax.set_xlabel("Pitch class")
    ax.set_ylabel("Occurrences")
    ax.set_title("Events per pitch class", loc="left", color=TEXT_PRIMARY)
    ax.grid(axis="x", visible=False)
    _integer_yaxis(ax)
    return _fig_to_png(fig)


def static_duration_histogram_png(events: pd.DataFrame, mean: float | None = None, median: float | None = None) -> bytes:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    if events is not None and not events.empty:
        d = events["duration_seconds"].astype(float)
        ax.hist(d, bins=min(40, max(8, int(np.sqrt(len(d)) * 2))), color=ACCENT, edgecolor=SURFACE, linewidth=0.5)
        if mean is not None and np.isfinite(mean):
            ax.axvline(mean, color=SERIES_COLORS[1], linewidth=1.5, label=f"mean {mean:.3f} s")
        if median is not None and np.isfinite(median):
            ax.axvline(median, color=SERIES_COLORS[2], linewidth=1.5, linestyle="--", label=f"median {median:.3f} s")
        ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("Duration (s)")
    ax.set_ylabel("Events")
    ax.set_title("Note duration distribution", loc="left", color=TEXT_PRIMARY)
    _integer_yaxis(ax)
    return _fig_to_png(fig)


def static_polyphony_png(times: Sequence[float], counts: Sequence[int]) -> bytes:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 2.2))
    if len(times):
        ax.step(times, counts, where="post", color=ACCENT, linewidth=1.5)
        ax.fill_between(times, counts, step="post", color=ACCENT, alpha=0.10)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Simultaneous notes")
    ax.set_title("Polyphony over time", loc="left", color=TEXT_PRIMARY)
    _integer_yaxis(ax)
    return _fig_to_png(fig)
