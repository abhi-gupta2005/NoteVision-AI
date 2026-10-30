# NoteVision AI

[![tests](https://github.com/abhi-gupta2005/NoteVision-AI/actions/workflows/tests.yml/badge.svg)](https://github.com/abhi-gupta2005/NoteVision-AI/actions/workflows/tests.yml)
[![Python 3.10-3.14](https://img.shields.io/badge/python-3.10%20--%203.14-blue.svg)](#2-system-requirements)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://notevision-ai.streamlit.app)

**AI-powered musical note detection, timestamping, frequency analysis and statistical reporting.**

**Try it online:** <https://notevision-ai.streamlit.app> (free Streamlit Community Cloud instance -
no installation; see [hosted-version notes](#hosted-version-streamlit-community-cloud)).
**Run it locally:** clone the repo and run `scripts\setup.bat` (Windows) or `bash scripts/setup.sh`.

Upload one or more recordings, let the app transcribe them into timestamped note events, inspect
every note with its start/end time, duration, frequency and confidence, explore interactive
charts (waveform, spectrogram, piano roll, note counts, durations, timeline), compare several
songs, and download CSV, Excel and PDF reports. Everything is stored in a local SQLite database
so earlier analyses can be re-opened and exported again.

> **Disclaimer.** Automatic note detection is an estimate. It is good on clean monophonic
> material and solo harmonic instruments (piano, guitar), and considerably less reliable on
> full songs with vocals, percussion, effects and overlapping instruments. Always treat the
> results as a starting point and verify important findings by ear.

---

## Contents

1. [Features](#1-features)
2. [System requirements](#2-system-requirements)
3. [Installation](#3-installation)
4. [Running the application](#4-running-the-application)
5. [Supported formats](#5-supported-formats)
6. [Analysis modes](#6-analysis-modes)
7. [How the pipeline works](#7-how-the-pipeline-works)
8. [Output schema](#8-output-schema)
9. [Exports](#9-exports)
10. [Database and file storage](#10-database-and-file-storage)
11. [Testing](#11-testing)
12. [Project structure](#12-project-structure)
13. [Troubleshooting](#13-troubleshooting)
14. [Performance considerations](#14-performance-considerations)
15. [Known limitations](#15-known-limitations)
16. [Deploying your own copy](#16-deploying-your-own-copy)
17. [Contributing, credits and license](#17-contributing-credits-and-license)

---

## 1. Features

| Area | What you get |
|---|---|
| Upload | Single or multiple files, size/duration limits, corrupted-file detection, metadata (format, codec, sample rate, channels, duration, SHA-256), or analyse files already on disk by path |
| Preprocessing | Mono mix-down, resampling to 22.05 kHz, optional peak normalisation, optional light noise reduction, optional high-pass, optional silence trimming - every step is logged and the original timeline is preserved |
| Transcription | **Mode A** monophonic (pYIN + onset-aware segmentation), **Mode B** polyphonic (Spotify Basic Pitch, ONNX), **Mode C** full mix (Basic Pitch on the mix, optionally per Demucs stem) |
| Events | Note name, MIDI number, pitch class, octave, estimated f0 (Hz) with cents deviation, start/end/duration (s and MM:SS.mmm), confidence, overlap flag and count, source track, detection method |
| Statistics | Event counts vs. total durations per note / pitch class / octave, mean & median durations, min/max, most & least frequent notes, timeline distribution, simultaneous-note count over time, duration outliers, excluded-event count |
| Dashboard | Overview cards, waveform with range slider, spectrogram, piano roll, note-frequency charts, duration analysis, timeline with event inspector, filterable/sortable data table |
| Multi-song | Per-song results plus a comparison page: normalised pitch-class and octave distributions, cosine similarity, Jensen-Shannon distance, approximate note-sequence similarity |
| Exports | CSV (one row per event), Excel workbook (metadata, events, note/pitch-class/octave statistics, processing information), PDF report with charts and the event table |
| Persistence | SQLite database of songs, analyses, events and statistics; duplicate-safe saves; history page to re-open, re-export or delete |
| Engineering | Result caching, background processing with progress and cancellation, logging, validated model outputs, 149 automated tests on synthetic audio with known ground truth |

## 2. System requirements

* **Python 3.10 - 3.14** (the automated test-suite passes on 3.10, 3.11 and 3.14; 3.11/3.12 are the
  most battle-tested choices). Python 2.x interpreters that some scientific tools install
  (MGLTools, old Anaconda) are detected and skipped by the setup script.
* Windows 10/11, macOS or Linux. About 1 GB of disk for the virtual environment.
* **CPU only.** All inference runs on the CPU through `onnxruntime`; a GPU is neither required nor
  used. Rough speed on a laptop CPU: Basic Pitch transcribes a 3-minute song in 10-30 s;
  pYIN (monophonic mode) takes 1-2x the audio length. The first Basic Pitch run after start-up
  takes an extra 10-20 s while ONNX optimises the graph.
* RAM: 2 GB free is enough for 15-minute recordings.
* Optional: `ffmpeg` on the PATH for M4A/AAC input; Demucs + PyTorch for source separation
  (`requirements-optional.txt`, large download, slow on CPU).

## 3. Installation

Everything is a plain `pip install -r requirements.txt`. The Spotify Basic Pitch model ships
inside the repository as a 230 KB ONNX file (`models/basic_pitch/nmp.onnx`, Apache-2.0) and runs
through `onnxruntime`, so there is no TensorFlow and no separate model download.

```powershell
git clone https://github.com/abhi-gupta2005/NoteVision-AI.git
cd NoteVision-AI
```

### Windows - scripted (recommended)

Double-click **`scripts\setup.bat`** in Explorer, or from a terminal in the project folder:

```powershell
scripts\setup.bat
```

The `.bat` wrappers start PowerShell with `-ExecutionPolicy Bypass`, so they work even when
PowerShell scripts are disabled on your machine (the default on Windows). The script finds a
Python 3.10-3.14 interpreter (`py` launcher, `python3.x`, python.org, Microsoft Store or conda
locations), creates the virtual environment, installs everything, writes `.vscode\settings.json`
and verifies the install. To use a specific interpreter:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -PythonExe "C:\Path\To\python.exe"
```

If the project folder is inside OneDrive, the virtual environment is created in
`%LOCALAPPDATA%\NoteVision-AI\venv` instead of `.venv` (a venv is tens of thousands of small
files; keeping it out of the synced folder avoids OneDrive sync churn and locked-file errors).
`run.bat` / `test.bat` and the generated VS Code settings find it there automatically.

### Windows - manual

```powershell
cd <project folder>
py -3.12 -m venv .venv              # or: py -3.11 / python3.14 / "C:\Path\To\python.exe"
.\.venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip wheel
pip install -r requirements.txt
```

Make sure `python --version` inside the activated environment says 3.10-3.14. If plain
`python` on your machine is a different program (for example MGLTools' Python 2.7), always use
the `py -3.x` launcher or the full path to `python.exe` to create the environment.

### macOS / Linux

```bash
bash scripts/setup.sh
# or manually:
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel
pip install -r requirements.txt
```

### Conda

```powershell
conda create -n notevision python=3.12 -y
conda activate notevision
pip install -r requirements.txt
```

### Exact, verified versions

`requirements-lock-py311.txt` lists the exact versions the test-suite was verified with
(Python 3.11). Use it instead of `requirements.txt` for a fully reproducible environment.

### Verify

```powershell
python -c "from src.basic_pitch_onnx import is_available; print(is_available())"
# expected: (True, 'Spotify Basic Pitch (ICASSP 2022, nmp) via onnxruntime x.y.z (CPU)')
```

### Optional extras

```powershell
pip install -r requirements-optional.txt     # Demucs + PyTorch for source separation (Mode C)
```

## 4. Running the application

```powershell
scripts\run.bat                      # Windows: starts the app (no execution-policy issues)
# equivalent:  powershell -ExecutionPolicy Bypass -File scripts\run.ps1
# or, with the environment activated:  streamlit run app.py
```

The browser opens at <http://localhost:8501>. In VS Code you can also press **F5** with the
"Run NoteVision AI (Streamlit)" launch configuration (`.vscode/launch.json`; the setup scripts
create `.vscode/` from `vscode-config/` if it does not exist yet - see `vscode-config/README.md`).

Workflow:

1. **Upload & Analyze** - drop files (or add a local path), pick a mode, optionally open
   *Advanced settings*, click **Analyze**. Progress is shown per file; **Cancel** stops after the
   current step. Identical file + settings are served from the database instantly.
2. **Results Dashboard** - select a recording and move through the tabs.
3. **Compare Songs** - choose two or more analysed recordings.
4. **History** - re-open or delete stored analyses.
5. **Help & About** - explanations, limitations, installed versions.

## 5. Supported formats

| Extension | Decoder | Notes |
|---|---|---|
| `.wav`, `.flac`, `.ogg`, `.aiff/.aif` | libsndfile (via `soundfile`) | lossless / native |
| `.mp3` | libsndfile >= 1.1 (bundled with the `soundfile` wheels), fallback ffmpeg | |
| `.m4a` | ffmpeg via `audioread` | needs `ffmpeg` on the PATH |

Limits (editable in `config/settings.py` or via environment variables `NOTEVISION_MAX_FILE_MB`,
`NOTEVISION_MAX_DURATION_S`): 200 MB and 15 minutes per file, minimum 0.1 s. Any sample rate and
channel count is accepted; audio is mixed to mono and resampled to 22.05 kHz for analysis.

## 6. Analysis modes

| Mode | Engine | Use it for | Confidence means |
|---|---|---|---|
| **A - Monophonic** | `librosa.pyin` f0 tracking -> onset detection -> segmentation (`src/note_segmentation.py`) | one voice / one melodic instrument | mean pYIN voicing probability of the note's frames |
| **B - Polyphonic** | Spotify Basic Pitch (ICASSP 2022 "nmp" model) through onnxruntime | piano, guitar, chords, several simultaneous notes | the model's note activation (0-1) |
| **C - Full music recording** | Basic Pitch on the mix; with Demucs installed and enabled, separately on vocals / bass / other stems | complete songs | as mode B; stem name is stored in `source_track` |

Each mode runs only the components it needs (pYIN is never run in mode B, Basic Pitch never in
mode A). If the model required by a mode is not installed, the app explains why and suggests a
compatible mode.

Repeated notes: in mode A a steady pitch is split into separate events **only** where the onset
detector finds a real attack (spectral-flux peak **and** an energy rise); in mode B the model's
own onset head provides that evidence. Overlapping events are never merged.

## 7. How the pipeline works

```
validate -> decode -> preprocess -> transcribe -> tempo estimate -> statistics -> persist
```

* `src/audio_loader.py` - validation, metadata, decoding (original file is never modified).
* `src/audio_preprocessing.py` - the configurable pipeline; each applied step is recorded.
  Only *silence trimming* changes the timeline, and its offset is added back to every timestamp.
* `src/note_transcription.py` - the three modes, the `NoteEvent` schema, validation
  (finite timestamps, `end >= start`, MIDI 0-127, valid frequency), overlap computation.
  Basic Pitch is driven window-by-window so progress and cancellation work; its pitch-contour
  output refines each note's frequency to ~1/3-semitone resolution with interpolation
  (calibrated on test tones to within +/-3 cents).
* `src/basic_pitch_onnx.py` - runs the bundled Basic Pitch ONNX model with `onnxruntime` and
  contains a line-by-line port of the original note-decoding algorithm (verified to produce
  identical notes to the `basic-pitch` package on the test signals).
* `src/note_statistics.py`, `src/song_comparison.py`, `src/visualization.py`,
  `src/report_generator.py`, `database/database.py` - statistics, comparison, charts, exports,
  persistence. `src/analysis_pipeline.py` orchestrates everything and `app.py` is the UI.
* `src/evaluation.py` - precision / recall / F1 and onset / offset timing errors against
  ground-truth annotations (`mir_eval`), used by the tests and available for your own annotated
  recordings (`load_ground_truth_csv`).

## 8. Output schema

One row per detected note event (CSV / Excel "Note Events" / database table `note_events`):

| Field | Type | Description |
|---|---|---|
| `event_id` | str | `<song_id[:8]>-<index>` ordered by start time |
| `song_id` | str | first 16 hex chars of the file's SHA-256 |
| `note_name` | str | e.g. `C4`, `D#5` (sharps, scientific pitch notation, C4 = MIDI 60) |
| `midi_note` | int | `round(69 + 12 * log2(f / 440))` |
| `pitch_class` | str | `C` ... `B` |
| `octave` | int | `midi // 12 - 1` |
| `frequency_hz` | float | estimated fundamental; see `frequency_source` |
| `start_time_seconds`, `end_time_seconds`, `duration_seconds` | float | original-recording timeline, rounded to 4 decimals (0.1 ms); model frame resolution ~11.6 ms |
| `confidence` | float or empty | see mode table above |
| `instrument_label` | str or empty | reserved (not predicted by the current models) |
| `source_track` | str | `mix`, or a stem name (`vocals`, `bass`, `other`) in mode C with separation |
| `onset_strength` | float or empty | normalised onset-strength at the note start (mode A) |
| `detection_method` | str | `pyin` or `basic_pitch` |
| `cents_deviation` | float | deviation of `frequency_hz` from the equal-tempered pitch |
| `frequency_source` | str | `measured_f0_median` (pYIN), `basic_pitch_contour`, or `equal_tempered` (fallback) |
| `overlaps_other_events`, `overlap_count` | bool, int | whether / how many other events sound at the same time |
| `is_valid`, `validation_error` | bool, str | invalid or confidence-filtered events are kept separately and never enter statistics |

Example:

```csv
event_id,song_id,note_name,midi_note,pitch_class,octave,frequency_hz,start_time_seconds,end_time_seconds,duration_seconds,confidence,...
52b45ed0-00001,52b45ed0a7fa371a,C4,60,C,4,261.6256,0.0,0.4586,0.4586,0.9678,...
52b45ed0-00002,52b45ed0a7fa371a,D4,62,D,4,293.6648,0.4934,0.9578,0.4644,0.9496,...
```

## 9. Exports

All exports are generated on demand from the stored results and use timestamped names such as
`song_note_events_20260131-154500.csv`.

* **CSV** - every event, one row each, all fields above plus `start_time_formatted` /
  `end_time_formatted` (MM:SS.mmm). The data-table tab also exports the *filtered* view.
* **Excel** - sheets *Song Metadata*, *Analysis Summary*, *Note Events*, *Note Statistics*,
  *Pitch-Class Statistics*, *Octave Statistics*, *Processing Information*, *Excluded Events*.
* **PDF** - metadata, analysis summary, warnings, charts (piano roll, note counts, pitch classes,
  duration histogram, polyphony), note / pitch-class / octave statistics, the event table (first
  600 rows; the CSV has all of them), processing information and limitations.

From Python: `src.report_generator.export_all(result, "exports/")` writes all three files.

## 10. Database and file storage

* Database: `data/notevision.sqlite3` (tables `songs`, `analyses`, `note_events`). Each analysis
  is identified by song hash + configuration hash, so re-saving never duplicates events, and the
  same file analysed with the same settings is loaded from the database instead of re-processed
  (untick *Re-analyse even if an identical analysis is stored* to force a new run).
* Uploaded recordings are copied **persistently** to `data/uploads/<song_id>/<original name>`
  so that history entries can be re-opened with their waveform and spectrogram. They are deleted
  together with the song on the History page. Set `NOTEVISION_PERSIST_UPLOADS=0` if you prefer
  not to keep copies (results remain, audio plots will not be available after a restart).
  Files added by path are read in place and never copied.
* Exports written by `export_all` go to `exports/`; logs to `logs/notevision.log`.
* Nothing under `data/`, `exports/`, `logs/` or `.venv/` is committed to git (`.gitignore`).
* **Hosted instances** (Streamlit Community Cloud) run on temporary disks: the database and
  uploaded copies are cleared whenever the app restarts or is redeployed. Download your reports.

## 11. Testing

```powershell
scripts\test.bat                                 # ~40-60 s, 149 tests
scripts\test.bat -q tests\test_transcription.py  # extra arguments go to pytest
# or, with the environment activated:  python -m pytest
```

The tests synthesise audio with exactly known pitches, onsets and durations (`tests/conftest.py`)
and cover: loading of supported / unsupported / corrupted / empty files, size and duration
limits, stereo and 44.1 kHz input, frequency-to-note conversion including invalid values,
segmentation (glitches, gaps, repeated notes with and without onset evidence), overlapping
events, the real pYIN and Basic Pitch pipelines against ground truth (precision, recall, onset
and offset error via `mir_eval`), statistics, CSV / Excel / PDF exports, the SQLite layer
(round-trip, de-duplication, cascade delete, failure status), caching, comparison metrics,
chart construction, and the Streamlit pages through `streamlit.testing`.

Measured on the synthetic test signals (50 ms onset tolerance, 50 cents pitch tolerance):

| Signal | Mode | Precision | Recall | Mean onset error | Precision with confidence >= 0.5 |
|---|---|---|---|---|---|
| melody (6 notes, one repeated) | A monophonic | 1.00 | 1.00 | 8.6 ms | 1.00 |
| melody | B polyphonic | 0.86 | 1.00 | 7.3 ms | 1.00 |
| two triads (6 notes) | B polyphonic | 0.43 | 1.00 | 6.9 ms | 1.00 |
| overlapping arpeggio (4 notes) | B polyphonic | 0.36 | 1.00 | 5.5 ms | 0.80 |

Basic Pitch finds every true note but also reports harmonic "ghost" notes with low confidence
(0.3-0.45), which is why the confidence filter matters. Real recordings will score lower than
synthetic tones - no transcription is perfectly accurate.

## 12. Project structure

```
NoteVision-AI/
    app.py                         Streamlit UI
    config/settings.py             paths, limits, model defaults, logging
    src/
        audio_loader.py            validation, metadata, decoding, upload storage
        audio_preprocessing.py     mono / resample / normalise / denoise / trim pipeline
        pitch_detection.py         Hz <-> MIDI <-> note names, cents, pYIN tracking
        onset_detection.py         onsets, onset strength, energy rise, tempo
        note_segmentation.py       pitch track -> note events (mode A)
        note_transcription.py      NoteEvent schema, modes A/B/C, validation, overlaps
        basic_pitch_onnx.py        bundled Basic Pitch model runner (onnxruntime) + note decoder
        source_separation.py       optional Demucs wrapper (mode C)
        note_statistics.py         all statistics
        song_comparison.py         multi-song comparison
        visualization.py           Plotly (interactive) and Matplotlib (PDF) charts
        report_generator.py        CSV / Excel / PDF
        analysis_pipeline.py       orchestration, caching, AnalysisResult
        evaluation.py              precision / recall / timing error vs. annotations
        progress.py, utils.py      progress & cancellation, formatting, hashing
    database/database.py           SQLite persistence
    models/basic_pitch/            nmp.onnx (Spotify Basic Pitch, Apache-2.0) + LICENSE + NOTICE
    data/uploads/, data/processed/ runtime data (git-ignored)
    exports/                       generated reports (git-ignored)
    tests/                         pytest suite with synthetic audio fixtures
    samples/                       two synthetic demo recordings for a first run
    scripts/setup.bat, run.bat, test.bat   Windows entry points (bypass the execution policy)
    scripts/setup.ps1, run.ps1, test.ps1, venv.ps1, setup.sh
    vscode-config/                 VS Code settings + launch configs (copied to .vscode/ by setup)
    requirements*.txt, pytest.ini, LICENSE
    .github/workflows/tests.yml    CI: pytest on Ubuntu + Windows, Python 3.11 / 3.12
    .streamlit/config.toml, packages.txt, .python-version   Streamlit Community Cloud configuration
```

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| "Polyphonic / full-mix modes are unavailable" in the app | `onnxruntime` is missing or `models/basic_pitch/nmp.onnx` is absent (incomplete clone/zip). Re-run `pip install -r requirements.txt` and check the file exists; the *Verify* command above shows the reason |
| `... cannot be loaded because running scripts is disabled on this system` | Use the `.bat` wrappers (`scripts\setup.bat`, `run.bat`, `test.bat`), or run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and retry |
| `Using interpreter: python (Python 2.7)` / `No module named venv` | Plain `python` is another program (e.g. MGLTools). Re-run `scripts\setup.bat` (it now skips such interpreters) or pass `-PythonExe` with the full path of a Python 3.10-3.14 |
| `No usable Python 3.10 - 3.14 interpreter was found` | Install Python 3.12 from python.org (tick *Add python.exe to PATH*), or pass `-PythonExe "C:\...\python.exe"` |
| `.\.venv\Scripts\python.exe is not recognized` | The environment was not created (see the two rows above) or it lives in `%LOCALAPPDATA%\NoteVision-AI\venv` because the project is inside OneDrive - use `scripts\run.bat` / `test.bat`, which find it |
| VS Code runs the wrong Python (status bar shows the global interpreter) | Reload the window after setup (Ctrl+Shift+P -> *Developer: Reload Window*) or pick the venv via *Python: Select Interpreter*; the path is printed at the end of setup |
| Project extracted from the zip sits in `NoteVision-AI\NoteVision-AI\` | Open the inner folder in VS Code (File -> Open Folder) so `app.py` is at the workspace root, or move its contents one level up |
| MP3 cannot be decoded | Update `soundfile` (`pip install -U soundfile`) or install ffmpeg and add it to PATH; or convert to WAV |
| M4A rejected | Install ffmpeg (`winget install Gyan.FFmpeg`) and restart the terminal |
| First polyphonic analysis is slow | Expected: onnxruntime optimises the graph once per process; later runs are fast |
| `numba` compile warnings / slow first pYIN call | Normal; numba caches compiled kernels after the first call |
| Port 8501 already in use | `streamlit run app.py --server.port 8600` |
| Too many spurious high notes in polyphonic mode | They are usually harmonics with confidence < 0.4: raise *Discard events below confidence* or the frame threshold in *Advanced settings* |
| A held note is split into several events (mode A) | Raise `ONSET_SPLIT_RELATIVE_THRESHOLD` / `ONSET_ENERGY_RISE_RATIO` in `config/settings.py`, or untick *Split repeated notes on onset evidence* |
| Repeated notes are merged (mode A) | Lower `ONSET_SPLIT_RELATIVE_THRESHOLD` / `ONSET_ENERGY_RISE_RATIO` in `config/settings.py` |
| Database locked | Close other app instances; the database is `data/notevision.sqlite3` (delete it to start fresh) |

Logs: `logs/notevision.log` (set `NOTEVISION_LOG_LEVEL=DEBUG` for more detail).

## 14. Performance considerations

* The Basic Pitch model is loaded once per process and shared between analyses; the Streamlit
  app runs analyses in a background thread so the UI stays responsive, with per-window
  progress and cooperative cancellation.
* Audio is processed as 32-bit float mono at 22.05 kHz (about 5 MB per minute in memory).
  Waveform and spectrogram plots are down-sampled for the browser.
* Results are cached by (file hash, configuration hash) in SQLite; re-uploading the same file
  with the same settings is instant.
* Source separation (optional) is the only step that benefits from a GPU; it is used only if a
  CUDA-enabled PyTorch build is installed **and** `torch.cuda.is_available()` is true. The app
  reports the device it used and never claims GPU acceleration otherwise.

## 15. Known limitations

* Monophonic mode reports one pitch per frame; chords and accompaniment are invisible to it.
* Neural transcription (Basic Pitch) can add harmonics as extra notes (usually low confidence),
  make octave errors, miss soft notes, and blur the boundaries of repeated legato notes.
* Full songs with drums, vocals and effects are transcribed approximately at best; source
  separation improves this but is slow on CPU.
* `instrument_label` is not predicted - Basic Pitch is instrument-agnostic.
* Tempo is a single global estimate from onset periodicity (unreliable for rubato / short clips).
* Confidence values are not calibrated probabilities and are not comparable across modes.
* Similar pitch-class distributions between songs do not imply a shared melody; the sequence
  similarity is an approximate order comparison, not a melody matcher.
* Accuracy figures in the tests come from synthetic signals; evaluate on your own annotated
  recordings with `src/evaluation.py` before relying on the numbers.

## 16. Deploying your own copy

### Hosted version (Streamlit Community Cloud)

The public demo at <https://notevision-ai.streamlit.app> runs on Streamlit Community Cloud's free
tier (shared CPU, about 1 GB RAM). Keep uploads short there (a few minutes of audio), expect the
first analysis after a cold start to take ~20 s longer, and remember that stored history is
cleared on every restart. For heavy use, run the app locally.

To host your own copy:

1. Fork or push this repository to your GitHub account.
2. Go to <https://share.streamlit.io>, sign in with GitHub, click **Create app** -> *Deploy a
   public app from GitHub*.
3. Repository: `<your-user>/NoteVision-AI`, branch `main`, main file path `app.py`.
4. *Advanced settings* -> Python version **3.12**. No secrets are needed.
5. **Deploy**. The first build takes 3-5 minutes (it installs `requirements.txt` and the apt
   packages in `packages.txt`, i.e. `ffmpeg`). Every later push to `main` redeploys automatically.

Resource limits for the hosted app live in `.streamlit/config.toml` (`maxUploadSize`) and
`config/settings.py` (`MAX_DURATION_SECONDS`, `MAX_FILE_SIZE_MB`).

### Docker (optional, any server)

```bash
docker build -t notevision-ai .
docker run -p 8501:8501 -v notevision-data:/app/data notevision-ai
```

The `Dockerfile` installs `ffmpeg`, the Python requirements and starts Streamlit on port 8501;
the named volume keeps the SQLite database and uploads across container restarts.

## 17. Contributing, credits and license

* Issues and pull requests are welcome at <https://github.com/abhi-gupta2005/NoteVision-AI>.
  Run `python -m pytest` before opening a PR; CI runs the suite on Ubuntu and Windows.
* Built with [librosa](https://librosa.org), [Spotify Basic Pitch](https://github.com/spotify/basic-pitch)
  (model weights and note-decoding algorithm, Apache-2.0 - see `models/basic_pitch/NOTICE.md`),
  [onnxruntime](https://onnxruntime.ai), [Streamlit](https://streamlit.io), Plotly, pandas,
  OpenPyXL, ReportLab and [mir_eval](https://mir-evaluation.github.io/mir_eval/).
* Copyright (c) 2026 Abhishek Kumar Gupta. Released under the [MIT License](LICENSE).
