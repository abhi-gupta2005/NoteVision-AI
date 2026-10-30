"""Headless Streamlit smoke tests (streamlit.testing.v1.AppTest)."""
import time
from pathlib import Path

import pytest

from config import settings

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover
    AppTest = None

pytestmark = pytest.mark.skipif(AppTest is None, reason="streamlit testing framework unavailable")
APP = Path(settings.PROJECT_ROOT) / "app.py"


@pytest.fixture()
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "DB_PATH", tmp_path / "app.sqlite3")


def _run(at: AppTest) -> AppTest:
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_all_pages_render_without_results(isolated_db):
    at = AppTest.from_file(str(APP), default_timeout=120)
    _run(at)
    options = at.sidebar.radio[0].options
    assert options == ["Upload & Analyze", "Results Dashboard", "Compare Songs", "History", "Help & About"]
    for page in options[1:]:
        at.sidebar.radio[0].set_value(page)   # re-fetch: elements are rebuilt on every run
        _run(at)
        assert page in at.header[0].value
    # dashboard without results shows an empty state
    at.sidebar.radio[0].set_value("Results Dashboard")
    _run(at)
    assert any("No analysed recordings" in i.value for i in at.info)


def test_dashboard_renders_with_injected_result(isolated_db, melody_wav: Path):
    from src.analysis_pipeline import run_analysis
    from src.note_transcription import TranscriptionConfig

    result = run_analysis(melody_wav, TranscriptionConfig(mode="monophonic"))
    at = AppTest.from_file(str(APP), default_timeout=120)
    _run(at)
    at.session_state["results"] = {result.song_id: result}
    at.session_state["selected_song"] = result.song_id
    at.sidebar.radio[0].set_value("Results Dashboard")
    _run(at)
    labels = [t.label for t in at.tabs]
    for expected in ("Overview", "Waveform", "Spectrogram", "Piano Roll", "Note Frequency", "Duration Analysis", "Timeline",
                     "Data Table", "Export"):
        assert expected in labels
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Detected note events"] == "6"
    assert metrics["Most common note"] == "C4"
    assert any(b.label.startswith("Download") for b in at.get("download_button"))


def test_local_path_analysis_job(isolated_db, melody_wav: Path):
    """Add a local file, run the background job through the fragment, and check the report."""
    at = AppTest.from_file(str(APP), default_timeout=180)
    _run(at)
    at.text_input(key="local_path_input").set_value(str(melody_wav))
    _run(at)
    at.button(key="add_local").click()
    _run(at)
    assert any(it["name"] == "melody.wav" for it in at.session_state["uploads"].values())
    at.radio[0].set_value("monophonic")   # analysis-mode radio (sidebar radio is separate)
    _run(at)
    analyze = [b for b in at.button if b.label.startswith("Analyze")][0]
    assert not analyze.disabled
    analyze.click()
    _run(at)
    assert at.session_state["job"] is not None
    for _ in range(120):
        job = at.session_state["job"]
        if job is None or job.done:
            break
        time.sleep(0.5)
    _run(at)                              # the fragment finalises the finished job
    assert at.session_state["job"] is None
    report = at.session_state["job_report"]
    assert report["errors"] == {} and report["ok"] == ["melody.wav"]
    assert len(at.session_state["results"]) == 1
    assert any("Analysed 1 recording" in s.value for s in at.success)
