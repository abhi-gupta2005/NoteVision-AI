@echo off
REM NoteVision AI - start the app (opens http://localhost:8501).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
