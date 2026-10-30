@echo off
REM NoteVision AI - run the automated tests.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0test.ps1" %*
pause
