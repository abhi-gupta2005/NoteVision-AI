@echo off
REM NoteVision AI - set up the Python environment (works even if PowerShell scripts are disabled).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
if errorlevel 1 (
  echo.
  echo Setup did not finish - read the messages above.
)
pause
