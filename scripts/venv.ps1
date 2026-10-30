# Shared helper: locate the project's virtual environment (dot-source this file).
#   . (Join-Path $PSScriptRoot "venv.ps1")
#   $venvPython = Find-VenvPython -ProjectRoot $ProjectRoot
#
# Search order: $env:NOTEVISION_VENV, <project>\.venv, %LOCALAPPDATA%\NoteVision-AI\venv
# (the last one is where setup.ps1 puts the environment when the project lives inside OneDrive).

function Find-VenvPython {
    param([string]$ProjectRoot)
    $candidates = @()
    if ($env:NOTEVISION_VENV) { $candidates += (Join-Path $env:NOTEVISION_VENV "Scripts\python.exe") }
    $candidates += (Join-Path $ProjectRoot ".venv\Scripts\python.exe")
    if ($env:LOCALAPPDATA) { $candidates += (Join-Path $env:LOCALAPPDATA "NoteVision-AI\venv\Scripts\python.exe") }
    foreach ($c in $candidates) {
        if (Test-Path $c) { return $c }
    }
    return $null
}
