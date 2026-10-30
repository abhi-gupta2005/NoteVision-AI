# Run the NoteVision AI test-suite (Windows PowerShell).
#   Easiest:  scripts\test.bat
#   Manual:   powershell -ExecutionPolicy Bypass -File scripts\test.ps1 [extra pytest arguments]
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
. (Join-Path $PSScriptRoot "venv.ps1")

$venvPython = Find-VenvPython -ProjectRoot $ProjectRoot
if (-not $venvPython) {
    Write-Host "No virtual environment found. Run scripts\setup.bat first." -ForegroundColor Yellow
    exit 1
}
& $venvPython -m pytest @args
exit $LASTEXITCODE
