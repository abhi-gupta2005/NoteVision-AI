# Start the NoteVision AI Streamlit app (Windows PowerShell).
#   Easiest:  scripts\run.bat
#   Manual:   powershell -ExecutionPolicy Bypass -File scripts\run.ps1 [-Port 8600]
param(
    [int]$Port = 8501
)
$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot
. (Join-Path $PSScriptRoot "venv.ps1")

$venvPython = Find-VenvPython -ProjectRoot $ProjectRoot
if (-not $venvPython) {
    Write-Host "No virtual environment found. Run scripts\setup.bat first." -ForegroundColor Yellow
    exit 1
}
Write-Host "Starting NoteVision AI with $venvPython on http://localhost:$Port ..."
& $venvPython -m streamlit run app.py --server.port $Port --browser.gatherUsageStats false
