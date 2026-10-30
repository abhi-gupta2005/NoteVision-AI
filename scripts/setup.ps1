# NoteVision AI - one-shot environment setup for Windows PowerShell.
#
#   Easiest:  double-click scripts\setup.bat   (or run it from a terminal)
#   Manual:   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
#
# Options:
#   -PythonExe "C:\Path\To\python.exe"   use this interpreter (Python 3.10 - 3.14)
#   -VenvPath  "D:\envs\notevision"      put the virtual environment somewhere else
#
# What it does: finds a suitable Python 3.10-3.14, creates a virtual environment,
# installs requirements.txt, creates .vscode\ settings pointing VS Code at the
# environment, and verifies the install (the Basic Pitch model is bundled).

param(
    [string]$PythonExe = "",
    [string]$VenvPath = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $ProjectRoot

$MinMinor = 10   # Python 3.10
$MaxMinor = 14   # Python 3.14 (newer versions are untested but will be tried last)

function Get-PythonVersion {
    # Returns "3.x" for a working CPython 3 that has the venv module, otherwise $null.
    param([string]$Exe, [string[]]$ExtraArgs = @())
    if (-not $ExtraArgs) { $ExtraArgs = @() }
    try {
        $code = "import sys; v = '%d.%d' % sys.version_info[:2]; exec('try:\n import venv\n ok = 1\nexcept Exception:\n ok = 0'); print(v if ok else v + '-novenv')"
        $out = & $Exe @ExtraArgs -c $code 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { return ([string]$out).Trim() }
    } catch { }
    return $null
}

function Test-AcceptableVersion {
    param([string]$Version)
    if (-not $Version) { return $false }
    if ($Version -like "*-novenv*") { return $false }   # e.g. MGLTools / old Anaconda interpreters
    $parts = $Version.Split(".")
    if ($parts.Length -lt 2) { return $false }
    return ([int]$parts[0] -eq 3 -and [int]$parts[1] -ge $MinMinor)
}

# ---------------------------------------------------------------- find Python
$exe = $null
$extra = @()
$version = $null
$seen = @()

if ($PythonExe -ne "") {
    $version = Get-PythonVersion -Exe $PythonExe
    if (-not (Test-AcceptableVersion $version)) {
        throw "'$PythonExe' is not a usable Python 3.$MinMinor+ interpreter (reported: '$version')."
    }
    $exe = $PythonExe
} else {
    # Order of preference: tested versions first, then whatever "py" / "python3" / "python" resolve to.
    $candidates = @()
    foreach ($v in @("3.12", "3.11", "3.13", "3.14", "3.10")) {
        $candidates += ,@("py", @("-$v"))
        $candidates += ,@("python$v", @())
    }
    $candidates += ,@("py", @("-3"))
    $candidates += ,@("python3", @())
    $candidates += ,@("python", @())
    # Well-known install folders (python.org, Microsoft Store, conda) in case nothing is on the PATH.
    $folders = @()
    if ($env:LOCALAPPDATA) {
        $folders += Join-Path $env:LOCALAPPDATA "Programs\Python"
        $folders += Join-Path $env:LOCALAPPDATA "Microsoft\WindowsApps"
        $folders += Join-Path $env:LOCALAPPDATA "miniconda3"
        $folders += Join-Path $env:LOCALAPPDATA "anaconda3"
    }
    if ($env:ProgramFiles) { $folders += $env:ProgramFiles }
    if ($env:USERPROFILE) { $folders += (Join-Path $env:USERPROFILE "miniconda3"); $folders += (Join-Path $env:USERPROFILE "anaconda3") }
    if ($env:ProgramData) { $folders += (Join-Path $env:ProgramData "miniconda3"); $folders += (Join-Path $env:ProgramData "anaconda3") }
    $folders += "C:\"
    foreach ($folder in $folders) {
        if (-not (Test-Path $folder)) { continue }
        try {
            $hits = @(Get-ChildItem -Path $folder -Filter "python*.exe" -File -ErrorAction SilentlyContinue)
            $hits += @(Get-ChildItem -Path $folder -Directory -Filter "Python3*" -ErrorAction SilentlyContinue |
                ForEach-Object { Get-ChildItem -Path $_.FullName -Filter "python.exe" -File -ErrorAction SilentlyContinue })
            foreach ($hit in $hits) {
                if ($hit -and $hit.FullName) { $candidates += ,@($hit.FullName, @()) }
            }
        } catch { }
    }

    $fallback = $null
    foreach ($candidate in $candidates) {
        $cExe = $candidate[0]
        $cArgs = $candidate[1]
        $v = Get-PythonVersion -Exe $cExe -ExtraArgs $cArgs
        if (-not $v) { continue }
        $label = ("$cExe $cArgs").Trim()
        $seen += "$label -> Python $v"
        if (-not (Test-AcceptableVersion $v)) { continue }
        $minor = [int]$v.Split(".")[1]
        if ($minor -le $MaxMinor) {
            $exe = $cExe; $extra = $cArgs; $version = $v
            break
        } elseif (-not $fallback) {
            $fallback = @($cExe, $cArgs, $v)   # newer than tested: use only if nothing else works
        }
    }
    if (-not $exe -and $fallback) {
        $exe = $fallback[0]; $extra = $fallback[1]; $version = $fallback[2]
        Write-Host "Note: Python $version is newer than the versions NoteVision AI was tested with (3.$MinMinor-3.$MaxMinor)." -ForegroundColor Yellow
    }
}

if (-not $exe) {
    Write-Host ""
    Write-Host "No usable Python 3.$MinMinor - 3.$MaxMinor interpreter was found." -ForegroundColor Red
    if ($seen.Count -gt 0) {
        Write-Host "Interpreters that were found but are not suitable:"
        foreach ($s in $seen) { Write-Host "   $s" }
    }
    Write-Host ""
    Write-Host "Install Python 3.12 (recommended) from https://www.python.org/downloads/windows/"
    Write-Host "  - tick 'Add python.exe to PATH' and keep the 'py launcher' option enabled,"
    Write-Host "  - then run this script again, or pass the interpreter explicitly:"
    Write-Host "    powershell -ExecutionPolicy Bypass -File scripts\setup.ps1 -PythonExe `"C:\Users\<you>\AppData\Local\Programs\Python\Python312\python.exe`""
    exit 1
}
$label = ("$exe $extra").Trim()
Write-Host "Using interpreter: $label (Python $version)" -ForegroundColor Green

# ---------------------------------------------------------- choose venv path
if ($VenvPath -eq "") {
    if ($env:NOTEVISION_VENV) {
        $VenvPath = $env:NOTEVISION_VENV
    } elseif ($ProjectRoot -like "*\OneDrive*") {
        # A virtual environment has tens of thousands of small files; keeping it out of a
        # OneDrive-synced folder avoids sync churn and locked-file errors during pip installs.
        $VenvPath = Join-Path $env:LOCALAPPDATA "NoteVision-AI\venv"
        Write-Host "The project folder is synced by OneDrive, so the virtual environment goes to:" -ForegroundColor Yellow
        Write-Host "   $VenvPath" -ForegroundColor Yellow
    } else {
        $VenvPath = Join-Path $ProjectRoot ".venv"
    }
}
$venvPython = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment at $VenvPath ..."
    & $exe @extra -m venv $VenvPath
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) {
        throw "Creating the virtual environment failed (exit code $LASTEXITCODE)."
    }
} else {
    Write-Host "Re-using existing virtual environment at $VenvPath"
}

# ------------------------------------------------------------------- install
function Invoke-Pip {
    param([string[]]$PipArgs)
    & $venvPython -m pip @PipArgs
    if ($LASTEXITCODE -ne 0) { throw "pip failed: pip $($PipArgs -join ' ')" }
}
Write-Host "Upgrading pip ..."
Invoke-Pip @("install", "--upgrade", "pip", "wheel")
Write-Host "Installing requirements (this downloads ~300 MB the first time) ..."
Invoke-Pip @("install", "-r", "requirements.txt")

# ------------------------------------------------------- VS Code integration
$vscodeDir = Join-Path $ProjectRoot ".vscode"
$configDir = Join-Path $ProjectRoot "vscode-config"
if (Test-Path $configDir) {
    if (-not (Test-Path $vscodeDir)) { New-Item -ItemType Directory -Path $vscodeDir | Out-Null }
    foreach ($name in @("launch.json", "extensions.json")) {
        $target = Join-Path $vscodeDir $name
        if (-not (Test-Path $target)) { Copy-Item (Join-Path $configDir $name) $target }
    }
    $settingsTarget = Join-Path $vscodeDir "settings.json"
    $settingsSource = Join-Path $configDir "settings.json"
    $placeholder = '${workspaceFolder}/.venv/Scripts/python.exe'
    $sourceFile = $null
    if (Test-Path $settingsTarget) {
        # Only touch an existing settings.json if it still carries the template placeholder.
        if ((Get-Content $settingsTarget -Raw) -like "*$placeholder*") { $sourceFile = $settingsTarget }
    } elseif (Test-Path $settingsSource) {
        $sourceFile = $settingsSource
    }
    if ($sourceFile -and ($venvPython -ne (Join-Path $ProjectRoot ".venv\Scripts\python.exe"))) {
        $jsonPath = $venvPython.Replace("\", "/")
        $settingsText = (Get-Content $sourceFile -Raw).Replace($placeholder, $jsonPath)
        [System.IO.File]::WriteAllText($settingsTarget, $settingsText)   # UTF-8 without BOM
        Write-Host "Wrote .vscode\settings.json (interpreter: $venvPython)"
    } elseif ($sourceFile -and -not (Test-Path $settingsTarget)) {
        Copy-Item $sourceFile $settingsTarget
    }
}

# -------------------------------------------------------------------- verify
Write-Host ""
Write-Host "Verifying the installation ..."
& $venvPython -c "import sys, librosa, streamlit, onnxruntime, numpy; from src.basic_pitch_onnx import is_available; ok, detail = is_available(); print('Python', sys.version.split()[0], '| numpy', numpy.__version__, '| librosa', librosa.__version__, '| streamlit', streamlit.__version__, '|', detail); sys.exit(0 if ok else 1)"
if ($LASTEXITCODE -ne 0) { throw "Verification failed - see the messages above." }

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "  Start the app:   scripts\run.bat        (or: powershell -ExecutionPolicy Bypass -File scripts\run.ps1)"
Write-Host "  Run the tests:   scripts\test.bat"
Write-Host "  In VS Code:      reload the window (Ctrl+Shift+P -> 'Developer: Reload Window'), then press F5."
Write-Host "  Interpreter:     $venvPython"
