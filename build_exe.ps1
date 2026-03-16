$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$appName = "AirBarAudioSeparator"
$distDir = Join-Path $PSScriptRoot "dist\$appName"

Write-Host "[1/4] Installing PyInstaller..."
.\.venv\Scripts\python.exe -m pip install pyinstaller

Write-Host "[2/4] Removing incompatible typing backport if present..."
$previousErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
.\.venv\Scripts\python.exe -m pip uninstall -y typing 2> $null | Out-Null
$ErrorActionPreference = "Continue"
.\.venv\Scripts\python.exe -m pip uninstall -y pandas 2> $null | Out-Null
$ErrorActionPreference = $previousErrorAction

Write-Host "[3/4] Building EXE..."
.\.venv\Scripts\python.exe -m PyInstaller --noconfirm --clean --windowed --onedir --name $appName --collect-data resemblyzer --add-data ".\background.jpg;." --add-data ".\ok;ok" --exclude-module pandas --exclude-module matplotlib --exclude-module IPython --exclude-module jupyter_client --exclude-module jupyter_core .\gui_resemblyzer.py
if ($LASTEXITCODE -ne 0) {
	throw "PyInstaller build failed with exit code $LASTEXITCODE"
}

Write-Host "[4/4] Done."
Write-Host "EXE path: $distDir\$appName.exe"
