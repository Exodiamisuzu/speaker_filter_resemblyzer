$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "[1/4] Installing PyInstaller..."
.\.venv\Scripts\python.exe -m pip install pyinstaller

Write-Host "[2/4] Removing incompatible typing backport if present..."
$previousErrorAction = $ErrorActionPreference
$ErrorActionPreference = "Continue"
.\.venv\Scripts\python.exe -m pip uninstall -y typing 2> $null | Out-Null
$ErrorActionPreference = $previousErrorAction

Write-Host "[3/4] Building EXE..."
.\.venv\Scripts\pyinstaller.exe --noconfirm --clean --windowed --onefile --name ResemblyzerSpeakerFilter .\gui_resemblyzer.py

Write-Host "[4/4] Done."
Write-Host "EXE path: .\dist\ResemblyzerSpeakerFilter.exe"
