param(
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

if (-not $PythonExe) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            py -3.12 --version *> $null
            if ($LASTEXITCODE -eq 0) {
                $PythonExe = "py -3.12"
            }
        } catch {}

        if (-not $PythonExe) {
            try {
                py -3.13 --version *> $null
                if ($LASTEXITCODE -eq 0) {
                    $PythonExe = "py -3.13"
                }
            } catch {}
        }
    }

    if (-not $PythonExe) {
        $PythonExe = "python"
    }
}

Write-Host "[1/4] Creating virtual environment (.venv)..."
Invoke-Expression "$PythonExe -m venv .venv"

Write-Host "[2/4] Upgrading pip/setuptools/wheel..."
.\.venv\Scripts\python.exe -m pip install --upgrade pip wheel

Write-Host "[3/4] Installing dependencies from requirements.txt..."
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "[4/4] Done."
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Then run: python build_reference.py ... and python classify_and_label.py ..."
