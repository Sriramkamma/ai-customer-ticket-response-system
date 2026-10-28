$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'

try {
    if (-not (Test-Path -LiteralPath $venvPython)) {
        $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
        if ($pythonLauncher) {
            & $pythonLauncher.Source -3 -m venv .venv
        } else {
            $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
            if (-not $pythonCommand) {
                throw 'Python 3.10 or newer was not found. Install Python and try again.'
            }
            & $pythonCommand.Source -m venv .venv
        }

        if ($LASTEXITCODE -ne 0) {
            throw 'Could not create the .venv environment.'
        }
    }

    $pythonVersion = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($LASTEXITCODE -ne 0 -or [version]$pythonVersion -lt [version]'3.10') {
        throw "The .venv Python version ($pythonVersion) is too old. Python 3.10 or newer is required."
    }

    Write-Host "Installing Python packages from requirements.txt using Python $pythonVersion."
    Write-Host 'This may install PyTorch and FAISS packages. The embedding model and Ollama model are not downloaded by this step.'
    & $venvPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw 'Dependency installation failed. Check the messages above and your package-index connection.'
    }

    Write-Host 'Starting the AI Customer Ticket Response System.'
    & $venvPython server.py
    if ($LASTEXITCODE -ne 0) {
        throw 'The application exited with an error. Check the messages above.'
    }
} catch {
    [Console]::Error.WriteLine("Setup or startup failed: $($_.Exception.Message)")
    exit 1
}
