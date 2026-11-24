$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Project virtual environment not found: $python" }
$env:PYTHONPATH = Join-Path $projectRoot "src"
& $python -m calibfreehands.app.main
