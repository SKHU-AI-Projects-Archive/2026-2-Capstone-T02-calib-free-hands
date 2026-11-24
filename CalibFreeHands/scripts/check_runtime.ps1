$ErrorActionPreference = "Stop"
Write-Host "Python:"
python --version
foreach ($tool in @("ffmpeg", "ffprobe")) {
    $found = Get-Command $tool -ErrorAction SilentlyContinue
    if ($found) { & $tool -version | Select-Object -First 1 }
    else { Write-Warning "$tool was not found on PATH. Install a Windows FFmpeg build and add its bin directory to PATH." }
}
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi --query-gpu=name,driver_version --format=csv,noheader }
