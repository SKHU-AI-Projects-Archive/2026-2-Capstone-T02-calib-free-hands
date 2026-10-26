$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$wilorDir = Join-Path $repoRoot "..\WiLoR\pretrained_models"
New-Item -ItemType Directory -Force -Path $wilorDir | Out-Null

Write-Host "Downloading official WiLoR assets to $wilorDir"
Invoke-WebRequest "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt" -OutFile (Join-Path $wilorDir "wilor_final.ckpt")
Invoke-WebRequest "https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt" -OutFile (Join-Path $wilorDir "detector.pt")

Write-Host "MANO is not downloaded automatically. Obtain the licensed MANO model from the official MANO website and place the required files under the path expected by the WiLoR configuration."
Write-Host "AnyCalib pretrained weights are loaded by the official AnyCalib model_id/Hugging Face mechanism after the AnyCalib package and PyTorch are installed."
