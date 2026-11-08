# Assamese Dubber - one-time setup for Windows 10/11
# Run from this folder:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
# Safe to run again: it skips what is already installed.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot
function Say($msg)  { Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "  OK  $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "  !!  $msg" -ForegroundColor Yellow }

# 1. Python 3.11 --------------------------------------------------------------
Say "1/6  Python 3.11"
$py = $null
try { & py -3.11 --version *> $null; if ($LASTEXITCODE -eq 0) { $py = "py -3.11" } } catch {}
if (-not $py) {
    Warn "Python 3.11 not found. Install it from https://www.python.org/downloads/release/python-3119/"
    Warn "(tick 'Add python.exe to PATH'), then run this script again."
    exit 1
}
Ok (& py -3.11 --version)

# 2. Virtual environment ----------------------------------------------------
Say "2/6  Project environment (.venv)"
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    & py -3.11 -m venv .venv
    Ok "created .venv"
} else { Ok ".venv already exists" }
$vpy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
& $vpy -m pip install --upgrade pip --quiet

# 3. PyTorch (GPU or CPU) ---------------------------------------------------
Say "3/6  PyTorch"
$gpu = $false
try { & nvidia-smi *> $null; if ($LASTEXITCODE -eq 0) { $gpu = $true } } catch {}
if ($gpu) {
    Ok "NVIDIA GPU found - installing CUDA 12.8 build (about 2.5 GB)"
    & $vpy -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
} else {
    Warn "No NVIDIA GPU found - installing CPU build. Dubbing will be MUCH slower."
    & $vpy -m pip install torch==2.11.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu
}
if ($LASTEXITCODE -ne 0) { Warn "PyTorch install failed."; exit 1 }

# 4. Python packages --------------------------------------------------------
Say "4/6  Python packages (Whisper, Parler-TTS, Demucs, pyannote ...)"
$gitOk = $false
try { & git --version *> $null; if ($LASTEXITCODE -eq 0) { $gitOk = $true } } catch {}
if (-not $gitOk) {
    Warn "Git is needed to install Parler-TTS. Install it:  winget install Git.Git   then run this script again."
    exit 1
}
# Exact, tested versions - installed without dependency solving (pip cannot solve
# this mix of libraries on its own; it fails with "resolution-too-deep").
& $vpy -m pip install --no-deps -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { Warn "Package install failed - see the error above."; exit 1 }
Ok "packages installed"

# 5. FFmpeg -----------------------------------------------------------------
Say "5/6  FFmpeg"
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { Ok "ffmpeg found" }
else {
    Warn "FFmpeg not found. Install it with:   winget install Gyan.FFmpeg"
    Warn "then CLOSE and reopen PowerShell so it is on PATH."
}

# 6. Ollama + translation models --------------------------------------------
Say "6/6  Ollama models (Sarvam-Translate + Llama 3.2 3B)"
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    & ollama pull hf.co/fischerman/sarvam-translate-gguf:Q4_K_S
    & ollama pull llama3.2:3b
    Ok "translation models ready"
} else {
    Warn "Ollama not found. Install it with:   winget install Ollama.Ollama"
    Warn "then run this script again to download the translation models."
}

# Check ---------------------------------------------------------------------
Say "Checking that everything loads"
& $vpy -c "import numpy, torch, parler_tts, faster_whisper, demucs, pysbd; print('Python packages OK - GPU:', torch.cuda.is_available())"

Write-Host ""
Write-Host "Setup finished. Two one-time steps on Hugging Face (free account):" -ForegroundColor Cyan
Write-Host "  1. Accept the terms:  https://huggingface.co/ai4bharat/indic-parler-tts"
Write-Host "     (for --speakers auto also: https://huggingface.co/pyannote/speaker-diarization-community-1)"
Write-Host "  2. Create a READ token: https://huggingface.co/settings/tokens"
Write-Host "     and before the FIRST dub run, paste it (it stays hidden) with:"
Write-Host "     `$env:HF_TOKEN = [System.Net.NetworkCredential]::new('', (Read-Host 'Paste your Hugging Face token' -AsSecureString)).Password"
Write-Host ""
Write-Host "Then dub a video:"
Write-Host "  .\dub.bat --url `"https://youtu.be/VIDEO_ID`""
Write-Host "  .\dub.bat --file `"D:\videos\lesson.mp4`" --speakers auto"
