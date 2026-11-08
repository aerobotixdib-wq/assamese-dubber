"""Stage: split the original audio into voice and background (music + effects)
with Demucs (MIT licence), so the Assamese voice can go back over the music.

Uses the Demucs Python API and reads/writes audio with soundfile, avoiding
torchaudio's file I/O (which changed in newer torchaudio versions).
Install once:  pip install demucs
"""
from pathlib import Path

import numpy as np
import soundfile as sf


def separate_background(wav_path, out_path, device="auto", model_name="htdemucs", vocals_path=None):
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[separate] Demucs {model_name} on {device}", flush=True)
    model = get_model(model_name).to(device).eval()

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    if sr != model.samplerate:
        raise RuntimeError(f"expected {model.samplerate} Hz audio, got {sr}")
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    wav = torch.from_numpy(audio.T.copy())
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.inference_mode():
        sources = apply_model(model, wav[None].to(device), split=True, overlap=0.25,
                              progress=True)[0]
    sources = sources * (ref.std() + 1e-8) + ref.mean()
    names = model.sources                    # e.g. drums, bass, other, vocals
    background = sum(sources[i] for i, n in enumerate(names) if n != "vocals")
    sf.write(str(out_path), background.cpu().numpy().T, sr)
    if vocals_path and "vocals" in names:            # clean voice: better speaker detection
        sf.write(str(vocals_path), sources[names.index("vocals")].cpu().numpy().T, sr)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"[separate] saved {out_path}", flush=True)
