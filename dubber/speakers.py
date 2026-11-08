"""Stage: who speaks each sentence, and which Assamese voice they get.

1. pyannote (speaker diarization) finds "who spoke when" -> SPEAKER_00, 01, ...
   Model: pyannote/speaker-diarization-community-1 (free; accept its terms on
   Hugging Face once).  Install: pip install pyannote.audio==4.0.7 "numpy<2"
2. Each sentence gets the speaker who talks most during it.
3. Each speaker's median voice pitch decides male / female
   (men ~85-155 Hz, women ~165-255 Hz; threshold 160 Hz).
4. Voices: male -> Amit, Rakesh;  female -> Sita, Poonam  (Indic Parler-TTS).

Run as a separate process (python -m dubber.speakers ...) so its many
dependencies and any crash stay out of the main pipeline.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

MODEL = "pyannote/speaker-diarization-community-1"
VOICES = {"male": ["Amit", "Rakesh"], "female": ["Sita", "Poonam"]}
PITCH_SPLIT_HZ = 160.0


# ---------- 1. diarization (needs pyannote) --------------------------------
def diarize(wav16k, out_json, device="auto", num_speakers=None):
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub" / \
        ("models--" + MODEL.replace("/", "--"))
    if cache.exists():
        os.environ["HF_HUB_OFFLINE"] = "1"          # cached: never wait on the network
    import torch
    from pyannote.audio import Pipeline

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[speakers] {MODEL} on {device}", flush=True)
    pipe = Pipeline.from_pretrained(MODEL, token=os.environ.get("HF_TOKEN") or None)
    if pipe is None:
        raise SystemExit("Could not load pyannote. Accept the terms at "
                         f"https://huggingface.co/{MODEL} and set HF_TOKEN.")
    pipe.to(torch.device(device))
    audio, sr = sf.read(str(wav16k), dtype="float32", always_2d=True)
    wave = torch.from_numpy(audio.mean(axis=1))[None]
    kw = {"num_speakers": num_speakers} if num_speakers else {}
    out = pipe({"waveform": wave, "sample_rate": sr}, **kw)
    ann = getattr(out, "exclusive_speaker_diarization", None) or \
        getattr(out, "speaker_diarization", out)
    segs = [{"start": round(t.start, 3), "end": round(t.end, 3), "speaker": spk}
            for t, _, spk in ann.itertracks(yield_label=True)]
    Path(out_json).write_text(json.dumps(segs, indent=1), encoding="utf-8")
    print(f"[speakers] {len(segs)} turns, {len({s['speaker'] for s in segs})} speakers", flush=True)


# ---------- 2-4. assignment (numpy only) ------------------------------------
def _f0_frames(x, sr, a, b):
    """Rough pitch (Hz) of voiced 40 ms frames between seconds a and b."""
    seg = x[int(a * sr):int(b * sr)]
    n, hop = int(0.04 * sr), int(0.02 * sr)
    lo, hi = int(sr / 400), int(sr / 70)
    if len(seg) < n:
        return []
    frames = np.lib.stride_tricks.sliding_window_view(seg, n)[::hop]
    rms = np.sqrt((frames ** 2).mean(axis=1))
    thr = max(rms.max() * 0.2, 1e-4)
    out = []
    for f in frames[rms > thr]:
        f = f - f.mean()
        ac = np.fft.irfft(np.abs(np.fft.rfft(f, 2 * n)) ** 2)[:n]
        if ac[0] <= 0:
            continue
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] / ac[0] > 0.45:                   # clearly periodic = voiced
            out.append(sr / lag)
    return out


def assign(units, turns, wav16k, default_voice="Amit", overrides=None):
    """Adds 'speaker' and 'voice' to every unit. Returns a speaker summary."""
    overrides = overrides or {}
    for u in units:
        best, spk = 0.0, None
        for t in turns:
            ov = min(u["speech_end"], t["end"]) - max(u["start"], t["start"])
            if ov > best:
                best, spk = ov, t["speaker"]
        if spk is None and turns:                     # no overlap: nearest turn
            spk = min(turns, key=lambda t: abs(t["start"] - u["start"]))["speaker"]
        u["speaker"] = spk or "SPEAKER_00"

    x, sr = sf.read(str(wav16k), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    talk, f0 = {}, {}
    for t in turns:
        talk[t["speaker"]] = talk.get(t["speaker"], 0) + t["end"] - t["start"]
        f0.setdefault(t["speaker"], []).extend(_f0_frames(x, sr, t["start"], t["end"]))

    used = {"male": 0, "female": 0}
    summary = {}
    for spk in sorted(talk, key=talk.get, reverse=True):     # main speaker first
        hz = float(np.median(f0[spk])) if f0.get(spk) else None
        gender = "female" if (hz and hz >= PITCH_SPLIT_HZ) else "male"
        o = overrides.get(spk)
        if o in ("male", "female"):
            gender, o = o, None
        if o:                                                  # a voice name
            voice = o
        else:
            names = VOICES[gender]
            voice = names[used[gender] % len(names)]
            used[gender] += 1
        summary[spk] = {"talk_sec": round(talk[spk], 1),
                        "pitch_hz": round(hz) if hz else None,
                        "gender": gender, "voice": voice}
    for u in units:
        u["voice"] = summary.get(u["speaker"], {}).get("voice", default_voice)
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("wav16k"); ap.add_argument("out")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--num-speakers", type=int, default=None)
    a = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    diarize(a.wav16k, a.out, a.device, a.num_speakers)
