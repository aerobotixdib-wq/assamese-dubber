"""Stage: speech-to-text with word timestamps (Faster-Whisper).

Output (transcript.json):
{
  "language": "en",
  "duration": 86.75,
  "words": [{"start": 0.0, "end": 0.42, "word": "Quantum", "prob": 0.98}, ...]
}
Only words are kept; Whisper's own segments are NOT used as dubbing units,
because they often cut sentences in half.
"""
import json
from pathlib import Path


PROMPTS = {
    "en": "Hello, everyone. Welcome to today's video. Let's begin.",
    "hi": "नमस्ते दोस्तों। आज के वीडियो में आपका स्वागत है। चलिए शुरू करते हैं।",
}


def pick_device(requested: str = "auto"):
    """Return (device, compute_type) for Faster-Whisper."""
    if requested in ("cpu", "cuda"):
        dev = requested
    else:
        try:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cpu"
    return dev, ("float16" if dev == "cuda" else "int8")


def transcribe(audio_path, out_path, model_name="small", language=None,
               device="auto", beam_size=5, task="transcribe"):
    from faster_whisper import WhisperModel

    dev, ctype = pick_device(device)
    print(f"[transcribe] model={model_name} device={dev} ({ctype})", flush=True)
    try:   # use the cached copy if present, so no network call can hang
        model = WhisperModel(model_name, device=dev, compute_type=ctype, local_files_only=True)
    except Exception:
        model = WhisperModel(model_name, device=dev, compute_type=ctype)

    # Read 16 kHz mono WAV with soundfile instead of PyAV (avoids PyAV version breaks)
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16000:
        raise RuntimeError("transcribe() expects a 16 kHz WAV (dub.py makes one)")

    # A short, well-punctuated prompt makes Whisper write punctuation.
    # Without it, fast or casual speech (YouTube Shorts) often comes back as one
    # long unpunctuated run - and then no complete sentences can be found.
    lang = language
    if lang is None:
        try:
            lang = model.detect_language(audio)[0]
        except Exception:
            lang = None
    prompt = PROMPTS.get(lang) if task == "transcribe" else PROMPTS["en"]

    segments, info = model.transcribe(
        audio,
        language=language,               # None = auto-detect
        initial_prompt=prompt,
        task=task,                       # "translate" = any language -> English
        beam_size=beam_size,
        word_timestamps=True,
        vad_filter=True,                 # skips music/silence, fewer hallucinations
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,  # stops repeated-text loops on long videos
    )

    words = []
    for seg in segments:
        for w in (seg.words or []):
            # Keep Whisper's own spacing: " word" starts a new word, "-intuitive"
            # continues the previous one. Joining raw tokens rebuilds the text exactly.
            if not w.word.strip() or w.start is None or w.end is None:
                continue
            words.append({"start": round(w.start, 3), "end": round(w.end, 3),
                          "word": w.word, "prob": round(w.probability, 3)})
        print(f"[transcribe] {seg.end:7.2f}s  {seg.text.strip()[:70]}", flush=True)

    if not words:
        raise RuntimeError("Whisper found no speech in this audio.")

    result = {"language": info.language, "duration": round(info.duration, 3),
              "words": words}
    Path(out_path).write_text(json.dumps(result, ensure_ascii=False, indent=1),
                              encoding="utf-8")
    print(f"[transcribe] {len(words)} words, language={info.language}", flush=True)
    return result


if __name__ == "__main__":
    # Run as a separate process so a GPU/driver crash can fall back to CPU.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("audio"); ap.add_argument("out")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--language", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--task", default="transcribe")
    a = ap.parse_args()
    transcribe(a.audio, a.out, a.model, a.language, a.device, task=a.task)
