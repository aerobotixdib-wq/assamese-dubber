"""Stage: fit each Assamese clip into its time slot and build the voice track.

Order of tricks, gentlest first:
  1. trim the silence TTS adds before/after speech
  2. use the pause after the sentence (slot = until the next sentence starts)
  3. start up to `lead_in_sec` early if the previous clip finished already
  4. speed up with FFmpeg atempo (pitch is kept), but never above `max_speed`
  5. still too long? -> report it (status "too_long"); the TTS stage can then
     regenerate at a faster pace or with a shorter translation, and only the
     final pass is allowed to spill into the next pause (pushing it slightly).

Nothing is ever cut off mid-word.
"""
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100


def load_mono(path, sr=SR):
    """Read any audio file as mono float32 at `sr` (uses FFmpeg to resample)."""
    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if file_sr != sr:
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / "r.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(path),
                            "-ac", "1", "-ar", str(sr), str(out)], check=True)
            audio, _ = sf.read(str(out), dtype="float32")
    return audio


def trim_silence(audio, sr=SR, threshold_db=-40.0, pad_sec=0.05, frame_sec=0.02):
    """Cut leading/trailing silence, keeping a small natural pad."""
    if audio.size == 0:
        return audio
    n = max(1, int(frame_sec * sr))
    frames = len(audio) // n
    if frames == 0:
        return audio
    rms = np.sqrt(np.mean(audio[:frames * n].reshape(frames, n) ** 2, axis=1) + 1e-12)
    peak = rms.max()
    loud = np.where(20 * np.log10(rms / peak + 1e-12) > threshold_db)[0]
    if loud.size == 0:
        return audio
    pad = int(pad_sec * sr)
    a = max(0, loud[0] * n - pad)
    b = min(len(audio), (loud[-1] + 1) * n + pad)
    return audio[a:b]


def shorten_pauses(audio, sr=SR, threshold_db=-35.0, max_pause_sec=0.25,
                   frame_sec=0.02):
    """Shorten long silent gaps INSIDE a clip to max_pause_sec.
    TTS often leaves 0.4-0.8 s gaps at commas; cutting them is inaudible
    and saves time before any speed-up is needed."""
    n = max(1, int(frame_sec * sr))
    frames = len(audio) // n
    if frames < 3:
        return audio
    rms = np.sqrt(np.mean(audio[:frames * n].reshape(frames, n) ** 2, axis=1) + 1e-12)
    quiet = 20 * np.log10(rms / rms.max() + 1e-12) < threshold_db
    keep_frames = int(max_pause_sec / frame_sec)
    pieces, i = [], 0
    while i < frames:
        j = i
        while j < frames and quiet[j] == quiet[i]:
            j += 1
        a, b = i * n, j * n
        if quiet[i] and (j - i) > keep_frames and i > 0 and j < frames:
            half = keep_frames * n // 2
            pieces += [audio[a:a + half], audio[b - half:b]]
        else:
            pieces.append(audio[a:b])
        i = j
    pieces.append(audio[frames * n:])
    return np.concatenate(pieces)


def atempo(audio, factor, sr=SR):
    """Speed up/down without changing pitch (FFmpeg atempo)."""
    if abs(factor - 1.0) < 0.01:
        return audio
    chain, f = [], factor
    while f > 2.0:                       # atempo is 0.5..2.0 per filter on older FFmpeg
        chain.append("atempo=2.0")
        f /= 2.0
    chain.append(f"atempo={f:.6f}")
    with tempfile.TemporaryDirectory() as t:
        src, dst = Path(t) / "in.wav", Path(t) / "out.wav"
        sf.write(str(src), audio, sr)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
                        "-af", ",".join(chain), "-ar", str(sr), "-ac", "1", str(dst)],
                       check=True)
        out, _ = sf.read(str(dst), dtype="float32")
    return out


def plan_unit(clip_sec, start, slot_end, prev_end, lead_in_sec=0.3, gap_sec=0.08,
              max_speed=1.2, hard_max_speed=1.35):
    """Decide where a clip goes and how much to speed it up.
    Returns dict(place_at, speed, status).
      fits / fits_early : no speed change
      sped_up           : speed <= max_speed (sounds natural)
      too_long          : needs more than max_speed -> regenerate if possible;
                          final placement uses up to hard_max_speed and may
                          spill a little into the next sentence."""
    earliest = max(start - lead_in_sec, prev_end + gap_sec, 0.0)
    place_at = max(earliest, min(start, slot_end))
    if clip_sec <= slot_end - place_at:
        return {"place_at": place_at, "speed": 1.0, "status": "fits", "need": 1.0}
    room = max(slot_end - earliest, 1e-3)
    if clip_sec <= room:
        return {"place_at": earliest, "speed": 1.0, "status": "fits_early", "need": 1.0}
    need = clip_sec / room
    if need <= max_speed:
        return {"place_at": earliest, "speed": need, "status": "sped_up", "need": need}
    return {"place_at": earliest, "speed": min(need, hard_max_speed),
            "status": "too_long", "need": need}


def prepare_clip(path, sr=SR):
    """Load a TTS clip and remove wasted silence. Returns (audio, seconds)."""
    audio = shorten_pauses(trim_silence(load_mono(path, sr), sr), sr)
    return audio, len(audio) / sr


LEAD_IN = 0.3      # natural: a line may start up to 0.3 s early (if the previous one is done)
GAP = 0.05         # natural: silence kept before the next line starts


def target_window(u, sync="natural"):
    """Seconds the Assamese line may use.
    strict:  exactly the original sentence (first word -> last word).
    natural: up to LEAD_IN early, and until just before the next sentence starts.
    flow:    until the next sentence starts (may run late)."""
    if sync == "strict":
        return max(u["speech_end"] - u["start"], 0.3)
    if sync == "natural":
        return max(u["slot_end"] - GAP - u["start"] + LEAD_IN, 0.3)
    return max(u["slot_end"] - u["start"], 0.3)


def build_voice_track(units, clips, total_sec, out_wav, report_path=None,
                      lead_in_sec=0.3, max_speed=1.2, hard_max_speed=1.35, sr=SR,
                      sync="natural", min_speed=0.85):
    """units: list from sentences.py (start, speech_end, slot_end...).
    clips: {unit_id: path_to_wav}. Returns the report list.

    sync="strict": every Assamese line starts exactly when the original sentence
    starts and ends when it ends - the clip is time-stretched to fit, so the
    dub can never drift. Lines needing more than max_speed are reported as
    "too_long" so the shortening stage rewrites them first.
    sync="natural" (default): every line starts at its original time (up to
    0.3 s early if needed) and must finish before the NEXT sentence starts -
    it may use the pause after its sentence, but can never push the next line,
    so the dub never drifts.
    sync="flow": older behaviour (may borrow pauses, may run a little late)."""
    timeline = np.zeros(int((total_sec + 5) * sr), dtype=np.float32)
    prev_end, report = 0.0, []
    for u in units:
        path = clips.get(u["id"])
        if not path:
            report.append({"id": u["id"], "status": "missing"})
            continue
        audio, raw = prepare_clip(path, sr)
        if sync == "strict":
            win = target_window(u, "strict")
            need = raw / win
            speed = max(need, min_speed)              # never slower than min_speed
            status = ("fits" if need <= 1.0 else "sped_up" if need <= max_speed
                      else "too_long")
            p = {"place_at": u["start"], "speed": speed, "status": status, "need": need}
        elif sync == "natural":
            earliest = max(u["start"] - LEAD_IN, prev_end + GAP, 0.0)
            latest_end = max(u["slot_end"] - GAP, u["speech_end"])
            if raw <= latest_end - u["start"]:            # fits from its own start
                p = {"place_at": u["start"], "speed": 1.0, "status": "fits", "need": 1.0}
            elif raw <= latest_end - earliest:            # fits if it starts a bit early
                p = {"place_at": latest_end - raw, "speed": 1.0, "status": "fits_early", "need": 1.0}
            else:
                need = raw / max(latest_end - earliest, 0.1)
                p = {"place_at": earliest, "speed": need,
                     "status": "sped_up" if need <= max_speed else "too_long", "need": need}
            p["latest_end"] = latest_end
        else:
            p = plan_unit(raw, u["start"], u["slot_end"], prev_end, lead_in_sec,
                          max_speed=max_speed, hard_max_speed=hard_max_speed)
        audio = atempo(audio, p["speed"], sr)
        if sync == "strict":                          # exact length, no spill
            audio = audio[:int(round(target_window(u, "strict") * sr))]
        elif sync == "natural":                       # never past the next sentence
            audio = audio[:max(int(round((p["latest_end"] - p["place_at"]) * sr)), 0)]
        a = int(round(p["place_at"] * sr))
        need = a + len(audio)
        if need > len(timeline):
            timeline = np.pad(timeline, (0, need - len(timeline)))
        timeline[a:need] += audio
        end = p["place_at"] + len(audio) / sr
        overflow = max(0.0, end - u["slot_end"])
        report.append({"id": u["id"], "clip_sec": round(raw, 2),
                       "slot_sec": round(target_window(u, sync), 2),
                       "speed": round(p["speed"], 3), "start_shift": round(p["place_at"] - u["start"], 2),
                       "overflow_sec": round(overflow, 2), "status": p["status"]})
        prev_end = end

    timeline = timeline[:int(total_sec * sr)] if total_sec else timeline
    peak = float(np.max(np.abs(timeline))) if timeline.size else 0
    if peak > 0.98:
        timeline *= 0.98 / peak
    sf.write(str(out_wav), timeline, sr)
    if report_path:
        Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    return report
