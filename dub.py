"""Assamese Dubber - dub an English / Hindi (or other) video into Assamese.

Free and local: Faster-Whisper -> pySBD sentences -> Sarvam-Translate (Ollama)
-> Indic Parler-TTS -> timing fit -> Demucs background music -> MP4 + SRT.

Usage (from this folder):
  python dub.py --url "https://youtu.be/VIDEO_ID"
  python dub.py --file "D:\\videos\\lesson.mp4"
Options:
  --from en|hi|auto   source language (default: auto-detect)
  --name NAME         project folder name (default: from the video)
  --no-music          do not keep the original background music
  --music-volume 1.0  original music level (default 1.0 = exactly as in the original video,
                      with the voice as loud as the original speaker); 0.5 = half, 1.5 = louder
  --whisper MODEL     Whisper model (default large-v3-turbo; "small" is faster)
  --max-speed 1.2     how much a line may be sped up before it is re-written
  --voice male        one voice for everyone: male (Amit) | female (Sita) | Amit | Sita | Rakesh | Poonam
  --speakers auto     detect each speaker (pyannote) and give each their own voice
  --speaker-voices "SPEAKER_01=female,SPEAKER_00=Rakesh"   fix a wrong guess
  --num-speakers 2    tell the speaker detector how many people talk (optional)
  --sync natural      natural (default): each line starts on time and finishes before the next
                      sentence starts (never drifts); strict: also ends exactly when the original
                      sentence ends (more lines sound rushed); flow: may run slightly late

Every stage saves its result in projects\\NAME\\ ; run the same command again
to continue after a crash. Only dub videos you own or may adapt.
"""
import argparse
import gc
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def run(cmd, **kw):
    print("  $", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], **kw)


def need(tool):
    if not shutil.which(tool):
        raise SystemExit(f"'{tool}' was not found. Install it and make sure it is on PATH.")


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def step(title):
    print("\n" + "=" * 60 + f"\n{title}\n" + "=" * 60, flush=True)


def mix_levels(setting, vocals_wav, bg_wav):
    """Match the ORIGINAL video's sound:
    - the Assamese voice gets the same loudness as the original speaker
    - the original music keeps its original level (factor 1.0), no ducking
    so the voice/music balance and overall loudness stay as in the original.
    --music-volume NUMBER multiplies the music (0.5 = half, 1.5 = louder).
    Returns (voice_lufs, music_gain)."""
    try:
        gain = float(setting)
    except (TypeError, ValueError):
        gain = 1.0
    voice_lufs = -16.0
    try:
        import pyloudnorm as pyln
        import soundfile as sf
        v, sr = sf.read(str(vocals_wav), dtype="float32")
        lv = pyln.Meter(sr).integrated_loudness(v)
        if lv != float("-inf"):
            voice_lufs = max(-20.0, min(-11.0, lv))
    except Exception as e:
        print(f"  (could not measure the original speech: {e}; voice at -16 LUFS)", flush=True)
    return voice_lufs, gain


def srt_time(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main():
    ap = argparse.ArgumentParser(description="Dub a video into Assamese (free, local).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="YouTube (or other yt-dlp) link")
    src.add_argument("--file", help="local video file, e.g. lesson.mp4")
    ap.add_argument("--from", dest="lang", default="auto")
    ap.add_argument("--name")
    ap.add_argument("--no-music", action="store_true")
    ap.add_argument("--whisper", default="large-v3-turbo")
    ap.add_argument("--max-speed", type=float, default=1.2)
    ap.add_argument("--voice", default="male")
    ap.add_argument("--speakers", choices=["off", "auto"], default="off")
    ap.add_argument("--speaker-voices", default="")
    ap.add_argument("--num-speakers", type=int, default=None)
    ap.add_argument("--sync", choices=["natural", "strict", "flow"], default="natural")
    ap.add_argument("--music-volume", type=float, default=1.0)
    a = ap.parse_args()
    default_voice = {"male": "Amit", "female": "Sita"}.get(a.voice.lower(), a.voice.capitalize())
    if default_voice not in ("Amit", "Sita", "Rakesh", "Poonam"):
        raise SystemExit("--voice must be male, female, Amit, Sita, Rakesh or Poonam")

    print("Reminder: only dub videos you own, have permission for, or that are licensed "
          "for changes (e.g. CC BY).\n", flush=True)
    need("ffmpeg")
    from dubber.translate_sarvam import check_ready
    check_ready()                                   # fail early if Ollama/model missing
    from dubber.translate_sarvam import gpu_warnings
    for w in gpu_warnings():                        # warn only - never stops the dub
        print(f"\n  !! {w}\n  !! Dubbing will be much slower until it is freed.", flush=True)

    name = a.name or (Path(a.file).stem if a.file else
                      (re.search(r"(?:v=|youtu\.be/|shorts/)([\w-]{6,})", a.url) or [None, "video"])[1])
    name = re.sub(r"[^\w\-]+", "_", name).strip("_") or "video"
    P = ROOT / "projects" / name
    P.mkdir(parents=True, exist_ok=True)
    print(f"Project folder: {P}")

    # 1. source video ----------------------------------------------------------
    step("1/8  Source video")
    video = P / "source.mp4"
    if a.file:
        video = Path(a.file).resolve()
        if not video.exists():
            raise SystemExit(f"File not found: {video}")
    elif not video.exists():
        run([sys.executable, "-m", "yt_dlp", "-f", "bv*+ba/b", "--merge-output-format", "mp4",
             "-o", str(video), a.url], check=True)
    print("Using", video)

    # 2. audio -----------------------------------------------------------------
    step("2/8  Extract audio")
    a16, a44 = P / "audio_16k.wav", P / "audio_44k.wav"
    if not a16.exists():
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vn", "-ac", "1", "-ar", "16000", a16], check=True)
    if not a44.exists():
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vn", "-ac", "2", "-ar", "44100", a44], check=True)

    # background music + clean voice (Demucs) - early, so speaker detection can use clean voice
    step("2b   Separate music and voice (Demucs)")
    bg, vocals, v16 = P / "background.wav", P / "vocals.wav", P / "vocals_16k.wav"
    if not bg.exists():
        try:
            from dubber.separate import separate_background
            separate_background(a44, bg, vocals_path=vocals)
        except ImportError:
            print("Demucs is not installed - continuing without background music.\n"
                  "  To keep music: pip install demucs", flush=True)
        except Exception as e:
            print(f"Background separation failed ({e}) - continuing without music.", flush=True)
        free_gpu()
    if vocals.exists() and not v16.exists():
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", vocals, "-ac", "1", "-ar", "16000", v16], check=True)

    # 3. transcription ---------------------------------------------------------
    step("3/8  Transcribe (Whisper, word timestamps)")
    from dubber.translate_sarvam import LANG_NAMES
    tr = P / "transcript.json"
    lang = None if a.lang == "auto" else a.lang
    if not tr.exists():
        base = [sys.executable, "-m", "dubber.transcribe", a16, tr, "--model", a.whisper]
        if lang:
            base += ["--language", lang]
        if run(base, cwd=ROOT).returncode != 0:
            print("GPU transcription failed - retrying on CPU (slower).", flush=True)
            run(base + ["--device", "cpu"], cwd=ROOT, check=True)
    data = json.loads(tr.read_text(encoding="utf-8"))
    src_lang = data["language"]
    if src_lang not in LANG_NAMES:
        print(f"Language '{src_lang}' is not supported by Sarvam directly; "
              "using Whisper's English translation first (experimental).", flush=True)
        tr_en = P / "transcript_en.json"
        if not tr_en.exists():
            run([sys.executable, "-m", "dubber.transcribe", a16, tr_en, "--model", a.whisper,
                 "--task", "translate", "--language", src_lang], cwd=ROOT, check=True)
        tr, src_lang = tr_en, "en"
    print("Source language:", src_lang)

    # 4. sentences -------------------------------------------------------------
    # speakers first, so a sentence is never shared by two voices
    turns_path = P / "speaker_turns.json"
    clean = v16 if v16.exists() else a16
    turns = None
    if a.speakers == "auto":
        step("3b   Who speaks when (pyannote)")
        if not turns_path.exists():
            cmd = [sys.executable, "-m", "dubber.speakers", clean, turns_path]
            if a.num_speakers:
                cmd += ["--num-speakers", a.num_speakers]
            if run(cmd, cwd=ROOT).returncode != 0:
                print("Speaker detection failed - using one voice for everyone.\n"
                      "  Accept the terms at https://huggingface.co/pyannote/speaker-diarization-community-1\n"
                      "  and set HF_TOKEN, then run again.", flush=True)
        if turns_path.exists():
            turns = json.loads(turns_path.read_text(encoding="utf-8"))

    step("4/8  Complete sentences (pySBD)")
    from dubber.sentences import build_units
    units_path = P / "units.json"
    fixes_file = ROOT / "word_fixes.json"            # optional {"Cubits": "Qubits"}
    fixes = json.loads(fixes_file.read_text(encoding="utf-8")) if fixes_file.exists() else {}
    if not units_path.exists():
        build_units(tr, units_path, word_fixes=fixes, turns=turns)
    units = json.loads(units_path.read_text(encoding="utf-8"))
    for u in units:
        print(f"  {u['id']:>3}  {u['start']:7.2f}s  {u['text']}")

    # 4b. speakers -------------------------------------------------------------
    voice_of, spk_summary = {}, {}
    if a.speakers == "auto":
        step("4b   Voice for each speaker")
        from dubber.speakers import assign
        if turns_path.exists():
            overrides = dict(kv.split("=", 1) for kv in a.speaker_voices.split(",") if "=" in kv)
            overrides = {k.strip(): v.strip().capitalize() if v.strip().lower() not in ("male", "female")
                         else v.strip().lower() for k, v in overrides.items()}
            turns = json.loads(turns_path.read_text(encoding="utf-8"))
            tmp_units = [dict(u) for u in units]
            spk_summary = assign(tmp_units, turns, clean, default_voice, overrides)
            voice_of = {u["id"]: (u["speaker"], u["voice"]) for u in tmp_units}
            for spk, info in spk_summary.items():
                print(f"  {spk}: {info['talk_sec']}s talking, pitch {info['pitch_hz']} Hz "
                      f"-> {info['gender']} -> voice {info['voice']}")
            (P / "speakers.json").write_text(json.dumps(spk_summary, indent=2), encoding="utf-8")

    # 5. translation -----------------------------------------------------------
    step("5/8  Translate to Assamese (Sarvam-Translate)")
    from dubber.translate_sarvam import translate_units, unload
    as_path = P / "assamese.json"
    translate_units(units_path, as_path, src_lang=src_lang)
    unload()

    # 6. voice + timing --------------------------------------------------------
    step("6/8  Assamese voice (Indic Parler-TTS) + timing")
    from dubber.tts import voice_units
    from dubber.fit import build_voice_track
    from dubber.shorten import shorten_long_lines
    final_path = P / "assamese_final.json"
    units = json.loads((final_path if final_path.exists() else as_path).read_text(encoding="utf-8"))
    for u in units:
        u["speaker"], u["voice"] = voice_of.get(u["id"], ("SPEAKER_00", default_voice))
    clips = voice_units(None, P / "voice", units=units, retry_ratio=a.max_speed, sync=a.sync)
    free_gpu()
    total = data["duration"]
    voice_wav, rep_path = P / "voice.wav", P / "fit_report.json"
    report = build_voice_track(units, clips, total, voice_wav, rep_path, max_speed=a.max_speed,
                               sync=a.sync)
    if not final_path.exists():
        long_ids = [r["id"] for r in report if r.get("speed", 1) > a.max_speed + 1e-3]
        if long_ids:
            print("Lines still too long:", long_ids, "- trying shorter wording", flush=True)
            from dubber.fit import target_window
            window = {u["id"]: target_window(u, a.sync) for u in units}
            units = shorten_long_lines(units, P / "voice", long_ids, src_lang, window=window,
                                       max_speed=a.max_speed)
            free_gpu()
            report = build_voice_track(units, clips, total, voice_wav, rep_path,
                                       max_speed=a.max_speed, sync=a.sync)
        final_path.write_text(json.dumps(units, ensure_ascii=False, indent=2), encoding="utf-8")

    # 8. final video, subtitles, report ----------------------------------------
    step("8/8  Final video + subtitles")
    import textwrap
    rep = {r["id"]: r for r in report}

    def wrap(t, w=42):
        return "\n".join(textwrap.wrap(t, w)) or t

    # Assamese subtitles follow the Assamese voice; original-language subtitles
    # follow the original speech (start -> last word).
    srt_as, srt_src, lines = [], [], []
    for n, u in enumerate(units, 1):
        r = rep.get(u["id"], {})
        st = u["start"] + r.get("start_shift", 0)
        en = st + r.get("clip_sec", u["slot"]) / max(r.get("speed", 1), 1e-3)
        srt_as.append(f"{n}\n{srt_time(max(st, 0))} --> {srt_time(en)}\n{wrap(u['assamese'], 40)}\n")
        srt_src.append(f"{n}\n{srt_time(u['start'])} --> {srt_time(u['speech_end'])}\n{wrap(u['text'])}\n")
        lines.append(f"[{u['id']}] {u['start']:.2f}s  {u.get('speaker', '')} voice {u.get('voice', '')}  "
                     f"speed {r.get('speed', '-')}x  {r.get('status', '-')}\n"
                     f"  EN: {u['text']}\n"
                     + (f"  EN (shortened{', minor details dropped' if u.get('condensed_strong') else ''}): "
                        f"{u['text_short']}\n" if u.get("text_short") else "")
                     + f"  AS: {u['assamese']}\n")
    sub_as = P / f"{name}_assamese.srt"
    sub_src = P / f"{name}_{src_lang}.srt"
    # utf-8-sig adds a BOM so Windows players read Assamese script correctly
    sub_as.write_text("\n".join(srt_as), encoding="utf-8-sig")
    sub_src.write_text("\n".join(srt_src), encoding="utf-8-sig")

    iso3 = {"en": "eng", "hi": "hin", "bn": "ben", "as": "asm"}.get(src_lang, "und")
    out = P / f"{name}_assamese.mp4"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", video]
    if bg.exists() and not a.no_music:
        # Same sound as the original: voice as loud as the original speaker,
        # original music at its original level, no ducking.
        voice_lufs, music_gain = mix_levels(a.music_volume, vocals, bg)
        print(f"  voice {voice_lufs:.1f} LUFS (as the original speaker), music x{music_gain:.2f}",
              flush=True)
        cmd += ["-i", bg, "-i", voice_wav, "-i", sub_as, "-i", sub_src, "-filter_complex",
                f"[2:a]aformat=channel_layouts=stereo,loudnorm=I={voice_lufs:.1f}:TP=-1.5:LRA=11,"
                "aresample=44100[v];"
                f"[1:a]volume={music_gain:.4f}[b];"
                "[b][v]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.97[a]",
                "-map", "0:v:0", "-map", "[a]", "-map", "3:0", "-map", "4:0"]
    else:
        voice_lufs, _ = mix_levels(1.0, vocals, bg)
        cmd += ["-i", voice_wav, "-i", sub_as, "-i", sub_src, "-filter_complex",
                f"[1:a]aformat=channel_layouts=stereo,loudnorm=I={voice_lufs:.1f}:TP=-1.5:LRA=11,"
                "aresample=44100[a]",
                "-map", "0:v:0", "-map", "[a]", "-map", "2:0", "-map", "3:0"]
    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-c:s", "mov_text",
            "-metadata:s:a:0", "language=asm",
            "-metadata:s:s:0", "language=asm", "-metadata:s:s:0", "title=Assamese",
            "-metadata:s:s:1", f"language={iso3}", "-metadata:s:s:1", f"title={src_lang}",
            "-disposition:s:0", "0", "-disposition:s:1", "0", out]
    # no "-shortest": it would end the video at the last subtitle / last word and
    # cut a silent ending (credits, music). The full original video length is kept.
    run(cmd, check=True)

    worst = max((r.get("speed", 1) for r in report), default=1)
    (P / "report.txt").write_text(
        f"Assamese dub report - {name}\nSource language: {src_lang}\n"
        f"Lines: {len(units)}   worst speed-up: {worst:.2f}x\n"
        + "".join(f"{k}: {v}\n" for k, v in spk_summary.items()) + "\n" + "\n".join(lines),
        encoding="utf-8")

    print(f"\nDONE\n  Video:     {out}  (subtitle tracks: Assamese + {src_lang}, switch them on in your player)"
          f"\n  Subtitles: {sub_as}\n             {sub_src}"
          f"\n  Report:    {P / 'report.txt'}  (worst speed-up {worst:.2f}x)")


if __name__ == "__main__":
    main()
