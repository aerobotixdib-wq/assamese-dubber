"""Stage: make lines that are too long fit their time window.

Strict sync keeps every line exactly inside the original sentence's time, so a
line that is too long would have to be spoken too fast. Instead, its WORDING is
made shorter, in up to `rounds` passes, each pass more aggressive
(from round 2, lines still needing > 1.5x may drop minor details):

  a) the English sentence is condensed by a small local model (condense.py),
     aiming at the length the time window allows, then translated by Sarvam
  b) Sarvam's alternative translations of the sentence
The candidates are voiced and the shortest audio is kept. A line stops as soon
as it fits at <= max_speed.
"""
import json
import os
from pathlib import Path

STRONG_ABOVE = 1.5     # lines needing more than 1.5x speed may lose minor details


def _save_manifest(voice_dir, u, sec):
    man = Path(voice_dir) / "manifest.json"
    if not man.exists():
        return
    rows = json.loads(man.read_text(encoding="utf-8"))
    for m in rows:
        if m["id"] == u["id"] and m.get("voice", "Amit") == u.get("voice", "Amit"):
            from .numbers_as import numbers_to_words
            m["assamese"], m["sec"], m["pace"] = u["assamese"], round(sec, 2), "slightly fast"
            m["spoken"] = numbers_to_words(u["assamese"])
    man.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def shorten_long_lines(units, voice_dir, long_ids, src_lang="en", window=None,
                       max_speed=1.2, rounds=3):
    """window = {unit_id: seconds available}. Returns the updated units."""
    from .translate_sarvam import variants, translate_text, unload
    from . import condense as cd
    from .tts import AssameseTTS
    from .fit import prepare_clip
    import soundfile as sf

    voice_dir = Path(voice_dir)
    window = window or {}
    use_condense = cd.available()
    if not use_condense:
        print(f"[shorten] '{cd.CONDENSE_MODEL}' not in Ollama - only Sarvam alternatives "
              f"(install: ollama pull {cd.CONDENSE_MODEL})", flush=True)
    by_id = {u["id"]: u for u in units}
    cur_sec = {i: prepare_clip(voice_dir / f"unit_{i:03d}.wav")[1] for i in long_ids}
    todo = [i for i in long_ids if cur_sec[i] / window.get(i, 1e9) > max_speed]

    for rnd in range(1, rounds + 1):
        if not todo:
            break
        print(f"[shorten] round {rnd}: lines {todo}", flush=True)

        # 1) shorter English (small model) - each round starts from the last one
        short_en, strong_ids = {}, set()
        if use_condense:
            for i in todo:
                u = by_id[i]
                base = u.get("text_short") or (u["text"] if src_lang == "en"
                                               else translate_text(u["text"], src_lang, "en"))
                need = cur_sec[i] / window[i] * (1.0 + 0.1 * rnd)   # a bit tighter each round
                strong = rnd >= 2 and cur_sec[i] / window[i] > STRONG_ABOVE
                s = cd.condense(base, need, strong=strong)
                if s:
                    short_en[i] = s
                    if strong:
                        strong_ids.add(i)
                    print(f"[shorten]  {i}: {base}\n[shorten]  {i}: -> {s}"
                          f"{'   (minor details dropped)' if strong else ''}", flush=True)
            unload(cd.CONDENSE_MODEL)

        # 2) Assamese candidates from Sarvam
        cands = {}
        for i in todo:
            u = by_id[i]
            c = []
            if i in short_en:
                c += [(a, short_en[i]) for a in variants(short_en[i], "en", n=2)]
            if rnd == 1:
                c += [(a, None) for a in variants(u["text"], src_lang, n=3)]
            seen, uniq = set(), []
            for a, s in sorted(c, key=lambda x: len(x[0])):
                if a not in seen and len(a) < len(u["assamese"]):
                    seen.add(a); uniq.append((a, s))
            cands[i] = uniq[:3]
        unload()
        if not any(cands.values()):
            break

        # 3) voice them, keep the shortest audio
        tts = AssameseTTS()
        for i in todo:
            u = by_id[i]
            best = None
            for k, (a, s) in enumerate(cands.get(i, [])):
                f = voice_dir / f"alt_{i:03d}_{k}.wav"
                sf.write(str(f), tts.speak(a, "slightly fast", u.get("voice", "Amit")), tts.sr)
                sec = prepare_clip(f)[1]
                print(f"[shorten]  {i} option {k+1} ({'short-EN' if s else 'alt'}): "
                      f"{sec:.2f}s for {window[i]:.2f}s window", flush=True)
                if sec < cur_sec[i] and (best is None or sec < best[0]):
                    best = (sec, a, s, f)
            if best:
                os.replace(best[3], voice_dir / f"unit_{i:03d}.wav")
                u.setdefault("assamese_original", u["assamese"])
                u["assamese"] = best[1]
                if best[2]:
                    u["text_short"] = best[2]
                    u["condensed_strong"] = i in strong_ids
                cur_sec[i] = best[0]
                _save_manifest(voice_dir, u, best[0])
            for k in range(3):
                f = voice_dir / f"alt_{i:03d}_{k}.wav"
                if f.exists():
                    f.unlink()
        del tts
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        todo = [i for i in todo if cur_sec[i] / window[i] > max_speed]

    if todo:
        print(f"[shorten] no shorter wording found for {todo} - these lines will be spoken faster", flush=True)
    return units
