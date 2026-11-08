"""Stage: rebuild complete sentences from Whisper words and give each a time slot.

Input : transcript.json  (from transcribe.py)
Output: units.json       one dubbing unit per complete sentence

Each unit:
  start       first word starts (Assamese voice starts here)
  speech_end  last word ends
  slot_end    when the NEXT unit starts (or end of audio) - the voice may use
              the pause after the sentence, so slot = slot_end - start
  text        the complete source sentence

Rules (all tunable in config):
  - sentences come from pySBD, never from Whisper segment cuts
  - a sentence longer than max_unit_sec is split at its longest pause,
    else at a comma, else in the middle (so the voice never drifts far)
  - a pause longer than split_pause_sec inside a sentence also splits it
    (the speaker stopped; the dub should stop too)
  - tiny units (< min_unit_sec or < min_words) merge into a close neighbour
"""
import json
from pathlib import Path

import pysbd

PYSBD_LANGS = {"en", "hi", "mr", "bn", "ur", "ta", "te", "kn", "ml", "gu",
               "pa", "or", "as"}
CLAUSE_END = (",", ";", ":", "—", "–")


def _join(words):
    return "".join(w["word"] for w in words).strip()


def _sentence_groups(words, lang):
    """Group word dicts into pySBD sentences using character offsets."""
    full, spans = "", []
    for w in words:
        a = len(full)
        full += w["word"]
        spans.append((a, len(full)))

    seg = pysbd.Segmenter(language=lang if lang in PYSBD_LANGS else "en",
                          clean=False)
    bounds, cursor = [], 0
    for s in seg.segment(full):
        s = s.strip()
        if not s:
            continue
        a = full.find(s, cursor)
        if a < 0:                      # should not happen with clean=False
            raise RuntimeError(f"pySBD sentence not found in transcript: {s[:60]}")
        bounds.append((a, a + len(s)))
        cursor = a + len(s)

    groups, wi = [], 0
    for k, (a, b) in enumerate(bounds):
        g = []
        # a word belongs to the sentence that contains its first real character
        while wi < len(words):
            ws, we = spans[wi]
            first = ws + (len(words[wi]["word"]) - len(words[wi]["word"].lstrip()))
            if k + 1 < len(bounds) and first >= bounds[k + 1][0]:
                break
            g.append(words[wi])
            wi += 1
        if g:
            groups.append(g)
    if wi < len(words):
        groups.append(words[wi:])
    return groups


def _split_long(g, max_sec, split_pause):
    """Recursively split a word group at pauses / commas / middle."""
    if len(g) < 2:
        return [g]
    gaps = [(g[i + 1]["start"] - g[i]["end"], i) for i in range(len(g) - 1)]
    big_gap, at = max(gaps)
    dur = g[-1]["end"] - g[0]["start"]
    if big_gap >= split_pause:
        cut = at
    elif dur > max_sec:
        if big_gap >= 0.25:
            cut = at
        else:
            commas = [i for i in range(len(g) - 1)
                      if g[i]["word"].rstrip().endswith(CLAUSE_END)]
            mid = len(g) / 2
            cut = min(commas, key=lambda i: abs(i - mid)) if commas else int(mid) - 1
    else:
        return [g]
    return (_split_long(g[:cut + 1], max_sec, split_pause)
            + _split_long(g[cut + 1:], max_sec, split_pause))


def _merge_short(groups, min_sec, min_words, max_sec, max_gap=0.6):
    def short(g):
        return (g[-1]["end"] - g[0]["start"] < min_sec) or len(g) < min_words

    changed = True
    while changed and len(groups) > 1:
        changed = False
        for i, g in enumerate(groups):
            if not short(g):
                continue
            cands = []
            if i > 0:
                gap = g[0]["start"] - groups[i - 1][-1]["end"]
                cands.append((gap, i - 1))
            if i + 1 < len(groups):
                gap = groups[i + 1][0]["start"] - g[-1]["end"]
                cands.append((gap, i + 1))
            cands = [c for c in cands if c[0] <= max_gap]
            if not cands:
                continue
            _, j = min(cands)
            a, b = sorted((i, j))
            merged = groups[a] + groups[b]
            if merged[-1]["end"] - merged[0]["start"] > max_sec * 1.25:
                continue
            groups[a:b + 1] = [merged]
            changed = True
            break
    return groups


def _clean_words(words, fixes):
    """Drop Whisper's repeated words ("Companies Companies") and apply the
    optional user word fixes (e.g. {"Cubits": "Qubits"})."""
    out = []
    for w in words:
        w = dict(w)
        core = w["word"].strip()
        bare = core.strip(".,;:!?\"'")
        if bare in fixes:
            core = core.replace(bare, fixes[bare])
            w["word"] = w["word"][:len(w["word"]) - len(w["word"].lstrip())] + core
        if out:
            prev = out[-1]["word"].strip().strip(".,;:!?\"'").lower()
            if (w["word"].startswith(" ") and prev == core.lower()
                    and w["start"] - out[-1]["end"] < 0.5):
                out[-1]["end"] = w["end"]          # keep one copy, cover both times
                continue
        out.append(w)
    return out


def build_units(transcript_path, out_path, max_unit_sec=12.0, split_pause_sec=1.2,
                min_unit_sec=1.2, min_words=3, tail_room_sec=2.0, word_fixes=None):
    data = json.loads(Path(transcript_path).read_text(encoding="utf-8"))
    words, lang = data["words"], data.get("language", "en")
    if not words:
        raise RuntimeError("Transcript has no words.")
    words = _clean_words(words, word_fixes or {})

    groups = []
    for g in _sentence_groups(words, lang):
        groups += _split_long(g, max_unit_sec, split_pause_sec)
    groups = _merge_short(groups, min_unit_sec, min_words, max_unit_sec)

    units = []
    audio_end = data.get("duration") or words[-1]["end"]
    for i, g in enumerate(groups):
        start, speech_end = g[0]["start"], g[-1]["end"]
        if i + 1 < len(groups):
            slot_end = groups[i + 1][0]["start"]
        else:
            slot_end = min(audio_end, speech_end + tail_room_sec)
        low_conf = [w["word"].strip() for w in g if w.get("prob", 1) < 0.5]
        units.append({
            "id": i + 1,
            "start": round(start, 3),
            "speech_end": round(speech_end, 3),
            "slot_end": round(max(slot_end, speech_end), 3),
            "slot": round(max(slot_end, speech_end) - start, 3),
            "text": _join(g),
            "unsure_words": low_conf,
        })

    Path(out_path).write_text(json.dumps(units, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"[sentences] {len(words)} words -> {len(units)} units", flush=True)
    return units
