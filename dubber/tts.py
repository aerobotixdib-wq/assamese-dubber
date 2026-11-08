"""Stage: Assamese speech with AI4Bharat Indic Parler-TTS (Apache-2.0).

One clip per sentence, same voice every time (fixed description + fixed seed).
If a clip is clearly too long for its slot, it is regenerated once with a
faster speaking pace before the timing stage has to speed it up.
Resumable: finished clips are kept in manifest.json.
"""
import json
import os
import time
from pathlib import Path

MODEL_ID = "ai4bharat/indic-parler-tts"
VOICE = ("{name} speaks in Assamese with a clear, natural voice, at a {pace} pace, "
         "in a calm educational narration style. The recording has very clear audio "
         "with no background noise.")
PACES = ["moderate", "slightly fast", "fast"]
MAX_CPS = 21          # characters per second above which a take is treated as cut short
MAX_PIECE = 90        # longest text (characters) given to Parler in one go


def split_for_tts(text, limit=MAX_PIECE):
    """Split at sentence ends, then at commas, into pieces of <= limit chars."""
    import re
    sents = [x for x in re.split(r"(?<=[।?!.])\s+", text.strip()) if x]
    pieces = []
    for s in sents:
        if len(s) <= limit:
            pieces.append(s); continue
        cur = ""
        for part in re.split(r"(?<=,)\s+", s):
            if cur and len(cur) + 1 + len(part) > limit:
                pieces.append(cur); cur = part
            else:
                cur = f"{cur} {part}".strip()
        if cur:
            pieces.append(cur)
    merged = []                                   # no tiny leftover pieces
    for x in pieces:
        if merged and len(x) < 15 and len(merged[-1]) + len(x) < limit + 20:
            merged[-1] += " " + x
        else:
            merged.append(x)
    return merged or [text]


class AssameseTTS:
    def __init__(self, device="auto", voice=VOICE, seed=1234):
        import torch
        from transformers import AutoTokenizer
        from parler_tts import ParlerTTSForConditionalGeneration
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device, self.voice, self.seed = device, voice, seed
        print(f"[tts] loading {MODEL_ID} on {device}", flush=True)
        def load(local):
            m = ParlerTTSForConditionalGeneration.from_pretrained(
                MODEL_ID, torch_dtype=torch.float32, local_files_only=local)
            t = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=local)
            d = AutoTokenizer.from_pretrained(m.config.text_encoder._name_or_path,
                                              local_files_only=local)
            return m, t, d
        try:        # cached copy first: no network call that could hang
            m, self.tok, self.dtok = load(True)
        except Exception:
            print("[tts] downloading model (first run; needs HF_TOKEN and accepted terms)", flush=True)
            m, self.tok, self.dtok = load(False)
        self.model = m.to(device).eval()
        self.sr = self.model.config.sampling_rate

    def _generate(self, text, pace, name, seed):
        torch = self.torch
        d = self.dtok(self.voice.format(pace=pace, name=name), return_tensors="pt").to(self.device)
        p = self.tok(text, return_tensors="pt").to(self.device)
        torch.manual_seed(seed)
        with torch.inference_mode():
            audio = self.model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                        prompt_input_ids=p.input_ids,
                                        prompt_attention_mask=p.attention_mask)
        return audio.float().cpu().numpy().squeeze()

    def _checked(self, text, pace, name):
        """Parler-TTS sometimes stops early and skips words. Assamese speech is
        about 12-17 characters per second; much faster than MAX_CPS means words
        were dropped, so try again with another seed (keep the longest take)."""
        best = None
        for k in range(3):
            audio = self._generate(text, pace, name, self.seed + k)
            sec = max(len(audio) / self.sr, 0.01)
            if best is None or len(audio) > len(best):
                best = audio
            if len(text.strip()) / sec <= MAX_CPS:
                return audio
            print(f"[tts] take {k+1} too short ({sec:.2f}s for {len(text)} chars) - retrying", flush=True)
        return best

    def speak(self, text, pace="moderate", name="Amit"):
        """Long lines are spoken in short pieces (Parler is more reliable on
        short text) and joined with a tiny natural pause."""
        import numpy as np
        from .numbers_as import numbers_to_words
        pieces = split_for_tts(numbers_to_words(text))     # "২০০,০০০" -> "দুই লাখ"
        gap = np.zeros(int(0.12 * self.sr), dtype=np.float32)
        out = []
        for i, piece in enumerate(pieces):
            if i:
                out.append(gap)
            out.append(np.asarray(self._checked(piece, pace, name), dtype=np.float32))
        return np.concatenate(out) if out else np.zeros(1, dtype=np.float32)


def voice_units(units_path, out_dir, device="auto", retry_ratio=1.2, units=None, sync="natural"):
    import soundfile as sf
    from .fit import prepare_clip
    from .numbers_as import numbers_to_words

    if units is None:
        units = json.loads(Path(units_path).read_text(encoding="utf-8"))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.json"
    manifest = {}
    if man_path.exists():
        for m in json.loads(man_path.read_text(encoding="utf-8")):
            if (out_dir / m["file"]).exists():
                manifest[(m["id"], m.get("spoken", m["assamese"]), m.get("voice", "Amit"))] = m

    tts = None
    for u in units:
        key = (u["id"], numbers_to_words(u["assamese"]), u.get("voice", "Amit"))
        if key in manifest:
            continue
        if tts is None:
            tts = AssameseTTS(device)
        from .fit import target_window
        slot = target_window(u, sync)
        tried, best = [], None
        for pace in PACES:
            t0 = time.time()
            audio = tts.speak(u["assamese"], pace, u.get("voice", "Amit"))
            f = out_dir / f"unit_{u['id']:03d}_{pace.replace(' ', '_')}.wav"
            sf.write(str(f), audio, tts.sr)
            _, dur = prepare_clip(f)
            tried.append({"pace": pace, "sec": round(dur, 2), "file": f.name})
            print(f"[tts] {u['id']}/{len(units)} pace={pace:13s} {dur:5.2f}s for slot {slot:5.2f}s "
                  f"({time.time()-t0:.1f}s)", flush=True)
            if best is None or dur < best["sec"]:
                best = tried[-1]
            if dur <= slot * retry_ratio:
                break
        final = out_dir / f"unit_{u['id']:03d}.wav"
        os.replace(out_dir / best["file"], final)
        for t in tried:                               # remove the takes not chosen
            extra = out_dir / t["file"]
            if extra.exists():
                extra.unlink()
        manifest[key] = {"id": u["id"], "assamese": u["assamese"], "spoken": key[1],
                         "voice": u.get("voice", "Amit"),
                         "file": final.name,
                         "pace": best["pace"], "sec": best["sec"], "slot": round(slot, 2),
                         "tries": tried}
        tmp = out_dir / "manifest.tmp"
        tmp.write_text(json.dumps(list(manifest.values()), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, man_path)
    def k(u):
        return (u["id"], numbers_to_words(u["assamese"]), u.get("voice", "Amit"))
    return {u["id"]: str(out_dir / manifest[k(u)]["file"]) for u in units if k(u) in manifest}
