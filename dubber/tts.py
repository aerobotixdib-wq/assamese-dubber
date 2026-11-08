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

    def speak(self, text, pace="moderate", name="Amit"):
        torch = self.torch
        d = self.dtok(self.voice.format(pace=pace, name=name), return_tensors="pt").to(self.device)
        p = self.tok(text, return_tensors="pt").to(self.device)
        torch.manual_seed(self.seed)
        with torch.inference_mode():
            audio = self.model.generate(input_ids=d.input_ids, attention_mask=d.attention_mask,
                                        prompt_input_ids=p.input_ids,
                                        prompt_attention_mask=p.attention_mask)
        return audio.float().cpu().numpy().squeeze()


def voice_units(units_path, out_dir, device="auto", retry_ratio=1.2, units=None, sync="natural"):
    import soundfile as sf
    from .fit import prepare_clip

    if units is None:
        units = json.loads(Path(units_path).read_text(encoding="utf-8"))
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    man_path = out_dir / "manifest.json"
    manifest = {}
    if man_path.exists():
        for m in json.loads(man_path.read_text(encoding="utf-8")):
            if (out_dir / m["file"]).exists():
                manifest[(m["id"], m["assamese"], m.get("voice", "Amit"))] = m

    tts = None
    for u in units:
        key = (u["id"], u["assamese"], u.get("voice", "Amit"))
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
        manifest[key] = {"id": u["id"], "assamese": u["assamese"], "voice": u.get("voice", "Amit"),
                         "file": final.name,
                         "pace": best["pace"], "sec": best["sec"], "slot": round(slot, 2),
                         "tries": tried}
        tmp = out_dir / "manifest.tmp"
        tmp.write_text(json.dumps(list(manifest.values()), ensure_ascii=False, indent=2),
                       encoding="utf-8")
        os.replace(tmp, man_path)
    def k(u):
        return (u["id"], u["assamese"], u.get("voice", "Amit"))
    return {u["id"]: str(out_dir / manifest[k(u)]["file"]) for u in units if k(u) in manifest}
