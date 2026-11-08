"""Stage (alternative): translate complete sentences with Sarvam-Translate via Ollama.

Sarvam-Translate (Sarvam AI, built on Gemma 3 4B, GPL-3.0) reads whole sentences
with context, so it may sound more natural than IndicTrans2.
Runs locally through Ollama - no API key, no internet after download.

    ollama pull hf.co/fischerman/sarvam-translate-gguf:Q4_K_S

The open model translates English <-> Indian languages. For Hindi videos,
pivot: Hindi -> English -> Assamese.
"""
import json
import os
import time
import urllib.request
from pathlib import Path

def _ollama_url():
    """OLLAMA_HOST is often set to a server bind address like "0.0.0.0" or
    "0.0.0.0:11434"; turn it into a URL a client can call."""
    h = os.environ.get("OLLAMA_HOST", "").strip() or "127.0.0.1:11434"
    if "://" not in h:
        h = "http://" + h
    scheme, rest = h.split("://", 1)
    host = rest.rstrip("/")
    if host.startswith("0.0.0.0"):
        host = "127.0.0.1" + host[len("0.0.0.0"):]
    if ":" not in host.split("/")[0]:
        host += ":11434"
    return f"{scheme}://{host}"


OLLAMA_URL = _ollama_url()
DEFAULT_MODEL = "hf.co/fischerman/sarvam-translate-gguf:Q4_K_S"
# Whisper language code -> name Sarvam-Translate understands
LANG_NAMES = {"en": "English", "as": "Assamese", "hi": "Hindi", "bn": "Bengali",
              "mr": "Marathi", "ta": "Tamil", "te": "Telugu", "gu": "Gujarati",
              "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "or": "Odia",
              "ur": "Urdu", "ne": "Nepali", "sa": "Sanskrit", "sd": "Sindhi"}


def _chat(model, system, text, timeout=300, temperature=0.01, seed=None):
    opts = {"temperature": temperature, "num_predict": 512}
    if seed is not None:
        opts["seed"] = seed
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": text}],
        "stream": False,
        "options": opts,
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))["message"]["content"].strip()


def translate_text(text, src="en", tgt="as", model=DEFAULT_MODEL):
    if src != "en" and tgt != "en":                      # Indic -> Indic: pivot via English
        text = translate_text(text, src, "en", model)
    return _chat(model, f"Translate the text below to {LANG_NAMES.get(tgt, tgt)}.", text)


def check_ready(model=DEFAULT_MODEL):
    """Fail early, with a clear message, if Ollama or the model is missing."""
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=10) as r:
            names = [m["name"] for m in json.loads(r.read().decode("utf-8"))["models"]]
    except Exception:
        raise SystemExit(f"Ollama is not running at {OLLAMA_URL}. Start the Ollama app and retry.")
    if not any(n.lower() == model.lower() or n.lower().startswith(model.lower() + ":")
               for n in names):
        raise SystemExit(f"Sarvam-Translate is not downloaded. Run:\n  ollama pull {model}")


def translate_units(units_path, out_path, src_lang="en", model=DEFAULT_MODEL, fixes=None):
    units = json.loads(Path(units_path).read_text(encoding="utf-8"))
    done = {}
    if Path(out_path).exists():                               # resume
        for u in json.loads(Path(out_path).read_text(encoding="utf-8")):
            if u.get("assamese"):
                done[(u["id"], u["text"])] = u
    fixes = fixes or {}
    out = []
    print(f"[sarvam] model={model} via {OLLAMA_URL}", flush=True)
    for u in units:
        key = (u["id"], u["text"])
        if key in done:
            out.append(done[key]); continue
        t0 = time.time()
        a = translate_text(u["text"], src_lang, "as", model)
        for x, y in fixes.items():
            a = a.replace(x, y)
        row = dict(u, assamese=a, candidates=[a])
        out.append(row)
        tmp = str(out_path) + ".tmp"
        Path(tmp).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, out_path)
        print(f"[sarvam] {u['id']}/{len(units)} ({time.time()-t0:.1f}s) {a}", flush=True)
    return out


def unload(model=DEFAULT_MODEL):
    """Free the GPU memory Ollama holds, so the voice model has room."""
    try:
        body = json.dumps({"model": model, "keep_alive": 0}).encode("utf-8")
        req = urllib.request.Request(OLLAMA_URL + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=30).read()
    except Exception as e:
        print(f"[sarvam] could not unload model ({e}); continuing", flush=True)


def variants(text, src="en", n=5, model=DEFAULT_MODEL):
    """Several faithful alternative translations (sampled), shortest first.
    Used only for lines whose Assamese voice is too long for the video slot."""
    if src != "en":
        text = translate_text(text, src, "en", model)
    out = []
    for k in range(n):
        a = _chat(model, "Translate the text below to Assamese.", text,
                  temperature=0.7, seed=100 + k)
        if a and a not in out:
            out.append(a)
    return sorted(out, key=len)


def gpu_warnings(own=(DEFAULT_MODEL, "llama3.2:3b"), min_free_gb=5.0):
    """Warn (never stop) if something else is holding GPU memory."""
    msgs = []
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/ps", timeout=10) as r:
            for m in json.loads(r.read().decode("utf-8")).get("models", []):
                name = m.get("name", "")
                if not any(name == o or name.startswith(o.split(":")[0] + ":") and o in name
                           for o in own):
                    gb = m.get("size_vram", m.get("size", 0)) / 1e9
                    msgs.append(f"Ollama is holding another model on the GPU: {name} ({gb:.1f} GB)\n"
                                f"     Free it with:  ollama stop {name}")
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            if free / 1e9 < min_free_gb:
                msgs.append(f"Only {free/1e9:.1f} GB of {total/1e9:.1f} GB GPU memory is free. "
                            "Close games, other AI apps or agents before dubbing.")
    except Exception:
        pass
    return msgs
