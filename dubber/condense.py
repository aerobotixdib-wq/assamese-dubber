"""Shorten an English sentence (keeping its meaning) before translation.

Used only for lines whose Assamese voice is too long for the video slot.
Runs locally with a small model in Ollama (default: Meta Llama 3.2 3B,
"Built with Llama").  Install once:   ollama pull llama3.2:3b

Safety checks: the result must be shorter, and must still contain every
number and every capitalised name of the original - otherwise it is rejected
and the original sentence is kept.
"""
import json
import re
import urllib.request

from .translate_sarvam import OLLAMA_URL

CONDENSE_MODEL = "llama3.2:3b"
SYSTEM = ("You shorten sentences for video dubbing. Rewrite the user's sentence so it "
          "is about {pct}% of its original length. Keep every name, number and key "
          "fact. Remove filler words (and, so, really, actually, just, very). Keep "
          "the same meaning and the same speaker's point of view. Reply with ONLY "
          "the rewritten sentence - no quotes, no explanation.")


def available(model=CONDENSE_MODEL):
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=10) as r:
            names = [m["name"] for m in json.loads(r.read().decode("utf-8"))["models"]]
        return any(n == model or n.startswith(model + ":") or n == model + ":latest"
                   for n in names)
    except Exception:
        return False


SYSTEM_STRONG = ("You adapt sentences for video dubbing, where the translation must fit a "
                 "short time. Rewrite the user's sentence in plain English at about {pct}% of "
                 "its length. Keep the main point and every number. You MAY drop minor "
                 "details, side remarks and descriptions. Keep the same speaker's point of "
                 "view. Reply with ONLY the rewritten sentence - no quotes, no explanation.")


def _ask(text, pct, model, seed, strong=False):
    body = json.dumps({
        "model": model, "stream": False,
        "messages": [{"role": "system", "content": (SYSTEM_STRONG if strong else SYSTEM).format(pct=pct)},
                     {"role": "user", "content": text}],
        "options": {"temperature": 0.3, "seed": seed, "num_predict": 200},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read().decode("utf-8"))["message"]["content"].strip()
    out = out.splitlines()[0].strip() if out else ""
    return out.strip('"“”\' ')


def _must_keep(text):
    """Numbers, and capitalised words that are not just the first word."""
    nums = set(re.findall(r"\d+(?:[.,]\d+)?", text))
    words = re.findall(r"[A-Za-z][\w'.-]*", text)
    names = {w.strip(".'") for i, w in enumerate(words)
             if i > 0 and (w[0].isupper() or w.isupper())}
    names |= {w for w in words if len(w) > 1 and w.isupper()}   # acronyms incl. first word
    return nums, {n for n in names if n}


def is_safe(original, short, strong=False):
    """Normal: every number and every name kept.
    Strong: every number kept, and at least half of the names (so the sentence
    is still about the same people/things); minor details may go."""
    if not short or len(short) >= len(original):
        return False
    nums, names = _must_keep(original)
    low = short.lower()
    if not all(n in short for n in nums):
        return False
    kept = sum(n.lower() in low for n in names)
    return kept == len(names) if not strong else kept * 2 >= len(names)


def condense(text, need, model=CONDENSE_MODEL, tries=3, strong=False):
    """need = how many times too long the voice is (e.g. 1.5).
    strong=True: may drop minor details (used only for lines that are far too long).
    Returns a shorter English sentence, or None if no safe version was found."""
    pct = max(35 if strong else 50, min(90, int(100 / need)))
    best = None
    for k in range(tries):
        try:
            s = _ask(text, pct, model, seed=7 + k, strong=strong)
        except Exception as e:                 # slow/busy Ollama must not stop the dub
            print(f"[condense] skipped ({e})", flush=True)
            return best
        if is_safe(text, s, strong) and (best is None or len(s) < len(best)):
            best = s
    return best
