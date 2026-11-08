# Assamese Dubber 🎙️

**Free, local, one-command dubbing of English / Hindi videos into Assamese.**
Made by **Aerobotix Edtech Institute, Dibrugarh** for Assamese students, teachers and creators.

```
.\dub.bat --url "https://youtu.be/VIDEO_ID" --speakers auto
```

Everything runs on your own PC. No paid service, no API key, no upload of your video.

> **অসমীয়াত পঢ়িবলৈ তললৈ যাওক → [অসমীয়া](#অসমীয়া)**

---

## ⚠️ Use it responsibly

Only dub videos that **you own**, that you have **the creator's permission** to use, or that have a
**Creative Commons licence that allows changes** (e.g. CC BY). Do not re-upload dubbed versions of other
people's videos - commercial or not. This tool is for education.

---

## What it does

| Step | Tool (free) | What happens |
|---|---|---|
| 1 | yt-dlp / your file | Gets the video |
| 2 | Demucs | Separates the voice from the background music |
| 3 | Whisper large-v3-turbo | Writes down every word with its exact time |
| 4 | pySBD | Rebuilds **complete sentences** (no half sentences) |
| 5 | pyannote *(optional)* | Finds who speaks each sentence, and guesses male / female from pitch |
| 6 | Sarvam-Translate (via Ollama) | Translates each sentence into Assamese |
| 7 | Indic Parler-TTS | Speaks it in Assamese (voices: Amit, Rakesh, Sita, Poonam) |
| 8 | Llama 3.2 3B (via Ollama) | If a line is too long for its time, shortens the sentence and tries again |
| 9 | FFmpeg | Puts the Assamese voice back over the **original music**, adds subtitles |

You get:
- `projects\NAME\NAME_assamese.mp4` - the dubbed video (with Assamese + original subtitle tracks)
- `NAME_assamese.srt` and `NAME_en.srt` - subtitle files (upload them to YouTube / Facebook)
- `report.txt` - English and Assamese side by side, which lines were shortened, and the speed of every line

---

## What you need

| | Minimum | Recommended (tested) |
|---|---|---|
| OS | Windows 10 / 11 | Windows 11 |
| GPU | NVIDIA, 6 GB VRAM | NVIDIA RTX, **8 GB VRAM** (tested: RTX 5060) |
| RAM | 16 GB | 32 GB |
| Disk | ~25 GB free | SSD |
| Internet | only for setup and the first run (model downloads) | |

Without an NVIDIA GPU it still works on CPU, but it is **very slow** (roughly 10-20x).

Software (setup tells you the exact install command if one is missing):
[Python 3.11](https://www.python.org/downloads/release/python-3119/) ·
[Git](https://git-scm.com/) · FFmpeg (`winget install Gyan.FFmpeg`) ·
[Ollama](https://ollama.com) (`winget install Ollama.Ollama`)

---

## Install (one time)

1. Download this repository (green **Code** button → **Download ZIP**, then unzip) or `git clone` it.
2. Open **PowerShell** in the folder and run:
   ```
   powershell -ExecutionPolicy Bypass -File setup.ps1
   ```
   It installs everything (about 20 GB in total, so it takes a while the first time).
3. On [Hugging Face](https://huggingface.co) (free account):
   - Accept the terms of **[ai4bharat/indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts)** (Assamese voice)
   - For `--speakers auto`, also accept **[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)**
   - Create a **Read** token: [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Before your **first** dub, give the token to PowerShell (it is only needed to download the models once).
   This line keeps it hidden - only `*****` appears on screen:
   ```
   $env:HF_TOKEN = [System.Net.NetworkCredential]::new('', (Read-Host 'Paste your Hugging Face token' -AsSecureString)).Password
   ```

---

## Use

```
.\dub.bat --url "https://youtu.be/VIDEO_ID"                    # YouTube video, one male voice
.\dub.bat --file "D:\videos\lesson.mp4" --voice female         # your own video, female voice
.\dub.bat --url "https://youtu.be/VIDEO_ID" --speakers auto    # each speaker gets their own voice
.\dub.bat --file "talk.mp4" --from hi                          # Hindi video
```

| Option | Meaning |
|---|---|
| `--url` / `--file` | YouTube link, or a video on your PC |
| `--name NAME` | Project folder name (default: from the video) |
| `--from en / hi / auto` | Language of the video (default: auto-detect) |
| `--voice male / female / Amit / Sita / Rakesh / Poonam` | One voice for everyone (default: male = Amit) |
| `--speakers auto` | Detect each speaker and give each their own voice |
| `--speaker-voices "SPEAKER_01=female"` | Fix a wrong male/female guess (see `report.txt` for speaker names) |
| `--num-speakers 2` | Tell the speaker detector how many people talk |
| `--sync natural / strict / flow` | Timing rule (see below). Default: **natural** |
| `--no-music` | Do not keep the original background music |
| `--whisper small` | Smaller, faster speech model (less accurate) |
| `--max-speed 1.2` | How much a line may be sped up before it is re-written shorter |

**Stopped in the middle?** Run the same command again - finished steps are reused.

### Timing (`--sync`)

- **natural** (default) - every Assamese line starts when the original sentence starts, and must finish
  before the next sentence starts. It may use the small pause after its sentence. **It never drifts.**
- **strict** - each line also ends exactly when the original sentence ends. Perfect timing, but fast
  speakers make more lines sound rushed.
- **flow** - most relaxed sound, but on fast dialogue the voice can fall a second or two behind.

Assamese usually needs more syllables than English. When a line is too long, the tool first makes the
**wording** shorter (Llama rewrites the English sentence, Sarvam translates it again) and only then speeds
the voice up. `report.txt` shows every shortened line, and marks lines where minor details were dropped.

---

## Honest limits

- The voice is clear but calm; it does not copy the original speaker's emotion.
- Very fast dialogue (news, comedy) will still have a few quick lines.
- Translation is good and understandable, not always how a native speaker would say it.
  Check `report.txt` before you publish.
- Lip sync is not done.
- Male / female is guessed from voice pitch and can be wrong - fix it with `--speaker-voices`.
- Other languages (not English / Indian languages) go through English first - experimental.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| PowerShell seems frozen, nothing moves | You probably clicked inside the window (Windows pauses the program while text is selected). Press **Esc**. Don't click inside the window while it runs. |
| `!! Ollama is holding another model on the GPU` / everything is very slow | Another app is using the GPU. Run the `ollama stop ...` command shown, close games / AI agents, then run again. |
| `400 Bad Request` or `401` from Hugging Face | Token missing or pasted wrong. Check: `$env:HF_TOKEN.Length` should be about 37. Also check that you accepted the model terms. |
| `Sarvam-Translate is not downloaded` | `ollama pull hf.co/fischerman/sarvam-translate-gguf:Q4_K_S` |
| Assamese subtitles show as `à¦¨à¦®...` | Use **VLC** (Windows Media Player has weak Assamese support). |
| `Speaker detection failed` | Accept the pyannote terms and set `HF_TOKEN`. The dub still finishes, with one voice. |
| YouTube download fails | Update yt-dlp: `.venv\Scripts\python.exe -m pip install -U yt-dlp` |

---

## Credits & licences

This repository's own code: **MIT** (see `LICENSE`). It uses these free models and tools - thank you:

| Component | By | Licence |
|---|---|---|
| [Whisper](https://github.com/openai/whisper) / [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | OpenAI / SYSTRAN | MIT |
| [pySBD](https://github.com/nipunsadvilkar/pySBD) | Nipun Sadvilkar | MIT |
| [Demucs](https://github.com/adefossez/demucs) | Meta / A. Défossez | MIT |
| [pyannote speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1) | pyannote | CC-BY-4.0 |
| [Sarvam-Translate](https://huggingface.co/sarvamai/sarvam-translate) (4-bit copy by [fischerman](https://huggingface.co/fischerman/sarvam-translate-gguf)) | Sarvam AI (built on Google Gemma 3) | GPL-3.0 + Gemma terms |
| [Llama 3.2 3B](https://ollama.com/library/llama3.2) | Meta | Llama 3.2 Community License - **Built with Llama** |
| [Indic Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts) | AI4Bharat, IIT Madras | Apache-2.0 |
| [Ollama](https://ollama.com), [FFmpeg](https://ffmpeg.org), [yt-dlp](https://github.com/yt-dlp/yt-dlp) | | MIT / LGPL-GPL / Unlicense |

Models are downloaded by you from their official sources; they are **not** included in this repository.

---

<a id="অসমীয়া"></a>
## অসমীয়া

### এইটো কি?

**Assamese Dubber** এটা free tool, যিয়ে English বা Hindi video এটা অসমীয়ালৈ dub কৰি দিয়ে।
এটা command দিলেই কাম হয়। সকলো আপোনাৰ নিজৰ PC-তে চলে, কোনো paid service বা API key নালাগে।
**Aerobotix Edtech Institute, Dibrugarh**-এ অসমীয়া ছাত্ৰ-ছাত্ৰী, শিক্ষক আৰু creator-সকলৰ বাবে বনাইছে।

### ⚠️ সাৱধান

নিজৰ video, creator-ৰ অনুমতি থকা video, বা Creative Commons licence থকা video-হে dub কৰক।
আনৰ video dub কৰি আকৌ upload নকৰিব, টকা উপাৰ্জন নকৰিলেও। এই tool-টো শিক্ষাৰ বাবে।

### কেনেকৈ কাম কৰে?

1. Video-টো download কৰে (YouTube link বা আপোনাৰ PC-ৰ file)।
2. Demucs-এ মাত আৰু background music বেলেগ কৰে।
3. Whisper-এ প্ৰতিটো শব্দ আৰু তাৰ সঠিক time লিখি লয়।
4. pySBD-এ সম্পূৰ্ণ sentence বনায়, আধা sentence নাথাকে।
5. pyannote-এ (ইচ্ছা কৰিলে) কোনে কোন sentence কৈছে উলিয়ায়, আৰু pitch চাই male নে female সেইটো অনুমান কৰে।
6. Sarvam-Translate-এ প্ৰতিটো sentence অসমীয়ালৈ translate কৰে।
7. Indic Parler-TTS-এ অসমীয়া মাতত কয় (voice: Amit, Rakesh, Sita, Poonam)।
8. কোনো line নিজৰ time-ত নধৰিলে, Llama-এ sentence-টো চুটি কৰে আৰু আকৌ চেষ্টা কৰে।
9. FFmpeg-এ অসমীয়া মাতটো original music-ৰ ওপৰত বহুৱাই দিয়ে, subtitle-ও যোগ কৰে।

### কি লাগিব?

- Windows 10 বা 11 PC
- NVIDIA GPU, 8 GB VRAM ভাল (RTX 5060-ত test কৰা হৈছে); 16 GB RAM বা তাতকৈ বেছি
- প্ৰায় 25 GB খালী disk space
- Python 3.11, Git, FFmpeg আৰু Ollama (কিবা নাথাকিলে setup-এ install কৰা command দেখুৱাই দিব)

NVIDIA GPU নাথাকিলেও CPU-ত চলে, কিন্তু বহুত লাহে লাহে।

### Install (এবাৰহে কৰিব লাগে)

1. এই repository-টো download কৰক (সেউজীয়া **Code** button → **Download ZIP**, তাৰ পিছত unzip কৰক)।
2. Folder-টোত PowerShell খুলি এইটো চলাওক:
   ```
   powershell -ExecutionPolicy Bypass -File setup.ps1
   ```
3. [Hugging Face](https://huggingface.co)-ত free account খুলি:
   - [indic-parler-tts](https://huggingface.co/ai4bharat/indic-parler-tts)-ৰ terms accept কৰক
   - `--speakers auto` ব্যৱহাৰ কৰিলে [pyannote](https://huggingface.co/pyannote/speaker-diarization-community-1)-ৰ terms-ও accept কৰক
   - [Settings → Tokens](https://huggingface.co/settings/tokens)-ত এটা **Read** token বনাওক
4. প্ৰথমবাৰ dub কৰাৰ আগতে PowerShell-ত এই line-টো দিয়ক আৰু token-টো paste কৰক (screen-ত কেৱল `*****` দেখা যাব):
   ```
   $env:HF_TOKEN = [System.Net.NetworkCredential]::new('', (Read-Host 'Paste your Hugging Face token' -AsSecureString)).Password
   ```

### ব্যৱহাৰ

```
.\dub.bat --url "https://youtu.be/VIDEO_ID"                    # YouTube video, এটা male voice
.\dub.bat --file "D:\videos\lesson.mp4" --voice female         # নিজৰ video, female voice
.\dub.bat --url "https://youtu.be/VIDEO_ID" --speakers auto    # প্ৰতিজন speaker-ৰ বেলেগ voice
.\dub.bat --file "talk.mp4" --from hi                          # Hindi video
```

শেষত `projects` folder-ত পাব:
- `NAME_assamese.mp4` - dub কৰা video
- `NAME_assamese.srt` আৰু `NAME_en.srt` - subtitle file
- `report.txt` - English আৰু অসমীয়া একেলগে, কোনটো line চুটি কৰা হ'ল, প্ৰতিটো line-ৰ speed

মাজতে বন্ধ হৈ গ'লে একেটা command আকৌ দিয়ক, শেষ হোৱা কামবোৰ আকৌ নকৰে।

### মনত ৰাখিব

- Voice-টো clear, কিন্তু original speaker-ৰ emotion নিদিয়ে।
- বহুত দ্ৰুত কথা কোৱা video-ত (news, comedy) কিছুমান line অলপ খৰকৈ শুনা যাব।
- Translation বুজিব পৰা ধৰণৰ, কিন্তু সদায় মানুহে কোৱাৰ দৰে নহয়। Publish কৰাৰ আগতে `report.txt` চাই লওক।
- Lip sync নহয়।
- Male/female অনুমান ভুল হ'লে `--speaker-voices "SPEAKER_01=female"` দি ঠিক কৰক।

### সমস্যা হ'লে

- **PowerShell ৰৈ গ'ল যেন লাগিছে?** Window-ৰ ভিতৰত click কৰিলে Windows-এ program-টো ৰখাই দিয়ে। **Esc** টিপক, আৰু চলি থকাৰ সময়ত window-ৰ ভিতৰত click নকৰিব।
- **বহুত লাহে চলিছে?** আন কোনো app-এ GPU ব্যৱহাৰ কৰি আছে। দেখুওৱা `ollama stop ...` command-টো দিয়ক, game বা আন AI app বন্ধ কৰক।
- **Hugging Face-ৰ `400` বা `401` error?** Token ভুলকৈ paste হৈছে বা terms accept কৰা নাই।
- **অসমীয়া subtitle `à¦¨à¦®...` দেখা গৈছে?** **VLC** player ব্যৱহাৰ কৰক।

---

Made with ❤️ in Assam by **Aerobotix Edtech Institute, Dibrugarh**.
