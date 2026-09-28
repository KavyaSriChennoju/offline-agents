# offline-agent

A multimodal assistant that sees through your webcam, hears you, and talks back.
It runs on your laptop. No API keys, no cloud, and it still works with the wifi off.

Built for the workshop **Open-Source Multimodal Agents That Run Entirely on Your Laptop**
(The AI Conference 2026, Day Zero).

```
 camera ──> shrink + JPEG ──┐
                            ├──> llama-server (a vision LLM, on your laptop) ──> text ──> your OS voice
 mic ────> whisper (local) ─┘
```

Four notebooks plus two bonuses. No IDE needed: every knob is a line in a cell you can change and re-run.

| Notebook | You end up with |
|---|---|
| `1_setup.ipynb` | a local LLM answering you, offline, plus a camera/mic/voice check |
| `2_eyes.ipynb` | it describes what your webcam sees, with a slider panel for image size, crop, grayscale |
| `3_ears.ipynb` | you talk, it transcribes, it answers out loud |
| `4_agent.ipynb` | the whole thing: 9 personalities, a talk/type panel, and three build challenges |
| `5_omni.ipynb` (bonus) | one model that takes your voice and the camera frame together, no whisper |
| `6_hands.ipynb` (bonus) | show red, green or blue and a robot arm in MuJoCo presses that button |

---

## Before the workshop (please do this at home, the venue wifi will thank you)

About 15 minutes and 2-4 GB of downloads.

### 1. Install uv and llama.cpp

The quick way for each OS. **[INSTALL.md](INSTALL.md)** has the full guide: other package
managers, prebuilt zips, GPU builds, building from source, and fixes for common snags.

| OS | Commands |
|---|---|
| macOS | `brew install uv llama.cpp` (no Homebrew? see INSTALL.md) |
| Windows | `winget install --id=astral-sh.uv -e` then `winget install llama.cpp`, then open a new terminal |
| Linux | `curl -LsSf https://astral.sh/uv/install.sh \| sh`, then llama.cpp via Homebrew, a prebuilt zip or a source build (INSTALL.md) |

Check both worked, in a new terminal: `uv --version` and `llama-server --version`.

### 2. Get the code and install the Python bits

Download the zip from **tinyurl.com/offline-agents**, unzip it, and open a terminal in the
`offline-agents-main` folder. Or clone it:

```bash
git clone https://github.com/KavyaSriChennoju/offline-agents && cd offline-agents
uv sync
```

(If you downloaded the zip, just run `uv sync` inside the unzipped folder.)

uv fetches Python 3.12 if you don't have it, creates `.venv`, and installs the exact versions
pinned in `uv.lock`. No activating anything.

Linux only: `sudo apt install libportaudio2 espeak-ng` (mic + voice).

### 3. Open the notebooks, download the model

```bash
uv run jupyter lab
```

Your browser opens (if not, copy the `http://localhost:8888/lab?token=...` link the terminal
prints). Double-click `1_setup.ipynb`. The **Step 0** cells pick a model, download it with a
progress bar, and start `llama-server` in the background for you:

| `MODEL` | For | Download |
|---|---|---|
| `"gemma-4-E2B"` | most laptops (16 GB RAM, any Apple Silicon Mac, or a GPU) | ~3-4 GB |
| `"smolvlm2-2.2B"` | 8 GB RAM or older machines | ~1.7 GB |
| `"potato"` | anything. Tiny, fast, a bit dim | ~0.3 GB |
| `"gemma-4-E4B"` | big machines that want smarter | ~5-6 GB |
| `"qwen2.5-omni-3B"` | sees *and* hears, for the bonus notebook (Gemma 4 hears too) | ~3.6 GB |

Models land in `models/<name>/`. Run the rest of `1_setup.ipynb` too: the mic cell downloads
whisper (~145 MB) the first time. macOS will ask for camera and mic permission. Say yes.

The server keeps running in the background across notebooks and kernel restarts.
`server.stop()` ends it; `server.tail()` shows its log.

Prefer the terminal? `llama-server -hf ggml-org/gemma-4-E2B-it-GGUF:Q4_0 -c 8192` does the same
download-and-start in one line. Skip the Step 0 cells if you go that way.

Prefer VS Code? Open the folder, open any `.ipynb`, pick the `.venv` Python as the kernel (after `uv sync`).

No uv? `python -m venv .venv`, activate it, `pip install -r requirements.txt`, then `jupyter lab`.

No camera or mic? You can still do everything: set `CAMERA` to a photo path (or a folder
of photos) in any notebook, and every voice cell has a typed or `.wav` alternative.

---

## Tips for the notebooks

- **Shift+Enter** runs a cell. Change a value, run it again. That's the whole workflow.
- Cells marked `...` hold **solutions**. Click to reveal. Peeking is fine.
- Something weird? **Kernel > Restart**, then re-run the first cell of the notebook.
- Only one notebook can use the camera at a time. Each notebook ends with an `eyes.close()` cell.
- A loop that won't stop (the "speak up" challenge): press the ■ button.

---

## Where things live

```
1_setup.ipynb ... 6_hands.ipynb       the workshop (5 and 6 are bonuses)
hands_viewer.py    the MuJoCo window for notebook 6 (the notebook starts it for you)
INSTALL.md         uv + llama.cpp install guides for macOS, Windows, Linux
config.py          defaults: server URL, camera, image size, whisper model, voice
modes.py           shared prompt bits + the words that make the agent "look"
agent/
  llm.py           ~170 lines. The entire "API client": one POST with some JSON
  eyes.py          camera thread + preprocess (resize / crop / grey / JPEG)
  ears.py          recording + faster-whisper
  mouth.py         OS text-to-speech, and speaking sentence by sentence while streaming
  brain.py         the response layer: builds the prompt from words + frames + memory
  nb.py            notebook helpers and the two control panels
  arm.py           the MuJoCo arm: scene, inverse kinematics, colour detectors, window control
  server.py        downloads models and starts/stops llama-server from a notebook
tests/             fake_llama_server.py (no model needed), run_notebooks.py, test_server.py
models/            downloaded GGUF files (created on first download)
samples/           test images, including one that tries to hijack the model
```

---

## When things go wrong

| You see | Try |
|---|---|
| `NOT RUNNING` in the first cell | Run the Step 0 cells in `1_setup.ipynb`. Or it's on another port: set `config.SERVER_URL`, or start Jupyter with `AGENT_SERVER=http://127.0.0.1:8081 uv run jupyter lab` |
| `400` / `500` mentioning image | You loaded a text-only model, or the vision projector didn't download. Use a model from the table above. |
| `llama-server: NOT FOUND` in Step 0 | Install llama.cpp ([INSTALL.md](INSTALL.md)), then stop Jupyter (ctrl-c) and `uv run jupyter lab` again so it sees the new PATH. Installed from a zip? Set `AGENT_LLAMA_SERVER` to the file. |
| Server won't start | `server.tail()` shows why. Often: not enough memory, pick a smaller `MODEL`. |
| Venue wifi too slow | Copy the model folder from a USB stick into `models/`. The download cell then skips the internet. |
| Camera won't open | macOS: System Settings > Privacy & Security > Camera, allow your terminal. Close Zoom/Teams. Another notebook still holding it? Run its `eyes.close()`. Try `CAMERA = 1`. |
| Mic is silent | Same privacy settings, for Microphone. Check the default input device. |
| `PortAudio library not found` | Linux: `sudo apt install libportaudio2`. Or type instead of talking. |
| Panels show as text, not buttons | Make sure you started Jupyter with `uv run jupyter lab`, not a Jupyter installed elsewhere. |
| `jupyter: command not found` | Use `uv run jupyter lab` (it runs Jupyter from the project's `.venv`). |
| http://127.0.0.1:8080 shows `"File Not Found"` (404) | Your llama.cpp build has no built-in chat page. The server is fine and the notebooks don't use that page. Check with `curl http://127.0.0.1:8080/health`. Want the page? `brew upgrade llama.cpp` / `winget upgrade llama.cpp`, or a prebuilt zip (INSTALL.md). |
| Notebook 6: MuJoCo window doesn't open | It falls back to `VIEW = "inline"` (the arm plays as a video in the notebook). On a Mac the window needs `mjpython`, which ships with MuJoCo; the notebook uses it automatically. On Linux without a screen, inline only. |
| First image takes ages | Normal. The vision encoder warms up on the first request. |
| Replies are empty, or `[empty reply: the model spent N tokens thinking]` | Gemma 4 and Qwen3 "think" before answering and can use up every token. We ask them not to, but the server has to agree: re-run `server.start(...)` (it restarts with `--reasoning off`). Started it in a terminal? Add `--reasoning off` (older llama.cpp: `--reasoning-budget 0`). Want to see the thinking? `AGENT_THINKING=1`, and read `reply.thinking`. |
| Out of memory / laptop fan takes off | Drop down a row in the model table, or lower `-c`. |
| No voice on Linux | `sudo apt install espeak-ng`. Or `AGENT_SPEAK=0` to go text-only. |

## Better voice (optional)

The OS voices are fine but robotic. Piper and Kokoro are open-source TTS models that run
fully offline and sound much better. Install one and change `_command()` in `agent/mouth.py`
to call it. Keep `StreamingMouth`: speaking sentence by sentence is what keeps it feeling fast.

## For facilitators

```bash
uv sync --group dev                         # adds the test tools
uv run python tests/fake_llama_server.py &  # pretends to be llama-server, no model needed
uv run python tests/run_notebooks.py        # runs all 6 notebooks + clicks the panels
uv run python tests/test_server.py          # model download + server start/stop, all faked
uv run python tests/test_thinking.py        # thinking models still answer
uv run python tests/test_arm.py             # arm reaches every button, colour detectors, window API
```

Run `run_notebooks.py` against the real server the night before. The fake server is also handy
for checking someone's Python setup while their model is still downloading.
