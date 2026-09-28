"""
Run every notebook top to bottom with no camera, mic or whisper model, against
tests/fake_llama_server.py (or a real server). Then click the widget panels.

    python tests/fake_llama_server.py &        # or a real llama-server
    python tests/run_notebooks.py

Facilitators: run this against the real server the night before.
"""
import glob
import os
import sys

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

FAKES = r'''
import os, pathlib, numpy as np
_m = pathlib.Path("models/gemma-4-E2B"); _m.mkdir(parents=True, exist_ok=True)   # pretend it's downloaded
(_m / "fake-Q4_0.gguf").write_bytes(b"x"); (_m / "mmproj-fake-Q8_0.gguf").write_bytes(b"x")
os.environ.setdefault("AGENT_CAMERA", "samples/test_card.png")
os.environ["AGENT_SPEAK"] = "0"
os.environ["AGENT_ARM_HEADLESS"] = "1"          # MuJoCo "window" without a screen
os.environ.setdefault("MUJOCO_GL", "osmesa")
import agent.ears as _ears

class _Seg:
    avg_logprob, text = -0.3, " computer what do you see"

class _W:
    def transcribe(self, audio, **k):
        return iter([_Seg()]), None

_ears.record_seconds = lambda s: (0.1 * np.sin(np.arange(int(16000 * s)) / 16000 * 2 * np.pi * 440)).astype(np.float32)
_ears.load_whisper = lambda *a, **k: _W()
_ears.transcribe = lambda audio, **k: ("computer what do you see", 0.05)

class _Rec:
    recording, started_at = False, 0.0
    def start(self): self.recording = True
    def stop(self):
        self.recording = False
        return np.full(16000, 0.1, np.float32)
_ears.BackgroundRecorder = _Rec
'''

CLICKS = {
    "2_eyes.ipynb": r'''
c = panel.controls
c["side"].value = 128; c["gray"].value = True; c["crop"].value = 0.5
c["ask"].click(); c["snap"].click()
assert "prompt tokens" in c["answer"].value, c["answer"].value
print("eyes panel ok:", c["answer"].value[:80])
''',
    "4_agent.ipynb": r'''
c = panel.controls
c["text"].value = "what do you see here"; c["send"].click()
c["talk"].value = True; c["talk"].value = False
for i in (3, 4, 5, 7):                       # i spy (kickoff), what changed, charades, memory palace
    c["mode"].value = c["mode"].options[i]
    c["text"].value = "is it a mug? what do you see"; c["send"].click()
c["memory"].value = "all_images"; c["reset"].click()
c["mode"].value = c["mode"].options[-1]; c["text"].value = "look at this"; c["send"].click()
assert "error" not in c["log"].value.lower(), c["log"].value
print("agent panel ok:", len(c["log"].value), "chars of log")
''',
}

PATCH_SOURCE = {"WATCH_FOR = 60": "WATCH_FOR = 3", "SECONDS = 60 ": "SECONDS = 4 ",
                "frame = eyes.frame()            # re-run": "frame = cv2.imread('samples/card_green.png')  # re-run"}

failed = False
for path in sorted(glob.glob("[0-9]_*.ipynb")):
    nb = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        for a, b in PATCH_SOURCE.items():
            cell.source = cell.source.replace(a, b)
    nb.cells.insert(0, new_code_cell(FAKES))
    if path in CLICKS:
        idx = next(i for i, c in enumerate(nb.cells) if "panel = nb." in c.source)
        nb.cells.insert(idx + 1, new_code_cell(CLICKS[path]))
    try:
        NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": ROOT}}).execute()
        print(f"PASS {path}")
    except Exception as e:  # noqa: BLE001
        failed = True
        print(f"FAIL {path}\n{str(e)[-2500:]}")
import shutil
shutil.rmtree("models/gemma-4-E2B", ignore_errors=True)
sys.exit(1 if failed else 0)
