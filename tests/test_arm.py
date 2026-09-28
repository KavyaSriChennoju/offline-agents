"""The arm reaches every button, the colour detectors work, and the window's command API works (headless)."""
import os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "osmesa")
import cv2  # noqa: E402
from agent import arm  # noqa: E402

s = arm.ArmSim(width=320, height=240)
for c in arm.COLORS:
    seen = set()
    s.on_step = lambda: seen.add(s.pressed())
    frames = s.tap(c)
    assert c in seen, f"never pressed {c}"
    assert frames and frames[0].shape == (240, 320, 3)
print("arm reaches all 3 buttons")

for c in arm.COLORS:
    card = cv2.imread(str(ROOT / f"samples/card_{c}.png"))
    got, scores = arm.color_by_pixels(card)
    assert got == c, (c, got, scores)
got, _ = arm.color_by_pixels(cv2.imread(str(ROOT / "samples/test_card.png")))
print("pixel detector ok; test card ->", got)

arm.VIEWER_PORT = 8799
assert arm.open_viewer(headless=True)
r = arm.send("tap", "blue")
assert r["ok"] and r["color"] == "blue", r
arm.send("home")
import requests  # noqa: E402
assert requests.get(arm._url("/state")).json()["count"] == 1
assert requests.post(arm._url("/do"), json={"action": "tap", "color": "purple"}).status_code == 400
arm.close_viewer()
time.sleep(0.5)
assert not arm.viewer_up()
print("viewer API ok")
