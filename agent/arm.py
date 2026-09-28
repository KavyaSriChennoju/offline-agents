"""
Hands: a small robot arm in MuJoCo, and three coloured buttons to press.

    sim = ArmSim()              # physics + an off-screen camera
    frames = sim.tap("red")     # reach over, press the red button, come home
    open_viewer(); send("tap", "blue")   # same, in a real MuJoCo window

No learning here: the arm is moved with plain geometry (inverse kinematics).
The *decision* (which colour?) is where the vision model comes in.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Linux with no screen (a server, a container): render off-screen with EGL
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

COLORS = ("red", "green", "blue")                 # the only three the arm knows
RGBA = {"red": (0.90, 0.20, 0.20), "green": (0.20, 0.75, 0.30), "blue": (0.20, 0.40, 0.95)}
ANGLE_DEG = {"red": 45, "green": 0, "blue": -45}  # where each button sits around the arm
RADIUS = 0.32                                     # metres from the arm's base
SHOULDER_Z, L1, L2 = 0.20, 0.22, 0.20             # arm geometry (must match the XML)
HOME = np.array([0.0, 1.2, -2.2])                 # yaw, shoulder, elbow (radians): tucked up
VIEWER_PORT = int(os.environ.get("AGENT_ARM_PORT", "8765"))
ROOT = Path(__file__).resolve().parent.parent


def _button(color: str) -> str:
    a = math.radians(ANGLE_DEG[color])
    x, y = RADIUS * math.cos(a), RADIUS * math.sin(a)
    r, g, b = RGBA[color]
    return f"""
    <body name="btn_{color}" pos="{x:.3f} {y:.3f} 0">
      <geom name="pad_{color}" type="cylinder" size=".055 .008" pos="0 0 .008" rgba=".15 .15 .17 1"/>
      <geom name="btn_{color}" type="cylinder" size=".045 .012" pos="0 0 .02" rgba="{r*.6:.2f} {g*.6:.2f} {b*.6:.2f} 1"/>
      <geom name="glow_{color}" type="cylinder" size=".075 .002" pos="0 0 .002" rgba="{r} {g} {b} 0"
            contype="0" conaffinity="0"/>
    </body>"""


SCENE = f"""
<mujoco model="agent_arm">
  <compiler angle="degree"/>
  <option timestep="0.002"/>
  <visual><global offwidth="1280" offheight="960" azimuth="150" elevation="-28"/>
          <quality shadowsize="2048"/></visual>
  <default>
    <joint armature="0.02" damping="2"/>
    <geom contype="0" conaffinity="0"/>
    <position kp="40"/>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".86 .85 .82" rgb2=".80 .79 .76" width="300" height="300"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1=".32 .36 .46" rgb2=".07 .08 .11" width="256" height="256"/>
    <material name="floor" texture="grid" texrepeat="8 8"/>
    <material name="arm" rgba=".16 .18 .24 1"/>
    <material name="accent" rgba="1 .54 .24 1"/>
  </asset>
  <worldbody>
    <light pos="0.3 -0.5 1.6" dir="-0.2 0.3 -1" diffuse=".9 .9 .9"/>
    <light pos="-0.6 0.6 1.2" dir="0.4 -0.4 -1" diffuse=".35 .35 .35" castshadow="false"/>
    <geom name="floor" type="plane" size="1.2 1.2 .05" material="floor" contype="1" conaffinity="1"/>
    <camera name="front" pos="0.78 -0.62 0.55" mode="targetbody" target="turret"/>
    <body name="base">
      <geom type="cylinder" size=".07 .03" pos="0 0 .03" material="arm"/>
      <body name="turret" pos="0 0 .06" gravcomp="1">
        <joint name="yaw" axis="0 0 1" range="-120 120"/>
        <geom type="cylinder" size=".045 .07" pos="0 0 .07" material="arm"/>
        <body name="upper" pos="0 0 {SHOULDER_Z - 0.06:.3f}" gravcomp="1">
          <joint name="shoulder" axis="0 -1 0" range="-30 150"/>
          <geom type="sphere" size=".038" material="accent"/>
          <geom type="capsule" fromto="0 0 0 {L1} 0 0" size=".024" material="arm"/>
          <body name="fore" pos="{L1} 0 0" gravcomp="1">
            <joint name="elbow" axis="0 -1 0" range="-165 20" damping="1.5"/>
            <geom type="sphere" size=".032" material="accent"/>
            <geom type="capsule" fromto="0 0 0 {L2} 0 0" size=".019" material="arm"/>
            <geom name="finger" type="sphere" pos="{L2} 0 0" size=".022" material="accent" contype="1" conaffinity="1"/>
            <site name="tip" pos="{L2} 0 0" size=".005"/>
          </body>
        </body>
      </body>
    </body>
    {''.join(_button(c) for c in COLORS)}
  </worldbody>
  <actuator>
    <position name="yaw" joint="yaw"/>
    <position name="shoulder" joint="shoulder"/>
    <position name="elbow" joint="elbow"/>
  </actuator>
</mujoco>
"""


class ArmSim:
    """The arm, the buttons, and (optionally) an off-screen camera to film it."""

    def __init__(self, width: int = 640, height: int = 480, render: bool = True, realtime: bool = False):
        import mujoco
        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(SCENE)
        self.data = mujoco.MjData(self.model)
        self.data.qpos[:3] = HOME
        self.data.ctrl[:3] = HOME
        mujoco.mj_forward(self.model, self.data)
        self.renderer = mujoco.Renderer(self.model, height, width) if render else None
        self.realtime = realtime
        self.on_step = None          # called after every physics step (the live viewer uses this)
        self.fps = 30
        self.log: list[str] = []

    # -- geometry ---------------------------------------------------------------
    def ik(self, x: float, y: float, z: float) -> np.ndarray:
        """Joint angles that put the fingertip at (x, y, z). Two-link arm on a turntable."""
        yaw = math.atan2(y, x)
        d, h = math.hypot(x, y), z - SHOULDER_Z
        c2 = (d * d + h * h - L1 * L1 - L2 * L2) / (2 * L1 * L2)
        if abs(c2) > 1:
            raise ValueError(f"({x:.2f}, {y:.2f}, {z:.2f}) is out of reach")
        elbow = -math.acos(c2)                                   # negative = elbow up
        shoulder = math.atan2(h, d) - math.atan2(L2 * math.sin(elbow), L1 + L2 * math.cos(elbow))
        return np.array([yaw, shoulder, elbow])

    def above(self, color: str, height: float) -> np.ndarray:
        a = math.radians(ANGLE_DEG[color])
        return self.ik(RADIUS * math.cos(a), RADIUS * math.sin(a), height)

    def tip(self) -> np.ndarray:
        return self.data.site("tip").xpos.copy()

    # -- motion -------------------------------------------------------------------
    def move_to(self, q: np.ndarray, seconds: float, frames: list | None = None):
        """Glide the joint targets from where they are to q (smooth start and stop)."""
        m, d = self.model, self.data
        q0 = d.ctrl[:3].copy()
        steps = max(1, int(seconds / m.opt.timestep))
        every = max(1, int(1 / (self.fps * m.opt.timestep)))
        t_wall = time.perf_counter()
        for i in range(steps):
            s = (1 - math.cos(math.pi * (i + 1) / steps)) / 2      # 0 -> 1, eased
            d.ctrl[:3] = q0 + (q - q0) * s
            self.mj.mj_step(m, d)
            self._update_glow()
            if self.on_step:
                self.on_step()
            if frames is not None and i % every == 0:
                frames.append(self.render())
            if self.realtime:
                ahead = (i + 1) * m.opt.timestep - (time.perf_counter() - t_wall)
                if ahead > 0:
                    time.sleep(ahead)

    def hold(self, seconds: float, frames: list | None = None):
        self.move_to(self.data.ctrl[:3].copy(), seconds, frames)

    def home(self, frames: list | None = None):
        self.move_to(HOME, 0.9, frames)

    def tap(self, color: str, film: bool = True) -> list:
        """Reach over the button, press it, come home. Returns video frames (BGR) if film=True."""
        color = color.lower().strip()
        if color not in COLORS:
            raise ValueError(f"The arm only knows {COLORS}, not {color!r}")
        frames = [] if (film and self.renderer) else None
        self.move_to(self.above(color, 0.16), 0.9, frames)       # hover
        self.move_to(self.above(color, 0.045), 0.5, frames)      # press
        self.hold(0.35, frames)
        self.move_to(self.above(color, 0.16), 0.4, frames)       # lift
        self.home(frames)
        self.log.append(color)
        return frames or []

    # -- buttons light up while pressed -----------------------------------------------
    def _update_glow(self):
        tip = self.tip()
        for c in COLORS:
            top = self.data.geom(f"btn_{c}").xpos
            pressed = np.linalg.norm(tip[:2] - top[:2]) < 0.05 and tip[2] < 0.075
            r, g, b = RGBA[c]
            k = 1.0 if pressed else 0.6
            self.model.geom(f"btn_{c}").rgba[:] = (min(1, r * k + (0.1 if pressed else 0)),
                                                  min(1, g * k + (0.1 if pressed else 0)),
                                                  min(1, b * k + (0.1 if pressed else 0)), 1)
            self.model.geom(f"glow_{c}").rgba[3] = 0.55 if pressed else 0.0

    def pressed(self) -> str | None:
        for c in COLORS:
            if self.model.geom(f"glow_{c}").rgba[3] > 0:
                return c
        return None

    # -- camera -------------------------------------------------------------------
    def render(self, camera: str = "front") -> np.ndarray:
        """One BGR frame (OpenCV colour order, so nb.show works on it)."""
        if self.renderer is None:
            raise RuntimeError("created with render=False")
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()[:, :, ::-1].copy()


# ---------------------------------------------------------------------------
# deciding the colour
# ---------------------------------------------------------------------------

def color_by_pixels(frame: np.ndarray, min_fraction: float = 0.08, center: float = 0.6):
    """
    Old-school computer vision: count strongly coloured pixels in the middle of the frame.
    Returns (colour or None, {colour: fraction of pixels}).
    """
    import cv2
    h, w = frame.shape[:2]
    ch, cw = int(h * center), int(w * center)
    roi = frame[(h - ch) // 2:(h + ch) // 2, (w - cw) // 2:(w + cw) // 2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sat = (hsv[..., 1] > 110) & (hsv[..., 2] > 70)
    hue = hsv[..., 0]                                   # OpenCV hue: 0-180
    masks = {
        "red": sat & ((hue < 8) | (hue > 170)),
        "green": sat & (hue > 40) & (hue < 85),
        "blue": sat & (hue > 95) & (hue < 130),
    }
    scores = {c: float(m.mean()) for c, m in masks.items()}
    best = max(scores, key=scores.get)
    return (best if scores[best] >= min_fraction else None), scores


COLOR_SCHEMA = {
    "type": "object",
    "properties": {"color": {"type": "string", "enum": [*COLORS, "none"]}},
    "required": ["color"],
}

COLOR_PROMPT = (
    "Look at the object the person is holding up to the camera. Which colour is it: "
    "red, green or blue? If it's none of those, or nothing is held up, answer none."
)


def color_by_llm(llm, jpeg: bytes, prompt: str = COLOR_PROMPT):
    """
    Ask the vision model, but only let it answer with one of four words.
    response_format + json_schema makes llama-server *constrain* the output to match.
    Returns (colour or None, the raw reply).
    """
    from agent.llm import user
    r = llm.chat([user(prompt, images=[jpeg])], stream=False, temperature=0, max_tokens=20,
                 response_format={"type": "json_schema", "json_schema": {"name": "color", "schema": COLOR_SCHEMA}})
    try:
        color = json.loads(r.text).get("color")
    except (json.JSONDecodeError, AttributeError):   # model ignored the schema: fall back to spotting a word
        color = next((c for c in COLORS if c in r.text.lower()), None)
    return (color if color in COLORS else None), r


# ---------------------------------------------------------------------------
# the real MuJoCo window, in its own process
# ---------------------------------------------------------------------------

def _viewer_python() -> str:
    """macOS needs MuJoCo's own launcher (mjpython) to open a window from a script."""
    if sys.platform == "darwin":
        mj = Path(sys.executable).with_name("mjpython")
        if mj.exists():
            return str(mj)
    return sys.executable


_viewer_proc: subprocess.Popen | None = None


def _url(path: str) -> str:
    return f"http://127.0.0.1:{VIEWER_PORT}{path}"


def viewer_up() -> bool:
    import requests
    try:
        return requests.get(_url("/health"), timeout=1).ok
    except requests.RequestException:
        return False


def open_viewer(headless: bool | None = None, wait_s: float = 30) -> bool:
    """Open the MuJoCo window (a separate program). Returns True when it's ready for commands."""
    global _viewer_proc
    if headless is None:
        headless = os.environ.get("AGENT_ARM_HEADLESS") == "1"
    if viewer_up():
        print("MuJoCo window already open")
        return True
    cmd = [_viewer_python(), str(ROOT / "hands_viewer.py"), "--port", str(VIEWER_PORT)]
    if headless:
        cmd.append("--headless")
    _viewer_proc = subprocess.Popen(cmd, cwd=ROOT)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if viewer_up():
            print("MuJoCo window is open" + (" (headless)" if headless else "") +
                  ". Drag to orbit, scroll to zoom, double-click a body to track it.")
            return True
        if _viewer_proc.poll() is not None:
            print("The MuJoCo window closed or failed to open. Use the inline view instead (VIEW = 'inline').")
            return False
        time.sleep(0.3)
    print("MuJoCo window didn't answer in time. Use VIEW = 'inline'.")
    return False


def send(action: str, color: str | None = None, wait: bool = True) -> dict:
    """Tell the window's arm what to do: send('tap', 'red'), send('home')."""
    import requests
    r = requests.post(_url("/do"), json={"action": action, "color": color, "wait": wait}, timeout=30)
    r.raise_for_status()
    return r.json()


def close_viewer():
    global _viewer_proc
    import requests
    try:
        requests.post(_url("/quit"), timeout=2)
    except requests.RequestException:
        pass
    if _viewer_proc is not None:
        try:
            _viewer_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _viewer_proc.kill()
        _viewer_proc = None
    print("MuJoCo window closed")
