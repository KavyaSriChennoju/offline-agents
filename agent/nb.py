"""
Notebook helpers: show pictures, record from the mic, and a couple of
little control panels built with ipywidgets.

You don't need to read this file to do the workshop. It's here so the
notebooks can stay short and the interesting code stays visible there.
"""
from __future__ import annotations

import html
import time

import cv2
import numpy as np
from IPython.display import Image, display

import config
from agent import ears, mouth
from agent.eyes import preprocess


# ---------------------------------------------------------------------------
# pictures
# ---------------------------------------------------------------------------

def to_jpeg(img) -> bytes:
    """Frame (numpy) or JPEG bytes -> JPEG bytes, for display."""
    if isinstance(img, (bytes, bytearray)):
        return bytes(img)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def show(*imgs, width: int = 420):
    """Show one or more frames / JPEGs side by side."""
    if len(imgs) == 1:
        display(Image(data=to_jpeg(imgs[0]), width=width))
        return
    import ipywidgets as w
    display(w.HBox([w.Image(value=to_jpeg(i), format="jpeg", width=width) for i in imgs]))


def model_view(frame: np.ndarray, **kw) -> bytes:
    """What the model actually receives, given preprocess() settings."""
    return preprocess(frame, **kw)


def blow_up(jpeg: bytes, height: int = 360) -> bytes:
    """Scale a small JPEG back up with blocky pixels, so you can see what was lost."""
    small = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    w = int(small.shape[1] * height / small.shape[0])
    return to_jpeg(cv2.resize(small, (w, height), interpolation=cv2.INTER_NEAREST))


def play_frames(frames, fps: int = 30, width: int = 480):
    """Play a list of frames (e.g. from ArmSim.tap) as a little video in the notebook."""
    import ipywidgets as w
    if not frames:
        return
    img = w.Image(format="jpeg", width=width)
    display(img)
    for f in frames:
        img.value = to_jpeg(f)
        time.sleep(1 / fps)


def stream(piece: str):
    """Use as on_token= to print a reply word by word."""
    print(piece, end="", flush=True)


# ---------------------------------------------------------------------------
# microphone
# ---------------------------------------------------------------------------

def record(seconds: float = 4) -> np.ndarray:
    """Count down, then record from the default mic."""
    for n in (3, 2, 1):
        print(f"\rrecording in {n}...", end="", flush=True)
        time.sleep(0.6)
    print(f"\rrecording for {seconds:.0f}s. Talk now!   ", flush=True)
    audio = ears.record_seconds(seconds)
    lvl = ears.loudness(audio)
    print(f"done. level {lvl:.3f}" + ("  (very quiet: mic muted or wrong device?)" if lvl < 0.003 else ""))
    return audio


def play(audio: np.ndarray):
    """Listen back to what you recorded."""
    from IPython.display import Audio
    display(Audio(audio, rate=config.SAMPLE_RATE))


# ---------------------------------------------------------------------------
# panel: eyes (Part 2)
# ---------------------------------------------------------------------------

def eyes_panel(llm, eyes, question: str, system_prompt: str, max_tokens: int = 120):
    """Snap a frame, fiddle with the preprocessing, ask. All with buttons."""
    import ipywidgets as w
    from agent.llm import system, user

    state = {"frame": eyes.frame()}
    img = w.Image(format="jpeg", layout=w.Layout(max_width="560px"))
    side = w.SelectionSlider(options=[64, 128, 224, 384, 512, 768, 1024], value=512, description="max_side")
    crop = w.FloatSlider(value=1.0, min=0.2, max=1.0, step=0.05, description="crop")
    gray = w.Checkbox(value=False, description="grayscale")
    q = w.Text(value=question, description="ask", layout=w.Layout(width="560px"))
    snap_b, ask_b = w.Button(description="Snap", icon="camera"), w.Button(description="Ask", button_style="primary")
    info = w.HTML()
    answer = w.HTML(layout=w.Layout(width="560px"))

    def settings():
        return dict(max_side=side.value, grayscale=gray.value, center_crop=crop.value)

    def refresh(*_):
        jpeg = preprocess(state["frame"], **settings())
        img.value = blow_up(jpeg)
        info.value = f"<code>model sees: {side.value}px max, {len(jpeg) / 1024:.1f} KB</code>"

    def on_snap(_):
        eyes.next_file()
        state["frame"] = eyes.frame()
        refresh()

    def on_ask(_):
        ask_b.disabled, answer.value = True, "<i>thinking...</i>"
        text = []
        last = [0.0]

        def grow(piece):
            text.append(piece)
            if time.time() - last[0] > 0.1:
                answer.value = html.escape("".join(text))
                last[0] = time.time()
        try:
            jpeg = preprocess(state["frame"], **settings())
            r = llm.chat([system(system_prompt), user(q.value, images=[jpeg])],
                         on_token=grow, temperature=0.2, max_tokens=max_tokens)
            answer.value = f"<b>{html.escape(r.text.strip())}</b><br><code>{r.stats()}</code>"
        except Exception as e:  # noqa: BLE001
            answer.value = f"<span style='color:#c33'>{html.escape(str(e))}</span>"
        ask_b.disabled = False

    for c in (side, crop, gray):
        c.observe(refresh, names="value")
    snap_b.on_click(on_snap)
    ask_b.on_click(on_ask)
    refresh()
    box = w.VBox([img, info, w.HBox([side, crop, gray]), q, w.HBox([snap_b, ask_b]), answer])
    box.controls = dict(side=side, crop=crop, gray=gray, q=q, snap=snap_b, ask=ask_b, answer=answer)
    display(box)
    return box


# ---------------------------------------------------------------------------
# panel: the agent (Part 4)
# ---------------------------------------------------------------------------

def agent_panel(bot, modes: list[dict], speak: bool = True):
    """
    The whole assistant in one widget: pick a mode, then either press
    "Start talking" / "Stop" or type and press Send.
    """
    import ipywidgets as w

    rec = ears.BackgroundRecorder()
    names = [f"{i + 1}. {m['name']}" for i, m in enumerate(modes)]
    mode_dd = w.Dropdown(options=names, value=names[0], description="mode", layout=w.Layout(width="360px"))
    mem_dd = w.Dropdown(options=["latest_image", "all_images", "captions"], description="memory")
    talk_b = w.ToggleButton(description="Start talking", icon="microphone", button_style="success")
    text = w.Text(placeholder="...or type here and press Send", layout=w.Layout(width="420px"))
    send_b = w.Button(description="Send", button_style="primary")
    reset_b = w.Button(description="Reset", tooltip="forget the conversation (and retake the BEFORE photo)")
    status = w.HTML()
    seen = w.Image(format="jpeg", layout=w.Layout(max_width="300px"))
    log = w.HTML(layout=w.Layout(width="620px", max_height="420px", overflow_y="auto",
                                 border="1px solid #ddd", padding="8px"))
    history: list[str] = []

    def render(live: str = ""):
        body = "".join(history) + (f"<p><b>agent</b> {html.escape(live)}</p>" if live else "")
        log.value = body or "<i>say hi</i>"

    def set_status(s):
        status.value = f"<code>{html.escape(s)}</code>"

    def run(said: str | None, kickoff: bool = False, stt_s: float | None = None):
        if said:
            history.append(f"<p><b>you</b> {html.escape(said)}</p>")
        voice = mouth.StreamingMouth() if speak else None
        buf, last = [], [0.0]

        def grow(piece):
            buf.append(piece)
            if voice:
                voice.feed(piece)
            if time.time() - last[0] > 0.1:
                render("".join(buf))
                last[0] = time.time()

        try:
            res = bot.kickoff(grow, set_status) if kickoff else bot.turn(said, grow, set_status)
            if res is None:
                return
            t = bot.turns[-1]
            if t.images:
                seen.value = t.images[-1]
            bits = [res.reply.stats(), f"{res.looked} image(s)"]
            if stt_s is not None:
                bits.insert(0, f"whisper {stt_s:.2f}s")
            if "caption" in res.extra:
                bits.append(f"remembered: {res.extra['caption']}")
            history.append(f"<p><b>agent</b> {html.escape(res.shown)}<br>"
                           f"<small><code>{html.escape(' | '.join(bits))}</code></small></p>")
            render()
        except Exception as e:  # noqa: BLE001
            history.append(f"<p style='color:#c33'>{html.escape(str(e))}</p>")
            render()
        finally:
            set_status("")
            if voice:
                voice.finish(wait=False)

    def on_mode(change):
        bot.set_mode(modes[names.index(change["new"])])
        mem_dd.value = bot.mode["memory"]
        history.clear()
        history.append(f"<p><i>mode: {html.escape(bot.mode['name'])}</i></p>")
        if bot.mode["look"] == "before_after":
            history.append("<p><i>Took the BEFORE photo. Change something, then ask what changed.</i></p>")
        render()
        if bot.mode.get("kickoff"):
            run(None, kickoff=True)

    def on_mem(change):
        bot.mode["memory"] = change["new"]

    def on_talk(change):
        if change["new"]:
            rec.start()
            talk_b.description, talk_b.button_style = "Stop", "danger"
            set_status("listening...")
        else:
            talk_b.description, talk_b.button_style = "Start talking", "success"
            audio = rec.stop()
            set_status("transcribing")
            said, stt_s = ears.transcribe(audio)
            if not said:
                set_status("didn't catch that")
                return
            run(said, stt_s=stt_s)

    def on_send(_):
        if text.value.strip():
            said, text.value = text.value.strip(), ""
            run(said)

    def on_reset(_):
        bot.reset()
        history.clear()
        render()

    mode_dd.observe(on_mode, names="value")
    mem_dd.observe(on_mem, names="value")
    talk_b.observe(on_talk, names="value")
    send_b.on_click(on_send)
    text.on_submit(lambda _: on_send(None))
    reset_b.on_click(on_reset)

    mem_dd.value = bot.mode["memory"]
    render()
    box = w.VBox([
        w.HBox([mode_dd, mem_dd, reset_b]),
        w.HBox([log, w.VBox([w.HTML("<small>last thing it saw</small>"), seen])]),
        w.HBox([talk_b, text, send_b]),
        status,
    ])
    box.controls = dict(mode=mode_dd, memory=mem_dd, talk=talk_b, text=text, send=send_b,
                        reset=reset_b, log=log, seen=seen)
    display(box)
    if bot.mode.get("kickoff"):
        run(None, kickoff=True)
    return box
