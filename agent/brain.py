"""
The response layer: turns "what you said" + "what the camera sees" + "what
happened before" into one prompt, and keeps the conversation going.

This is where most of the interesting decisions live. The model is fixed;
what you choose to put in front of it is not.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime

from agent.eyes import Eyes, preprocess
from agent.llm import LocalLLM, Reply, assistant, system, user
import modes


@dataclass
class Turn:
    said: str
    images: list[bytes]
    reply: str = ""                   # full reply, including hidden lines
    caption: str | None = None        # one-line description of what the camera saw
    hidden_kickoff: bool = False


@dataclass
class TurnResult:
    shown: str                        # what the user sees / hears
    reply: Reply
    looked: int                       # how many images went in
    extra: dict = field(default_factory=dict)


class Agent:
    def __init__(self, llm: LocalLLM, eyes: Eyes | None, mode: dict):
        self.llm, self.eyes = llm, eyes
        self.turns: list[Turn] = []
        self.before: bytes | None = None
        self.set_mode(mode)

    # ------------------------------------------------------------------ modes
    def set_mode(self, mode: dict):
        self.mode = {
            "look": "when_asked", "memory": "latest_image", "max_turns": 8,
            "temperature": 0.7, "max_tokens": 160, "burst": 3, "burst_gap": 0.6,
            **mode,
        }
        self.reset()

    def reset(self):
        self.turns = []
        self.before = None
        if self.mode["look"] == "before_after" and self.eyes:
            self.before = self._snap()

    # ------------------------------------------------------------------ seeing
    def _snap(self) -> bytes:
        return preprocess(self.eyes.frame())

    def wants_to_look(self, said: str) -> bool:
        s = said.lower()
        return any(re.search(rf"\b{re.escape(w)}\b", s) for w in modes.LOOK_WORDS)

    def gather_images(self, said: str, on_status=None) -> list[bytes]:
        look = self.mode["look"]
        if not self.eyes or look == "never":
            return []
        if look == "when_asked" and not self.wants_to_look(said):
            return []
        if look == "before_after":
            return [self.before or self._snap(), self._snap()]
        if look == "burst":
            frames = []
            for i in range(self.mode["burst"]):
                if on_status:
                    on_status(f"recording frame {i + 1}/{self.mode['burst']}")
                frames.append(self._snap())
                if i < self.mode["burst"] - 1:
                    time.sleep(self.mode["burst_gap"])
            return frames
        return [self._snap()]

    # ------------------------------------------------------------------ prompt building
    def build_messages(self, said: str, images: list[bytes]) -> list[dict]:
        m = self.mode
        prompt = m["system"].replace("{now}", datetime.now().strftime("%A %I:%M %p"))
        msgs = [system(prompt)]

        for t in self.turns[-m["max_turns"]:]:
            if m["memory"] == "all_images":
                msgs.append(user(t.said, images=t.images))
            elif m["memory"] == "captions" and t.caption:
                msgs.append(user(f"[camera: {t.caption}]\n{t.said}"))
            else:
                note = f" [{len(t.images)} camera frame(s) were shown here]" if t.images else ""
                msgs.append(user(t.said + note))
            msgs.append(assistant(t.reply))

        msgs.append(user(said, images=images))
        return msgs

    def hide(self, text: str) -> str:
        pat = self.mode.get("hide")
        return re.sub(pat, "", text).strip() if pat else text.strip()

    # ------------------------------------------------------------------ one turn
    def turn(self, said: str, on_token=None, on_status=None, hidden: bool = False) -> TurnResult:
        images = self.gather_images(said, on_status)
        msgs = self.build_messages(said, images)
        if on_status:
            on_status("thinking" + (f" ({len(images)} image{'s' if len(images) != 1 else ''})" if images else ""))

        # if part of the reply must stay secret, we can't stream it straight to the screen/voice
        stream_cb = None if self.mode.get("hide") else on_token
        reply = self.llm.chat(
            msgs, on_token=stream_cb,
            temperature=self.mode["temperature"], max_tokens=self.mode["max_tokens"],
        )
        shown = self.hide(reply.text)
        if on_token and stream_cb is None:
            on_token(shown)

        t = Turn(said, images, reply.text, hidden_kickoff=hidden)
        extra = {}
        if self.mode["memory"] == "captions" and images:
            t0 = time.perf_counter()
            t.caption = self.caption(images[-1])
            extra["caption"] = t.caption
            extra["caption_s"] = time.perf_counter() - t0
        self.turns.append(t)
        return TurnResult(shown, reply, len(images), extra)

    def kickoff(self, on_token=None, on_status=None) -> TurnResult | None:
        if not self.mode.get("kickoff"):
            return None
        return self.turn(self.mode["kickoff"], on_token, on_status, hidden=True)

    def caption(self, jpeg: bytes) -> str:
        r = self.llm.chat(
            [user("Describe this webcam frame in one short line: main objects and where they are.",
                  images=[jpeg])],
            stream=False, temperature=0, max_tokens=40,
        )
        return r.text.strip().replace("\n", " ")

    # ------------------------------------------------------------------ introspection
    def context_report(self) -> str:
        imgs = sum(len(t.images) for t in self.turns[-self.mode["max_turns"]:]) if self.mode["memory"] == "all_images" else 0
        return f"{len(self.turns)} turns in memory, {imgs} old images re-sent each turn"
