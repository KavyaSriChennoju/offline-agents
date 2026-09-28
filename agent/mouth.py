"""
Text to speech using whatever voice your OS already has. Zero downloads.

macOS: `say`      Windows: SAPI via PowerShell      Linux: espeak-ng / espeak / spd-say

Want a nicer voice? See CHALLENGES.md, "Better mouth" (Piper / Kokoro run fully offline).
"""
from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import sys
import threading

import config


def _command(text: str) -> list[str] | None:
    if sys.platform == "darwin":
        cmd = ["say"]
        if config.VOICE:
            cmd += ["-v", config.VOICE]
        return cmd + [text]
    if os.name == "nt":
        safe = text.replace("'", "''")
        ps = (
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe}')"
        )
        return ["powershell", "-NoProfile", "-Command", ps]
    for exe in ("espeak-ng", "espeak", "spd-say"):
        if shutil.which(exe):
            return [exe, text] if exe != "spd-say" else [exe, "-w", text]
    return None


def clean_for_speech(text: str) -> str:
    """Strip markdown and emoji-ish stuff so the voice doesn't read out asterisks."""
    text = re.sub(r"[*_#`>|]", "", text)
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"[^\w\s.,!?;:'\"()%$-]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def speak(text: str, block: bool = True):
    if not config.SPEAK:
        return
    text = clean_for_speech(text)
    if not text:
        return
    cmd = _command(text)
    if cmd is None:
        return  # no TTS on this machine; the text is already on screen
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if block:
        p.wait()


class StreamingMouth:
    """
    Feed it tokens as they stream in; it speaks each sentence as soon as the
    sentence is complete. The first sentence starts playing while the model is
    still writing the second. That's the whole trick behind "feels real-time".
    """

    _END = re.compile(r"([.!?])(\s|$)")

    def __init__(self):
        self._buf = ""
        self._q: queue.Queue[str | None] = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def _run(self):
        while True:
            s = self._q.get()
            if s is None:
                break
            speak(s, block=True)

    def feed(self, token: str):
        self._buf += token
        while True:
            m = self._END.search(self._buf)
            if not m:
                break
            sentence, self._buf = self._buf[: m.end()].strip(), self._buf[m.end():]
            if sentence:
                self._q.put(sentence)

    def finish(self, wait: bool = True):
        if self._buf.strip():
            self._q.put(self._buf.strip())
        self._buf = ""
        self._q.put(None)
        if wait:
            self._worker.join()
