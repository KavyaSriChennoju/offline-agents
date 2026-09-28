"""
Microphone in, text out. Whisper runs locally via faster-whisper.

Recording is push-to-talk: press Enter to start, Enter to stop.
No wake word, no voice-activity detection. Simple beats clever at 9am.
"""
from __future__ import annotations

import io
import threading
import time
import wave

import numpy as np

import config

_whisper = None


# ---------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------

def _sd():
    try:
        import sounddevice as sd
    except OSError as e:  # PortAudio missing (some Linux boxes)
        raise RuntimeError(
            "sounddevice can't find PortAudio. Linux: sudo apt install libportaudio2. "
            "Or skip the mic with --text / --wav."
        ) from e
    return sd


def record_until_enter(prompt: str = "  [recording... press Enter to stop]") -> np.ndarray:
    """Record from the default mic until the user hits Enter. Returns float32 mono @ 16 kHz."""
    sd = _sd()
    chunks: list[np.ndarray] = []

    def callback(indata, frames, t, status):
        chunks.append(indata.copy())

    with sd.InputStream(samplerate=config.SAMPLE_RATE, channels=1, dtype="float32", callback=callback):
        input(prompt)
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)[:, 0]


def record_seconds(seconds: float) -> np.ndarray:
    sd = _sd()
    audio = sd.rec(int(seconds * config.SAMPLE_RATE), samplerate=config.SAMPLE_RATE, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


class BackgroundRecorder:
    """Start/stop recording without blocking. Used by the live camera window."""

    def __init__(self):
        self._chunks: list[np.ndarray] = []
        self._stream = None
        self.started_at = 0.0

    @property
    def recording(self) -> bool:
        return self._stream is not None

    def start(self):
        sd = _sd()
        self._chunks = []
        self._stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE, channels=1, dtype="float32",
            callback=lambda indata, *_: self._chunks.append(indata.copy()),
        )
        self._stream.start()
        self.started_at = time.time()

    def stop(self) -> np.ndarray:
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        self._stream.stop()
        self._stream.close()
        self._stream = None
        if not self._chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self._chunks)[:, 0]


# ---------------------------------------------------------------------------
# wav helpers
# ---------------------------------------------------------------------------

def to_wav_bytes(audio: np.ndarray, sr: int = config.SAMPLE_RATE) -> bytes:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def load_wav(path: str) -> np.ndarray:
    """Load a wav as float32 mono @ 16 kHz (resamples crudely if needed)."""
    with wave.open(path, "rb") as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width not in (2, 4):
        raise ValueError(f"{path}: only 16/32-bit PCM wavs, please (got {width * 8}-bit)")
    dtype = {2: np.int16, 4: np.int32}[width]
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float32) / np.iinfo(dtype).max
    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)
    if sr != config.SAMPLE_RATE:
        n = int(len(audio) * config.SAMPLE_RATE / sr)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio)
    return audio.astype(np.float32)


def loudness(audio: np.ndarray) -> float:
    """RMS level, 0..1. Below ~0.005 you probably recorded silence."""
    return float(np.sqrt(np.mean(audio**2))) if len(audio) else 0.0


# ---------------------------------------------------------------------------
# speech to text
# ---------------------------------------------------------------------------

def load_whisper(model: str = config.WHISPER_MODEL):
    """First call downloads the model (once) and loads it. Takes a few seconds."""
    global _whisper
    if _whisper is None or _whisper[0] != model:
        from faster_whisper import WhisperModel

        compute = "int8"  # fast on CPU, fine on GPU
        _whisper = (model, WhisperModel(model, device=config.WHISPER_DEVICE, compute_type=compute))
    return _whisper[1]


def transcribe(
    audio: np.ndarray,
    model: str = config.WHISPER_MODEL,
    language: str | None = "en",
    vocab: str | None = None,
) -> tuple[str, float]:
    """
    Returns (text, seconds_it_took).
    vocab: words whisper should expect, e.g. "llama.cpp, Gemma, Arm, MuJoCo".
           It nudges spelling. It's a hint, not a guarantee.
    """
    if len(audio) < config.SAMPLE_RATE * 0.3:
        return "", 0.0
    w = load_whisper(model)
    t0 = time.perf_counter()
    segments, _info = w.transcribe(
        audio,
        language=language,
        beam_size=1,        # greedy. beam_size=5 is a little better and a lot slower
        vad_filter=True,    # trims silence so whisper doesn't hallucinate "Thank you." into it
        initial_prompt=vocab,
    )
    text = " ".join(s.text.strip() for s in segments).strip()
    return text, time.perf_counter() - t0


def warm_up_in_background(model: str = config.WHISPER_MODEL):
    """Load whisper while the user is doing something else."""
    threading.Thread(target=lambda: load_whisper(model), daemon=True).start()
