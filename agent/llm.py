"""
A tiny client for llama-server's OpenAI-compatible API.

No SDK on purpose: it's one POST request with some JSON. Read this file once
and you know exactly what goes over the wire.
"""
from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field

import requests

import config


# ---------------------------------------------------------------------------
# building messages
# ---------------------------------------------------------------------------

def image_part(jpeg_bytes: bytes) -> dict:
    """One image, in the shape the OpenAI chat API (and llama-server) expects."""
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}


def audio_part(wav_bytes: bytes) -> dict:
    """One audio clip. Only audio-capable models (Gemma 4, Qwen2.5-Omni, Voxtral...) accept this."""
    b64 = base64.b64encode(wav_bytes).decode("ascii")
    return {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}


def user(text: str, images: list[bytes] | None = None, audio: list[bytes] | None = None) -> dict:
    """A user message. Images first, then audio clips (WAV bytes), then the text."""
    if not images and not audio:
        return {"role": "user", "content": text}
    parts = [image_part(img) for img in images or []]
    parts += [audio_part(wav) for wav in audio or []]
    parts.append({"type": "text", "text": text})
    return {"role": "user", "content": parts}


def system(text: str) -> dict:
    return {"role": "system", "content": text}


def assistant(text: str) -> dict:
    return {"role": "assistant", "content": text}


# ---------------------------------------------------------------------------
# talking to the server
# ---------------------------------------------------------------------------

@dataclass
class Reply:
    text: str
    seconds: float                 # wall clock for the whole request
    first_token_s: float | None    # time until the first word showed up (streaming only)
    timings: dict = field(default_factory=dict)  # llama-server's own numbers, if it sent them
    thinking: str = ""             # "reasoning" text some models write before answering (not spoken)

    @property
    def prompt_tokens(self) -> int | None:
        return self.timings.get("prompt_n")

    @property
    def tokens_per_s(self) -> float | None:
        return self.timings.get("predicted_per_second")

    def stats(self) -> str:
        bits = [f"{self.seconds:.2f}s total"]
        if self.first_token_s is not None:
            bits.append(f"first word {self.first_token_s:.2f}s")
        if self.prompt_tokens is not None:
            bits.append(f"{self.prompt_tokens} prompt tokens")
        if self.tokens_per_s:
            bits.append(f"{self.tokens_per_s:.1f} tok/s")
        return " | ".join(bits)


class ServerDown(RuntimeError):
    pass


class LocalLLM:
    def __init__(self, url: str = config.SERVER_URL):
        self.url = url.rstrip("/")

    # -- health ---------------------------------------------------------------
    def health(self) -> bool:
        try:
            r = requests.get(f"{self.url}/health", timeout=3)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def model_name(self) -> str:
        try:
            r = requests.get(f"{self.url}/v1/models", timeout=3)
            data = r.json().get("data") or []
            return data[0]["id"] if data else "unknown"
        except (requests.RequestException, ValueError, KeyError, IndexError):
            return "unknown"

    # -- the one call that matters -----------------------------------------------
    def chat(
        self,
        messages: list[dict],
        *,
        stream: bool = True,
        on_token=None,               # called with each chunk of text as it arrives
        temperature: float = 0.7,
        max_tokens: int = 256,
        **extra,                     # anything else llama-server understands (top_p, seed, ...)
    ) -> Reply:
        payload = {
            "messages": messages,
            "stream": stream,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **extra,
        }
        if not config.THINKING:
            # Ask thinking models (Gemma 4, Qwen3...) to answer straight away. Ignored by others.
            payload.setdefault("chat_template_kwargs", {"enable_thinking": False})
        t0 = time.perf_counter()
        try:
            r = requests.post(
                f"{self.url}/v1/chat/completions",
                json=payload,
                stream=stream,
                timeout=config.REQUEST_TIMEOUT_S,
            )
        except requests.ConnectionError as e:
            raise ServerDown(
                f"Can't reach llama-server at {self.url}. Is it running? (see README, step 1)"
            ) from e

        if r.status_code != 200:
            hint = ""
            if r.status_code in (400, 500) and "image" in r.text.lower():
                hint = "\n  -> This model may not support images. Did you start the server with a vision model?"
            elif r.status_code in (400, 500) and "audio" in r.text.lower():
                hint = ("\n  -> This model can't hear. Use one that takes audio: MODEL = \"gemma-4-E2B\" "
                        "or \"qwen2.5-omni-3B\" in 1_setup.ipynb.")
            raise RuntimeError(f"llama-server said {r.status_code}: {r.text[:400]}{hint}")

        if not stream:
            data = r.json()
            msg = data["choices"][0]["message"]
            reply = Reply(msg.get("content") or "", time.perf_counter() - t0, None,
                          data.get("timings", {}), msg.get("reasoning_content") or "")
            _warn_if_only_thinking(reply)
            return reply

        # streaming: server-sent events, one JSON blob per "data:" line
        text, first, timings, thinking = [], None, {}, []
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            body = raw[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            if "timings" in chunk:
                timings = chunk["timings"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                if delta.get("reasoning_content"):
                    thinking.append(delta["reasoning_content"])   # kept, but not shown or spoken
                piece = delta.get("content")
                if piece:
                    if first is None:
                        first = time.perf_counter() - t0
                    text.append(piece)
                    if on_token:
                        on_token(piece)
        reply = Reply("".join(text), time.perf_counter() - t0, first, timings, "".join(thinking))
        _warn_if_only_thinking(reply)
        return reply


def _warn_if_only_thinking(reply: Reply):
    """Thinking models can spend every token 'thinking' and never answer. Say so, loudly."""
    if not reply.text.strip() and (reply.thinking or (reply.timings.get("predicted_n") or 0) > 0):
        n = reply.timings.get("predicted_n", "all its")
        print(f"\n[empty reply: the model spent {n} tokens thinking and never answered.\n"
              " Fix: re-run server.start(...) in 1_setup.ipynb (it restarts with thinking off),\n"
              " or restart llama-server with --reasoning off. Or raise max_tokens.]")
