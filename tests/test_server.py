"""
Check agent/server.py without the internet or a real llama.cpp:
fake Hugging Face calls, a fake llama-server binary on PATH.

    python tests/test_server.py
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["PATH"] = str(ROOT / "tests" / "fake_bin") + os.pathsep + os.environ["PATH"]

import huggingface_hub  # noqa: E402

import config  # noqa: E402
from agent import server  # noqa: E402

config.SERVER_URL = "http://127.0.0.1:8093"
server.MODELS_DIR = Path(tempfile.mkdtemp())
server.LOG, server.PIDFILE = server.MODELS_DIR / "log.txt", server.MODELS_DIR / "pid"

FILES = ["README.md", "mtp-gemma-4-E2B-it-Q4_0.gguf", "gemma-4-E2B-it-BF16.gguf", "gemma-4-E2B-it-Q8_0.gguf", "gemma-4-E2B-it-Q4_0.gguf",
         "mmproj-gemma-4-E2B-it-BF16.gguf", "mmproj-gemma-4-E2B-it-Q8_0.gguf"]
calls = []


class FakeApi:
    def list_repo_files(self, repo):
        calls.append(("list", repo))
        return FILES


def fake_dl(repo, filename, local_dir):
    calls.append(("dl", filename))
    p = Path(local_dir) / filename
    p.write_bytes(b"0" * 1000)
    return str(p)


huggingface_hub.HfApi, huggingface_hub.hf_hub_download = FakeApi, fake_dl

# picks the right files
m, p = server.download("gemma-4-E2B")
assert m.name == "gemma-4-E2B-it-Q4_0.gguf", m
assert p.name == "mmproj-gemma-4-E2B-it-Q8_0.gguf", p

# second call: no network
n = len(calls)
server.download("gemma-4-E2B")
assert len(calls) == n, "should reuse local files"

# picker on other repos
assert server._pick(["SmolVLM2-2.2B-Instruct-Q4_K_M.gguf", "SmolVLM2-2.2B-Instruct-Q8_0.gguf",
                     "SmolVLM2-2.2B-Instruct-f16.gguf", "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf",
                     "mmproj-SmolVLM2-2.2B-Instruct-f16.gguf"], "Q4_K_M") == \
    ("SmolVLM2-2.2B-Instruct-Q4_K_M.gguf", "mmproj-SmolVLM2-2.2B-Instruct-Q8_0.gguf")
assert server._pick(["SmolVLM-256M-Instruct-f16.gguf", "SmolVLM-256M-Instruct-Q8_0.gguf",
                     "mmproj-SmolVLM-256M-Instruct-f16.gguf"], "Q8_0")[0] == "SmolVLM-256M-Instruct-Q8_0.gguf"

# start / already running / stop
pid = server.start(m, p, wait_s=20)
assert pid and server.is_up()
pid2 = server.start(m, p, wait_s=20)        # running + ours -> restart
assert pid2 and pid2 != pid and server.is_up()
server.stop()
import time  # noqa: E402
time.sleep(1)
assert not server.is_up(), "server should be down"
print("\nserver tests ok")

# check() finds the fake binary, and honours AGENT_LLAMA_SERVER
assert server.check()
config.LLAMA_SERVER = "/nope/llama-server"
assert not server.check()
config.LLAMA_SERVER = str(ROOT / "tests" / "fake_bin" / "llama-server")
assert server.find_binary() == config.LLAMA_SERVER
print("check() ok")

# thinking gets switched off with whichever flag this llama-server knows
assert server._thinking_flags(server.find_binary(), thinking=False) == ["--reasoning", "off"]
assert server._thinking_flags(server.find_binary(), thinking=True) == []
print("thinking flags ok")
