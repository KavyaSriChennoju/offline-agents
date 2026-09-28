"""A 'thinking' model must still answer (we ask it not to think), and if it only thinks we say so."""
import os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config  # noqa: E402
from agent.llm import LocalLLM, user  # noqa: E402

p = subprocess.Popen([sys.executable, str(ROOT / "tests/fake_llama_server.py"), "--port", "8095"],
                     env={**os.environ, "FAKE_THINKING": "1"})
time.sleep(1.5)
try:
    llm = LocalLLM("http://127.0.0.1:8095")
    for stream in (True, False):
        config.THINKING = False
        r = llm.chat([user("tell a poem")], stream=stream)
        assert r.text, "thinking off -> should get an answer"
        config.THINKING = True
        r = llm.chat([user("tell a poem")], stream=stream)
        assert not r.text and r.thinking, "thinking on -> fake model only thinks"
    print("thinking tests ok")
finally:
    p.terminate()
