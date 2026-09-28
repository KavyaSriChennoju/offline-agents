"""
Every knob in the workshop lives here. Change a value, re-run the script.

You can also override most of these with environment variables, e.g.
    AGENT_SERVER=http://192.168.1.20:8080 python 2_eyes.py
"""
import os

# ---- the brain -------------------------------------------------------------
# Where llama-server is listening. Default port is 8080.
SERVER_URL = os.environ.get("AGENT_SERVER", "http://127.0.0.1:8080")
# Where llama-server lives, if it isn't on your PATH (prebuilt zip / source build). See INSTALL.md.
LLAMA_SERVER = os.environ.get("AGENT_LLAMA_SERVER")
REQUEST_TIMEOUT_S = 180          # first image request on a slow laptop can take a while

# Some models (Gemma 4, Qwen3) "think" before answering. Slow, and it can eat every token.
# False = ask them to answer straight away (recommended). True = let them think.
THINKING = os.environ.get("AGENT_THINKING", "0") == "1"

# ---- the eyes --------------------------------------------------------------
# 0 = your default webcam (try 1 if you get the wrong one), or a photo / folder path
CAMERA = os.environ.get("AGENT_CAMERA", "0")
IMAGE_MAX_SIDE = 512             # longest edge in pixels before we send the frame
JPEG_QUALITY = 80                # 1-100. Lower = smaller payload, uglier image
GRAYSCALE = False                # colour costs nothing extra in tokens. Try it anyway.

# ---- the ears --------------------------------------------------------------
# tiny.en (75 MB, fastest) | base.en (145 MB) | small.en (480 MB, best of the three)
WHISPER_MODEL = os.environ.get("AGENT_WHISPER", "base.en")
WHISPER_DEVICE = "auto"          # "cpu" if you get CUDA errors
SAMPLE_RATE = 16_000             # whisper wants 16 kHz mono. Don't touch.

# ---- the mouth -------------------------------------------------------------
SPEAK = os.environ.get("AGENT_SPEAK", "1") == "1"   # AGENT_SPEAK=0 to keep it quiet
VOICE = None                     # macOS: "Samantha", "Daniel", "Karen"...  `say -v '?'` lists them
