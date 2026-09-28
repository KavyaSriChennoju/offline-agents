"""
Download a vision model and start llama-server, from inside a notebook.

    from agent import server
    model, mmproj = server.download("gemma-4-E2B")   # progress bars, ~3-4 GB, once
    server.start(model, mmproj)                        # runs in the background
    server.stop()                                      # when you're done

Files land in models/<name>/. If that folder already has a model .gguf and an
mmproj .gguf (say, copied from a USB stick), nothing is downloaded.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

import config

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
LOG = MODELS_DIR / "llama-server.log"
PIDFILE = MODELS_DIR / "llama-server.pid"

# name -> (Hugging Face repo, preferred quant, context size)
MODELS = {
    "gemma-4-E2B":   ("ggml-org/gemma-4-E2B-it-GGUF", "Q4_0", 8192),     # most laptops, ~3-4 GB
    "smolvlm2-2.2B": ("ggml-org/SmolVLM2-2.2B-Instruct-GGUF", "Q4_K_M", 8192),  # 8 GB RAM, ~1.7 GB
    "potato":        ("ggml-org/SmolVLM-256M-Instruct-GGUF", "Q8_0", 4096),     # tiny, ~0.3 GB
    "gemma-4-E4B":   ("ggml-org/gemma-4-E4B-it-GGUF", "Q4_0", 8192),     # big machine, ~5-6 GB
    "qwen2.5-omni-3B": ("ggml-org/Qwen2.5-Omni-3B-GGUF", "Q4_K_M", 8192),  # sees + hears, ~3.6 GB
}


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------

def _pick(files: list[str], quant: str) -> tuple[str, str]:
    """From a list of .gguf names, pick the model file and the vision projector."""
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    projs = [f for f in ggufs if "mmproj" in f.lower()]
    # skip vision/audio projectors and "mtp" draft heads (small speculative-decoding helpers)
    mains = [f for f in ggufs if "mmproj" not in f.lower() and not f.lower().startswith("mtp")]
    if not mains or not projs:
        raise RuntimeError(f"Expected a model .gguf and an mmproj .gguf, found: {ggufs}")

    def rank_main(f):
        f_l = f.lower()
        return (quant.lower() not in f_l, "bf16" in f_l or "f16" in f_l or "f32" in f_l, len(f))

    def rank_proj(f):
        f_l = f.lower()
        return (0 if "q8" in f_l else 1 if "f16" in f_l else 3 if "f32" in f_l else 2, len(f))

    return sorted(mains, key=rank_main)[0], sorted(projs, key=rank_proj)[0]


def local_files(name: str) -> tuple[Path, Path] | None:
    """Model + projector already sitting in models/<name>/ ? (downloaded before, or from a USB stick)"""
    folder = MODELS_DIR / name
    if not folder.is_dir():
        return None
    files = [p.name for p in folder.glob("*.gguf")]
    try:
        main, proj = _pick(files, MODELS.get(name, ("", "Q4", 0))[1])
    except RuntimeError:
        return None
    return folder / main, folder / proj


def download(name: str = "gemma-4-E2B") -> tuple[Path, Path]:
    """Download (once) the model and its vision projector. Returns their paths."""
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Pick one of: {', '.join(MODELS)}")
    have = local_files(name)
    if have:
        print(f"already have {name}:\n  {have[0].name}\n  {have[1].name}")
        return have

    from huggingface_hub import HfApi, hf_hub_download

    repo, quant, _ = MODELS[name]
    folder = MODELS_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    main, proj = _pick(HfApi().list_repo_files(repo), quant)
    print(f"downloading from {repo}:\n  {main}\n  {proj}\n(this happens once; grab a coffee on slow wifi)")
    paths = [Path(hf_hub_download(repo, f, local_dir=folder)) for f in (main, proj)]
    total = sum(p.stat().st_size for p in paths) / 1e9
    print(f"done: {total:.2f} GB in models/{name}/")
    return paths[0], paths[1]


# ---------------------------------------------------------------------------
# run llama-server in the background
# ---------------------------------------------------------------------------

def _port() -> int:
    return int(config.SERVER_URL.rsplit(":", 1)[-1].split("/")[0])


def is_up() -> bool:
    try:
        return requests.get(f"{config.SERVER_URL}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def find_binary() -> str | None:
    if config.LLAMA_SERVER:
        path = os.path.expanduser(config.LLAMA_SERVER)
        return path if os.path.isfile(path) else None
    return shutil.which("llama-server") or shutil.which("llama-server.exe")


INSTALL_HINT = {
    "darwin": "brew install llama.cpp",
    "win32": "winget install llama.cpp   (then open a new terminal and restart Jupyter)",
    "linux": "brew install llama.cpp, or a prebuilt zip / source build (see INSTALL.md)",
}


def check() -> bool:
    """Is llama-server installed? Prints the version, or how to install it."""
    exe = find_binary()
    if exe is None:
        where = f"AGENT_LLAMA_SERVER={config.LLAMA_SERVER} (file not found)" if config.LLAMA_SERVER else "your PATH"
        print(f"llama-server: NOT FOUND in {where}")
        print(f"  install it: {INSTALL_HINT.get(sys.platform, 'see INSTALL.md')}")
        print("  then stop Jupyter (ctrl-c in its terminal) and run `uv run jupyter lab` again.")
        print("  Full guide for every OS: INSTALL.md")
        return False
    try:
        out = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20)
        version = next((l for l in (out.stdout + out.stderr).splitlines() if "version" in l.lower()), "").strip()
    except (OSError, subprocess.TimeoutExpired):
        version = ""
    print(f"llama-server: {exe}  {version}")
    return True


def _thinking_flags(exe: str, thinking: bool) -> list[str]:
    """Turn 'thinking' off with whichever flag this llama-server version understands."""
    if thinking:
        return []
    try:
        out = subprocess.run([exe, "--help"], capture_output=True, text=True, timeout=20)
        helptext = out.stdout + out.stderr
    except (OSError, subprocess.TimeoutExpired):
        return []
    if "--reasoning " in helptext or "--reasoning," in helptext or "--reasoning\n" in helptext:
        return ["--reasoning", "off"]
    if "--reasoning-budget" in helptext:
        return ["--reasoning-budget", "0"]
    return []


def start(model: Path, mmproj: Path, ctx: int | None = None, extra: list[str] | None = None,
          wait_s: int = 300, thinking: bool = config.THINKING) -> int | None:
    """
    Start llama-server in the background and wait until it's ready. If one started from here is
    already running, it's restarted (so a new MODEL or setting takes effect). Safe to re-run.
    It keeps running if you restart the kernel or close this notebook; use stop() to end it.
    """
    if is_up():
        if not PIDFILE.exists():
            print(f"a llama-server you started yourself (in a terminal?) is already running at {config.SERVER_URL}.\n"
                  "Using it as is. To let this notebook manage it, stop that one (ctrl-c) and run this cell again.")
            return None
        print("restarting the server started from here, so it picks up the model and settings in this cell")
        stop()
        for _ in range(30):
            if not is_up():
                break
            time.sleep(0.5)
    exe = find_binary()
    if exe is None:
        check()
        raise RuntimeError("llama-server isn't installed yet. See the message above, or INSTALL.md.")
    if ctx is None:
        ctx = next((c for n, (_, _, c) in MODELS.items() if str(model).startswith(str(MODELS_DIR / n))), 8192)

    cmd = [exe, "-m", str(model), "--mmproj", str(mmproj), "-c", str(ctx), "--port", str(_port()),
           *_thinking_flags(exe, thinking), *(extra or [])]
    MODELS_DIR.mkdir(exist_ok=True)
    log = open(LOG, "w")
    kwargs = {"start_new_session": True} if os.name != "nt" else {
        "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "DETACHED_PROCESS", 0)}
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, **kwargs)
    PIDFILE.write_text(str(proc.pid))
    print("starting:", " ".join(Path(c).name if i in (0, 2, 4) else c for i, c in enumerate(cmd)))

    t0 = time.time()
    while time.time() - t0 < wait_s:
        if proc.poll() is not None:
            raise RuntimeError(f"llama-server exited early. Last lines of the log:\n{tail(15)}")
        if is_up():
            print(f"\nready in {time.time() - t0:.0f}s at {config.SERVER_URL} (log: models/llama-server.log)")
            return proc.pid
        sys.stdout.write(".")
        sys.stdout.flush()
        time.sleep(1)
    raise TimeoutError(f"llama-server didn't become ready in {wait_s}s. Log:\n{tail(15)}")


def stop():
    """Stop the llama-server that start() launched."""
    if not PIDFILE.exists():
        print("no server started from here (if you started one in a terminal, stop it there with ctrl-c)")
        return
    pid = int(PIDFILE.read_text())
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
        else:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        print(f"stopped llama-server (pid {pid})")
    except (ProcessLookupError, PermissionError, OSError):
        print("it had already stopped")
    PIDFILE.unlink(missing_ok=True)


def tail(n: int = 20) -> str:
    """Last lines of the server log. Useful when something goes wrong."""
    if not LOG.exists():
        return "(no log yet)"
    return "\n".join(LOG.read_text(errors="replace").splitlines()[-n:])
