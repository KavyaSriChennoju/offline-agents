# Installing uv and llama.cpp

You need two things on your laptop before the workshop:

- **uv** installs Python and all the workshop libraries, and runs Jupyter.
- **llama.cpp** gives you `llama-server`, the program that runs the model.

Pick your OS below. Each section has a **quick way** (one or two commands) and
**other ways** if the quick one doesn't suit you. Finish with [Check it worked](#check-it-worked).

About 10 minutes. Then go back to the [README](README.md), step 2.

---

## macOS

### Quick way: Homebrew

No Homebrew yet? Install it first (it asks for your password once):

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then:

```bash
brew install uv llama.cpp
```

On Apple Silicon (M1 and later) llama.cpp uses the GPU through Metal automatically. Nothing to configure.

### Other ways

| Tool | uv | llama.cpp |
|---|---|---|
| No Homebrew | `curl -LsSf https://astral.sh/uv/install.sh \| sh` | prebuilt zip (see [Prebuilt binaries](#prebuilt-binaries-any-os)) |
| MacPorts | `sudo port install uv` | `sudo port install llama.cpp` |
| Nix | | `nix profile install nixpkgs#llama-cpp` |
| conda / mamba | | `conda install -c conda-forge llama.cpp` |

---

## Windows

Use **PowerShell** or **Windows Terminal**.

### Quick way: winget (built into Windows 10 and 11)

```powershell
winget install --id=astral-sh.uv -e
winget install llama.cpp
```

Then **close the terminal and open a new one**, so it picks up the new commands.

### Other ways

| Tool | uv | llama.cpp |
|---|---|---|
| No winget | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` | prebuilt zip (see [Prebuilt binaries](#prebuilt-binaries-any-os)) |
| Scoop | `scoop install main/uv` | |
| conda / mamba | | `conda install -c conda-forge llama.cpp` |

### Using your NVIDIA or AMD GPU

A GPU makes the model several times faster, but it's optional: everything works on the CPU.
The prebuilt zips come in versions for NVIDIA (**cuda**) and for most GPUs (**vulkan**). If
llama-server's startup log doesn't mention your GPU, download the matching zip from the releases page.

---

## Linux

### Quick way: installer script + Homebrew

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # uv
brew install llama.cpp                               # if you use Homebrew on Linux
```

No Homebrew? Use a [prebuilt binary](#prebuilt-binaries-any-os) or [build it](#build-from-source-any-os).
Either takes about 2 minutes.

Also install the mic and voice libraries:

```bash
sudo apt install libportaudio2 espeak-ng        # Debian / Ubuntu
sudo dnf install portaudio espeak-ng            # Fedora
```

Open a new terminal afterwards (or `source ~/.bashrc`) so `uv` is on your PATH.

### Other ways

| Tool | uv | llama.cpp |
|---|---|---|
| no curl | `wget -qO- https://astral.sh/uv/install.sh \| sh` | |
| pipx | `pipx install uv` | |
| Nix | | `nix profile install nixpkgs#llama-cpp` |
| conda / mamba | | `conda install -c conda-forge llama.cpp` (CUDA and Vulkan builds available) |

---

## Prebuilt binaries (any OS)

Good when there's no package manager, or you want a specific GPU build.

1. Go to the **Releases** page of github.com/ggml-org/llama.cpp.
2. Download the zip for your machine. The names look like:

   | File name contains | For |
   |---|---|
   | `macos-arm64` | Apple Silicon Mac |
   | `win-cpu-x64` | Windows, no GPU |
   | `win-cuda` | Windows + NVIDIA GPU (also grab the matching `cudart` zip) |
   | `win-vulkan-x64` | Windows + most other GPUs |
   | `ubuntu-x64` | Linux, CPU |
   | `ubuntu-vulkan-x64` | Linux + GPU |

3. Unzip it somewhere permanent, e.g. `~/llama.cpp` or `C:\llama.cpp`.
4. Tell the workshop where `llama-server` is. Either add that folder to your PATH, or set:

   ```bash
   export AGENT_LLAMA_SERVER=~/llama.cpp/llama-server              # macOS / Linux
   ```
   ```powershell
   setx AGENT_LLAMA_SERVER "C:\llama.cpp\llama-server.exe"          # Windows, then open a new terminal
   ```

macOS may block an unzipped binary the first time. Run `xattr -dr com.apple.quarantine ~/llama.cpp`.

---

## Build from source (any OS)

Needs git, cmake and a C++ compiler (Xcode command line tools on macOS, `build-essential` on Linux,
Visual Studio Build Tools on Windows).

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build                       # add -DGGML_CUDA=ON for NVIDIA, -DGGML_VULKAN=ON for Vulkan
cmake --build build --config Release -j
```

`llama-server` ends up in `build/bin/`. Point the workshop at it:

```bash
export AGENT_LLAMA_SERVER=$PWD/build/bin/llama-server
```

---

## Check it worked

In a **new** terminal:

```bash
uv --version
llama-server --version
```

Both should print a version number. (Used `AGENT_LLAMA_SERVER`? Run that path with `--version` instead.)

Then in the workshop folder:

```bash
uv sync
uv run jupyter lab
```

The first cell of `1_setup.ipynb` checks both again and tells you exactly what's missing.

---

## When it doesn't work

| You see | Try |
|---|---|
| `uv: command not found` right after installing | Open a new terminal. Still missing? Run `source $HOME/.local/bin/env` (macOS/Linux). |
| `llama-server: command not found` | New terminal. Installed from a zip or source? Add it to PATH or set `AGENT_LLAMA_SERVER`. |
| `brew: command not found` | Install Homebrew (macOS section), then follow the lines it prints at the end to add it to your PATH. |
| Windows: `winget` not recognised | Update "App Installer" from the Microsoft Store, or use the PowerShell installer / zip. |
| Windows: script blocked by execution policy | Use the exact `powershell -ExecutionPolicy ByPass -c ...` line above. |
| macOS: "cannot be opened because the developer cannot be verified" | `xattr -dr com.apple.quarantine <folder>` on the unzipped llama.cpp folder. |
| Jupyter says it can't find llama-server, but the terminal can | Jupyter was started before you installed it. Stop it (ctrl-c) and run `uv run jupyter lab` again. |
| Behind a corporate proxy | Set `HTTPS_PROXY` before `uv sync`, or install at home. |

Updating later: `uv self update` (or `brew upgrade uv llama.cpp` / `winget upgrade llama.cpp`).
