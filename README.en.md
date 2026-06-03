<h1 align="center">NEURONOVO · Talking Heads</h1>

<p align="center">
  <a href="https://нейроново.рф"><img src="https://img.shields.io/badge/Site-NEURONOVO.RF-7c3aed"></a>
  <a href="https://github.com/neuronovo-X/neuronovo-talkinghead"><img src="https://img.shields.io/badge/GitHub-neuronovo--talkinghead-181717?logo=github"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue"></a>
  <a href="https://github.com/antgroup/ditto-talkinghead"><img src="https://img.shields.io/badge/Engine-Ditto-purple"></a>
</p>

<p align="center"><a href="README.md">Русский</a> · <b>English</b></p>

A simple app: drop in an **audio** clip and an **avatar photo** — get a **talking-head video**.
A friendly Gradio UI (Russian) on top of the [Ditto](https://github.com/antgroup/ditto-talkinghead) engine by Ant Group (PyTorch variant, no TensorRT).

> Part of the [NEURONOVO](https://нейроново.рф) toolkit — AI image, video and audio generation.

<p align="center">
  <video src="https://github.com/user-attachments/assets/bf39b0a5-652d-4c2d-a2f4-c3b52561f9ee" controls width="80%"></video>
</p>

---

## ⚡ TL;DR

1. Install **Python 3.10+**, **ffmpeg** and **NVIDIA** drivers (a CUDA GPU is required).
2. Download this repository.
3. Run **`install.bat`** (Windows) — it creates the environment, installs dependencies and downloads the models (several GB on first run).
4. Run **`start.bat`** — the browser opens the UI. Add audio + photo, click **“Generate”**.

> Linux/macOS: same steps via `./install.sh` and `./start.sh` (see below).

<p align="center">
  <img src="https://github.com/user-attachments/assets/6d2dbc0e-f51a-46a8-af00-2db1f244c0fa" width="80%" alt="Example">
</p>

---

## 🪟 Windows — step by step

> This is the primary supported path.

### 1. Requirements

| Component | Version | How to install |
|-----------|---------|----------------|
| GPU | **NVIDIA**, ≥ 6–8 GB VRAM, recent driver | [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx) |
| Python | **3.10 – 3.12** (tick *Add Python to PATH*) | [python.org/downloads](https://www.python.org/downloads/) |
| ffmpeg | any recent | `winget install Gyan.FFmpeg` |
| C++ Build Tools | to compile the engine’s `.pyx` | [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) → “Desktop development with C++” |
| Git | recommended (else the engine code is fetched as ZIP) | [git-scm.com](https://git-scm.com/download/win) |

### 2. Install

1. Get the repo: **Code → Download ZIP** at
   <https://github.com/neuronovo-X/neuronovo-talkinghead>, unzip into a folder
   (prefer a path without long non-ASCII names).
2. Double-click **`install.bat`**. It will:
   - create the `.venv` virtual environment;
   - install PyTorch (CUDA 12.1) and dependencies;
   - download the Ditto engine code and **PyTorch weights** into `ditto/checkpoints/` (several GB).
3. Wait for **“Installation complete”**.

### 3. Run

Double-click **`start.bat`** — the app opens in your browser
(`http://127.0.0.1:…`). Done.

---

## 🐧 Linux

```bash
# system packages (Ubuntu/Debian)
sudo apt update && sudo apt install -y python3 python3-venv ffmpeg git build-essential

# project
git clone https://github.com/neuronovo-X/neuronovo-talkinghead.git
cd neuronovo-talkinghead
chmod +x install.sh start.sh
./install.sh        # env + deps + model download
./start.sh          # run
```

Proprietary NVIDIA drivers and a CUDA-capable GPU are required.

## 🍎 macOS

Ditto targets **CUDA GPUs**. macOS has no full CUDA support — the PyTorch variant will
technically run on CPU/MPS but is **very slow** and unsupported. For experiments only:

```bash
brew install python ffmpeg git
./install.sh && ./start.sh
```

---

## 🎬 Usage

Use only your own images and audio, or material you have the rights holder’s
permission to use.

1. **Audio** — speech that drives the lips (wav/mp3/…).
2. **Photo** — the avatar’s face (frontal, well-lit).
3. **“Generate”** — a video appears in seconds/minutes.
4. **Log** — expandable at the bottom; useful for troubleshooting.

> Create a photo and voice for your Avatar at [NEURONOVO.RF](https://нейроново.рф) · Images · Video · Audio.

### Settings (collapsible)

| Parameter | Effect |
|-----------|--------|
| `crop_scale` | face crop scale: larger = more area around the face |
| `crop_vx_ratio` / `crop_vy_ratio` | crop shift horizontally / vertically |
| `crop_flag_do_rot` | align head rotation |
| `mask_ratio_w` / `mask_ratio_h` | size of the “hard” mask zone (smaller = wider soft blend edge) |
| Mask file | optional custom RGB mask instead of the built-in one |

Settings are stored in `settings.json`. **“Reset to defaults”** restores the originals.

---

## 📦 What gets downloaded

`download_models.py` (called by `install`/`start`) creates:

```
ditto/                       ← antgroup/ditto-talkinghead engine code
ditto/inference_neuronovo.py ← our wrapper (overlay; lives in ditto_overlay/)
ditto/checkpoints/ditto_pytorch/  ← PyTorch weights
ditto/checkpoints/ditto_cfg/      ← configs
hf_cache/                    ← Hugging Face cache
```

Handy commands:

```bat
.venv\Scripts\python.exe download_models.py --yes               REM code + weights
.venv\Scripts\python.exe download_models.py --only code --yes   REM engine code only
.venv\Scripts\python.exe download_models.py --only weights --yes
.venv\Scripts\python.exe download_models.py --force-code --yes  REM refresh engine code
```

---

## ⚙️ Environment variables (optional)

| Variable | Purpose |
|----------|---------|
| `NEURONOVO_DITTO_BASE` | engine directory (default `./ditto`) |
| `NEURONOVO_DITTO_DATA_ROOT` | model directory (default `…/checkpoints/ditto_pytorch`) |
| `NEURONOVO_DITTO_CFG_PKL` | cfg path (`…/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl`) |
| `NEURONOVO_DITTO_PYTHON` | alternate Python for inference (default: current) |
| `NEURONOVO_HOST` / `NEURONOVO_PORT` | UI host and port (default `127.0.0.1`, random port) |

For TensorRT acceleration, place engines in `checkpoints/ditto_trt_*` and set
`NEURONOVO_DITTO_TRT_LIB` / `NEURONOVO_DITTO_TRT_BIN` / `NEURONOVO_DITTO_CUDNN_BIN`.

---

## 🛠️ Troubleshooting

- **`ffmpeg` not found** → the final video isn’t muxed. Install ffmpeg and retry.
- **`.pyx` / Cython build error** → install *Build Tools for Visual Studio* (C++ component).
- **`CUDA out of memory`** → close other GPU apps, use a smaller photo.
- **Weight download interrupted** → run `download_models.py --only weights --yes` again; HF resumes.
- **Slow generation** → this is the PyTorch variant; set up TensorRT for speed (see vars above).

---

## 📄 License & credits

- The NEURONOVO · Talking Heads app — **Apache License 2.0** (see [LICENSE](LICENSE) and [NOTICE](NOTICE)).
- The **Ditto** engine — [antgroup/ditto-talkinghead](https://github.com/antgroup/ditto-talkinghead),
  © Ant Group, Apache-2.0. Weights — [digital-avatar/ditto-talkinghead](https://huggingface.co/digital-avatar/ditto-talkinghead).
- Thanks to the Ditto authors and to the Gradio, PyTorch, MediaPipe and ONNX Runtime projects.

> Creating videos with a real person’s face and voice without their consent may violate the law.
> You are responsible for the material you use.

<p align="center"><a href="https://нейроново.рф">нейроново.рф</a></p>
