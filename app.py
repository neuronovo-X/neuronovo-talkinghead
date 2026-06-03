# -*- coding: utf-8 -*-
"""
НЕЙРОНОВО · Говорящие головы (Talking Heads)

Самостоятельное Gradio-приложение: аудио + фото аватара -> видео.
Под капотом — движок Ditto (antgroup/ditto-talkinghead), PyTorch-вариант (без TensorRT).

Инференс запускается отдельным процессом через ditto/inference_neuronovo.py,
чтобы освобождать VRAM после каждой генерации.

Сайт: https://нейроново.рф
Репозиторий: https://github.com/neuronovo-X/neuronovo-talkinghead
Движок Ditto: https://github.com/antgroup/ditto-talkinghead (Apache-2.0)
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import gradio as gr

BASE_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = BASE_DIR / "settings.json"

NEURONOVO_SITE_URL = "https://нейроново.рф"
NEURONOVO_REPO_URL = "https://github.com/neuronovo-X/neuronovo-talkinghead"


# ──────────────────────────── Настройки ────────────────────────────

def _default_settings() -> dict:
    """Параметры StreamSDK.setup для Ditto (см. stream_pipeline_offline setup kwargs)."""
    return {
        "mask_ratio_w": 0.9,
        "mask_ratio_h": 0.9,
        "crop_scale": 2.3,
        "crop_vx_ratio": 0.0,
        "crop_vy_ratio": -0.125,
        "crop_flag_do_rot": True,
        "mask_template_path": "",
    }


def _load_settings() -> dict:
    d = _default_settings()
    if SETTINGS_PATH.exists():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if isinstance(saved, dict):
                for k in d:
                    if k in saved:
                        d[k] = saved[k]
        except Exception:
            pass
    return d


def _save_settings(**patch) -> None:
    d = _load_settings()
    for k, v in patch.items():
        if k in d:
            d[k] = v
    try:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"  Ошибка сохранения настроек: {exc}")


# ──────────────────────── Ditto: пути и запуск ──────────────────────

def _ditto_base_dir() -> Path:
    raw = (os.environ.get("NEURONOVO_DITTO_BASE") or "").strip()
    if raw:
        return Path(raw)
    return BASE_DIR / "ditto"


def _ditto_inference_script(base: Path) -> Path:
    nn = base / "inference_neuronovo.py"
    return nn if nn.is_file() else base / "inference.py"


def _ditto_resolve_data_and_cfg(base: Path) -> tuple[Path, Path]:
    """PyTorch-вариант предпочтителен; TensorRT — если кто-то положил движки вручную."""
    dr_raw = (os.environ.get("NEURONOVO_DITTO_DATA_ROOT") or "").strip()
    cp_raw = (os.environ.get("NEURONOVO_DITTO_CFG_PKL") or "").strip()
    ckpt = base / "checkpoints"

    if dr_raw:
        data_root = Path(dr_raw)
    else:
        data_root = None
        # Сначала PyTorch (этот репозиторий качает именно его).
        pt = ckpt / "ditto_pytorch"
        if pt.is_dir():
            data_root = pt
        if data_root is None:
            for cand_name in ("ditto_trt_Ampere_Plus", "ditto_trt_3090"):
                c = ckpt / cand_name
                if c.is_dir():
                    data_root = c
                    break
        if data_root is None:
            data_root = ckpt / "ditto_pytorch"

    if cp_raw:
        cfg_pkl = Path(cp_raw)
    else:
        cfg_dir = ckpt / "ditto_cfg"
        if data_root.name == "ditto_pytorch":
            cfg_pkl = cfg_dir / "v0.4_hubert_cfg_pytorch.pkl"
        else:
            cfg_pkl = cfg_dir / "v0.4_hubert_cfg_trt.pkl"
    return data_root, cfg_pkl


def _ditto_python_exe() -> str:
    """Python окружения движка. По умолчанию — текущий интерпретатор (одно окружение)."""
    env_p = (os.environ.get("NEURONOVO_DITTO_PYTHON") or "").strip()
    if env_p:
        p = Path(env_p)
        if p.is_file():
            return str(p)
    return sys.executable


def _ditto_path_prepend(py_exe: str) -> str:
    """PATH для опционального TensorRT/cuDNN — только если переменные заданы и существуют."""
    chunks: list[str] = []
    for key in ("NEURONOVO_DITTO_CUDNN_BIN", "NEURONOVO_DITTO_TRT_LIB", "NEURONOVO_DITTO_TRT_BIN"):
        v = (os.environ.get(key) or "").strip()
        if v and Path(v).exists():
            chunks.append(v)
    env_root = Path(py_exe).resolve().parent
    for p in (env_root, env_root / "Library" / "bin", env_root / "Scripts"):
        if p.exists():
            chunks.append(str(p))
    seen: set[str] = set()
    out: list[str] = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return os.pathsep.join(out)


def run_talking_head(
    audio_file,
    image_file,
    mask_ratio_w,
    mask_ratio_h,
    crop_scale,
    crop_vx_ratio,
    crop_vy_ratio,
    crop_flag_do_rot,
    mask_template_path_ui,
):
    """Субпроцесс: inference_neuronovo.py из репозитория Ditto."""
    if not audio_file:
        raise gr.Error("Не загружено аудио.")
    if not image_file:
        raise gr.Error("Не загружено изображение.")

    py = _ditto_python_exe()

    base = _ditto_base_dir()
    infer = _ditto_inference_script(base)
    if not infer.is_file():
        raise gr.Error(
            f"Не найден скрипт инференса Ditto ({infer.name}) в каталоге: {base}. "
            "Запустите install.bat / install.sh или python download_models.py."
        )

    data_root, cfg_pkl = _ditto_resolve_data_and_cfg(base)
    if not data_root.exists():
        raise gr.Error(
            f"Не найден каталог моделей Ditto (data_root): {data_root}. "
            "Запустите загрузку моделей: python download_models.py."
        )
    if not cfg_pkl.is_file():
        raise gr.Error(
            f"Не найден файл конфигурации cfg_pkl: {cfg_pkl}. "
            "Запустите загрузку моделей: python download_models.py."
        )

    tmp_root = base / "tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    job_id = uuid.uuid4().hex[:8]
    ap_in = Path(str(audio_file))
    ip_in = Path(str(image_file))
    audio_ext = ap_in.suffix or ".wav"
    image_ext = ip_in.suffix or ".png"

    audio_path = tmp_root / f"audio_{job_id}{audio_ext}"
    image_path = tmp_root / f"source_{job_id}{image_ext}"
    output_path = tmp_root / f"result_{job_id}.mp4"

    shutil.copy2(str(ap_in), audio_path)
    shutil.copy2(str(ip_in), image_path)

    setup_blob: dict = {
        "mask_ratio_w": float(mask_ratio_w),
        "mask_ratio_h": float(mask_ratio_h),
        "crop_scale": float(crop_scale),
        "crop_vx_ratio": float(crop_vx_ratio),
        "crop_vy_ratio": float(crop_vy_ratio),
        "crop_flag_do_rot": bool(crop_flag_do_rot),
    }
    mtp_raw = (mask_template_path_ui or "").strip()
    if mtp_raw:
        mtp_path = Path(mtp_raw)
        if not mtp_path.is_file():
            raise gr.Error(f"Файл маски не найден: {mtp_path}")
        setup_blob["mask_template_path"] = str(mtp_path.resolve())

    setup_json_path = BASE_DIR / "ditto_setup_runtime.json"
    try:
        with open(setup_json_path, "w", encoding="utf-8") as f:
            json.dump(setup_blob, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        raise gr.Error(f"Не удалось записать параметры Ditto: {exc}") from exc

    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    prepend = _ditto_path_prepend(py)
    if prepend:
        env["PATH"] = prepend + os.pathsep + env.get("PATH", "")

    cmd = [
        py,
        str(infer),
        "--data_root", str(data_root),
        "--cfg_pkl", str(cfg_pkl),
        "--audio_path", str(audio_path),
        "--source_path", str(image_path),
        "--output_path", str(output_path),
        *(
            ["--setup_json", str(setup_json_path)]
            if infer.name == "inference_neuronovo.py"
            else []
        ),
    ]

    result = subprocess.run(
        cmd,
        cwd=str(base),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    log_text = (
        f"$ {' '.join(cmd)}\n\n"
        f"--- stdout ---\n{result.stdout or '(пусто)'}\n"
        f"--- stderr ---\n{result.stderr or '(пусто)'}\n"
        f"--- exit code: {result.returncode} ---"
    )

    if result.returncode != 0 or not output_path.is_file():
        # возвращаем лог чтобы пользователь видел ошибку, затем бросаем gr.Error
        # gr.Error показывает всплывающее уведомление, лог остаётся видимым
        return None, log_text

    return str(output_path), log_text


# ──────────────────────────── Интерфейс ────────────────────────────

CSS = """
.gradio-container {
    background: linear-gradient(160deg, #06000f 0%, #10052a 50%, #0a0220 100%) !important;
    min-width: 1060px !important;
    max-width: 1100px !important;
    margin: 0 auto !important;
}

.app-hdr { text-align:center; padding:30px 0 14px; }
.app-hdr h1 { margin:0; line-height:1.15; }
.app-hdr .app-hdr-title-link {
    display:inline-block;
    background: linear-gradient(135deg, #c084fc, #a855f7, #7c3aed);
    -webkit-background-clip:text; background-clip:text;
    -webkit-text-fill-color:transparent;
    color:transparent;
    font-size:1.3em !important; font-weight:800 !important;
    letter-spacing:0.05em;
    text-decoration:none !important;
    cursor:pointer;
    transition: filter 0.15s ease, transform 0.15s ease;
}
.app-hdr .app-hdr-title-link:hover { filter: brightness(1.12) saturate(1.05); transform: translateY(-1px); }
.app-hdr .app-hdr-title-link:focus-visible {
    outline: 2px solid rgba(196, 181, 253, 0.65);
    outline-offset: 4px;
    border-radius: 4px;
}
.app-hdr p { color:#8b7aaf !important; font-size:0.95em; margin-top:6px; }

.nn-cross-promo {
    box-sizing: border-box;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-wrap: wrap;
    gap: 10px;
    text-align: center;
    padding: 10px 14px;
    border-radius: 10px;
    font-size: 0.88em;
    line-height: 1.5;
    color: #c9bcfc;
    background: linear-gradient(125deg, rgba(52, 28, 92, 0.42), rgba(14, 10, 36, 0.72));
    border: 1px solid rgba(139, 92, 246, 0.26);
    box-shadow: 0 1px 0 rgba(255,255,255,0.04) inset;
}
.nn-cross-promo-glyph {
    flex: 0 0 auto;
    opacity: 0.72;
    font-size: 1.05em;
    color: #e9d5ff;
}
.nn-cross-promo-body { flex: 0 1 auto; min-width: 0; text-align: center; }
.nn-cross-promo a {
    color: #f5ebff;
    font-weight: 600;
    text-decoration: none;
    border-bottom: 1px solid rgba(245, 235, 255, 0.28);
    transition: border-color 0.15s ease, color 0.15s ease;
}
.nn-cross-promo a:hover {
    color: #fff;
    border-bottom-color: rgba(255, 255, 255, 0.55);
}
.nn-cross-promo--ditto { margin: 6px 0 18px; }

.voice-btn {
    font-size:1.2em !important;
    padding:15px 0 !important;
    border-radius:12px !important;
    letter-spacing:0.03em;
    margin-top:4px !important;
}

.ditto-hdr { text-align:center; }
.ditto-hdr h2 {
    background: linear-gradient(135deg, #c084fc, #a855f7, #7c3aed);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    font-size:1.85em !important; font-weight:800 !important;
    margin: 0.2em 0 0.35em;
    letter-spacing:0.03em;
}
.ditto-hdr p { color:#8b7aaf !important; font-size:0.95em; margin: 0 0 0.5em; }
.ditto-hdr code { color:#c4b5fd; font-size:0.9em; }

footer { display:none !important; }
"""

theme = gr.themes.Base(
    primary_hue=gr.themes.colors.purple,
    secondary_hue=gr.themes.colors.violet,
    neutral_hue=gr.themes.colors.slate,
    # Без Google Fonts: иначе при блокировке fonts.googleapis.com интерфейс зависает на «Загрузка…».
    font=("Segoe UI", "system-ui", "sans-serif"),
)


def build_app() -> gr.Blocks:
    s = _load_settings()
    with gr.Blocks(title="НЕЙРОНОВО · Говорящие головы") as app:
        gr.HTML(
            '<div class="app-hdr">'
            "<h1>"
            f'<a class="app-hdr-title-link" href="{NEURONOVO_SITE_URL}" '
            'target="_blank" rel="noopener noreferrer">НЕЙРОНОВО</a>'
            "</h1>"
            "</div>"
        )

        gr.Markdown(
            '<div class="ditto-hdr">'
            "<h2>Говорящие головы</h2>"
            "<p>Загрузите аудио и изображение аватара — получите видео</p>"
            "</div>"
        )
        gr.HTML(
            '<div class="nn-cross-promo nn-cross-promo--ditto">'
            '<span class="nn-cross-promo-glyph" aria-hidden="true">✦</span>'
            '<span class="nn-cross-promo-body">Создай фото и голос для Аватара в '
            f'<a href="{NEURONOVO_SITE_URL}" target="_blank" rel="noopener noreferrer">'
            "НЕЙРОНОВО.РФ"
            "</a>"
            " · Изображения · Видео · Аудио"
            "</span></div>"
        )

        with gr.Row():
            ditto_audio = gr.Audio(label="Аудио (wav, mp3, …)", type="filepath")
            ditto_image = gr.Image(
                label="Фото аватара (JPG, PNG)",
                type="filepath",
                image_mode="RGB",
            )

        ditto_btn = gr.Button("Сгенерировать", variant="primary", elem_classes=["voice-btn"])
        ditto_video = gr.Video(label="Результат")
        with gr.Accordion("Лог", open=False):
            ditto_logs = gr.Textbox(
                label="Вывод процесса",
                lines=20,
                max_lines=40,
            )

        with gr.Accordion("Настройки", open=False):
            gr.Markdown(
                "**Кадр лица** — `crop_scale` (больше — шире область вокруг лица), смещения и поворот. "
                "**Растушёвка** — `mask_ratio_w` / `mask_ratio_h` (центральная зона маски; меньше — шире мягкая кромка). "
                "Необязательный **файл маски** заменяет встроенную маску (RGB, как в PutBack)."
            )
            ditto_mask_rw = gr.Slider(
                label="mask_ratio_w — ширина «твёрдой» зоны маски",
                minimum=0.5, maximum=0.99, step=0.01,
                value=float(s["mask_ratio_w"]),
            )
            ditto_mask_rh = gr.Slider(
                label="mask_ratio_h — высота «твёрдой» зоны маски",
                minimum=0.5, maximum=0.99, step=0.01,
                value=float(s["mask_ratio_h"]),
            )
            ditto_crop_scale = gr.Slider(
                label="crop_scale — масштаб кропа лица",
                minimum=1.4, maximum=3.5, step=0.05,
                value=float(s["crop_scale"]),
            )
            ditto_crop_vx = gr.Slider(
                label="crop_vx_ratio — сдвиг кропа по горизонтали",
                minimum=-0.35, maximum=0.35, step=0.01,
                value=float(s["crop_vx_ratio"]),
            )
            ditto_crop_vy = gr.Slider(
                label="crop_vy_ratio — сдвиг кропа по вертикали",
                minimum=-0.35, maximum=0.15, step=0.01,
                value=float(s["crop_vy_ratio"]),
            )
            ditto_crop_rot = gr.Checkbox(
                label="crop_flag_do_rot — выравнивать поворот головы",
                value=bool(s["crop_flag_do_rot"]),
            )
            ditto_mask_file = gr.Textbox(
                label="Путь к файлу маски (пусто — встроенная)",
                value=str(s.get("mask_template_path") or ""),
                max_lines=1,
                placeholder=r"например D:\masks\soft_mask.png",
            )
            ditto_reset_btn = gr.Button("Сбросить по умолчанию", variant="secondary", size="sm")

        ditto_btn.click(
            run_talking_head,
            inputs=[
                ditto_audio, ditto_image,
                ditto_mask_rw, ditto_mask_rh,
                ditto_crop_scale, ditto_crop_vx, ditto_crop_vy,
                ditto_crop_rot, ditto_mask_file,
            ],
            outputs=[ditto_video, ditto_logs],
            concurrency_limit=1,
            trigger_mode="always_last",
        )

        ditto_mask_rw.change(lambda v: _save_settings(mask_ratio_w=float(v)), inputs=[ditto_mask_rw])
        ditto_mask_rh.change(lambda v: _save_settings(mask_ratio_h=float(v)), inputs=[ditto_mask_rh])
        ditto_crop_scale.change(lambda v: _save_settings(crop_scale=float(v)), inputs=[ditto_crop_scale])
        ditto_crop_vx.change(lambda v: _save_settings(crop_vx_ratio=float(v)), inputs=[ditto_crop_vx])
        ditto_crop_vy.change(lambda v: _save_settings(crop_vy_ratio=float(v)), inputs=[ditto_crop_vy])
        ditto_crop_rot.change(lambda v: _save_settings(crop_flag_do_rot=bool(v)), inputs=[ditto_crop_rot])

        def _save_mask_path(val: str):
            _save_settings(mask_template_path=(val or "").strip())

        ditto_mask_file.change(_save_mask_path, inputs=[ditto_mask_file])
        ditto_mask_file.blur(_save_mask_path, inputs=[ditto_mask_file])

        def _reset_ui():
            d = _default_settings()
            for k, v in d.items():
                _save_settings(**{k: v})
            return (
                gr.update(value=float(d["mask_ratio_w"])),
                gr.update(value=float(d["mask_ratio_h"])),
                gr.update(value=float(d["crop_scale"])),
                gr.update(value=float(d["crop_vx_ratio"])),
                gr.update(value=float(d["crop_vy_ratio"])),
                gr.update(value=bool(d["crop_flag_do_rot"])),
                gr.update(value=str(d.get("mask_template_path") or "")),
            )

        ditto_reset_btn.click(
            _reset_ui,
            outputs=[
                ditto_mask_rw, ditto_mask_rh, ditto_crop_scale,
                ditto_crop_vx, ditto_crop_vy, ditto_crop_rot, ditto_mask_file,
            ],
        )

    return app


if __name__ == "__main__":
    _app = build_app()
    _allowed = [str(_ditto_base_dir() / "tmp")]
    _favicon_svg = (
        "data:image/svg+xml,"
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
        "<text y='.9em' font-size='88'>🗣️</text></svg>"
    )
    _head = f'<link rel="icon" href="{_favicon_svg}">'
    _app.queue().launch(
        inbrowser=True,
        server_name=os.environ.get("NEURONOVO_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("NEURONOVO_PORT", "0")) or None,
        theme=theme,
        css=CSS,
        head=_head,
        allowed_paths=_allowed,
    )
