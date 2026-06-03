# -*- coding: utf-8 -*-
"""
Обёртка для Нейроново над inference antgroup/ditto-talkinghead.
Передаёт setup_kwargs из JSON (маска, кроп), как ожидает stream_pipeline_offline.run().
Официальный inference.py репозитория: https://github.com/antgroup/ditto-talkinghead

Особенность Windows: MediaPipe (C-библиотека) не открывает файлы по путям с
кириллицей. Функция _ensure_ascii_data_root() копирует checkpoints во временный
каталог с ASCII-путём и возвращает его — оригинальные файлы не трогаются.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import pickle
import torch

from stream_pipeline_offline import StreamSDK


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pkl(pkl):
    with open(pkl, "rb") as f:
        return pickle.load(f)


def _has_non_ascii(path: str) -> bool:
    try:
        path.encode("ascii")
        return False
    except UnicodeEncodeError:
        return True


def _ensure_ascii_data_root(data_root: str, cfg_pkl: str) -> tuple[str, str]:
    """
    Если data_root или cfg_pkl содержат не-ASCII символы — копируем нужные
    каталоги во временный ASCII-путь и возвращаем новые пути.
    MediaPipe на Windows не умеет открывать файлы по путям с кириллицей.
    """
    if not _has_non_ascii(data_root) and not _has_non_ascii(cfg_pkl):
        return data_root, cfg_pkl

    print("[neuronovo] Путь содержит кириллицу — копирую веса во временный каталог...")
    tmp_dir = Path(tempfile.gettempdir()) / "neuronovo_ditto"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Копируем data_root
    dr_src = Path(data_root)
    dr_dst = tmp_dir / dr_src.name
    if not dr_dst.exists():
        print(f"[neuronovo]   {dr_src} -> {dr_dst}")
        shutil.copytree(str(dr_src), str(dr_dst))
    else:
        print(f"[neuronovo]   {dr_dst} (уже есть)")

    # Копируем ditto_cfg (родительская папка cfg_pkl)
    cfg_src = Path(cfg_pkl)
    cfg_cfg_src = cfg_src.parent
    cfg_cfg_dst = tmp_dir / cfg_cfg_src.name
    if not cfg_cfg_dst.exists():
        print(f"[neuronovo]   {cfg_cfg_src} -> {cfg_cfg_dst}")
        shutil.copytree(str(cfg_cfg_src), str(cfg_cfg_dst))
    else:
        print(f"[neuronovo]   {cfg_cfg_dst} (уже есть)")

    new_data_root = str(dr_dst)
    new_cfg_pkl = str(cfg_cfg_dst / cfg_src.name)
    print(f"[neuronovo] data_root -> {new_data_root}")
    print(f"[neuronovo] cfg_pkl   -> {new_cfg_pkl}")
    return new_data_root, new_cfg_pkl


def run(SDK: StreamSDK, audio_path: str, source_path: str, output_path: str, more_kwargs: str | dict | None = None):
    if more_kwargs is None:
        more_kwargs = {}
    if isinstance(more_kwargs, str):
        more_kwargs = load_pkl(more_kwargs)
    setup_kwargs = more_kwargs.get("setup_kwargs", {})
    run_kwargs = more_kwargs.get("run_kwargs", {})

    SDK.setup(source_path, output_path, **setup_kwargs)

    audio, _sr = librosa.core.load(audio_path, sr=16000)
    num_f = math.ceil(len(audio) / 16000 * 25)

    fade_in = run_kwargs.get("fade_in", -1)
    fade_out = run_kwargs.get("fade_out", -1)
    ctrl_info = run_kwargs.get("ctrl_info", {})
    SDK.setup_Nd(N_d=num_f, fade_in=fade_in, fade_out=fade_out, ctrl_info=ctrl_info)

    online_mode = SDK.online_mode
    if online_mode:
        chunksize = run_kwargs.get("chunksize", (3, 5, 2))
        audio = np.concatenate([np.zeros((chunksize[0] * 640,), dtype=np.float32), audio], 0)
        split_len = int(sum(chunksize) * 0.04 * 16000) + 80
        for i in range(0, len(audio), chunksize[1] * 640):
            audio_chunk = audio[i : i + split_len]
            if len(audio_chunk) < split_len:
                audio_chunk = np.pad(audio_chunk, (0, split_len - len(audio_chunk)), mode="constant")
            SDK.run_chunk(audio_chunk, chunksize)
    else:
        aud_feat = SDK.wav2feat.wav2feat(audio)
        SDK.audio2motion_queue.put(aud_feat)
        SDK.close()

    cmd = (
        f'ffmpeg -loglevel error -y -i "{SDK.tmp_output_path}" -i "{audio_path}" '
        f'-map 0:v -map 1:a -c:v copy -c:a aac "{output_path}"'
    )
    print(cmd)
    os.system(cmd)
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="./checkpoints/ditto_pytorch",
        help="каталог PyTorch / TensorRT модели",
    )
    parser.add_argument(
        "--cfg_pkl",
        type=str,
        default="./checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl",
        help="pickle конфигурации",
    )
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument(
        "--setup_json",
        type=str,
        default="",
        help="JSON с полями для setup_kwargs (маска, кроп)",
    )
    args = parser.parse_args()

    data_root, cfg_pkl = _ensure_ascii_data_root(args.data_root, args.cfg_pkl)

    SDK = StreamSDK(cfg_pkl, data_root)

    more: dict = {"setup_kwargs": {}, "run_kwargs": {}}
    if args.setup_json and str(args.setup_json).strip():
        p = Path(args.setup_json)
        if not p.is_file():
            raise SystemExit(f"setup_json не найден: {p}")
        with open(p, encoding="utf-8") as f:
            blob = json.load(f)
        if isinstance(blob, dict):
            more["setup_kwargs"] = blob

    # OpenCV тоже не открывает файлы по путям с кириллицей на Windows.
    # Копируем входные файлы во временный ASCII-каталог.
    audio_path  = args.audio_path
    source_path = args.source_path
    output_path = args.output_path
    orig_output = args.output_path

    if _has_non_ascii(audio_path) or _has_non_ascii(source_path):
        tmp_io = Path(tempfile.gettempdir()) / "neuronovo_ditto" / "io"
        tmp_io.mkdir(parents=True, exist_ok=True)
        ap = Path(audio_path);  new_audio  = tmp_io / ap.name;  shutil.copy2(str(ap), str(new_audio))
        sp = Path(source_path); new_source = tmp_io / sp.name;  shutil.copy2(str(sp), str(new_source))
        new_output = tmp_io / Path(output_path).name
        audio_path  = str(new_audio)
        source_path = str(new_source)
        output_path = str(new_output)
        print(f"[neuronovo] IO -> {tmp_io}")

    run(SDK, audio_path, source_path, output_path, more)

    # Копируем результат назад в оригинальный путь (где ждёт app.py)
    if output_path != orig_output and Path(output_path).is_file():
        shutil.copy2(output_path, orig_output)
        print(f"[neuronovo] result -> {orig_output}")
