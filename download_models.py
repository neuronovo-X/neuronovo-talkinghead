#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
НЕЙРОНОВО · Говорящие головы — загрузка движка и весов.

Делает две вещи:
  1) Кладёт в ./ditto код antgroup/ditto-talkinghead (git clone или ZIP),
     сохраняя поверх наш inference_neuronovo.py (overlay).
  2) Качает PyTorch-веса с Hugging Face digital-avatar/ditto-talkinghead
     в ./ditto/checkpoints/ (только ditto_pytorch/* и ditto_cfg/* — без тяжёлых TensorRT-движков).

Примеры:
  python download_models.py            # код + PyTorch-веса (с паузой и текстом о лицензиях)
  python download_models.py --yes      # без паузы (для install.bat / install.sh)
  python download_models.py --only code --yes   # только код движка, без весов
  python download_models.py --force-code --yes  # обновить код движка, даже если он уже есть

Код Ditto — Apache-2.0 (https://github.com/antgroup/ditto-talkinghead).
Веса — карточка репозитория https://huggingface.co/digital-avatar/ditto-talkinghead.
Ответственность за соблюдение лицензий лежит на пользователе.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

BASE = Path(__file__).resolve().parent

# Локальный кэш Hugging Face — рядом со скриптом, чтобы не засорять профиль пользователя.
_hf = BASE / "hf_cache"
_hf.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_hf))
os.environ.setdefault("HF_HUB_CACHE", str(_hf / "hub"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(_hf / "hub"))

DITTO_HF_REPO = "digital-avatar/ditto-talkinghead"
DITTO_GITHUB_GIT = "https://github.com/antgroup/ditto-talkinghead.git"
DITTO_GITHUB_ZIP = "https://github.com/antgroup/ditto-talkinghead/archive/refs/heads/main.zip"

# PyTorch-вариант: качаем только нужные подкаталоги (TensorRT-движки пропускаем).
DITTO_ALLOW_PATTERNS = ["ditto_pytorch/**", "ditto_cfg/**"]

# Наши файлы, которые сохраняем поверх дистрибутива antgroup после слияния.
_NEURONOVO_OVERLAY: tuple[str, ...] = ("inference_neuronovo.py",)

# Обрывы IncompleteRead на больших файлах — повтор и меньше воркеров.
_SNAPSHOT_MAX_ATTEMPTS = 12
_SNAPSHOT_MAX_WORKERS = 2


def _section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  {title}\n{bar}")


def _is_retryable(exc: BaseException) -> bool:
    retry_names = frozenset({
        "ChunkedEncodingError", "IncompleteRead", "ProtocolError", "ConnectionError",
        "ConnectionResetError", "ReadTimeout", "Timeout", "SSLError",
    })
    visited: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        e = stack.pop()
        if id(e) in visited:
            continue
        visited.add(id(e))
        if type(e).__name__ in retry_names:
            return True
        if isinstance(e, (BrokenPipeError, TimeoutError, ConnectionResetError, ConnectionAbortedError)):
            return True
        if isinstance(e, OSError) and getattr(e, "errno", None) in (10054, 104, 110):
            return True
        if e.__cause__ is not None:
            stack.append(e.__cause__)
        ctx = getattr(e, "__context__", None)
        if ctx is not None and ctx is not e.__cause__:
            stack.append(ctx)
    return False


def _backoff_sec(attempt_index: int) -> float:
    return min(180.0, 5.0 * (2 ** attempt_index))


def _download_url_to_file(url: str, dest: Path, *, timeout_sec: int = 900) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Neuronovo-talkinghead/1.0"})
    with urlopen(req, timeout=timeout_sec) as resp:
        dest.write_bytes(resp.read())


def _merge_item(src_item: Path, dest_parent: Path) -> None:
    dest = dest_parent / src_item.name
    if src_item.is_dir():
        dest.mkdir(parents=True, exist_ok=True)
        for ch in sorted(src_item.iterdir(), key=lambda p: p.name.lower()):
            _merge_item(ch, dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_dir():
        shutil.rmtree(dest)
    shutil.copy2(src_item, dest)


def _merge_upstream_root(src_root: Path, ditto_dest: Path) -> None:
    ditto_dest.mkdir(parents=True, exist_ok=True)
    for item in sorted(src_root.iterdir(), key=lambda p: p.name.lower()):
        if item.name in (".git", ".github", "checkpoints"):
            continue
        if item.name in _NEURONOVO_OVERLAY:
            continue
        _merge_item(item, ditto_dest)


def _apply_overlay(ditto_dest: Path) -> None:
    """Кладёт наши файлы (inference_neuronovo.py) поверх ditto/."""
    bundle = BASE / "ditto_overlay"
    for fname in _NEURONOVO_OVERLAY:
        src = bundle / fname
        if not src.is_file():
            print(f"  ВНИМАНИЕ: не найден overlay-файл {src} — пропуск.", file=sys.stderr)
            continue
        dst = ditto_dest / fname
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"  Применён файл Нейроново: ditto/{fname}")


def _git_clone_shallow(repo_url: str, target_dir: Path) -> bool:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
            check=True, capture_output=True, text=True, timeout=600,
        )
        return True
    except FileNotFoundError:
        print("  git не найден в PATH — пробуем скачивание ZIP с GitHub…")
        return False
    except subprocess.CalledProcessError as e:
        snippet = ((e.stderr or "") + (e.stdout or "")).strip()[:400]
        print(f"  git clone завершился с ошибкой, пробуем ZIP…{(chr(10) + snippet) if snippet else ''}")
        return False
    except subprocess.TimeoutExpired:
        print("  git clone слишком долго — пробуем ZIP…")
        return False


def _fetch_via_zip(scratch: Path, dest_clone: Path) -> bool:
    zpath = scratch / "ditto-upstream-main.zip"
    print(f"  Загрузка архива:\n    {DITTO_GITHUB_ZIP}")
    _download_url_to_file(DITTO_GITHUB_ZIP, zpath)
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(scratch)
    candidates = sorted(p for p in scratch.iterdir() if p.is_dir() and p.name.startswith("ditto-talkinghead"))
    root = candidates[0] if len(candidates) == 1 else None
    if root is None or not root.is_dir():
        print("  Не удалось распознать корень после распаковки ZIP.", file=sys.stderr)
        return False
    if dest_clone.exists():
        shutil.rmtree(dest_clone)
    shutil.move(str(root), str(dest_clone))
    return True


def ensure_upstream_code(*, force: bool) -> None:
    ditto_dest = BASE / "ditto"
    marker = ditto_dest / "stream_pipeline_offline.py"
    if marker.is_file() and not force:
        print("  Код Ditto уже на месте (stream_pipeline_offline.py найден); пропуск загрузки кода.")
        _apply_overlay(ditto_dest)
        return

    _section("Ditto: код инференса (GitHub)" + (" — принудительное обновление" if force else ""))
    ditto_dest.mkdir(parents=True, exist_ok=True)
    checkpoints = ditto_dest / "checkpoints"
    had_ckpt = checkpoints.is_dir()

    with tempfile.TemporaryDirectory(prefix="neuronovo_ditto_src_") as tmp:
        tdir = Path(tmp)
        cloned = tdir / "upstream"
        ok = _git_clone_shallow(DITTO_GITHUB_GIT, cloned)
        if not ok:
            shutil.rmtree(cloned, ignore_errors=True)
            cloned = tdir / "upstream_zip"
            if not _fetch_via_zip(tdir, cloned):
                raise RuntimeError(
                    "Не удалось получить код Ditto: установите Git или проверьте доступ к github.com."
                )
        _merge_upstream_root(cloned, ditto_dest)

    _apply_overlay(ditto_dest)
    if had_ckpt:
        print("  Каталог ditto/checkpoints/ сохранён.")
    print("  Код Ditto готов.")


def download_checkpoints() -> None:
    from huggingface_hub import snapshot_download  # отложенный импорт: код движка не требует hf_hub

    _section("Ditto: PyTorch-веса (Hugging Face · digital-avatar)")
    dest = BASE / "ditto" / "checkpoints"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"  repo={DITTO_HF_REPO}")
    print(f"  шаблоны: {', '.join(DITTO_ALLOW_PATTERNS)}")
    print(f"  -> {dest}")
    print("  При обрыве сети загрузка повторяется автоматически; частичные файлы не удаляйте — HF докачает их.")

    last_err: BaseException | None = None
    for attempt in range(_SNAPSHOT_MAX_ATTEMPTS):
        try:
            snapshot_download(
                repo_id=DITTO_HF_REPO,
                local_dir=str(dest),
                local_dir_use_symlinks=False,
                allow_patterns=DITTO_ALLOW_PATTERNS,
                max_workers=_SNAPSHOT_MAX_WORKERS,
            )
            print("  PyTorch-веса Ditto готовы.")
            return
        except KeyboardInterrupt:
            raise
        except BaseException as err:
            last_err = err
            if not _is_retryable(err) or attempt >= _SNAPSHOT_MAX_ATTEMPTS - 1:
                if not _is_retryable(err):
                    raise
                break
            delay = _backoff_sec(attempt)
            print(
                f"\n  Временная ошибка сети ({type(err).__name__}: {err}). "
                f"Повтор через {delay:.0f} с ({attempt + 2}/{_SNAPSHOT_MAX_ATTEMPTS})…",
                file=sys.stderr,
            )
            time.sleep(delay)

    assert last_err is not None
    raise RuntimeError(
        "Не удалось скачать веса Ditto после нескольких попыток. Проверьте сеть/VPN и запустите позже."
    ) from last_err


def _print_preface() -> None:
    sep = "-" * 58
    print("\nНЕЙРОНОВО · Говорящие головы — загрузка кода и весов")
    print(sep)
    print("Будут выполнены задачи:")
    print(f"  • код движка GitHub `antgroup/ditto-talkinghead` → каталог `ditto/` (Apache-2.0)")
    print(f"  • PyTorch-веса HF `{DITTO_HF_REPO}` → `ditto/checkpoints/` (несколько ГБ)")
    print("\nОтказ от ответственности")
    print("Программное обеспечение предоставляется «как есть» (AS IS) без каких-либо гарантий.")
    print(
        "\nВеса и код загружаются со сторонних репозиториев (Hugging Face, GitHub). "
        "Условия использования определяются карточками репозиториев на момент загрузки. "
        "Вы самостоятельно проверяете применимость лицензий, в том числе для коммерческого использования."
    )
    print(
        "\nФото и голос. Создание видео с лицом и голосом реального человека без его согласия "
        "может нарушать законодательство (право на изображение и голос, защита персональных данных). "
        "Ответственность за используемые материалы лежит на пользователе."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Загрузка движка и весов Ditto (Говорящие головы).")
    parser.add_argument("--only", choices=["code", "weights", "all"], default="all",
                        help="code — только код движка; weights — только веса; all — всё (по умолчанию).")
    parser.add_argument("--yes", action="store_true", help="Без паузы Enter (для автоматизации).")
    parser.add_argument("--force-code", action="store_true",
                        help="Обновить код движка с GitHub, даже если он уже есть.")
    args = parser.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass

    if not args.yes:
        _print_preface()
        print("\nНажимая Enter, вы подтверждаете, что ознакомились с лицензиями и принимаете их.")
        input("Нажмите Enter для продолжения или закройте окно / Ctrl+C для отмены...")

    print("\n" + "=" * 72 + "\n  НЕЙРОНОВО · загрузка ресурсов\n" + "=" * 72)

    if args.only in ("code", "all"):
        ensure_upstream_code(force=args.force_code)
    if args.only in ("weights", "all"):
        download_checkpoints()

    print("\nГотово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
