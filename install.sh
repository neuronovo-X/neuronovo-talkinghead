#!/usr/bin/env bash
# НЕЙРОНОВО · Говорящие головы — установка (Linux / macOS)
set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo "  НЕЙРОНОВО · Говорящие головы — установка"
echo "============================================================"

# 1) Python 3.10+
if ! command -v python3 >/dev/null 2>&1; then
    echo "ОШИБКА: не найден python3. Установите Python 3.10+."
    exit 1
fi

# 2) ffmpeg
if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ВНИМАНИЕ: ffmpeg не найден. Он нужен для сборки видео."
    echo "  Ubuntu/Debian: sudo apt install ffmpeg"
    echo "  macOS:         brew install ffmpeg"
fi

# 3) Виртуальное окружение
if [ ! -x ".venv/bin/python" ]; then
    echo "Создаю виртуальное окружение .venv ..."
    python3 -m venv .venv
fi
PY=".venv/bin/python"

echo "Обновляю pip ..."
"$PY" -m pip install --upgrade pip wheel

# 4) PyTorch
echo
OS="$(uname -s)"
if [ "$OS" = "Darwin" ]; then
    echo "macOS: ставлю PyTorch (CPU/MPS). Внимание: Ditto рассчитан на CUDA-GPU, на macOS работа крайне медленная/ограниченная."
    "$PY" -m pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1
else
    echo "Устанавливаю PyTorch (CUDA 12.1) ..."
    "$PY" -m pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
fi

# 5) Остальные зависимости
echo
echo "Устанавливаю зависимости из requirements.txt ..."
"$PY" -m pip install -r requirements.txt

# 6) Загрузка движка и весов
echo
echo "Скачиваю движок Ditto и PyTorch-веса (несколько ГБ) ..."
"$PY" download_models.py --yes

echo
echo "============================================================"
echo "  Установка завершена. Запуск: ./start.sh"
echo "============================================================"
