#!/usr/bin/env bash
# НЕЙРОНОВО · Говорящие головы — запуск (Linux / macOS)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
    echo "Окружение .venv не найдено. Сначала запустите ./install.sh"
    exit 1
fi
PY=".venv/bin/python"
export PYTHONIOENCODING=utf-8
export NEURONOVO_DITTO_BASE="$(pwd)/ditto"

if [ ! -f "ditto/stream_pipeline_offline.py" ]; then
    echo "Код движка не найден — скачиваю ..."
    "$PY" download_models.py --yes
fi
if [ ! -d "ditto/checkpoints/ditto_pytorch" ]; then
    echo "PyTorch-веса не найдены — скачиваю ..."
    "$PY" download_models.py --only weights --yes
fi

echo "Запуск НЕЙРОНОВО · Говорящие головы ..."
exec "$PY" app.py
