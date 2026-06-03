<h1 align="center">НЕЙРОНОВО · Говорящие головы</h1>

<p align="center">
  <a href="https://нейроново.рф"><img src="https://img.shields.io/badge/Сайт-НЕЙРОНОВО.РФ-7c3aed"></a>
  <a href="https://github.com/neuronovo-X/neuronovo-talkinghead"><img src="https://img.shields.io/badge/GitHub-neuronovo--talkinghead-181717?logo=github"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue"></a>
  <a href="https://github.com/antgroup/ditto-talkinghead"><img src="https://img.shields.io/badge/Движок-Ditto-purple"></a>
</p>

<p align="center"><b>Русский</b> · <a href="README.en.md">English</a></p>

Простое приложение: загрузите **аудио** и **фото аватара** — получите **видео говорящей головы**.
Удобный интерфейс на русском языке (Gradio) поверх движка [Ditto](https://github.com/antgroup/ditto-talkinghead) от Ant Group (PyTorch-вариант, без TensorRT).

> Часть экосистемы [НЕЙРОНОВО.РФ](https://нейроново.рф) — генерация изображений, видео и аудио с ИИ.

<p align="center">
  <video src="https://github.com/user-attachments/assets/bf39b0a5-652d-4c2d-a2f4-c3b52561f9ee" controls width="80%"></video>
</p>

---

## ⚡ Кратко

1. Установите **Python 3.10+**, **ffmpeg** и драйверы **NVIDIA** (нужен GPU с CUDA).
2. Скачайте этот репозиторий.
3. Запустите **`install.bat`** (Windows) — он создаст окружение, поставит зависимости и скачает модели (несколько ГБ при первом запуске).
4. Запустите **`start.bat`** — браузер откроется на интерфейсе. Загрузите аудио и фото, нажмите **«Сгенерировать»**.

> Linux/macOS — те же шаги через `./install.sh` и `./start.sh` (см. ниже).

<p align="center">
  <img src="https://github.com/user-attachments/assets/6d2dbc0e-f51a-46a8-af00-2db1f244c0fa" width="80%" alt="Пример">
</p>

---

## 🪟 Windows — установка по шагам

> Это основной поддерживаемый сценарий.

### 1. Требования

| Компонент | Версия | Как поставить |
|-----------|--------|----------------|
| Видеокарта | **NVIDIA**, ≥ 6–8 ГБ VRAM, актуальный драйвер | [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx) |
| Python | **3.10 – 3.12** (галочка *Add Python to PATH*) | [python.org/downloads](https://www.python.org/downloads/) |
| ffmpeg | любой свежий | `winget install Gyan.FFmpeg` |
| Build Tools (C++) | для сборки `.pyx` движка | [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/) → «Разработка на C++» |
| Git | желательно (иначе код движка скачается ZIP-ом) | [git-scm.com](https://git-scm.com/download/win) |

### 2. Установка

1. Скачайте репозиторий: зелёная кнопка **Code → Download ZIP** на
   <https://github.com/neuronovo-X/neuronovo-talkinghead>, распакуйте в удобную папку
   (путь желательно без длинных кириллических имён).
2. Двойной клик по **`install.bat`**.
   Скрипт:
   - создаст виртуальное окружение `.venv`;
   - поставит PyTorch (CUDA 12.1) и зависимости;
   - скачает код движка Ditto и **PyTorch-веса** в `ditto/checkpoints/` (несколько ГБ).
3. Дождитесь надписи **«Установка завершена»**.

### 3. Запуск

Двойной клик по **`start.bat`** — приложение откроется в браузере
(`http://127.0.0.1:…`). Готово.

---

## 🐧 Linux

```bash
# зависимости системы (Ubuntu/Debian)
sudo apt update && sudo apt install -y python3 python3-venv ffmpeg git build-essential

# установка проекта
git clone https://github.com/neuronovo-X/neuronovo-talkinghead.git
cd neuronovo-talkinghead
chmod +x install.sh start.sh
./install.sh        # окружение + зависимости + загрузка моделей
./start.sh          # запуск
```

Нужны проприетарные драйверы NVIDIA и CUDA-совместимый GPU.

## 🍎 macOS

Ditto рассчитан на **CUDA-GPU**. На macOS (Apple Silicon / Intel) полноценной CUDA нет —
PyTorch-вариант формально запустится на CPU/MPS, но **очень медленно** и без гарантий.
Рекомендуется только для экспериментов:

```bash
brew install python ffmpeg git
./install.sh && ./start.sh
```

---

## 🎬 Использование

Используйте только собственные изображения и аудиоматериалы
или материалы с разрешения правообладателя.

1. **Аудио** — речь, под которую будут двигаться губы (wav/mp3/…).
2. **Фото** — изображение лица аватара (фронтальное, хорошо освещённое).
3. **«Сгенерировать»** — через несколько секунд/минут появится видео.
4. **Лог** — раскрывается внизу, помогает при ошибках.

> Создай фото и голос для Аватара в [НЕЙРОНОВО.РФ](https://нейроново.рф) · Изображения · Видео · Аудио.

### Настройки (раскрывающийся блок)

| Параметр | Что делает |
|----------|------------|
| `crop_scale` | масштаб кропа лица: больше — шире область вокруг лица |
| `crop_vx_ratio` / `crop_vy_ratio` | сдвиг кропа по горизонтали / вертикали |
| `crop_flag_do_rot` | выравнивать поворот головы |
| `mask_ratio_w` / `mask_ratio_h` | размер «твёрдой» зоны маски (меньше — шире мягкая кромка вклейки) |
| Файл маски | необязательная пользовательская маска (RGB) вместо встроенной |

Настройки сохраняются в `settings.json`. Кнопка **«Сбросить по умолчанию»** возвращает исходные значения.

---

## 📦 Что и куда скачивается

`download_models.py` (вызывается из `install`/`start`) кладёт:

```
ditto/                      ← код движка antgroup/ditto-talkinghead
ditto/inference_neuronovo.py← наша обёртка (overlay, в репозитории — ditto_overlay/)
ditto/checkpoints/ditto_pytorch/   ← PyTorch-веса
ditto/checkpoints/ditto_cfg/       ← конфиги
hf_cache/                   ← кэш Hugging Face
```

Полезные команды:

```bat
.venv\Scripts\python.exe download_models.py --yes              REM код + веса
.venv\Scripts\python.exe download_models.py --only code --yes  REM только код движка
.venv\Scripts\python.exe download_models.py --only weights --yes
.venv\Scripts\python.exe download_models.py --force-code --yes REM обновить код движка
```

---

## ⚙️ Переменные окружения (необязательно)

| Переменная | Назначение |
|------------|------------|
| `NEURONOVO_DITTO_BASE` | каталог движка (по умолчанию `./ditto`) |
| `NEURONOVO_DITTO_DATA_ROOT` | каталог модели (по умолчанию `…/checkpoints/ditto_pytorch`) |
| `NEURONOVO_DITTO_CFG_PKL` | путь к cfg (`…/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl`) |
| `NEURONOVO_DITTO_PYTHON` | другой Python для инференса (по умолчанию — текущий) |
| `NEURONOVO_HOST` / `NEURONOVO_PORT` | адрес и порт интерфейса (по умолчанию `127.0.0.1`, случайный порт) |

Опционально (для ускорения через TensorRT) можно положить TensorRT-движки в `checkpoints/ditto_trt_*`
и указать `NEURONOVO_DITTO_TRT_LIB` / `NEURONOVO_DITTO_TRT_BIN` / `NEURONOVO_DITTO_CUDNN_BIN`.

---

## 🛠️ Частые проблемы

- **`ffmpeg` не найден** → итоговое видео не собирается. Установите ffmpeg и перезапустите.
- **Ошибка сборки `.pyx` / Cython** → поставьте *Build Tools for Visual Studio* (компонент C++).
- **`CUDA out of memory`** → закройте другие GPU-приложения, используйте фото меньшего размера.
- **Обрыв загрузки весов** → запустите `download_models.py --only weights --yes` снова, HF докачает.
- **Долгая генерация** → это PyTorch-вариант; для скорости настройте TensorRT (см. переменные выше).

---

## 📄 Лицензии и благодарности

- Приложение НЕЙРОНОВО · Говорящие головы — **Apache License 2.0** (см. [LICENSE](LICENSE) и [NOTICE](NOTICE)).
- Движок **Ditto** — [antgroup/ditto-talkinghead](https://github.com/antgroup/ditto-talkinghead),
  © Ant Group, Apache-2.0. Веса — [digital-avatar/ditto-talkinghead](https://huggingface.co/digital-avatar/ditto-talkinghead).
- Спасибо авторам Ditto, а также проектам Gradio, PyTorch, MediaPipe, ONNX Runtime.

> Создание видео с лицом и голосом реального человека без его согласия может нарушать
> законодательство. Ответственность за используемые материалы лежит на пользователе.

<p align="center"><a href="https://нейроново.рф">нейроново.рф</a></p>
