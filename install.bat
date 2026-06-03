@echo off
chcp 1251 >nul
setlocal EnableExtensions
title НЕЙРОНОВО - Установка
cd /d "%~dp0"

echo ============================================================
echo   НЕЙРОНОВО - Говорящие головы - Установка
echo ============================================================
echo.

where python >nul 2>&1
if errorlevel 1 (
    echo ОШИБКА: Python не найден в PATH.
    echo Установите Python 3.10+ с https://www.python.org/downloads/
    pause
    exit /b 1
)

where ffmpeg >nul 2>&1
if errorlevel 1 (
    echo ВНИМАНИЕ: ffmpeg не найден - видео не будет собираться.
    echo Установите: winget install Gyan.FFmpeg
    echo.
)

if not exist ".venv\Scripts\python.exe" (
    echo Создаю виртуальное окружение .venv ...
    python -m venv .venv
    if errorlevel 1 (
        echo ОШИБКА: не удалось создать .venv
        pause
        exit /b 1
    )
)
set "PY=.venv\Scripts\python.exe"

echo Обновляю pip ...
"%PY%" -m pip install --upgrade pip wheel

echo.
echo Устанавливаю PyTorch CUDA 12.1 ...
"%PY%" -m pip install torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 (
    echo ОШИБКА установки PyTorch.
    pause
    exit /b 1
)

echo.
echo Устанавливаю зависимости из requirements.txt ...
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 (
    echo ОШИБКА установки зависимостей.
    echo Если ошибка Cython/pyx - установите Build Tools for Visual Studio (компонент C++).
    pause
    exit /b 1
)

echo.
echo Скачиваю движок Ditto и веса (несколько ГБ) ...
"%PY%" download_models.py --yes
if errorlevel 1 (
    echo ОШИБКА загрузки. Запустите позже:
    echo   .venv\Scripts\python.exe download_models.py --yes
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Установка завершена! Запустите start.bat
echo ============================================================
echo.
pause
endlocal