@echo off
chcp 1251 >nul
setlocal EnableExtensions
title НЕЙРОНОВО - Говорящие головы
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ОШИБКА: окружение .venv не найдено. Запустите install.bat
    pause
    exit /b 1
)
set "PY=.venv\Scripts\python.exe"
set "PYTHONIOENCODING=utf-8"
set "NEURONOVO_DITTO_BASE=%~dp0ditto"

if not exist "ditto\stream_pipeline_offline.py" (
    echo Код движка не найден - скачиваю...
    "%PY%" download_models.py --only code --yes
)
if not exist "ditto\checkpoints\ditto_pytorch" (
    echo Веса не найдены - скачиваю...
    "%PY%" download_models.py --only weights --yes
)

echo Запуск приложения...
"%PY%" app.py
echo.
echo Код выхода: %ERRORLEVEL%
pause
endlocal