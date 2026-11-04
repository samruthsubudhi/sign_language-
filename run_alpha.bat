@echo off
title Sign Language Translator Alpha
cd /d "%~dp0"
echo ===================================================
echo   Launching Sign Language Translator Alpha
echo ===================================================
set "PYTHON_EXE=python"
if exist "%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe" (
    set "PYTHON_EXE=%LOCALAPPDATA%\Python\pythoncore-3.14-64\python.exe"
)
"%PYTHON_EXE%" src\ui_app.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Application exited with code %ERRORLEVEL%.
    pause
)
