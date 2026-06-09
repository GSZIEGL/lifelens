@echo off
title LifeLens V7.1 AI telepites

echo ================================================
echo LifeLens V7.1 - Deep Analysis AI telepites
echo ================================================
echo.

py --version >nul 2>&1
if %errorlevel%==0 (
    py -m pip install --upgrade pip
    py -m pip install torch transformers
    pause
    exit /b
)

python --version >nul 2>&1
if %errorlevel%==0 (
    python -m pip install --upgrade pip
    python -m pip install torch transformers
    pause
    exit /b
)

echo Python nem talalhato.
echo Telepitsd innen: https://www.python.org/downloads/
echo Fontos: telepiteskor pipald be: Add Python to PATH
pause
