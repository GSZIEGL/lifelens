@echo off
title LifeLens V7.1 Family Analytics

echo ================================================
echo LifeLens V7.1 - Family Analytics inditasa
echo ================================================
echo.

py --version >nul 2>&1
if %errorlevel%==0 (
    py -m pip install -r requirements.txt
    py -m streamlit run lifelens_v7_1_family_analytics.py
    pause
    exit /b
)

python --version >nul 2>&1
if %errorlevel%==0 (
    python -m pip install -r requirements.txt
    python -m streamlit run lifelens_v7_1_family_analytics.py
    pause
    exit /b
)

echo Python nem talalhato.
echo Telepitsd innen: https://www.python.org/downloads/
echo Fontos: telepiteskor pipald be: Add Python to PATH
pause
