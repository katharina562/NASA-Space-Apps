@echo off
REM Mars Navigator: first run installs everything (a few minutes), later runs start straight away.
cd /d "%~dp0"
if not exist .venv (
    echo Creating Python environment...
    python -m venv .venv || (echo Python 3.10+ is needed: https://www.python.org/downloads/ & pause & exit /b 1)
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip
    pip install -r requirements.txt || (pause & exit /b 1)
) else (
    call .venv\Scripts\activate.bat
)
streamlit run app.py
pause
