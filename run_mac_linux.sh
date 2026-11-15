#!/usr/bin/env bash
# Mars Navigator: first run installs everything (a few minutes), later runs start straight away.
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Creating Python environment..."
  python3 -m venv .venv
  . .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
else
  . .venv/bin/activate
fi
streamlit run app.py
