# Face Recognition Assignment 2 — reproducible starter

## 1. Setup
Use Python 3.10+ if possible.

Windows:
    py -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt
    python main.py

macOS/Linux:
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    python -m pip install -r requirements.txt
    python main.py

The first run downloads the Olivetti faces dataset and may take a little time.
The script writes CSV files to `outputs/` and figures to `figures/`.

