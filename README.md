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

## 2. Important
- Run the code yourself and use the numbers it actually produces.
- Do not submit the report template with blank placeholders or claim experiments ran if they did not.
- The script is intentionally written in straightforward Python. Read the comments and make sure you can explain the key ideas.
- Runtime can be substantial because Q3 trains many models. If you are close to the deadline, run the full script once and do not repeatedly rerun it.
- The split is saved to `outputs/split_indices.npz`; subsequent runs reuse it so the clean test set stays fixed.
- The central masked test images are made by setting rows 24:40 and columns 24:40 to zero. Models are trained only on clean images.

## 3. Submission checklist
[ ] Run `main.py` successfully.
[ ] Inspect all CSV files and plots.
[ ] Fill `report_template.md` with real results and observations.
[ ] Export the report to a two-column CVPR-style PDF, maximum 8 pages excluding references.
[ ] Include source code, requirements, saved split, outputs, figures, exact run command and environment information in the repository.
[ ] Add your name, student number, repository link and final commit hash to page 1.
[ ] Check the course rules about permitted AI assistance and disclose assistance if required.
