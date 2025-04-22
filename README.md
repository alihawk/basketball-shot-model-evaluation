# Basketball Shot Model Evaluation

This repository contains the code and report for **HW1: Model Evaluation**  
(UL FRI Data Science, 2024–2025) by Muhammad Ali.

## 📂 Project Structure

- `part1_2.py`  
  Combined code for:
  1. Comparing four classifiers (Baseline, Logistic Regression, kNN train‑fold, kNN nested CV)  
     via 5‑fold CV, reporting log‑score and accuracy (with ±std).  
  2. Analyzing log‑loss error vs. shot distance and reweighting by true competition frequencies.
- `dataset.csv`  
  Basketball shot data (features + `ShotType`).
- `*.svg`  
  Generated plots:  
  - `overall_performance.svg`  
  - `distance_error_correlation.svg`  
  - `distance_bin_error.svg`  
  - `competition_performance.svg`
- `report/`  
  LaTeX source and compiled PDF of the write‑up.

## 📈 Results Highlights

- **Logistic Regression** was the top performer (lowest log‑score & highest accuracy).  
- **kNN** benefited from nested CV tuning but still trailed logistic regression.  
- There’s a moderate negative correlation (r = −0.46) between **Distance** and **log‑loss error**.  
- Reweighting competition types (NBA 0.6, others 0.1) shifts the overall performance metrics.

## 🔧 How to run

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   # or source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
