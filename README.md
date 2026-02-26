# 2026-ncomms-analysis-code
# Code for "A Continuous Intensification Rate Index for Tropical Cyclones Using Vortex-Scale Machine Learning"

This repository contains all custom computer code used to generate the results central to the manuscript submitted to **Nature Communications**.

The code implements three machine learning models — **Random Forest**, **Support Vector Regression (SVR)**, and **Artificial Neural Network (ANN)** — to derive a continuous intensification rate index for tropical cyclones based on vortex-scale features.

## System requirements
- Python 3.13 (developed and tested on Python 3.13)
- Operating system: Linux, macOS, or Windows
- Required packages (see `requirements.txt`):
  - numpy
  - pandas
  - matplotlib
  - scikit-learn
  - (any additional packages listed in requirements.txt)

- Recommended hardware: ≥8 GB RAM (for ANN training)
- Installation time: approximately 2 minutes

## Installation
```bash
# 1. Clone the repository (use the HTTPS link from your private repo)
git clone https://github.com/Seishou0724/2026-ncomms-analysis-code.git
cd 2026-ncomms-analysis-code

# 2. (Recommended) Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
