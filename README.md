# Cloud-Deployed Text Classifier (SVM: Manual vs Library)

**Owner**: Muhammad Abrar Ahmad

## Overview
This project implements a binary text classifier using Support Vector Machines (SVM).
It compares a **manual implementation** of SVM (using hinge, squared hinge, and logistic losses) against a **scikit-learn** baseline.
The best model is deployed as a FastAPI endpoint, and the entire development environment is configured for GitHub Codespaces.

## Structure
- `data/`: Datasets and vectorizers.
- `scripts/`: Python scripts for training, evaluation, and preprocessing.
- `app/`: FastAPI application.
- `notebooks/`: Jupyter notebooks for analysis and plotting.
- `models/`: Saved models.
- `report/`: Final report and artifacts.
- `.devcontainer/`: Configuration for GitHub Codespaces.

## Quickstart

### Local Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Data Preparation
```bash
python scripts/prepare_dataset.py --dataset imdb --out data/
```

### Training (Manual SVM)
```bash
python scripts/manual_svm.py --loss hinge --epochs 20 --lr 0.01 --save models/manual_hinge.joblib
```

### Training (Sklearn Baseline)
```bash
python scripts/sklearn_svm.py --save models/sklearn_linear_svc.joblib
```

### API
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```
