#!/bin/bash
set -e

echo "1. Preparing Dataset..."
python3 scripts/prepare_dataset.py --dataset imdb --out data/

echo "2. Training Manual SVM (Hinge)..."
python3 scripts/manual_svm.py --loss hinge --epochs 20 --save models/manual_hinge.joblib

echo "3. Training Manual SVM (Squared Hinge)..."
python3 scripts/manual_svm.py --loss squared_hinge --epochs 20 --save models/manual_squared_hinge.joblib

echo "4. Training Manual SVM (Logistic)..."
python3 scripts/manual_svm.py --loss logistic --epochs 20 --save models/manual_logistic.joblib

echo "5. Training Sklearn Baseline..."
python3 scripts/sklearn_svm.py --save models/sklearn_linear_svc.joblib

echo "6. Evaluating and Comparing..."
python3 scripts/evaluate_compare.py

echo "7. Plotting Losses..."
python3 scripts/plot_losses.py

echo "Done!"
