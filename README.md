# Cloud-Deployed Text Classification System with Manual SVM and Loss Function Analysis

## Project Overview
In this project, we built a binary text-classification system using:
1. **Manual SVM implementation** (no ML libraries for training)
2. **Three loss functions**:
    - Hinge Loss
    - Squared Hinge Loss
    - Logistic Loss
3. **Library-based SVM** (scikit-learn) for comparison
4. **Cloud deployment** of the best model using GitHub Codespaces.

This project covers end-to-end ML development, experimentation, and cloud deployment.

---

## Project Tasks

### Task 1 — Dataset Preparation
We used the **IMDb Movie Reviews dataset** for binary sentiment classification (Positive/Negative).
- **Source**: [Stanford AI Lab](http://ai.stanford.edu/~amaas/data/sentiment/)
- **Preprocessing**: 
    - HTML tag removal
    - TF-IDF Vectorization (Bigrams, 20,000 features)
    - 80/20 Train/Test Split

### Task 2 — Implement Manual SVM from Scratch
We implemented a linear SVM class `ManualSVM` in Python without using scikit-learn for the optimization loop.
- **Key Components**:
    - Weight vector `w` and bias `b` initialized using Xavier/Glorot initialization.
    - **Score computation**: `score = w • x + b`
    - **Optimization**: Adam Optimizer for faster convergence.
    - **Loss Functions**:
        - **Hinge Loss**: `max(0, 1 - y*score)`
        - **Squared Hinge Loss**: `max(0, 1 - y*score)^2`
        - **Logistic Loss**: `log(1 + exp(-y*score))`

### Task 3 — Compare the Loss Functions
We compared the three loss functions based on training stability and final accuracy.
- **Squared Hinge Loss** converged the fastest and achieved the highest accuracy among manual models (~90%).
- **Hinge Loss** was robust but slightly slower to converge.
- **Logistic Loss** provided a smooth loss curve but had slightly lower accuracy in this specific setting.

### Task 4 — Implement Library-Based SVM
We trained a `LinearSVC` from `scikit-learn` to serve as a baseline.
- **Comparison**: The library model achieved ~90.6% accuracy, very similar to our best manual model (Squared Hinge).
- **Conclusion**: Our manual implementation with the Adam optimizer effectively replicates the performance of optimized libraries.

---

## Cloud Component

### Task 5 — Choose and Set Up a Cloud Environment
We selected **GitHub Codespaces** for its seamless integration with the repository.
- **Setup**:
    - Created a Codespace from the main branch.
    - Installed dependencies via `pip install -r requirements.txt`.
    - Configured port forwarding for API testing on port 8000.

### Task 6 — Train Models on the Cloud
We successfully ran the training scripts in the cloud environment.

**1. Run Data Preparation:**
```bash
python3 scripts/prepare_dataset.py --out data/
```

**2. Run Manual SVM Training:**
```bash
python3 scripts/manual_svm.py --epochs 40 --lambda_param 0.000001 --save best_manual_svm.joblib
```
*Result: Achieved ~90.6% Test Accuracy with low loss.*

**3. Run Library SVM Training:**
```bash
python3 scripts/sklearn_svm.py --save sklearn_svm.joblib
```
*Result: Achieved ~90.2% Test Accuracy.*

### Task 7 — Deploy Your Best Model as an API
We created a **FastAPI** application to serve the model.

**API Features:**
- **Framework**: FastAPI
- **Endpoint**: `/predict` (POST)
- **Input**: `{"text": "Your movie review here"}`
- **Output**: `{"label": "Positive"}` or `{"label": "Negative"}`

**Running the API:**
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

**Testing the API:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was absolutely amazing and I loved it."}'
```
*Response:* `{"label": "Positive", "score": 0.0}`

---

## How to Run This Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/abrar898/Manual-SVM-Text-Classification-with-Loss-Function-Analysis.git
   cd Manual-SVM-Text-Classification-with-Loss-Function-Analysis
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train Models**:
   ```bash
   # Prepare Data
   python3 scripts/prepare_dataset.py --out data/
   
   # Train Manual SVM
   python3 scripts/manual_svm.py --epochs 40 --lambda_param 0.000001 --save best_manual_svm.joblib
   
   # Train Library SVM
   python3 scripts/sklearn_svm.py --save sklearn_svm.joblib
   ```

4. **Run API**:
   ```bash
   uvicorn app.api:app --host 0.0.0.0 --port 8000
   ```
