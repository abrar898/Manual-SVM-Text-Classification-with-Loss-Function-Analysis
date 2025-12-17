# Cloud-Deployed Text Classification System with Manual SVM and Loss Function Analysis

## Project Overview
This project involves building a complete, end-to-end binary text classification system to classify movie reviews as either **Positive** or **Negative**. Unlike standard projects that rely solely on libraries, this project focuses on the **mathematical implementation** of Support Vector Machines (SVM) from scratch. We implement the core optimization algorithms, gradients, and loss functions manually using NumPy to understand the underlying mechanics of machine learning. The project culminates in a robust cloud deployment where the trained model is served as a REST API using FastAPI on GitHub Codespaces, demonstrating a full machine learning lifecycle from algorithm design to production deployment.

---

## Dataset Information

We utilize the **IMDb Large Movie Review Dataset**, a standard benchmark for binary sentiment classification.

- **Source**: [Stanford AI Lab](http://ai.stanford.edu/~amaas/data/sentiment/)
- **Total Samples**: 50,000 reviews.
    - **Training Set**: 40,000 samples (80%)
    - **Test Set**: 10,000 samples (20%)
- **Classes**: Binary (Positive vs. Negative).
- **Features**: 20,000 (TF-IDF Vectors).

### Preprocessing & Feature Extraction
Before training, the raw text data undergoes several processing steps:
1. **HTML Tag Removal**: Cleaning `<br />` tags and other HTML artifacts.
2. **Tokenization**: Splitting text into individual words.
3. **N-grams**: We use **Bigrams** (pairs of consecutive words) in addition to unigrams. This captures context (e.g., "not good" is different from "good").
    - *Example*: "not good" -> ["not", "good", "not good"]
4. **TF-IDF Vectorization**: We convert text into numerical vectors using Term Frequency-Inverse Document Frequency.
    - **Max Features**: Limited to the top **20,000** most frequent n-grams to balance performance and memory.
    - **Stop Words**: We explicitly **kept** stop words (like "not", "no") because they are critical for sentiment analysis.

---

## Task Breakdown & Implementation Details

### Task 1 — Dataset Preparation
**Objective**: Download, clean, and format the data for machine learning.
- We implemented a script `scripts/prepare_dataset.py` that automatically downloads the 80MB dataset.
- It uses `scikit-learn`'s `TfidfVectorizer` to transform raw text into a sparse matrix of size `(50000, 20000)`.
- The data is saved as `.joblib` files (`train_data.joblib`, `test_data.joblib`) for fast loading during training.

### Task 2 — Implement Manual SVM from Scratch
**Objective**: Write a custom SVM class without using `sklearn.svm`.
- **File**: `scripts/models.py`
- **Model Parameters**:
    - **Weights ($w$)**: A vector of size 20,000. Initialized using a random normal distribution with small standard deviation ($0.01$) to ensure stable starting loss.
    - **Bias ($b$)**: A scalar intercept term.
- **Score Calculation**: The linear decision boundary is calculated as:
  $$ f(x) = w \cdot x + b $$
- **Optimization Algorithm**: We implemented the **Adam Optimizer** (Adaptive Moment Estimation) manually.
    - Unlike standard SGD, Adam adapts the learning rate for each parameter, leading to much faster convergence on sparse text data.
    - **Hyperparameters**: Learning Rate ($\alpha = 0.001$), Regularization ($\lambda = 1e-6$), Batch Size ($256$).

### Task 3 — Compare Loss Functions
**Objective**: Implement and analyze three different loss functions to guide the gradient descent.

1. **Hinge Loss**:
   - **Formula**: $L = \max(0, 1 - y \cdot f(x))$
   - **Description**: The standard loss for SVM. It penalizes predictions only if they are on the wrong side of the margin. It is robust to outliers but not differentiable at exactly 1.
   
2. **Squared Hinge Loss**:
   - **Formula**: $L = \max(0, 1 - y \cdot f(x))^2$
   - **Description**: A smoother version of Hinge loss. It penalizes large errors more severely (quadratically). In our experiments, this often converged faster and achieved higher accuracy.

3. **Logistic Loss**:
   - **Formula**: $L = \log(1 + e^{-y \cdot f(x)})$
   - **Description**: Technically makes the model a Logistic Regression classifier. It provides a smooth, probabilistic loss curve that is differentiable everywhere.

**Comparison Results**:
- **Squared Hinge** generally performed best for this dataset, achieving ~90.6% accuracy.
- **Hinge** was close behind but slightly less stable during early epochs.

### Task 4 — Implement Library-Based SVM
**Objective**: Establish a baseline using a production-grade library (`scikit-learn`).
- **File**: `scripts/library_svm.py`
- **Models Implemented**:
    - **Hinge Loss**: `LinearSVC(loss='hinge')`
    - **Squared Hinge Loss**: `LinearSVC(loss='squared_hinge')`
    - **Logistic Loss**: `LogisticRegression()`
- **Performance**: The library models achieved ~90.2% accuracy, serving as a strong benchmark for our manual implementation.
- **Conclusion**: Our manual implementation (90.6%) successfully matched and even slightly exceeded the library baseline.

---

## Cloud Deployment Details

### Task 5 — Cloud Environment Setup
We utilized **GitHub Codespaces** as our cloud development environment.
- **Environment**: Ubuntu Linux container.
- **Dependencies**: Python 3.10, NumPy, Pandas, Scikit-learn, FastAPI, Uvicorn.
- **Setup**: The environment is defined in `requirements.txt` and automatically provisioned.

### Task 6 — Train Models on the Cloud
We executed the training pipeline entirely in the cloud.
- **Training Script**: `scripts/manual_svm.py`
- **Compute**: The training runs on the cloud VM's CPU.
- **Artifacts**: The best performing model is serialized and saved as `best_manual_svm.joblib` (approx. 470KB).

### Task 7 — Deploy Best Model as an API
**Objective**: Serve the trained model so it can be accessed over the web.
- **Framework**: **FastAPI** (a modern, fast web framework for building APIs).
- **Server**: **Uvicorn** (an ASGI web server implementation).
- **Endpoint**: `POST /predict`
    - Accepts a JSON payload: `{"text": "Review content..."}`
    - Returns: `{"label": "Positive"}` or `{"label": "Negative"}`
- **Logic**:
    1. Receives text.
    2. Loads the saved `vectorizer.joblib`.
    3. Transforms text to vectors.
    4. Loads `best_manual_svm.joblib` (or library model).
    5. Computes dot product score.
    6. Returns label based on sign of score.

---

## How to Run This Project

### 1. Installation
Clone the repository and install the required Python packages:
```bash
git clone https://github.com/abrar898/Manual-SVM-Text-Classification-with-Loss-Function-Analysis.git
cd Manual-SVM-Text-Classification-with-Loss-Function-Analysis
pip install -r requirements.txt
```

### 2. Data Preparation
Download and preprocess the IMDb dataset:
```bash
python3 scripts/prepare_dataset.py --out data/
```

### 3. Training
Train the Manual SVM model (this may take 1-2 minutes):
```bash
python3 scripts/manual_svm.py --epochs 40 --lambda_param 0.000001 --save best_manual_svm.joblib
```

Train the Library Baseline (Squared Hinge):
```bash
python3 scripts/library_svm.py --loss squared_hinge --save library_svm.joblib
```
*(You can also use `--loss hinge` or `--loss logistic`)*

### 4. API Deployment
Start the web server:
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Test the API (in a new terminal):
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "This movie was absolutely amazing and I loved it."}'
```
