# Presentation Outline: SVM Text Classifier

## Slide 1: Title Slide
- **Project**: Cloud-Deployed Text Classifier (SVM: Manual vs Library)
- **Owner**: Muhammad Abrar Ahmad
- **Context**: Machine Learning Project

## Slide 2: Problem Statement
- **Goal**: Classify movie reviews as Positive or Negative.
- **Approach**: Binary Classification using Support Vector Machines (SVM).
- **Key Comparison**: Manual implementation (understanding the math) vs Scikit-learn (production baseline).

## Slide 3: Dataset & Preprocessing
- **Data**: IMDb Movie Reviews (50k samples).
- **Features**: TF-IDF Vectorization (5000 features).
- **Split**: 80% Train / 20% Test.

## Slide 4: Manual SVM Implementation
- **Objective**: Implement SVM from scratch using SGD.
- **Loss Functions Explored**:
  1. Hinge Loss (Standard SVM)
  2. Squared Hinge Loss (Smoother penalty)
  3. Logistic Loss (Approximation to Logistic Regression)
- **Gradients**: Computed manually for weight updates.

## Slide 5: Experimental Results
- **Loss Curves**: (Show plot of loss convergence)
- **Metrics**: Accuracy, Precision, Recall, F1.
- **Comparison**: How close is the manual implementation to sklearn?

## Slide 6: Deployment & Cloud
- **API**: FastAPI (`/predict` endpoint).
- **Environment**: GitHub Codespaces (DevContainer).
- **Reproducibility**: `requirements.txt` and automated scripts.

## Slide 7: Conclusion
- Summary of findings.
- Future work (Hyperparameter tuning, more features, deep learning).
