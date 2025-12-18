# Viva Preparation Guide: Cloud-Based SVM Sentiment Analysis

**Project Title:** Manual SVM Text Classification with Loss Function Analysis & Cloud Deployment  
**Student Name:** Muhammad Abrar Ahmad  
**Date:** December 17, 2024

---

## 1. Project Overview (The "Elevator Pitch")

**What is this project?**
This is an end-to-end Machine Learning project where I built a system to classify movie reviews as **Positive** or **Negative**.

**What makes it special?**
Instead of just using a library like `scikit-learn`, I **built the Support Vector Machine (SVM) algorithm from scratch** using Python and NumPy. I implemented three different "loss functions" to see which one learns best. Finally, I deployed the best model to the cloud (GitHub Codespaces) so anyone can use it via an API.

**Key Technologies:**
- **Language:** Python 3.10
- **Math Library:** NumPy (for manual SVM)
- **ML Library:** Scikit-learn (for comparison/baseline)
- **Web Framework:** FastAPI (for the API)
- **Cloud:** GitHub Codespaces

---

## 2. Understanding the Core Concepts (For Viva Questions)

### Q: What is an SVM (Support Vector Machine)?
**Simple Answer:** Imagine you have red balls (negative reviews) and blue balls (positive reviews) on a floor. An SVM tries to draw a straight line (or a "hyperplane" in 3D+) that best separates them.
- **Margin:** The gap between the line and the nearest balls. SVM wants this gap to be as wide as possible.
- **Support Vectors:** The specific balls that are closest to the line. They "support" or define where the line goes.

### Q: Why did you implement it manually?
**Answer:** To understand the math behind the magic. Libraries hide the details. By writing the code for gradients and updates myself, I learned exactly how the model "learns" from mistakes.

### Q: What is a "Loss Function"?
**Simple Answer:** It's a "scorecard" for how bad the model's mistakes are.
- If the model predicts "Positive" for a "Negative" review, the loss function gives it a high penalty score.
- The goal of training is to make this total penalty score as low as possible.

---

## 3. The Three Loss Functions (The Core Experiment)

I implemented three different ways to calculate this penalty:

### 1. Hinge Loss (The Standard SVM)
- **Concept:** "If you are on the correct side of the line and far enough away (margin > 1), zero penalty. If you are on the wrong side or too close, penalty increases linearly."
- **Analogy:** A strict teacher. If you pass, you get 0 punishment. If you fail, you get punished based on how badly you failed.
- **My Result:** Good accuracy (88.68%), but not the best.

### 2. Squared Hinge Loss (The Winner 🏆)
- **Concept:** Same as Hinge, but the penalty is **squared**.
- **Why it's different:** It punishes *big* mistakes much more severely than small ones. (e.g., a mistake of size 2 becomes penalty 4).
- **Benefit:** It makes the math "smoother" (differentiable everywhere), which helps the optimizer find the best solution faster.
- **My Result:** **Best Accuracy (90.34%)**.

### 3. Logistic Loss (The Probabilistic One)
- **Concept:** Instead of a hard line, it calculates the *probability* of being positive.
- **Analogy:** "I'm 80% sure this is positive."
- **My Result:** 87.65%. It was okay, but for this specific manual implementation, the Squared Hinge worked better.

---

## 4. How I Built It (Step-by-Step)

### Step 1: Data Preparation (`prepare_dataset.py`)
- **Dataset:** IMDb Movie Reviews (50,000 reviews).
- **Cleaning:** Removed HTML tags (`<br />`), lowercased text.
- **Vectorization (TF-IDF):** Computers can't read words. I converted words into numbers.
  - Used **Bigrams**: "not good" is treated as one unit, which is crucial for sentiment.
  - **Features:** Kept top 10,000 most frequent words/phrases (optimized for Cloud memory).

### Step 2: Manual Training (`manual_svm.py`)
- **The Math:** `f(x) = w · x + b` (Equation of a line).
- **The Optimizer (Adam):** I didn't use simple Gradient Descent. I used **Adam**, which is a smart optimizer that adapts the speed of learning. It speeds up when it's confident and slows down to be precise.
- **Training Loop:**
  1. Pick a batch of reviews.
  2. Make predictions.
  3. Calculate Loss (Penalty).
  4. Calculate Gradients (Direction to move).
  5. Update weights (`w`) and bias (`b`).
  6. Repeat for 40 epochs.

### Step 3: Library Comparison (`library_svm.py`)
- I trained the same models using `scikit-learn` to have a benchmark.
- **Surprise Finding:** My manual **Squared Hinge** model (90.34%) actually beat the library version (90.19%)! This proves my implementation and optimization (Adam) were very effective.

---

## 5. Cloud Deployment (The "MLOps" Part)

### Q: How is it deployed?
**Answer:** I used **GitHub Codespaces**. It's a cloud development environment (like a computer in the browser).

### The API (`app/api.py`)
- I used **FastAPI** to create a web server.
- **Endpoint:** `/predict`
- **Input:** JSON text (e.g., `{"text": "I loved this movie"}`)
- **Output:** JSON label (e.g., `{"label": "Positive"}`)

### Why Codespaces?
- **Zero Setup:** It comes with Python pre-installed.
- **Port Forwarding:** It automatically gives me a public URL (HTTPS) to test my API from anywhere.
- **Memory Challenge:** Codespaces only has 4GB RAM. I had to optimize my code (reduce features from 20k to 10k) to make it fit. This shows I can handle real-world resource constraints.

---

## 6. Key Results & Comparison

| Model | Accuracy | Verdict |
|-------|----------|---------|
| **Manual Squared Hinge** | **90.34%** | **WINNER 🏆 (Deployed)** |
| Library Squared Hinge | 90.19% | Very close 2nd |
| Library Hinge | 88.89% | Good |
| Manual Hinge | 88.68% | Good |
| Manual Logistic | 87.65% | Decent |
| Library Logistic | 86.58% | Lowest |

**Conclusion:**
My manual implementation of Squared Hinge Loss with the Adam optimizer was the **best performing model overall**, proving that understanding the math allows you to build high-performance systems.

---

## 7. Potential Viva Questions & Answers

**Q: Why did you choose TF-IDF instead of just counting words?**
**A:** TF-IDF (Term Frequency-Inverse Document Frequency) is smarter. It lowers the weight of common words like "the" or "is" that appear everywhere, and highlights unique, meaningful words like "amazing" or "terrible".

**Q: Why did you use Bigrams?**
**A:** Because "not good" means the opposite of "good". If I only used single words (unigrams), the model would see "good" and think it's positive, ignoring the "not". Bigrams capture "not good" as a single feature.

**Q: What is the 'Adam' optimizer?**
**A:** It stands for Adaptive Moment Estimation. It's an algorithm that updates the weights. Unlike standard Gradient Descent which uses a fixed step size, Adam changes the step size for each feature individually. It helps the model learn faster and converge better.

**Q: How do you handle overfitting?**
**A:**
1. **Regularization (L2):** I added a penalty term (`lambda * w^2`) to the loss function. This keeps the weights small and simple.
2. **Dataset Split:** I kept 20% of data (10,000 reviews) completely separate for testing. The 90.34% accuracy is on this unseen data, proving it generalizes well.

**Q: What was the biggest challenge?**
**A:** Getting the manual gradients right for the Squared Hinge loss. Also, running out of memory in the Cloud environment required me to optimize the feature size (reducing from 20k to 10k features).

---

## 8. Summary for the Teacher

"Sir/Ma'am, in this project, I didn't just use a library. I built the SVM mathematical engine from scratch to understand how it works. I implemented three different loss functions and found that the **Squared Hinge Loss** combined with the **Adam Optimizer** gave the best results (90.34%), even beating the standard library implementation. Finally, I deployed this model as a live API on the cloud using FastAPI and GitHub Codespaces, handling real-world constraints like memory optimization."
