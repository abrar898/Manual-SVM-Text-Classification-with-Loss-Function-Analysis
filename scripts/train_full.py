import os
import sys
import tarfile
import urllib.request
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time

# Set random seed
np.random.seed(42)

# Configuration
DATA_DIR = "data"
MODELS_DIR = "models"
REPORT_DIR = "report"
IMDB_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# --- 1. Data Preparation ---
def download_and_extract_imdb(data_dir):
    imdb_dir = os.path.join(data_dir, "aclImdb")
    if os.path.exists(imdb_dir):
        print(f"Dataset already exists at {imdb_dir}")
        return imdb_dir

    tar_path = os.path.join(data_dir, "aclImdb_v1.tar.gz")
    if not os.path.exists(tar_path):
        print(f"Downloading IMDb dataset from {IMDB_URL}...")
        urllib.request.urlretrieve(IMDB_URL, tar_path)
        print("Download complete.")

    print("Extracting dataset...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print("Extraction complete.")
    return imdb_dir

def load_imdb_data(imdb_dir):
    print("Loading training data...")
    train_data = load_files(os.path.join(imdb_dir, "train"), categories=["pos", "neg"], encoding="utf-8", shuffle=True, random_state=42)
    print("Loading test data...")
    test_data = load_files(os.path.join(imdb_dir, "test"), categories=["pos", "neg"], encoding="utf-8", shuffle=True, random_state=42)
    
    X = train_data.data + test_data.data
    y = np.concatenate([train_data.target, test_data.target])
    # Map 0 -> -1 (neg), 1 -> +1 (pos)
    y = np.where(y == 0, -1, 1)
    return X, y

def clean_text(text):
    # Remove HTML tags
    text = re.sub(r'<br />', ' ', text)
    return text

print("--- Step 1: Data Preparation ---")
imdb_dir = download_and_extract_imdb(DATA_DIR)
X_text_raw, y = load_imdb_data(imdb_dir)

print("Cleaning text...")
X_text = [clean_text(text) for text in X_text_raw]
print(f"Total samples: {len(X_text)}")

# Optimization:
# 1. ngram_range=(1, 2) for better context (e.g. "not good")
# 2. min_df=5 to remove rare noise and reduce feature space size (speeds up training)
# 3. sublinear_tf=True usually helps
# 4. REMOVED stop_words="english" because "not", "no", "never" are critical for sentiment!
print("Vectorizing data (TfidfVectorizer, ngram_range=(1, 2), min_df=5, no stopwords)...")
vectorizer = TfidfVectorizer(
    ngram_range=(1, 2), 
    min_df=5,
    max_features=20000, # Cap features to prevent memory explosion
    sublinear_tf=True
)
X = vectorizer.fit_transform(X_text)
print(f"Feature matrix shape: {X.shape}")

print("Splitting data 80/20...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

joblib.dump(vectorizer, os.path.join(DATA_DIR, "vectorizer.joblib"))

# --- 2. Manual SVM Implementation (Optimized) ---
class ManualSVM:
    def __init__(self, loss='hinge', learning_rate=0.001, lambda_param=0.0001, epochs=10, batch_size=256):
        self.loss_type = loss
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.batch_size = batch_size
        self.w = None
        self.b = 0
        self.history = {'loss': [], 'accuracy': []}

    def _init_weights(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0
        # Adam parameters
        self.m_w = np.zeros(n_features)
        self.v_w = np.zeros(n_features)
        self.m_b = 0
        self.v_b = 0
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0

    def _compute_loss(self, X, y):
        # Vectorized loss computation
        scores = X.dot(self.w) + self.b
        
        if self.loss_type == 'hinge':
            losses = np.maximum(0, 1 - y * scores)
            data_loss = np.mean(losses)
        elif self.loss_type == 'squared_hinge':
            losses = np.maximum(0, 1 - y * scores) ** 2
            data_loss = np.mean(losses)
        elif self.loss_type == 'logistic':
            z = -y * scores
            data_loss = np.mean(np.logaddexp(0, z))
        else:
            raise ValueError(f"Unknown loss: {self.loss_type}")
            
        reg_loss = self.lambda_param * np.sum(self.w ** 2)
        return data_loss + reg_loss

    def _compute_gradients(self, X_batch, y_batch):
        n_samples = X_batch.shape[0]
        scores = X_batch.dot(self.w) + self.b
        margins = y_batch * scores
        
        dw = np.zeros_like(self.w)
        db = 0
        
        if self.loss_type == 'hinge':
            mask = (1 - margins) > 0
            if np.any(mask):
                X_active = X_batch[mask]
                y_active = y_batch[mask]
                # Efficient sparse dot product
                dw_data = -X_active.T.dot(y_active) / n_samples
                db_data = -np.sum(y_active) / n_samples
                dw += dw_data
                db += db_data
                
        elif self.loss_type == 'squared_hinge':
            mask = (1 - margins) > 0
            if np.any(mask):
                X_active = X_batch[mask]
                y_active = y_batch[mask]
                scores_active = scores[mask]
                
                factors = 2 * (1 - y_active * scores_active)
                grad_scalars = -factors * y_active
                
                dw_data = X_active.T.dot(grad_scalars) / n_samples
                db_data = np.sum(grad_scalars) / n_samples
                dw += dw_data
                db += db_data
                
        elif self.loss_type == 'logistic':
            z = margins
            p = np.zeros_like(z)
            pos_mask = z >= 0
            neg_mask = ~pos_mask
            p[pos_mask] = 1 / (1 + np.exp(-z[pos_mask]))
            p[neg_mask] = np.exp(z[neg_mask]) / (1 + np.exp(z[neg_mask]))
            
            grad_scalars = (p - 1) * y_batch
            dw_data = X_batch.T.dot(grad_scalars) / n_samples
            db_data = np.sum(grad_scalars) / n_samples
            dw += dw_data
            db += db_data
            
        dw += 2 * self.lambda_param * self.w
        return dw, db

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._init_weights(n_features)
        
        print(f"Training on {n_samples} samples, {n_features} features...")
        
        for epoch in range(self.epochs):
            start_time = time.time()
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_idx = indices[start_idx:end_idx]
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                
                dw, db = self._compute_gradients(X_batch, y_batch)
                
                # Adam Update
                self.t += 1
                self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dw
                self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * db
                self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dw ** 2)
                self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (db ** 2)
                
                m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
                m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
                v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
                v_b_hat = self.v_b / (1 - self.beta2 ** self.t)
                
                self.w -= self.lr * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                self.b -= self.lr * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
            
            # Compute loss/acc less frequently if needed, but for 20 epochs it's fine
            loss = self._compute_loss(X, y)
            acc = self.score(X, y)
            self.history['loss'].append(loss)
            self.history['accuracy'].append(acc)
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {loss:.4f} - Acc: {acc:.4f} - Time: {epoch_time:.2f}s")

    def predict(self, X):
        scores = X.dot(self.w) + self.b
        return np.where(scores >= 0, 1, -1)

    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

print("\n--- Step 2: Training Manual Models ---")
losses = ['hinge', 'squared_hinge', 'logistic']
manual_models = {}

for loss in losses:
    print(f"\nTraining Manual SVM with {loss} loss...")
    # Increased batch size to 256 for speed
    model = ManualSVM(loss=loss, epochs=10, learning_rate=0.001, batch_size=256)
    model.fit(X_train, y_train)
    manual_models[loss] = model
    joblib.dump(model, os.path.join(MODELS_DIR, f"manual_{loss}.joblib"))

# --- 3. Sklearn Baseline ---
print("\n--- Step 3: Training Sklearn Baseline ---")
print("Training Sklearn LinearSVC...")
# Tuned C=0.1 often works better for text with many features (regularization)
sklearn_model = LinearSVC(C=0.1, max_iter=2000, random_state=42)
sklearn_model.fit(X_train, y_train)
joblib.dump(sklearn_model, os.path.join(MODELS_DIR, "sklearn_linear_svc.joblib"))

# --- 4. Evaluation ---
print("\n--- Step 4: Evaluation ---")
def evaluate(model, X, y, name):
    preds = model.predict(X)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y, preds),
        "Precision": precision_score(y, preds, pos_label=1),
        "Recall": recall_score(y, preds, pos_label=1),
        "F1": f1_score(y, preds, pos_label=1)
    }

results = []
for loss, model in manual_models.items():
    results.append(evaluate(model, X_test, y_test, f"Manual {loss.replace('_', ' ').title()}"))

results.append(evaluate(sklearn_model, X_test, y_test, "Sklearn LinearSVC"))

df_results = pd.DataFrame(results)
print(df_results)
df_results.to_csv(os.path.join(REPORT_DIR, "comparison_table.csv"), index=False)

# --- 5. Plotting ---
print("\n--- Step 5: Plotting ---")
plt.figure(figsize=(10, 6))
for loss, model in manual_models.items():
    plt.plot(model.history['loss'], label=f"Manual {loss}")

plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Comparison")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(REPORT_DIR, "loss_comparison.png"))
print("Done.")
