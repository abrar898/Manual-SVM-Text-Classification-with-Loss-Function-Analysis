import argparse
import os
import sys
import tarfile
import urllib.request
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files

# URL for the IMDb dataset
IMDB_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

def download_and_extract_imdb(data_dir):
    """Downloads and extracts the IMDb dataset if not already present."""
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
    """Loads the IMDb dataset using sklearn load_files."""
    # We only want 'pos' and 'neg' folders. load_files works on subfolders as categories.
    # The aclImdb structure is: train/pos, train/neg, test/pos, test/neg.
    # We will load train and test separately and combine them, or just load one if we want to do our own split.
    # The prompt says "train/test split (80/20)". The original dataset has 25k train and 25k test.
    # Let's load both and combine, then split 80/20 as requested, or just use the provided split if it matches.
    # However, to strictly follow "train/test split (80/20)", I'll combine and resplit to ensure the ratio is exactly what the user asked for, 
    # although 50/50 is standard for IMDb. Let's stick to the user's request of 80/20 split.
    
    print("Loading training data...")
    train_data = load_files(os.path.join(imdb_dir, "train"), categories=["pos", "neg"], encoding="utf-8", shuffle=True, random_state=42)
    print("Loading test data...")
    test_data = load_files(os.path.join(imdb_dir, "test"), categories=["pos", "neg"], encoding="utf-8", shuffle=True, random_state=42)
    
    X = train_data.data + test_data.data
    y = np.concatenate([train_data.target, test_data.target])
    
    # Map 0 to -1 (neg) and 1 to +1 (pos) for SVM
    # sklearn load_files maps categories alphabetically. neg -> 0, pos -> 1.
    # We want -1, +1.
    # 0 -> -1
    # 1 -> 1
    y = np.where(y == 0, -1, 1)
    
    return X, y

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset for SVM training")
    parser.add_argument("--dataset", type=str, default="imdb", help="Dataset name (currently only imdb)")
    parser.add_argument("--out", type=str, default="data/", help="Output directory for processed data")
    args = parser.parse_args()

    if args.dataset != "imdb":
        print("Only IMDb dataset is supported currently.")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    # 1. Download and Extract
    imdb_dir = download_and_extract_imdb(args.out)

    # 2. Load Data
    print("Loading data into memory...")
    X_text, y = load_imdb_data(imdb_dir)
    print(f"Total samples: {len(X_text)}")

    # 3. Vectorize
    print("Vectorizing data (TfidfVectorizer)...")
    # Optimization for better accuracy:
    # 1. ngram_range=(1, 2) includes bigrams (e.g., "not good") which are crucial for sentiment.
    # 2. Removed stop_words="english" because it strips words like "not", "no", "never".
    # 3. max_features=10000 balances accuracy and memory usage for cloud environments.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    X = vectorizer.fit_transform(X_text)

    # 4. Split 80/20
    print("Splitting data 80/20...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # 5. Save
    print("Saving processed data and vectorizer...")
    joblib.dump((X_train, y_train), os.path.join(args.out, "train_data.joblib"))
    joblib.dump((X_test, y_test), os.path.join(args.out, "test_data.joblib"))
    joblib.dump(vectorizer, os.path.join(args.out, "vectorizer.joblib"))
    
    # Create a README in data/
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write("# Dataset Info\n\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Source: {IMDB_URL}\n")
        f.write(f"Preprocessing: TfidfVectorizer (max_features=5000, stop_words='english')\n")
        f.write(f"Split: 80% Train, 20% Test\n")
        f.write(f"Train samples: {X_train.shape[0]}\n")
        f.write(f"Test samples: {X_test.shape[0]}\n")
        f.write(f"Labels: -1 (Negative), +1 (Positive)\n")

    print("Done.")

if __name__ == "__main__":
    main()
