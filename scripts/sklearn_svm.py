import argparse
import joblib
import os
import sys
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default="models/sklearn_linear_svc.joblib")
    parser.add_argument("--data_dir", type=str, default="data/")
    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, "train_data.joblib")
    test_path = os.path.join(args.data_dir, "test_data.joblib")
    
    if not os.path.exists(train_path):
        print("Data not found. Run prepare_dataset.py first.")
        sys.exit(1)
        
    print("Loading data...")
    X_train, y_train = joblib.load(train_path)
    X_test, y_test = joblib.load(test_path)
    
    print("Training Sklearn LinearSVC...")
    # LinearSVC minimizes squared hinge loss by default with l2 penalty.
    model = LinearSVC(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc:.4f}")
    
    if args.save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        joblib.dump(model, args.save)
        print(f"Model saved to {args.save}")

if __name__ == "__main__":
    main()
