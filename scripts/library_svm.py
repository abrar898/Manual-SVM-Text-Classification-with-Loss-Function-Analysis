import argparse
import joblib
import os
import sys
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, default="squared_hinge", choices=["hinge", "squared_hinge", "logistic"], help="Loss function to use")
    parser.add_argument("--save", type=str, default=None, help="Path to save the trained model")
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
    
    print(f"Training Library Model with {args.loss} loss...")
    
    if args.loss == "hinge":
        # LinearSVC with hinge loss
        # dual=True is required for hinge loss in LinearSVC
        model = LinearSVC(loss='hinge', C=0.1, max_iter=1000, random_state=42, dual=True)
    elif args.loss == "squared_hinge":
        # LinearSVC with squared hinge loss (default)
        model = LinearSVC(loss='squared_hinge', C=0.1, max_iter=1000, random_state=42, dual=False)
    elif args.loss == "logistic":
        # Logistic Regression (equivalent to logistic loss)
        model = LogisticRegression(C=0.1, max_iter=1000, random_state=42, solver='liblinear')
    
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print("-" * 60)
    print(f"{'Model':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 60)
    print(f"{f'Library {args.loss}':<25} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")
    print("-" * 60)
    
    if args.save:
        if os.path.dirname(args.save):
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
        joblib.dump(model, args.save)
        print(f"Model saved to {args.save}")
    else:
        # Default save behavior if not specified
        default_save = f"models/library_{args.loss}.joblib"
        os.makedirs(os.path.dirname(default_save), exist_ok=True)
        joblib.dump(model, default_save)
        print(f"Model saved to {default_save}")

if __name__ == "__main__":
    main()
