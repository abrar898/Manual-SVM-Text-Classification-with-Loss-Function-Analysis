import argparse
import joblib
import os
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add scripts directory to path to import ManualSVM
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import ManualSVM

def evaluate_model(model, X, y, name):
    preds = model.predict(X)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y, preds),
        "Precision": precision_score(y, preds, pos_label=1),
        "Recall": recall_score(y, preds, pos_label=1),
        "F1": f1_score(y, preds, pos_label=1)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--models_dir", type=str, default="models/")
    parser.add_argument("--out", type=str, default="report/comparison_table.csv")
    args = parser.parse_args()

    test_path = os.path.join(args.data_dir, "test_data.joblib")
    if not os.path.exists(test_path):
        print("Data not found.")
        sys.exit(1)
        
    print("Loading test data...")
    X_test, y_test = joblib.load(test_path)
    
    results = []
    
    # List of expected models
    model_files = {
        "Manual Hinge": "manual_hinge.joblib",
        "Manual Squared Hinge": "manual_squared_hinge.joblib",
        "Manual Logistic": "manual_logistic.joblib",
        "Sklearn LinearSVC": "sklearn_linear_svc.joblib"
    }
    
    for name, filename in model_files.items():
        path = os.path.join(args.models_dir, filename)
        if os.path.exists(path):
            print(f"Evaluating {name}...")
            try:
                model = joblib.load(path)
                metrics = evaluate_model(model, X_test, y_test, name)
                results.append(metrics)
            except Exception as e:
                print(f"Failed to load/evaluate {name}: {e}")
        else:
            print(f"Model {name} not found at {path}")

    if results:
        df = pd.DataFrame(results)
        print("\nComparison Table:")
        print(df)
        
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nSaved comparison table to {args.out}")
    else:
        print("No models evaluated.")

if __name__ == "__main__":
    main()
