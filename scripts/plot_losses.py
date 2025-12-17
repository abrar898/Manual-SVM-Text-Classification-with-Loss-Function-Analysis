import joblib
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Add scripts to path to load ManualSVM class
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import ManualSVM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="models/")
    parser.add_argument("--out", type=str, default="report/loss_comparison.png")
    args = parser.parse_args()

    models = {
        "Hinge": "manual_hinge.joblib",
        "Squared Hinge": "manual_squared_hinge.joblib",
        "Logistic": "manual_logistic.joblib"
    }
    
    plt.figure(figsize=(10, 6))
    
    found_any = False
    for name, filename in models.items():
        path = os.path.join(args.models_dir, filename)
        if os.path.exists(path):
            print(f"Loading {name} from {path}...")
            try:
                model = joblib.load(path)
                if hasattr(model, 'history') and 'loss' in model.history:
                    plt.plot(model.history['loss'], label=name)
                    found_any = True
                else:
                    print(f"Model {name} does not have history attribute.")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        else:
            print(f"Model {name} not found.")
            
    if found_any:
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Comparison")
        plt.legend()
        plt.grid(True)
        
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        plt.savefig(args.out)
        print(f"Plot saved to {args.out}")
    else:
        print("No models with history found to plot.")

if __name__ == "__main__":
    main()
