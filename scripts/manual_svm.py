import argparse
import joblib
import os
import sys

# Add current directory to path to find models.py if running as script
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import ManualSVM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss", type=str, default="hinge", choices=["hinge", "squared_hinge", "logistic"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data/")
    args = parser.parse_args()

    # Load data
    train_path = os.path.join(args.data_dir, "train_data.joblib")
    test_path = os.path.join(args.data_dir, "test_data.joblib")
    
    if not os.path.exists(train_path):
        print("Data not found. Run prepare_dataset.py first.")
        sys.exit(1)
        
    print("Loading data...")
    X_train, y_train = joblib.load(train_path)
    X_test, y_test = joblib.load(test_path)
    
    print(f"Training Manual SVM with {args.loss} loss...")
    model = ManualSVM(loss=args.loss, epochs=args.epochs, learning_rate=args.lr, batch_size=args.batch)
    model.fit(X_train, y_train)
    
    test_acc = model.score(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    
    if args.save:
        if os.path.dirname(args.save):
            os.makedirs(os.path.dirname(args.save), exist_ok=True)
        joblib.dump(model, args.save)
        print(f"Model saved to {args.save}")

if __name__ == "__main__":
    main()
