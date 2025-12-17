import joblib
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Text Classifier API", description="Binary Sentiment Analysis using SVM")

# Global variables for model and vectorizer
model = None
vectorizer = None

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    score: float = 0.0 # Manual SVM might not give probability, but we can give raw score

def load_artifacts():
    global model, vectorizer
    # Paths - assume running from root or app directory
    base_dir = os.getcwd()
    
    # 1. Setup paths for data and models
    # Check if we are in root (data/ exists) or in app/ (../data/ exists)
    if os.path.exists(os.path.join(base_dir, "data", "vectorizer.joblib")):
        data_dir = os.path.join(base_dir, "data")
        # Models might be in root or models/
        models_dir = base_dir 
    elif os.path.exists(os.path.join(base_dir, "..", "data", "vectorizer.joblib")):
        data_dir = os.path.join(base_dir, "..", "data")
        models_dir = os.path.join(base_dir, "..")
    else:
        # Fallback
        data_dir = "data"
        models_dir = "."

    # 2. Add scripts/ to sys.path so we can import 'models' module if needed for ManualSVM
    scripts_path = os.path.join(base_dir, "scripts")
    if not os.path.exists(scripts_path):
        scripts_path = os.path.join(base_dir, "..", "scripts")
    
    if os.path.exists(scripts_path):
        import sys
        if scripts_path not in sys.path:
            sys.path.append(scripts_path)
        try:
            # Try importing ManualSVM to ensure class is available for joblib
            from models import ManualSVM
            print("Successfully imported ManualSVM class")
        except ImportError as e:
            print(f"Could not import ManualSVM: {e}")

    # 3. Load Vectorizer
    vec_path = os.path.join(data_dir, "vectorizer.joblib")
    if os.path.exists(vec_path):
        vectorizer = joblib.load(vec_path)
        print(f"Loaded vectorizer from {vec_path}")
    else:
        print(f"Vectorizer not found at {vec_path}")

    # 4. Load Model
    # Priority: best_manual_svm.joblib (root) -> sklearn_svm.joblib (root) -> models/manual_hinge.joblib
    possible_models = [
        os.path.join(models_dir, "best_manual_svm.joblib"),
        os.path.join(models_dir, "sklearn_svm.joblib"),
        os.path.join(models_dir, "models", "manual_hinge.joblib"), # If in models subdir
        os.path.join(models_dir, "models", "sklearn_linear_svc.joblib")
    ]

    for path in possible_models:
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                print(f"Loaded model from {path}")
                break
            except Exception as e:
                print(f"Found model at {path} but failed to load: {e}")
    
    if model is None:
        print("No model loaded!")

@app.on_event("startup")
async def startup_event():
    load_artifacts()

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=503, detail="Model or vectorizer not loaded")
    
    # Vectorize
    features = vectorizer.transform([request.text])
    
    # Predict
    prediction = model.predict(features)[0]
    
    # Map -1/1 to labels
    label = "Positive" if prediction == 1 else "Negative"
    
    return PredictionResponse(label=label)

@app.get("/")
def read_root():
    return {"message": "Text Classifier API is running. Use /predict endpoint."}
