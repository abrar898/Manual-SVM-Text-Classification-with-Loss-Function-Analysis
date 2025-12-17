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
    # We try to find the artifacts relative to current working directory
    base_dir = os.getcwd()
    
    # Check if we are in 'app' dir or root
    if os.path.exists(os.path.join(base_dir, "data", "vectorizer.joblib")):
        data_dir = os.path.join(base_dir, "data")
        models_dir = os.path.join(base_dir, "models")
    elif os.path.exists(os.path.join(base_dir, "..", "data", "vectorizer.joblib")):
        data_dir = os.path.join(base_dir, "..", "data")
        models_dir = os.path.join(base_dir, "..", "models")
    else:
        # Fallback for Docker/Cloud
        data_dir = "/workspace/data" # Example
        models_dir = "/workspace/models"
        if not os.path.exists(data_dir):
             print("Warning: Artifacts not found.")
             return

    # Load Vectorizer
    vec_path = os.path.join(data_dir, "vectorizer.joblib")
    if os.path.exists(vec_path):
        vectorizer = joblib.load(vec_path)
        print(f"Loaded vectorizer from {vec_path}")
    else:
        print(f"Vectorizer not found at {vec_path}")

    # Load Model - Try to load the best one, or default to manual_hinge
    # Priority: sklearn_linear_svc -> manual_hinge
    model_name = "sklearn_linear_svc.joblib"
    model_path = os.path.join(models_dir, model_name)
    
    if not os.path.exists(model_path):
        model_name = "manual_hinge.joblib"
        model_path = os.path.join(models_dir, model_name)
    
    if os.path.exists(model_path):
        # We need to make sure ManualSVM class is available if loading a manual model
        # If it's a manual model, we might need to import the class.
        # joblib should handle it if the class is in path, but since we are in app/api.py, 
        # scripts/manual_svm.py might not be in path.
        # We'll try to add scripts to path.
        import sys
        if os.path.exists(os.path.join(base_dir, "scripts")):
            sys.path.append(os.path.join(base_dir, "scripts"))
        elif os.path.exists(os.path.join(base_dir, "..", "scripts")):
            sys.path.append(os.path.join(base_dir, "..", "scripts"))
            
        try:
            # If it's manual model, we need the class definition
            # We can try importing it
            try:
                from manual_svm import ManualSVM
            except ImportError:
                pass # Might be sklearn model
                
            model = joblib.load(model_path)
            print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Failed to load model: {e}")
    else:
        print(f"Model not found at {model_path}")

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
