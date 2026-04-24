from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI(title="Loan Default Risk Assessment API")

model = joblib.load("../../model.pkl")

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= 0.5)

    return {
        "default_probability": round(float(prob), 4),
        "predicted_class": pred,
        "risk_label": "High Risk" if pred == 1 else "Low Risk"
    }