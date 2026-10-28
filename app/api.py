from fastapi import FastAPI
from app.schemas import PredictionRequest, PredictionResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
import os

app = FastAPI()

current_dir = os.path.dirname(os.path.abspath(__file__))
ruta_modelos = os.path.join(current_dir, "models.joblib")

models = joblib.load(ruta_modelos)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://cristianclc.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    try:
        if request.model_name not in models:
            return {
                "error": "Modelo no encontrado",
                "available_models": list(models.keys()),
            }

        pipeline = models[request.model_name]

        input_df = pd.DataFrame([request.data.dict()])

        prediction_proba = pipeline.predict_proba(input_df)[0][1]
        prediction = int(prediction_proba > 0.5)

        return PredictionResponse(
            model_used=request.model_name,
            churn_probability=float(prediction_proba),
            prediction=prediction,
        )
    except Exception as e:
        return {"error": f"Ocurrió un error al predecir: {str(e)}"}
