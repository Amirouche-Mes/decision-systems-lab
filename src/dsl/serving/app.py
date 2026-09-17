from contextlib import asynccontextmanager
import logging
import pandas as pd
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import uvicorn

from dsl.serving.serving import load_best_model


logger = logging.getLogger("uvicorn.error")

class Settings(BaseSettings):
    experiment_name: str = "booking_conversion_clf"
    metric: str = "val_auc"
    host: str = "0.0.0.0"
    port: int = 8000
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_prefix = "SERVING_",
        extra="ignore",
    )
settings = Settings()

ml_models: dict[str, Any] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        f"Intialisation of the server | Experiment: {settings.experiment_name}"
    )
    try:
        model, run_id = load_best_model(
            experiment_name=settings.experiment_name,
            metric=settings.metric,
        )
        ml_models["model"] = model 
        ml_models["run_id"] = run_id
        logger.info(f"Model loaded with success using the run id:{run_id}")
    except Exception as e:
        logger.error(f"Failed to load the model: {e}")
        raise RuntimeError("Can't load the model at the start")

    yield

    logger.info("Stopping the server: liberating the resources")
    ml_models.clear()

app = FastAPI(
    title = "booking conversion serving API",
    lifespan=lifespan
)

class PredictionRequest(BaseModel):
    inputs: list[dict[str, Any]]

class PredictionOutput(BaseModel):
    run_id: str
    predictions: list[float]

@app.get("/health")
def health_check():
    """Check if the server returns an actif run_id."""
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    return {
        "status": "ok",
        "run_id": ml_models["run_id"],
        "experiment_name": settings.experiment_name,
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionRequest):
    """receives a json payload, validate, apply the model and return the probabilities"""
    if "model" not in ml_models:
        raise HTTPException(status_code=503, detail="The model is not available for inference.")
    
    try:
        df_input = pd.DataFrame(payload.inputs)
        probabilities = ml_models["model"].predict_proba(df_input)[:, 1].tolist()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"prediction failed: {e}")

    return PredictionOutput(
        run_id=ml_models["run_id"],
        predictions=probabilities,
    )
