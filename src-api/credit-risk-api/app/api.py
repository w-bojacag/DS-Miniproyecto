import json
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from loguru import logger
from model import __version__ as model_version
from model.predict import make_prediction

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()

# Ruta para verificar que la API se esté ejecutando correctamente
@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=model_version
    )

    return health.dict()

# Ruta para realizar las predicciones
@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: dict) -> Any:
    """
    Prediccion usando el modelo de riesgo crediticio
    """

    logger.info(f"Making prediction on inputs: {input_data}")

    # TODO: Integrar con modelo real de predicción
    results = {
        "probabilidad_incumplimiento": 0.36,
        "umbral_aplicado": 0.40,
        "clasificacion": "Riesgo Alto",
        "explicacion": {
            "unidad_aporte": "log_odds",
            "variables": [
                {
                    "variable": "moras_90",
                    "nombre": "Moras mayores a 90 días",
                    "valor_original": int(input_data.get("moras_90", 0)),
                    "aporte": 0.82,
                    "direccion": "aumenta"
                },
                {
                    "variable": "utilizacion",
                    "nombre": "Utilización del crédito",
                    "valor_original": round(input_data.get("utilizacion", 0.5), 2),
                    "aporte": 0.57,
                    "direccion": "aumenta"
                },
                {
                    "variable": "edad",
                    "nombre": "Edad",
                    "valor_original": int(input_data.get("edad", 35)),
                    "aporte": -0.31,
                    "direccion": "reduce"
                },
                {
                    "variable": "ingreso_mensual",
                    "nombre": "Ingreso mensual",
                    "valor_original": int(input_data.get("ingreso_mensual", 5000)),
                    "aporte": -0.18,
                    "direccion": "reduce"
                }
            ]
        }
    }
    return results
