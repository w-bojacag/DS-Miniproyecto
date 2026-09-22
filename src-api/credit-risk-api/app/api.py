from typing import Any

from fastapi import APIRouter
from loguru import logger

from app import __version__, schemas
from app.config import settings
from app.model_loader import cargar_modelo, predecir

api_router = APIRouter()

# Ruta para verificar que la API se esté ejecutando correctamente
@api_router.get("/health", response_model=schemas.Health, status_code=200)
def health() -> dict:
    """
    Root Get
    """
    modelo = cargar_modelo()
    health = schemas.Health(
        name=settings.PROJECT_NAME, api_version=__version__, model_version=modelo["model_name"]
    )

    return health.dict()

# Ruta para realizar las predicciones
@api_router.post("/predict", response_model=schemas.PredictionResults, status_code=200)
async def predict(input_data: schemas.CreditApplicationInput) -> Any:
    """
    Prediccion usando el modelo de riesgo crediticio
    """
    logger.info(f"Making prediction on inputs: {input_data}")
    return predecir(input_data.dict())
