from typing import List, Optional

from pydantic import BaseModel


# Esquema de entrada: solicitud de crédito a evaluar (dataset "Give Me Some
# Credit", columnas renombradas por src/preprocessing.py)
class CreditApplicationInput(BaseModel):
    utilizacion: float
    edad: int
    moras_30_59: int
    razon_deuda: float
    ingreso_mensual: Optional[float] = None
    lineas_abiertas: int
    moras_90: int
    creditos_inmob: int
    moras_60_89: int
    dependientes: Optional[float] = None

    class Config:
        schema_extra = {
            "example": {
                "utilizacion": 0.45,
                "edad": 38,
                "moras_30_59": 0,
                "razon_deuda": 0.35,
                "ingreso_mensual": 5000,
                "lineas_abiertas": 8,
                "moras_90": 0,
                "creditos_inmob": 1,
                "moras_60_89": 0,
                "dependientes": 2,
            }
        }


# Esquema para variables explicativas
class VariableExplicacion(BaseModel):
    variable: str
    nombre: str
    valor_original: Optional[float]
    aporte: float
    direccion: str


# Esquema para explicación del modelo
class Explicacion(BaseModel):
    unidad_aporte: str
    variables: List[VariableExplicacion]


# Esquema de los resultados de predicción
class PredictionResults(BaseModel):
    probabilidad_incumplimiento: float
    umbral_aplicado: float
    clasificacion: str
    explicacion: Explicacion
