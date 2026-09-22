"""Carga en memoria el modelo de riesgo crediticio (pipeline XGBoost) exportado
por src/export_model.py y expone utilidades para predecir con explicabilidad."""
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import shap

MODEL_PATH = Path(__file__).resolve().parent.parent / "model-pkg" / "model.joblib"

NOMBRES_LEGIBLES = {
    "utilizacion": "Utilización del crédito",
    "edad": "Edad",
    "moras_30_59": "Moras de 30 a 59 días",
    "razon_deuda": "Razón de deuda",
    "ingreso_mensual": "Ingreso mensual",
    "lineas_abiertas": "Líneas de crédito abiertas",
    "moras_90": "Moras mayores a 90 días",
    "creditos_inmob": "Créditos inmobiliarios",
    "moras_60_89": "Moras de 60 a 89 días",
    "dependientes": "Número de dependientes",
    "ingreso_faltante": "Ingreso mensual no reportado",
    "es_moro_cronico": "Historial de mora crónica",
}

_bundle: Dict[str, Any] = {}
_explainer = None


def _limpiar_nombre_columna(nombre: str) -> str:
    # ColumnTransformer antepone el nombre del transformador, ej "imputacion__ingreso_mensual"
    return nombre.split("__")[-1]


def cargar_modelo() -> Dict[str, Any]:
    global _bundle, _explainer
    if not _bundle:
        _bundle = joblib.load(MODEL_PATH)
        _explainer = shap.TreeExplainer(_bundle["pipeline"].named_steps["modelo"])
    return _bundle


def construir_fila(input_data: dict) -> pd.DataFrame:
    """Arma la fila con las mismas columnas derivadas que usa el entrenamiento
    (src/preprocessing.py), a partir de los campos que manda el usuario."""
    ingreso_mensual = input_data.get("ingreso_mensual")
    moras = [
        input_data.get("moras_30_59", 0),
        input_data.get("moras_60_89", 0),
        input_data.get("moras_90", 0),
    ]
    es_moro_cronico = int(any(m in (96, 98) for m in moras))

    fila = {
        "utilizacion": input_data.get("utilizacion", 0.0),
        "edad": input_data.get("edad", 0),
        "moras_30_59": 0 if input_data.get("moras_30_59", 0) in (96, 98) else input_data.get("moras_30_59", 0),
        "razon_deuda": input_data.get("razon_deuda", 0.0),
        "ingreso_mensual": ingreso_mensual,
        "lineas_abiertas": input_data.get("lineas_abiertas", 0),
        "moras_90": 0 if input_data.get("moras_90", 0) in (96, 98) else input_data.get("moras_90", 0),
        "creditos_inmob": input_data.get("creditos_inmob", 0),
        "moras_60_89": 0 if input_data.get("moras_60_89", 0) in (96, 98) else input_data.get("moras_60_89", 0),
        "dependientes": input_data.get("dependientes"),
        "ingreso_faltante": int(ingreso_mensual is None),
        "es_moro_cronico": es_moro_cronico,
    }
    return pd.DataFrame([fila])


def predecir(input_data: dict, umbral_override: float | None = None) -> Dict[str, Any]:
    bundle = cargar_modelo()
    pipeline = bundle["pipeline"]
    fila = construir_fila(input_data)

    probabilidad = float(pipeline.predict_proba(fila)[:, 1][0])
    # el umbral de entorno permite ajustar la politica de riesgo sin
    # reentrenar; si no se define, se usa el optimo calculado al entrenar
    umbral = umbral_override if umbral_override is not None else bundle["umbral"]
    clasificacion = "Riesgo Alto" if probabilidad >= umbral else "Riesgo Bajo"

    transformado = pipeline[:-1].transform(fila)
    nombres_columnas = [
        _limpiar_nombre_columna(n)
        for n in pipeline.named_steps["preprocessing"].get_feature_names_out()
    ]
    shap_values = _explainer.shap_values(transformado)[0]

    variables = sorted(
        zip(nombres_columnas, shap_values, fila.iloc[0][nombres_columnas]),
        key=lambda t: abs(t[1]),
        reverse=True,
    )[:4]

    explicacion = {
        "unidad_aporte": "log_odds",
        "variables": [
            {
                "variable": nombre,
                "nombre": NOMBRES_LEGIBLES.get(nombre, nombre),
                "valor_original": None if pd.isna(valor) else round(float(valor), 2),
                "aporte": round(float(aporte), 4),
                "direccion": "aumenta" if aporte > 0 else "reduce",
            }
            for nombre, aporte, valor in variables
        ],
    }

    return {
        "probabilidad_incumplimiento": round(probabilidad, 4),
        "umbral_aplicado": umbral,
        "clasificacion": clasificacion,
        "explicacion": explicacion,
    }
