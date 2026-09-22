"""
Entrena el modelo ganador (XGBoost, n_estimators=300, max_depth=5, balanceado por
clase) con los mismos datos/split que train_boosting.py, y lo empaqueta en un
unico archivo .joblib que la API de riesgo crediticio carga directamente en
tiempo de inferencia. Reemplaza al modelo mock y al paquete model-pkg de otro
dominio (churn) que traia la API por defecto.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# WinsorizerP99 vive en app.transformers (dentro de src-api) para que el
# .joblib exportado se pueda cargar despues desde la API sin depender de
# __main__ ni de este script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src-api" / "credit-risk-api"))
from app.transformers import WinsorizerP99  # noqa: E402

FEATURE_COLUMNS = [
    "utilizacion", "edad", "moras_30_59", "razon_deuda", "ingreso_mensual",
    "lineas_abiertas", "moras_90", "creditos_inmob", "moras_60_89", "dependientes",
]


def mejor_umbral_por_f1(pipeline, X_test, y_test):
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    umbrales = np.arange(0.05, 0.96, 0.05)
    mejor_umbral, mejor_f1 = 0.5, -1.0
    for t in umbrales:
        f1 = f1_score(y_test, (y_proba >= t).astype(int))
        if f1 > mejor_f1:
            mejor_umbral, mejor_f1 = round(float(t), 2), f1
    return mejor_umbral


def export(output_path="src-api/credit-risk-api/model-pkg/model.joblib"):
    df = pd.read_csv("data/processed/data_prepared.csv")
    # el modelo se entrena con los dos flags derivados (ingreso_faltante,
    # es_moro_cronico), pero esos los calcula la API a partir del resto de
    # variables, no se piden directamente al usuario
    X = df[FEATURE_COLUMNS + ["ingreso_faltante", "es_moro_cronico"]]
    y = df["incumplio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    pipeline = Pipeline([
        ("winsorizacion", WinsorizerP99(columna="utilizacion")),
        ("preprocessing", ColumnTransformer(
            transformers=[("imputacion", SimpleImputer(strategy="median"),
                           ["ingreso_mensual", "dependientes"])],
            remainder="passthrough",
        )),
        ("modelo", XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)

    umbral = mejor_umbral_por_f1(pipeline, X_test, y_test)

    joblib.dump({
        "pipeline": pipeline,
        "feature_columns": FEATURE_COLUMNS,
        "umbral": umbral,
        "model_name": "xgb_n300_d5_balanced",
    }, output_path)
    print(f"Modelo exportado a {output_path} (umbral operativo: {umbral})")


if __name__ == "__main__":
    export()
