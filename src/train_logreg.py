import os

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


DATA_PATH = "data/processed/data_prepared.csv"
EXPERIMENT_NAME = "riesgo_crediticio_v1"


class WinsorizerP99(BaseEstimator, TransformerMixin):
    """Recorta valores por encima del percentil 99"""

    def __init__(self, columnas=("utilizacion", "razon_deuda", "ingreso_mensual")):
        self.columnas = list(columnas)
        self.limites_ = {}

    def fit(self, X, y=None):
        self.limites_ = {
            c: X[c].quantile(0.99) for c in self.columnas if c in X.columns
        }
        return self

    def transform(self, X):
        X = X.copy()
        for c, lim in self.limites_.items():
            X[c] = X[c].clip(upper=lim)
        return X


def build_pipeline(C=1.0, penalty="l2", class_weight=None, solver="liblinear"):

    columnas_imputar = ["ingreso_mensual", "dependientes"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("imputacion", SimpleImputer(strategy="median"), columnas_imputar),
        ],
        remainder="passthrough",
    )

    return Pipeline(
        [
            (
                "winsorizacion",
                WinsorizerP99(["utilizacion", "razon_deuda", "ingreso_mensual"]),
            ),
            ("preprocessing", preprocessor),
            ("escalado", StandardScaler()),
            (
                "modelo",
                LogisticRegression(
                    C=C,
                    penalty=penalty,
                    class_weight=class_weight,
                    solver=solver,
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def train_logreg(C=1.0, penalty="l2", class_weight=None, solver="liblinear"):
    # 1. Datos ya limpios por src/preprocessing.py
    df = pd.read_csv(DATA_PATH)
    X = df.drop("incumplio", axis=1)
    y = df["incumplio"]

    # 2. Mismo split que src/train.py para poder comparar modelos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(C, penalty, class_weight, solver)

    # 3. MLflow: local por defecto o EC2 definiendo el parametro MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(EXPERIMENT_NAME)

    cw_tag = "balanced" if class_weight == "balanced" else "none"
    run_name = f"logreg_C{C}_{penalty}_cw{cw_tag}"

    with mlflow.start_run(run_name=run_name):
        pipeline.fit(X_train, y_train)

        # 4. Parametros
        mlflow.log_params(
            {
                "modelo": "regresion_logistica",
                "C": C,
                "penalty": penalty,
                "solver": solver,
                "class_weight": str(class_weight),
                "escalado": "StandardScaler",
                "imputacion": "median",
                "winsorizacion": "p99_util_razondeuda_ingreso",
                "test_size": 0.2,
                "stratify": True,
                "random_state": 42,
            }
        )

        # 5. Predicciones
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # 6. Metricas (ROC-AUC principal; precision/recall/F1 de la clase "incumplio")
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            "precision_clase1": precision_score(y_test, y_pred, zero_division=0),
            "recall_clase1": recall_score(y_test, y_pred),
            "f1_clase1": f1_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)

        # 7. Coeficientes
        nombres = pipeline.named_steps["preprocessing"].get_feature_names_out()
        coefs = pd.DataFrame(
            {
                "variable": nombres,
                "coeficiente": pipeline.named_steps["modelo"].coef_[0],
            }
        ).sort_values("coeficiente", key=abs, ascending=False)
        coefs.to_csv("coeficientes_logreg.csv", index=False)
        mlflow.log_artifact("coeficientes_logreg.csv")

        # 8. Reporte por consola
        print(f"\n== {run_name} ==")
        print(classification_report(y_test, y_pred, digits=4))
        print({k: round(v, 4) for k, v in metrics.items()})

        # 9. Guardar el pipeline completo
        mlflow.sklearn.log_model(
            pipeline,
            name="modelo_regresion_logistica",
            skops_trusted_types=["__main__.WinsorizerP99", "numpy.dtype"],
        )

        print(f"Fin {run_name} | ROC-AUC: {metrics['roc_auc']:.4f} | "
              f"Recall(1): {metrics['recall_clase1']:.4f}")


if __name__ == "__main__":
    # 3 configuraciones en paralelo

    # 1. Baseline
    train_logreg(C=1.0, penalty="l2", class_weight=None)

    # 2. Menor regularizacion (modelo mas flexible)
    train_logreg(C=10.0, penalty="l2", class_weight=None)

    # 3. Tratamiento del desbalance de clases
    train_logreg(C=10.0, penalty="l2", class_weight="balanced")
