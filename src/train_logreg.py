import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
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
        # ojo: aca no se puede transformar el parametro (ej. list(columnas)),
        # sklearn clona el pipeline en cada fold de la CV y exige que el
        # constructor guarde los parametros tal cual se los pasaron
        self.columnas = columnas
        self.limites_ = {}

    def fit(self, X, y=None):
        self.limites_ = {
            c: X[c].quantile(0.99) for c in list(self.columnas) if c in X.columns
        }
        return self

    def transform(self, X):
        X = X.copy()
        for c, lim in self.limites_.items():
            X[c] = X[c].clip(upper=lim)
        return X


# con un solo split no se sabe si una diferencia entre corridas es real o es solo
# suerte con esa particion de datos (esto lo señalaron en la retro de la entrega 2).
# aca se repite el entrenamiento 5 veces con particiones distintas y se saca
# promedio y desviacion de cada metrica
def validacion_cruzada(pipeline, X, y, cv_splits=5, random_state=42):
    """5-fold CV sobre todo el dataset. Complementa (no reemplaza) el split 80/20."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = ["roc_auc", "average_precision", "accuracy", "precision", "recall", "f1"]
    resultados = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)

    resumen = {}
    for m in scoring:
        scores = resultados[f"test_{m}"]
        resumen[f"cv_{m}_mean"] = scores.mean()
        resumen[f"cv_{m}_std"] = scores.std()

    mlflow.log_metrics(resumen)
    print("Validacion cruzada (5-fold):")
    for m in scoring:
        print(f"  {m}: {resumen[f'cv_{m}_mean']:.4f} +/- {resumen[f'cv_{m}_std']:.4f}")
    return resumen


# hasta ahora todo se evaluaba con el umbral 0.5 que pone sklearn por defecto,
# pero ese numero es arbitrario. aca se recorre de 0.05 a 0.95 para ver como se
# mueven precision y recall segun donde se ponga la linea de corte
def analisis_umbral(pipeline, X_test, y_test, archivo="umbrales.csv"):
    """Precision/recall/F1 y matriz de confusion en umbrales de 0.05 a 0.95."""
    umbrales = np.arange(0.05, 0.96, 0.05)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    filas = []
    for t in umbrales:
        y_pred_t = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
        precision_t = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_t = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_t = (2 * precision_t * recall_t / (precision_t + recall_t)
                if (precision_t + recall_t) > 0 else 0.0)
        filas.append({
            "umbral": round(t, 2), "precision": precision_t, "recall": recall_t,
            "f1": f1_t, "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        })

    tabla = pd.DataFrame(filas)
    tabla.to_csv(archivo, index=False)
    mlflow.log_artifact(archivo)

    mejor_f1 = tabla.loc[tabla["f1"].idxmax()]
    mlflow.log_metric("umbral_mejor_f1", mejor_f1["umbral"])
    mlflow.log_metric("f1_en_mejor_umbral", mejor_f1["f1"])
    print(f"Mejor umbral por F1: {mejor_f1['umbral']:.2f} (F1={mejor_f1['f1']:.4f})")
    return tabla


# un falso negativo (dejamos pasar a alguien que si va a incumplir) sale mas caro
# que un falso positivo (mandamos a revision manual a alguien que si iba a pagar).
# se asume una relacion de costo 10 a 1 y se escala a un volumen de solicitudes
# para poder hablar en numero de revisiones, no solo en porcentaje de precision
def cuantificar_costo(tabla_umbrales, costo_fn=10, costo_fp=1,
                      volumen_esperado=1000, tasa_base=0.067,
                      archivo="costo_umbrales.csv"):
    """Traduce cada umbral a revisiones manuales / incumplidores no detectados
    sobre un volumen hipotetico de solicitudes, y a un costo total."""
    tabla = tabla_umbrales.copy()
    incumplidores_esperados = volumen_esperado * tasa_base

    tabla["incumplidores_detectados"] = tabla["recall"] * incumplidores_esperados
    tabla["incumplidores_no_detectados"] = incumplidores_esperados - tabla["incumplidores_detectados"]
    tabla["marcados_total"] = tabla.apply(
        lambda r: r["incumplidores_detectados"] / r["precision"] if r["precision"] > 0 else 0,
        axis=1,
    )
    tabla["revisiones_innecesarias"] = tabla["marcados_total"] - tabla["incumplidores_detectados"]
    tabla["costo_total"] = (
        tabla["incumplidores_no_detectados"] * costo_fn
        + tabla["revisiones_innecesarias"] * costo_fp
    )

    tabla.to_csv(archivo, index=False)
    mlflow.log_artifact(archivo)

    mejor = tabla.loc[tabla["costo_total"].idxmin()]
    mlflow.log_metric("umbral_menor_costo", mejor["umbral"])
    mlflow.log_metric("costo_total_minimo", mejor["costo_total"])
    print(f"Umbral de menor costo: {mejor['umbral']:.2f} "
          f"(revisiones: {mejor['marcados_total']:.0f}, costo: {mejor['costo_total']:.1f})")
    return tabla


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

        # 7.1 cruzada + umbral + costo, lo que pedian en la retro de la entrega 2
        validacion_cruzada(pipeline, X, y)
        tabla_umbrales = analisis_umbral(pipeline, X_test, y_test, archivo=f"umbrales_{run_name}.csv")
        cuantificar_costo(tabla_umbrales, costo_fn=10, costo_fp=1, volumen_esperado=1000,
                          archivo=f"costo_{run_name}.csv")

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
    train_logreg(C=1.0,  penalty="l2", class_weight=None)
    train_logreg(C=10.0, penalty="l2", class_weight=None)
    train_logreg(C=10.0, penalty="l2", class_weight="balanced")
