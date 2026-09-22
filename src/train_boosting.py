import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, classification_report,
    confusion_matrix,
)
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


# misma clase que en train.py, la repito aca en vez de importarla para no depender
# de otro archivo si alguien corre este solo
class WinsorizerP99(BaseEstimator, TransformerMixin):
    def __init__(self, columna="utilizacion"):
        self.columna = columna
        self.limite_superior_ = None

    def fit(self, X, y=None):
        self.limite_superior_ = X[self.columna].quantile(0.99)
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columna] = X[self.columna].clip(upper=self.limite_superior_)
        return X


# con un solo split no se sabe si una diferencia entre corridas es real o es solo
# suerte con esa particion de datos (esto lo señalaron en la retro de la entrega 2).
# aca se repite el entrenamiento 5 veces con particiones distintas y se saca
# promedio y desviacion de cada metrica
def validacion_cruzada(pipeline, X, y, cv_splits=5, random_state=42):
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


def train_boosting(n_estimators=100, max_depth=3, learning_rate=0.1, balancear=False):
    # mismos datos y mismo split que RF y regresion logistica, para que los 3
    # modelos sean comparables entre si
    df = pd.read_csv("data/processed/data_prepared.csv")
    X = df.drop("incumplio", axis=1)
    y = df["incumplio"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # misma imputacion por mediana que usan los otros dos
    columnas_imputar = ["ingreso_mensual", "dependientes"]
    preprocessor = ColumnTransformer(
        transformers=[("imputacion", SimpleImputer(strategy="median"), columnas_imputar)],
        remainder="passthrough",
    )

    # xgboost no tiene el class_weight="balanced" de sklearn, el equivalente es
    # scale_pos_weight = negativos / positivos en el set de entrenamiento
    scale_pos_weight = 1.0
    if balancear:
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    pipeline = Pipeline([
        ("winsorizacion", WinsorizerP99(columna="utilizacion")),
        ("preprocessing", preprocessor),
        ("modelo", XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
        )),
    ])

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("riesgo_crediticio_v1")

    cw_tag = "balanced" if balancear else "none"
    run_name = f"xgb_n{n_estimators}_d{max_depth}_cw{cw_tag}"

    with mlflow.start_run(run_name=run_name):
        pipeline.fit(X_train, y_train)

        mlflow.log_params({
            "modelo": "xgboost",
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "scale_pos_weight": scale_pos_weight,
            "imputacion": "median",
            "winsorizacion": "p99_utilizacion",
            "test_size": 0.2,
            "stratify": True,
            "random_state": 42,
        })

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # las mismas 6 metricas que RF y logistica, para poder comparar los 3 modelos
        # en la misma tabla en mlflow
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "pr_auc": average_precision_score(y_test, y_proba),
            "precision_clase1": precision_score(y_test, y_pred, zero_division=0),
            "recall_clase1": recall_score(y_test, y_pred),
            "f1_clase1": f1_score(y_test, y_pred),
        }
        mlflow.log_metrics(metrics)

        # cruzada + umbral + costo, lo que pedian en la retro de la entrega 2
        validacion_cruzada(pipeline, X, y)
        tabla_umbrales = analisis_umbral(pipeline, X_test, y_test, archivo=f"umbrales_{run_name}.csv")
        cuantificar_costo(tabla_umbrales, costo_fn=10, costo_fp=1, volumen_esperado=1000,
                          archivo=f"costo_{run_name}.csv")

        # importancia de variables, es el equivalente en xgboost a los coeficientes
        # que se sacan en la regresion logistica
        nombres = pipeline.named_steps["preprocessing"].get_feature_names_out()
        importancias = pd.DataFrame({
            "variable": nombres,
            "importancia": pipeline.named_steps["modelo"].feature_importances_,
        }).sort_values("importancia", ascending=False)
        importancias.to_csv(f"importancias_{run_name}.csv", index=False)
        mlflow.log_artifact(f"importancias_{run_name}.csv")

        print(f"\n== {run_name} ==")
        print(classification_report(y_test, y_pred, digits=4))
        print({k: round(v, 4) for k, v in metrics.items()})

        mlflow.sklearn.log_model(
            pipeline,
            name="modelo_xgboost",
            # xgboost es libreria externa, hay que declarar sus clases como
            # confiables o skops no deja guardar el modelo
            skops_trusted_types=[
                "__main__.WinsorizerP99",
                "numpy.dtype",
                "xgboost.core.Booster",
                "xgboost.sklearn.XGBClassifier",
            ],
        )


if __name__ == "__main__":
    # mismo patron que ya usan RF y logistica: baseline, mas capacidad, y al final
    # con balanceo de clases, para poder ver el efecto de cada cambio por separado
    train_boosting(n_estimators=100, max_depth=3, learning_rate=0.1, balancear=False)
    train_boosting(n_estimators=300, max_depth=5, learning_rate=0.1, balancear=False)
    train_boosting(n_estimators=300, max_depth=5, learning_rate=0.1, balancear=True)
