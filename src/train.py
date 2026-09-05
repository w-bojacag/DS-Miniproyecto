import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin


class WinsorizerP99(BaseEstimator, TransformerMixin):
    def __init__(self, columna="utilizacion"):
        self.columna = columna
        self.limite_superior_ = None

    def fit(self, X, y=None):
        self.limite_superior_ = X[self.columna].quantile(0.99)
        return self

    def transform(self, X):
        X = X.copy()
        X[self.columna] = X[self.columna].clip(
            upper=self.limite_superior_
        )
        return X

def train_model(n_estimators=100, max_depth=5, class_weight=None):
    # 1. Cargar datos procesados
    df = pd.read_csv("data/processed/data_prepared.csv")
    
    # 1.1 Definir X (features) y y (target)
    X = df.drop("incumplio", axis=1)
    y = df["incumplio"]
    
    # 1.2 Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 1.3 Definir columnas a imputar y preprocesador
    columnas_imputar = [
        "ingreso_mensual",
        "dependientes"
    ]

    # 1.4 Crear preprocesador para imputar valores faltantes en las columnas seleccionadas
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "imputacion",
                SimpleImputer(strategy="median"),
                columnas_imputar
            )
        ],
        remainder="passthrough"
    )

    # 1.5 Crear pipeline que incluya el preprocesador y el modelo
    pipeline = Pipeline([
        ("winsorizacion", WinsorizerP99(columna="utilizacion")),
        ("preprocessing", preprocessor),
        ("modelo", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=42
        ))
    ])


    # 2. Configurar MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment("riesgo_crediticio_v1")
    
    with mlflow.start_run(run_name=f"rf_n{n_estimators}_d{max_depth}_cw{class_weight}"):
        # 3. Entrenar pipeline completo
        pipeline.fit(X_train, y_train)
        
        # 4. Registrar parámetros
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("imputacion", "median")
        mlflow.log_param("winsorizacion", "p99_utilizacion")
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("stratify", True)
        mlflow.log_param("class_weight", str(class_weight))
        mlflow.log_param("random_state", 42)
        
        # 5. Predicciones y Métricas
        predictions = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:, 1]

        # 6. Métricas de evaluación
        acc = accuracy_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probs)
        
        # 7. Registrar métricas
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # 8. Imprimir reporte detallado
        print(f"\nReporte de Clasificación (n={n_estimators}, depth={max_depth}):")
        print(classification_report(y_test, predictions))
        
        # 9. Guardar pipeline completo
        mlflow.sklearn.log_model(
            pipeline,
            name="modelo_random_forest",
            skops_trusted_types=[
                "__main__.WinsorizerP99",
                "numpy.dtype"
            ]
        )
        
        print(f"Entrenamiento finalizado. ROC-AUC: {roc_auc:.4f} | Accuracy: {acc:.4f}")

if __name__ == "__main__":
    # Baseline
    train_model(
        n_estimators=100,
        max_depth=5
    )

    # Mayor profundidad
    train_model(
        n_estimators=150,
        max_depth=10
    )

    # Tratamiento del desbalance
    train_model(
        n_estimators=150,
        max_depth=10,
        class_weight="balanced"
    )
