import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def train_model(n_estimators=100, max_depth=5):
    # 1. Cargar datos procesados
    df = pd.read_csv("data/processed/data_clean.csv")
    
    # Definir X (features) y y (target)
    X = df.drop("incumplio", axis=1)
    y = df["incumplio"]
    
    # Dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Configurar MLflow
    mlflow.set_experiment("riesgo_crediticio_v1")
    
    with mlflow.start_run():
        # 3. Entrenar modelo
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        
        # 4. Registrar parámetros
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        # 5. Predicciones y Métricas
        predictions = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] # Probabilidades para la clase 1
        
        acc = accuracy_score(y_test, predictions)
        roc_auc = roc_auc_score(y_test, probs)
        
        # Registrar métricas
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc_auc)
        
        # Imprimir reporte detallado
        print(f"\nReporte de Clasificación (n={n_estimators}, depth={max_depth}):")
        print(classification_report(y_test, predictions))
        
        # 6. Guardar modelo
        mlflow.sklearn.log_model(model, "modelo_random_forest")
        
        print(f"Entrenamiento finalizado. ROC-AUC: {roc_auc:.4f} | Accuracy: {acc:.4f}")

if __name__ == "__main__":
    train_model(n_estimators=100, max_depth=5)
    train_model(n_estimators=150, max_depth=10)
