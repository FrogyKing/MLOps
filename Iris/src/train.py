import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
from pathlib import Path
import joblib

# Paths
PROJECT_DIR = Path(__file__).resolve().parents[1]
TRAIN_DATA_PATH = PROJECT_DIR / "data" / "processed" / "train.csv"
MODEL_DIR = PROJECT_DIR / "models"

def train_and_log_model(C_value=1.0):
    # Crear experimento si no existe
    mlflow.set_experiment("Iris_Logistic_Regression")

    with mlflow.start_run():
        # 1. Cargar datos procesados
        train_data = pd.read_csv(TRAIN_DATA_PATH)
        X_train = train_data.drop(columns=["target"])
        y_train = train_data["target"]

        # 2. Entrenar modelo
        model = LogisticRegression(C=C_value, max_iter=200)
        model.fit(X_train, y_train)

        # 3. Métrica de entrenamiento
        train_acc = accuracy_score(y_train, model.predict(X_train))

        # 4. Logging con MLflow
        mlflow.log_param("C_regularization", C_value)
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.sklearn.log_model(model, artifact_path="model")

        # 5. Guardar modelo local
        MODEL_DIR.mkdir(exist_ok=True, parents=True)
        model_path = MODEL_DIR / "model.pkl"
        joblib.dump(model, model_path)

        print(f"Modelo entrenado (C={C_value}) y guardado en: {model_path}")

if __name__ == "__main__":
    train_and_log_model(C_value=1.0)
    train_and_log_model(C_value=10)
