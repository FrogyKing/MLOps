import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

def train_and_log_model(C_value=1.0):
    # 1. Configuración de MLflow para el Experimento
    # Usaremos el nombre de carpeta del proyecto como nombre del experimento
    mlflow.set_experiment("Iris_Logistic_Regression")

    with mlflow.start_run():
        # 2. Carga y preparación de datos (Asumimos que iris.csv existe)
        data = pd.read_csv(os.path.join("data", "raw", "iris.csv"))
        
        X = data.drop(columns=['target'])
        y = data['target']
        
        # 3. División de datos
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Entrenamiento del modelo
        model = LogisticRegression(C=C_value, max_iter=200)
        model.fit(X_train, y_train)

        # 5. Evaluación del modelo
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # 6. Logging (Registro) de MLOps con MLflow
        
        # Registro de Parámetros
        mlflow.log_param("C_regularization", C_value)
        mlflow.log_param("model_type", "LogisticRegression")
        
        # Registro de Métricas
        mlflow.log_metric("accuracy", accuracy)
        
        # Registro del Modelo (Lo guarda en la carpeta mlruns)
        mlflow.sklearn.log_model(model, "model")

        print(f"Modelo entrenado con C={C_value}. Precisión: {accuracy:.4f}")
        print(f"Experimento registrado en MLflow Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    # Puedes probar diferentes parámetros aquí
    train_and_log_model(C_value=0.1)
    train_and_log_model(C_value=1.0) 
    train_and_log_model(C_value=10.0)