import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path

# Rutas
PROJECT_DIR = Path(__file__).resolve().parents[1]
TEST_DATA_PATH = PROJECT_DIR / "data" / "processed" / "test.csv"
MODEL_PATH = PROJECT_DIR / "models" / "model.pkl"
METRICS_PATH = PROJECT_DIR / "metrics.json"

def evaluate_model():
    # Cargar datos y modelo
    test_data = pd.read_csv(TEST_DATA_PATH)
    model = joblib.load(MODEL_PATH)
    
    X_test = test_data.drop(columns=['target'])
    y_test = test_data['target']
    
    # Predecir
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Accuracy en Test: {accuracy}")
    
    # Guardar métricas en un JSON (Importante para DVC)
    metrics = {
        "test_accuracy": accuracy
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    evaluate_model()