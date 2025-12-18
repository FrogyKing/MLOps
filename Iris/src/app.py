from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from pathlib import Path
import uvicorn

app = FastAPI()

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"

# Cargar modelo
try:
    model = joblib.load(MODEL_PATH)
    print(f"Modelo cargado correctamente desde: {MODEL_PATH}")
except FileNotFoundError:
    print("Error: No se encontró el modelo.")
    model = None

# Input Schema (Nombres limpios para la API)
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict")
def predict(data: IrisInput):
    if model is None:
        return {"error": "Modelo no encontrado"}
    
    # 1. Convertir el input a DataFrame
    input_data = data.dict()
    df = pd.DataFrame([input_data])
    
    # --- AQUÍ ESTÁ EL ARREGLO ---
    # Diccionario de traducción: API -> Modelo
    column_mapping = {
        "sepal_length": "sepal length (cm)",
        "sepal_width": "sepal width (cm)",
        "petal_length": "petal length (cm)",
        "petal_width": "petal width (cm)"
    }
    
    # Renombrar columnas para que Scikit-Learn esté feliz
    df.rename(columns=column_mapping, inplace=True)
    # ----------------------------

    # 2. Predecir
    prediction = model.predict(df)
    
    # 3. Retornar resultado
    return {"class": int(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)