import pandas as pd
from sklearn.datasets import load_iris
import os

def fetch_and_save_data():
    """Carga el dataset de Iris y lo guarda como CSV en data/raw."""
    # 1. Cargar el dataset de sklearn
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    # 2. Definir la ruta de destino
    raw_data_path = os.path.join("data", "raw", "iris.csv")
    
    # Asegurarse de que la carpeta data/raw exista
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    
    # 3. Guardar el archivo
    df.to_csv(raw_data_path, index=False)
    print(f"Dataset de Iris guardado en: {raw_data_path}")

if __name__ == "__main__":
    fetch_and_save_data()