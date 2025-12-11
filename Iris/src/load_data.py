import pandas as pd
from sklearn.datasets import load_iris
from pathlib import Path
import os

# --- CONFIGURACIÓN DE RUTAS (Vital en MLOps) ---
# Esto encuentra la raíz del proyecto dinámicamente, sin importar desde dónde ejecutes el script
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "iris.csv"

def fetch_and_save_data():
    """
    INGESTIÓN: Simula descargar datos de una fuente externa y guardarlos en disco.
    Se usa en el pipeline de DVC (dvc run ...).
    """
    print("Descargando datos desde sklearn...")
    iris = load_iris(as_frame=True)
    df = iris.frame
    
    # Crear carpeta si no existe
    # parent es la carpeta 'data/raw'
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Dataset guardado exitosamente en: {RAW_DATA_PATH}")

def load_raw_data():
    """
    CARGA: Lee el CSV desde el disco para usarlo en Notebooks o Scripts de entrenamiento.
    """
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo {RAW_DATA_PATH}. Ejecuta 'python src/load_data.py' primero.")
    
    return pd.read_csv(RAW_DATA_PATH)

if __name__ == "__main__":
    # Si ejecutamos el archivo directamente, hacemos la INGESTIÓN
    fetch_and_save_data()