import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

# Configuración de rutas
PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_DIR / "data" / "raw" / "iris.csv"
PROCESSED_DATA_DIR = PROJECT_DIR / "data" / "processed"

def prepare_data():
    print("Preparando datos...")
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Dividir datos
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Crear carpeta processed si no existe
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Guardar CSVs separados
    train_path = PROCESSED_DATA_DIR / "train.csv"
    test_path = PROCESSED_DATA_DIR / "test.csv"
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"Datos divididos guardados en: {PROCESSED_DATA_DIR}")

if __name__ == "__main__":
    prepare_data()