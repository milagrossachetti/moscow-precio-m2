"""Configuración central del proyecto.

Todo lo que se repite entre notebooks (rutas, semilla, nombres de MLflow)
vive acá. Si algo cambia, se cambia en un solo lugar.
"""
from pathlib import Path

# --- Rutas -------------------------------------------------------------
# ROOT es la raíz del repo, sin importar desde qué carpeta se ejecute.
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"

SECONDARY_CSV = DATA_RAW / "secondary_market.csv"
DISTRICT_MONTHLY_CSV = DATA_RAW / "district_prices_monthly.csv"
METRO_CSV = DATA_RAW / "metro_stations.csv"

# Generados por los módulos de limpieza y feature engineering (notebook 01)
CLEAN_CSV = DATA_PROCESSED / "secondary_clean.csv"
FEATURES_CSV = DATA_PROCESSED / "features.csv"
FEATURE_SETS_JSON = DATA_PROCESSED / "feature_sets.json"

# --- Problema ----------------------------------------------------------
TARGET = "price_per_sqm"
# El modelo se entrena sobre log(price_per_sqm) (ver notebook 01, §4 y §12.1).
TARGET_LOG = "log_price_per_sqm"
# Columnas que NUNCA entran como predictoras (fuga de datos / identificadores).
# price_rub = price_per_sqm * total_area -> contiene el target.
# (secondary_price_per_sqm, del archivo mensual por distrito, tampoco se usa.)
EXCLUDED_COLS = ["id", "price_rub"]

# --- Reproducibilidad --------------------------------------------------
RANDOM_STATE = 42

# --- MLflow ------------------------------------------------------------
# Backend SQLite: un único archivo (mlflow.db) que se sube a GitHub,
# así el profesor puede abrir `mlflow ui` y ver todas las ejecuciones.
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
MLFLOW_EXPERIMENT = "moscow-secondary-price-per-sqm"
MLFLOW_ARTIFACTS = ROOT / "mlartifacts"


def _rutas_mlflow_relativas():
    """Reescribe en mlflow.db las rutas de artefactos como relativas a la raíz del repo.

    MLflow guarda rutas absolutas (file:///Users/.../mlartifacts/...), que no existen
    en otra computadora. Con rutas relativas ("mlartifacts/..."), MLflow las resuelve
    desde la carpeta en la que se ejecuta, así que `mlflow ui` lanzado desde la raíz
    del repo encuentra los artefactos en cualquier máquina. Es idempotente.
    """
    import re
    import sqlite3

    db = ROOT / "mlflow.db"
    if not db.exists():
        return 0
    absoluta = re.compile(r"^(?:file://)?/.*?/(mlartifacts(?:/.*)?)$")
    columnas = [("experiments", "artifact_location", "experiment_id"),
                ("runs", "artifact_uri", "run_uuid"),
                ("logged_models", "artifact_location", "model_id")]
    cambios = 0
    with sqlite3.connect(db) as con:
        for tabla, col, clave in columnas:
            for id_, ruta in con.execute(f"SELECT {clave}, {col} FROM {tabla}").fetchall():
                m = absoluta.match(ruta or "")
                if m:
                    con.execute(f"UPDATE {tabla} SET {col} = ? WHERE {clave} = ?", (m.group(1), id_))
                    cambios += 1
    return cambios


def setup_mlflow():
    """Conecta MLflow al backend del proyecto y activa el experimento.

    Trabaja desde la raíz del repo (os.chdir) y deja las rutas de artefactos
    relativas, para que mlflow.db y mlartifacts/ funcionen en cualquier computadora.
    """
    import os
    import mlflow

    os.chdir(ROOT)  # las rutas relativas de los artefactos se resuelven desde acá
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=MLFLOW_ARTIFACTS.as_uri())
    _rutas_mlflow_relativas()
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    return mlflow
