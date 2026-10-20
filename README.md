# Precio por m² en el mercado secundario de Moscú — Regresión lineal

Parcial — Introducción al Análisis de Datos (UTN FRT).

## Objetivo
Predecir `price_per_sqm` (rublos por m²) de departamentos en reventa en Moscú
(2020–2026) mediante regresión lineal, comparando las ejecuciones con MLflow.

**Dataset:** [Moscow Real Estate: Sales & Rentals (2020–2026)](https://www.kaggle.com/datasets/sergionefedov/moscow-real-estate-sales-and-rentals-20202026)
— archivo base: `secondary_market.csv`.

## Estructura
```
data/raw/          datos originales (no se modifican nunca)
data/processed/    datos limpios generados por los notebooks
notebooks/         00 (verificación del entorno) y 01 (EDA, limpieza, features y modelado)
src/config.py      rutas, semilla y configuración de MLflow
reports/figures/   gráficos exportados
mlflow.db          registro de todas las ejecuciones (MLflow)
mlartifacts/       artefactos de cada ejecución (modelos, gráficos)
```

## Notebooks
| # | Notebook | Etapa |
|---|---|---|
| 00 | `00_setup.ipynb` | Verificación del entorno |
| 01 | `01_eda.ipynb` | Módulo A: análisis exploratorio (§1–§10)<br>Módulo B: limpieza (§11) → `data/processed/secondary_clean.csv`<br>Módulo C: ingeniería de variables (§12–§13) → `data/processed/features.csv`, `feature_sets.json`<br>Módulo D: modelado y comparación en MLflow (§14–§15) |

## Cómo reproducir (en tu computadora)
Probado con Python 3.13.

```bash
git clone https://github.com/milagrossachetti/moscow-precio-m2.git
cd moscow-precio-m2
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook                   # abrir notebooks/ y correr 00 y después 01
```

En Jupyter: abrir el notebook → **Kernel → Restart Kernel and Run All Cells**.
`01_eda.ipynb` tarda alrededor de un minuto y regenera `data/processed/`, `reports/figures/`
y los runs de MLflow. Los notebooks ya están guardados **con sus salidas**, así que también
se pueden leer sin ejecutarlos (por ejemplo, directamente en GitHub).

## Ver las ejecuciones en MLflow

### Online (DagsHub), sin instalar nada
**https://dagshub.com/milagrossachetti/moscow-precio-m2.mlflow**

Es el mismo experimento `moscow-secondary-price-per-sqm`, alojado en el servidor MLflow de DagsHub
y abierto a cualquiera con el link.

**Registrar los runs en DagsHub** (solo para quien tenga permisos de escritura en el repo de DagsHub):

1. Generar un token en DagsHub: **User settings → Tokens**.
2. Copiar `.env.example` como `.env` en la raíz del repo y completar el token. `.env` está en
   `.gitignore`, así que el token nunca se sube a GitHub.
3. Ejecutar `01_eda.ipynb` completo. `config.setup_mlflow()` detecta `MLFLOW_TRACKING_URI` y
   registra los runs en DagsHub en lugar de `mlflow.db`. Los runs de una ejecución anterior quedan
   en la papelera del experimento en DagsHub.

Sin `.env` (ni la variable `MLFLOW_TRACKING_URI`), todo funciona en local como se describe abajo.
En Colab, las mismas tres variables se pueden definir con `os.environ[...]` antes de ejecutar.

### En local
El repositorio incluye `mlflow.db` y `mlartifacts/` con todas las ejecuciones, así que **no hace
falta ejecutar el notebook** para verlas.

**1. Abrir la interfaz.** Con el entorno virtual activado y desde la **raíz del repositorio**
(la carpeta que contiene `mlflow.db`):

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001
```

Abrir **http://127.0.0.1:5001** en el navegador. El comando queda corriendo; para cerrarlo,
`Ctrl+C` en la terminal.

> Se usa el puerto 5001 porque en macOS el 5000 suele estar ocupado por el Receptor AirPlay.
> En Windows o Linux también funciona con el puerto por defecto (sin `--port`, en http://127.0.0.1:5000).

**2. Qué mirar.**

| Dónde | Qué se ve |
|---|---|
| Panel izquierdo → experimento `moscow-secondary-price-per-sqm` | Los 21 runs del notebook: baseline, comparación de conjuntos, Ridge, ablaciones, modelo reducido y modelo final |
| Botón **Columns** de la tabla | Agregar `tags.etapa`, `params.n_variables`, `metrics.cv_rmse_log` y `metrics.test_r2_log`, y ordenar por `cv_rmse_log` para ver el ranking de modelos |
| Buscador de runs | Filtrar por etapa, por ejemplo `tags.etapa = 'ablacion'` (las 14 ablaciones) o `tags.etapa = 'comparacion_conjuntos'` |
| Tildar varios runs → **Compare** | Métricas y parámetros lado a lado, con gráficos |
| Run **`modelo_final`** | Parámetros, métricas de prueba (`test_*`), entrenamiento (`train_*`) y validación cruzada (`cv_*`). En **Artifacts**: gráficos de residuos y de coeficientes, tabla de coeficientes y lista de variables |
| Pestaña **Models** del experimento | El pipeline entrenado (estandarización + regresión lineal), listo para cargar |

**3. Cargar el modelo desde Python** (opcional). En la tabla de runs, copiar el *Run ID* de
`modelo_final`:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")      # ejecutar desde la raíz del repo
modelo = mlflow.sklearn.load_model("runs:/<run_id>/modelo")
```

Las rutas de los artefactos se guardan **relativas** a la raíz del repo (`mlartifacts/...`), así
que funcionan en cualquier computadora sin pasos extra. `config.setup_mlflow()` las mantiene así
cada vez que se ejecuta un notebook. Por eso `mlflow ui` (y el código de arriba) se ejecutan
**desde la raíz del repo**: desde otra carpeta, MLflow no encuentra los artefactos.

## Ejecutar en Google Colab
Colab permite ejecutar el notebook sin instalar nada, pero necesita el repositorio completo
(`src/config.py` y `data/raw/`), no solo el `.ipynb`.

**1. Abrir el notebook en Colab:**

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/milagrossachetti/moscow-precio-m2/blob/main/notebooks/01_eda.ipynb)

**2. Insertar una celda al principio** (`+ Código`) con lo siguiente y ejecutarla:

```python
import os
REPO = "/content/moscow-precio-m2"
if not os.path.exists(REPO):
    !git clone -q https://github.com/milagrossachetti/moscow-precio-m2.git {REPO}
    !pip install -q -r {REPO}/requirements.txt
%cd {REPO}/notebooks
```

La celda clona el repositorio, instala las versiones exactas de `requirements.txt` y se ubica en
`notebooks/`, que es lo que espera la primera celda del notebook.

**3. Reiniciar el entorno:** menú **Entorno de ejecución → Reiniciar sesión** (*Runtime → Restart
session*). Hace falta porque Colab trae otras versiones de pandas y numpy ya cargadas en memoria.

**4. Ejecutar todo:** **Entorno de ejecución → Ejecutar todas** (*Run all*). La celda del paso 2
detecta que el repositorio ya está clonado, no reinstala nada y solo vuelve a ubicarse en
`notebooks/`.

**Limitaciones de Colab:**
- **Los archivos se borran al cerrar la sesión.** Lo que genere el notebook (runs de MLflow,
  figuras, `data/processed/`) no se guarda en el repositorio. Para conservar los runs, ejecutar al
  final:
  ```python
  !cd {REPO} && zip -qr mlflow_colab.zip mlflow.db mlartifacts
  from google.colab import files; files.download(f"{REPO}/mlflow_colab.zip")
  ```
  y descomprimir el zip en la raíz del repo local para verlo con `mlflow ui`.
- **`mlflow ui` no se puede abrir desde Colab** (necesitaría un túnel para exponer el puerto).
  Para ver MLflow, usar la sección anterior en una computadora local.
- Si la versión de Python de Colab no fuera compatible con alguna versión fijada en
  `requirements.txt`, `pip` lo va a informar en el paso 2. El proyecto se probó con Python 3.13.
