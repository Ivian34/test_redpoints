# Redpoints Listing Pipeline

Proyecto de clasificación en 2 etapas para listings:

1. Stage 1: clasifica si un listing es `asset` o no.
2. Stage 2: solo si es `asset`, estima si es `suspicious`.
3. Motor de similitud para recuperar referencias parecidas.

## Estructura importante

- `src/train/asset_classificator.py`: entrenamiento stage 1.
- `src/train/suspicion_scorer.py`: entrenamiento stage 2.
- `src/api.py`: API FastAPI.
- `src/call_api.py`: cliente CLI para la API.
- `src/bin/`: modelos `.joblib`, splits train/validation y metadatos `.json`.
- `environment.test-redpoints.yml`: freeze de Conda del entorno `test-redpoints`.

### 5) Estructura de API

- `/health`: verificación de estado y disponibilidad de artefactos.
- `/analyze`: inferencia end-to-end de una muestra.
- `/analyzed-listings/by-threshold`: consulta histórica por umbral y stage.
- `/analyzed-listings/lastN`: últimas N inferencias guardadas.
- `/model-metadata`: metadata de entrenamiento de un stage (`stage=1|2`).

## Conda: cargar entorno correctamente

Desde la raíz del proyecto:

```bash
conda activate test-redpoints
```

Si no existe el entorno todavía:

```bash
conda env create -f environment.test-redpoints.yml
conda activate test-redpoints
```

Si ya existe y quieres actualizarlo con el freeze:

```bash
conda env update -n test-redpoints -f environment.test-redpoints.yml --prune
conda activate test-redpoints
```

Verificar que estás en el entorno correcto:

```bash
conda info --envs
python --version
```

## Freeze de Conda

Para regenerar el freeze del entorno actual:

```bash
conda env export -n test-redpoints --no-builds > environment.test-redpoints.yml
```

## Entrenamiento de modelos

### Stage 1 (asset classifier)

```bash
python -m src.train.asset_classificator
```

Genera:

- `src/bin/asset_classifier_pipeline.joblib`
- `src/bin/asset_classifier_train_split.tsv`
- `src/bin/asset_classifier_validation_split.tsv`
- `src/bin/assetClassificator_metadata.json`

### Stage 2 (suspicion scorer)

```bash
python -m src.train.suspicion_scorer
```

Genera:

- `src/bin/suspicion_scorer_pipeline.joblib`
- `src/bin/suspicion_scorer_train_split.tsv`
- `src/bin/suspicion_scorer_validation_split.tsv`
- `src/bin/suspicionScorer_metadata.json`

## Arrancar la API

```bash
uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

Base URL por defecto: `http://127.0.0.1:8000`

## Endpoints

### `GET /health`

Estado de la API y existencia de artefactos.

Entrada:

- Sin parámetros.

Salida:

- `status`: estado general del servicio (`"ok"`).
- `stage1_model_path`: ruta del modelo Stage 1.
- `stage1_model_exists`: indica si el archivo de Stage 1 existe.
- `stage2_model_path`: ruta del modelo Stage 2.
- `stage2_model_exists`: indica si el archivo de Stage 2 existe.
- `similarity_model_path`: ruta del índice/modelo de similitud.
- `similarity_model_exists`: indica si el índice/modelo de similitud existe.
- `reference_data_path`: ruta del dataset de referencias.
- `reference_data_exists`: indica si el dataset de referencias existe.

### `POST /analyze`

Body JSON:

```json
{
  "title": "iPhone 14 Pro Max 256GB nuevo, con factura",
  "top_k": 3
}
```

Entrada:

- `title`: título del listing a evaluar.
- `top_k`: número de referencias similares a devolver (1 a 50).

Salida:

- `title`: título procesado.
- `stage_1_ran`: `true` si Stage 1 se ejecutó.
- `similarity_ran`: `true` si el motor de similitud se ejecutó.
- `stage_2_ran`: `true` solo si el listing fue clasificado como asset y Stage 2 se ejecutó.
- `is_asset`: predicción binaria de Stage 1.
- `asset_score`: probabilidad estimada de clase positiva (asset) en Stage 1.
- `suspicion_flag`: predicción binaria de Stage 2 (`null` si no ejecuta Stage 2).
- `suspicion_score`: probabilidad estimada de clase positiva (suspicious) en Stage 2 (`null` si no ejecuta Stage 2).
- `similarity_score`: score del match más similar (`top1`).
- `top_k`: valor de `top_k` usado en la consulta.
- `top_k_most_similar_reference_listings`: lista de referencias similares.
- `reference_id` (dentro de cada item): identificador interno de referencia.
- `reference_title` (dentro de cada item): título de la referencia.
- `similarity_score` (dentro de cada item): score de similitud de ese vecino.

Errores típicos:

- `400`: título vacío.
- `500`: modelo o datos de referencia no disponibles.

### `GET /analyzed-listings/by-threshold`

Query params:

- `stage`: `1` o `2`
- `threshold`: `0.0` a `1.0`

Entrada:

- `stage`: define sobre qué score se filtra.
- `threshold`: umbral mínimo del score del stage indicado.

Salida:

- Lista de registros analizados que cumplen el filtro.
- Si `stage=1`, el filtro se aplica sobre `asset_score` para todos los listings.
- Si `stage=2`, el filtro se aplica sobre `suspicion_score` y solo en listings asset.
- `id`: identificador del registro en BD.
- `created_at_utc`: timestamp UTC de análisis.
- `title`: título analizado.
- `stage_1_ran`: indica si Stage 1 corrió para ese registro.
- `stage_2_ran`: indica si Stage 2 corrió para ese registro.
- `similarity_ran`: indica si la consulta de similitud corrió.
- `is_asset`: decisión binaria de Stage 1.
- `asset_score`: score de Stage 1.
- `suspicion_flag`: decisión binaria de Stage 2 o `null`.
- `suspicion_score`: score de Stage 2 o `null`.
- `similarity_score`: score top1 de similitud.
- `top_k`: vecinos solicitados en ese análisis.
- `top_k_most_similar_reference_listings`: vecinos almacenados para ese análisis.

Errores típicos:

- `400`: parámetros inválidos (`stage` fuera de 1/2, etc.).

### `GET /analyzed-listings/lastN`

Query params:

- `n`: `1` a `50`

Entrada:

- `n`: cuántos registros recientes devolver.

Salida:

- Lista con los últimos `n` análisis guardados.
- Estructura de cada item igual que en `/analyzed-listings/by-threshold`.

Errores típicos:

- `400`: `n` fuera de rango.

### `GET /model-metadata`

Query params:

- `stage`: `1` o `2`

Entrada:

- `stage`: stage cuyo metadata quieres consultar.

Forma de respuesta:

- `stage`: stage solicitado.
- `metadata_path`: ruta del JSON cargado.
- `metadata`: contenido completo del stage, incluyendo `artifacts`, `data` y `metrics`.

Errores típicos:

- `404`: metadata inexistente para ese stage.
- `500`: metadata corrupto o no legible.

## Llamadas con curl

### Health

```bash
curl "http://127.0.0.1:8000/health"
```

### Analyze

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"title":"iPhone 14 Pro Max 256GB nuevo, con factura","top_k":3}'
```

### By threshold

```bash
curl "http://127.0.0.1:8000/analyzed-listings/by-threshold?stage=1&threshold=0.7"
```

### Last N

```bash
curl "http://127.0.0.1:8000/analyzed-listings/lastN?n=10"
```

### Model metadata

```bash
curl "http://127.0.0.1:8000/model-metadata?stage=1"
curl "http://127.0.0.1:8000/model-metadata?stage=2"
```

## Llamadas con Python (`requests`)

```python
import requests

BASE = "http://127.0.0.1:8000"

print("health")
print(requests.get(f"{BASE}/health", timeout=15).json())

print("analyze")
print(
    requests.post(
        f"{BASE}/analyze",
        json={"title": "iPhone 14 Pro Max 256GB nuevo, con factura", "top_k": 3},
        timeout=15,
    ).json()
)

print("by-threshold")
print(
    requests.get(
        f"{BASE}/analyzed-listings/by-threshold",
        params={"stage": 1, "threshold": 0.7},
        timeout=15,
    ).json()
)

print("lastN")
print(
    requests.get(
        f"{BASE}/analyzed-listings/lastN",
        params={"n": 10},
        timeout=15,
    ).json()
)

print("model-metadata stage 1")
print(
    requests.get(
        f"{BASE}/model-metadata",
        params={"stage": 1},
        timeout=15,
    ).json()
)
```

## Llamadas con `call_api.py`

```bash
python src/call_api.py analyze --title "iPhone 14 Pro Max 256GB nuevo, con factura" --top-k 3
python src/call_api.py by-threshold --stage 1 --threshold 0.7
python src/call_api.py last-n-listings --n 10
python src/call_api.py model-metadata --stage 1
python src/call_api.py model-metadata --stage 2
```
