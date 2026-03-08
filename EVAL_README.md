# EVAL README

## Evaluation

Esta sección reporta primero decisiones de modelado de cada stage, métricas de ambos modelos y su análisis, no solo los scores.

## Decisiones de modelado (por qué)

### Stage 1 (Asset classifier)

- Modelo: `TF-IDF` de caracteres (`char_wb`, n-gramas 3-5) + `LogisticRegression`.
- Razón:
  - Los títulos son texto corto y ruidoso; los n-gramas de caracteres son robustos a typos, variaciones y mezclas de idioma.
  - `LogisticRegression` da un baseline fuerte, rápido de entrenar y fácil de interpretar/calibrar en clasificación binaria.
  - Se usa `class_weight="balanced"` para intentar reducir impacto del desbalance de clases.

### Stage 2 (Suspicion scorer)

- Modelo: `LogisticRegression` sobre features combinadas:
  - representación textual (`TF-IDF`),
  - `top1_similarity`,
  - `mean_topk_similarity` (ambas del motor de similitud).
- Razón:
  - Stage 2 necesita separar casos sospechosos vs descartados incorporando tanto señal textual como señal de cercanía a referencias.
  - No se usa solo `similarity_score` porque dos títulos pueden tener score alto por texto genérico y no ser infracción real.
  - También puede ocurrir lo contrario: títulos con score moderado pueden ser sospechosos por contexto textual.
  - Por eso se combinan `top1_similarity`, `mean_topk_similarity` y `TF-IDF` en lugar de una regla única con un solo umbral de similitud.

### Similarity metric (qué significa y por qué)

- El motor de similitud usa vecinos más cercanos con distancia coseno.
- Se transforma a score con `similarity_score = 1 - distance`.
- Interpretación:
  - score cercano a `1`: título muy cercano a referencias,
  - score cercano a `0`: poca similitud.
- Se usan `top1_similarity` y `mean_topk_similarity` para que Stage 2 no dependa solo de un único match.


## Resultados

Fuente de métricas:

- `src/bin/assetClassificator_metadata.json`
- `src/bin/suspicionScorer_metadata.json`

### Stage 1: Asset Classifier

Datos usados:

- Fuente: `src/data/Result_7.tsv`, `src/data/labels.tsv`
- Split: `train_test_split(test_size=0.2, random_state=42, stratify=True)`
- Train: 8000 filas (7042 positivas, 958 negativas)
- Validation: 2000 filas (1761 positivas, 239 negativas)

Métricas por clase:

- Clase 0: precision `0.5335`, recall `0.7322`, f1 `0.6173`, support `239`
- Clase 1: precision `0.9617`, recall `0.9131`, f1 `0.9368`, support `1761`

### Stage 2: Suspicion Scorer

Datos usados:

- Fuente: `src/data/Result_7.tsv`, `src/data/labels.tsv`
- Filtro de labels: `{7,5,9,10,15,16}`
- Target: `suspicion_target = (label_id != 7).astype(int)`
- Split: `train_test_split(test_size=0.2, random_state=42, stratify=True)`
- Train: 4873 filas (3804 positivas, 1069 negativas)
- Validation: 1219 filas (951 positivas, 268 negativas)

Métricas por clase:

- Clase 0: precision `0.7980`, recall `0.8993`, f1 `0.8456`, support `268`
- Clase 1: precision `0.9706`, recall `0.9359`, f1 `0.9529`, support `951`

## Consideraciones de análisis de datos

1. Desbalance de clases:
   Stage 1 está fuertemente desbalanceado hacia clase positiva (~88% positivos en train). Stage 2 también (~78% positivos). Esto explica en parte por qué el rendimiento en clase 0 es más débil que en clase 1.

2. Lectura de métricas:
   El F1 global alto está dominado por clase 1. En un caso de producción convendria vigilar especialmente `recall` de clase 0 en stage 1 y `precision` de clase 0 en stage 2.

3. Qué se podria explorar:
   - Otros modelos no lineales como RandomForest, SVM con estos datos para mejores resultados.
   - Sensibilidad del resultado a thresholds distintos de 0.5 en la elección de la clase por cada modelo.
   - Evaluar qué pasa con los falsos negativos del stage 1 y como afectan al stage 2.
   - Matriz de confusión por segmento (longitud de título, idioma, categorías).
   
