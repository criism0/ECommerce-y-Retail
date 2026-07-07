# Proyecto Comercio Electrónico

Se poseen dos pipelines de recomendación secuencial por sesiones:

- `ECommerce.ipynb`: pipeline principal sobre el dataset de e-commerce de noviembre 2019.
- `RetailRocket.ipynb`: adaptación del mismo enfoque al dataset **RetailRocket Recommender System Dataset**.

En ambos casos el problema se formula como **next-item prediction**: dado el prefijo de una sesión, predecir el siguiente producto con el que interactuará el usuario.

## Datasets

Se utilizan dos datasets públicos de Kaggle:

- **eCommerce Behavior Data from Multi Category Store** (usado en `ECommerce.ipynb`): eventos de comportamiento (`view`, `cart`, `purchase`) de una tienda online multi-categoría, octubre 2019 - abril 2020.
  https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store

- **RetailRocket eCommerce Dataset** (usado en `RetailRocket.ipynb`): eventos de navegación y compra junto con propiedades de ítems y jerarquía de categorías.
  https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

**Incluidos en el repo** (livianos):

- `category_tree.csv` - jerarquía de categorías de RetailRocket (14 KB).

**No incluidos en el repo** (ya que son muy pesados):

- `2019-Nov.csv` (~9 GB) - dataset principal de `ECommerce.ipynb`.
- `events.csv` (~94 MB) - log de eventos de RetailRocket.
- `item_properties_part1.csv` (~484 MB) y `item_properties_part2.csv` (~409 MB) - metadatos de productos de RetailRocket.

Para reproducir los notebooks, descarga estos cuatro archivos desde los links de arriba y ubícalos en la raíz del proyecto con los mismos nombres.

## Estructura del repo

- `2019-Nov.csv`
  Dataset principal del pipeline `ECommerce.ipynb`. **No incluido en el repo** (ver sección Datasets).

- `events.csv`
  Log de eventos usado en `RetailRocket.ipynb`. **No incluido en el repo** (ver sección Datasets).

- `item_properties_part1.csv`, `item_properties_part2.csv`
  Metadatos de productos de RetailRocket en formato key-value. **No incluidos en el repo** (ver sección Datasets).

- `category_tree.csv`
  Jerarquía de categorías de RetailRocket. **Incluido en el repo.**

- `ECommerce.ipynb`
  Notebook principal del proyecto original.

- `RetailRocket.ipynb`
  Notebook que replica el pipeline sobre RetailRocket.

- `ECommerce_Helper.py`
  Utilidades de carga, muestreo, split temporal y filtrado.

- `ECommerce_Experiments.py`
  Recomendadores clásicos, híbridos, popularidad reciente y reranking.

- `ECommerce_Models.py`
  Modelos secuenciales neuronales y utilidades de entrenamiento.

- `ECommerce_TALE.py`
  Integración de TALE y helpers de preparación temporal.

- `ECommerce_Recommendation_Examples.py`
  Utilidades para inspección cualitativa de recomendaciones en el pipeline principal.

- `ECommerce_Exports.py`
  Helper simple para exportar tablas finales de resultados desde los notebooks a archivos Excel.

## Pipeline principal: `ECommerce.ipynb`

Este notebook contiene el flujo principal del proyecto:

1. carga y muestreo reproducible del CSV grande
2. análisis exploratorio y formulación del problema
3. split temporal estricto
4. evaluación de baselines
5. modelos neuronales
6. reglas secuenciales e híbridos
7. reranking con metadatos
8. integración exploratoria de TALE
9. análisis segmentado
10. inspección cualitativa de recomendaciones

La lectura general de ese notebook se mantiene:

- `item_knn` fue un baseline muy fuerte.
- Los híbridos con reglas y reranking por metadatos fueron las mejores variantes comparables.
- `TALE` aparecio como extensión exploratoria prometedora, pero con una comparación metodologicamente distinta por su catálogo reducido.

## Pipeline RetailRocket: `RetailRocket.ipynb`

`RetailRocket.ipynb` adapta la lógica del notebook principal, pero sobre el dataset RetailRocket.

### Flujo del notebook

1. carga y limpieza de `events.csv`
2. reconstrucción de sesiones por `visitorid` usando un corte de 30 minutos de inactividad
3. construcción de secuencias por sesión y eliminación de duplicados consecutivos
4. split temporal estricto por `end_time`
5. evaluación de baselines
6. bloque neuronal con `GRU4Rec` y `SASRec`
7. reglas secuenciales e híbridos
8. metadatos mínimos por categoría usando `item_properties` y `category_tree`
9. popularidad reciente y ajuste de pesos del híbrido
10. análisis segmentado
11. integración exploratoria de `TALE`

En RetailRocket la lectura principal vuelve a favorecer señales locales:

- `item_knn` fue un baseline muy fuerte.
- `gru4rec` y `sasrec` quedaron muy por debajo.
- `seq_rules` y `hybrid_knn_rules` mejoraron sobre `item_knn`.
- El mejor híbrido ajustado en validación uso pesos `0.5 item_knn + 0.5 seq_rules + 0.0 recent_pop`.
- `recent_pop` por si solo tuvo desempeño bajo y no entro en la mejor mezcla.

## Modulos Python

### `ECommerce_Helper.py`

Se encarga de:

- preprocesar eventos
- muestrear sesiones de forma reproducible
- construir resumenes de carga
- separar en train, validation y test
- filtrar items por soporte usando solo train

### `ECommerce_Experiments.py`

Incluye:

- `RecomendadorPorScores`
- `ConstruirItemKnnRecomendador`
- `ConstruirSequentialRulesRecomendador`
- `ConstruirRecomendadorHibrido`
- `ConstruirRecomendadorPopularidadReciente`
- `ConstruirMetadatosItems`
- `ConstruirRecomendadorRerankMetadatos`

### `ECommerce_Models.py`

Incluye:

- utilidades de mapeo y batching,
- `DatasetSecuencial`,
- `RecomendadorSecuencialNeuronal`,
- `ModeloGru4Rec`,
- `ModeloSasRec`,
- `EntrenarGru4Rec`,
- `EntrenarSasRec`.

### `ECommerce_TALE.py`

Incluye:

- preparación de sesiones con timestamps,
- filtrado y proyección de splits para TALE,
- limitación de catalogo por popularidad,
- resumen de splits,
- entrenamiento y recomendacion con `TALE`.

### `ECommerce_Exports.py`

Incluye:

- `ExportarDataFramesExcel`
  Exporta un conjunto de `DataFrame` a un archivo Excel. Se utiliza en las celdas finales de `ECommerce.ipynb` y `RetailRocket.ipynb` para guardar resúmenes y resultados agregados.

## Declaración de uso de IA

> ⓘ Nota
>
> Este proyecto se desarrolló con asistencia de Inteligencia Artificial:
>
> - **Modelo:** Claude (familia Claude 4.x - Opus / Sonnet) y Codex 5.3
> - **Plataforma:** Claude Code / Codex
> - **Cómo se usó:** Se empleó IA generativa como apoyo en distintas etapas del desarrollo. En la parte de código, se usó para escribir, refactorizar y estructurar los módulos de `ECommerce_Helper.py`, `ECommerce_Experiments.py`, `ECommerce_Models.py`, `ECommerce_TALE.py` y `ECommerce_Exports.py`, buscando funciones reutilizables y consistentes entre los dos pipelines (`ECommerce.ipynb` y `RetailRocket.ipynb`), tanto en la arquitectura de los modelos secuenciales (`GRU4Rec`, `SASRec`, integración de `TALE`) como en los helpers de carga, muestreo, split temporal y evaluación. También se usó como apoyo en la fase exploratoria de datos, agilizando la búsqueda e identificación del dataset RetailRocket, permitiendo contrastarlo con el dataset eCommerce original y establecer un esquema de validación cruzada entre ambos que sirvió para corroborar y robustecer los hallazgos observados inicialmente. Adicionalmente se usó para revisión de código (detección de bugs y sugerencias) y para redactar este README. Todo el contenido generado fue revisado, ajustado y validado por el equipo antes de su uso; las decisiones de diseño, metodología y análisis fueron del equipo, con la IA como herramienta de apoyo.
