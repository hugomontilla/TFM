# TFM — Predicción del precio de Airbnb en Nueva York

Trabajo Fin de Máster en Data Science y Big Data (Universidad de Sevilla).

## Qué problema resuelve

Dado un anuncio **nuevo** de Airbnb en Nueva York (características del alojamiento, ubicación y perfil del
anfitrión), el proyecto predice su **precio por noche** y da un **intervalo del 80%**, para poder recomendar un
precio a un anfitrión. El objetivo del modelo es `log(price)`; `exp(ŷ)` estima la **mediana** del precio.

- **Datos:** dataset de Airbnb 2008-2021 (`listings.csv`, `reviews.csv`). Se empezó con 10 ciudades y se acotó a
  Nueva York, que tiene precio en USD, `district` completo y efectos de ubicación coherentes.
- **Tamaño:** 36.922 anuncios válidos → 29.538 en train y 7.384 en test (anfitriones distintos en cada lado; `particion.json` registra 29.586 / 7.398 antes de la regla de bloqueo).
- **Variables:** 79 candidatas en cinco bloques: producto, ubicación, condiciones de reserva, perfil del anfitrión y
  56 amenities.

## Resultados principales

Error medido con el RMSE sobre `log(price)` (menor es mejor):

| Modelo | RMSE log (validación cruzada) | RMSE log (test) | MAE en test |
|---|---|---|---|
| Mediana por tipo × capacidad (referencia) | 0,535 | — | — |
| Ridge / ElasticNet | 0,434 | — | — |
| Random Forest | 0,428 | — | — |
| **HGB** | 0,419 | **0,424** | $49 |
| **CatBoost** | 0,416 | 0,429 | $50 |

- HGB y CatBoost son los dos finalistas y quedan empatados en la práctica; HGB se ajusta unas 16 veces más rápido.
- El **producto** (tipo y capacidad) explica la mayor parte del precio, seguido de la **ubicación**; conocer al
  anfitrión mejora el resultado.
- El intervalo del 80% (cuantiles + calibración conformal, CQR) cubre el 79-80%, pero **falla en los precios altos**
  (cobertura ~58% en el quintil más caro): es la principal limitación.

Detalle completo en [docs/resumen_ejecutivo.md](docs/resumen_ejecutivo.md).

## Cómo está organizado

```
data/        particion/ (train y test fijos con su MD5) y modelado/ (parquet preparados y preparador)
scripts/     preparacion_datos.py (limpieza y variables), particion_datos.py, func_modelado.py, func_final.py, registro.py, config.py, ...
.env         parámetros comunes del experimento (semilla, folds, cobertura del intervalo...)
notebooks/   análisis y modelado, en orden numérico
models/      modelos finalistas guardados (CatBoost y HGB), cuantiles y fichas
outputs/     registro de experimentos (experimentos.csv), folds de CV, importancias y resultados en test
docs/        resumen ejecutivo, arquitectura, bitácora de decisiones, planes y enunciado
```

### Notebooks (ejecutar en este orden)

| Notebook | Contenido |
|---|---|
| `0-INTRODUCCION` | Comparación de las 10 ciudades y decisión de acotar a Nueva York |
| `1-EXPLORACION` | Calidad de datos y creación de la partición train/test |
| `2-ANALISIS` | EDA del precio, solo con train |
| `3-MODELADO` | Referencias, lineal, kNN, árboles, ensembles, selección de variables y comparación |
| `4-MODELO_FINAL` | Los dos finalistas, CatBoost y HGB, con el mismo procedimiento: intervalo CQR, evaluación en test, errores, interpretación y una comparación final |

## Decisiones de diseño clave

- **Sin fuga de información:** la partición se fija antes de mirar el precio, agrupada por `host_id` y estratificada por
  `room_type` × `district`. Todo lo que se aprende de los datos se ajusta solo con train.
- **Sin reseñas** ni variables derivadas del precio: no existen al publicar un anuncio nuevo.
- **Comparaciones pareadas:** validación cruzada de 5 folds × 3 repeticiones con folds guardados, y registro de todos
  los experimentos en `outputs/experimentos.csv`.

## Documentación

- [Resumen ejecutivo](docs/resumen_ejecutivo.md): resultados, variables y modelos explorados.
- [Arquitectura](docs/arquitectura.md): cómo está construido el proyecto.
- [Bitácora de decisiones](docs/bitacora_decisiones.md): qué se decidió y por qué.
- [Registro de experimentos](docs/registro_experimentos.md) y [planes](docs/Planes/).

## Datos y reproducción

`data/listings.csv` y `data/reviews.csv` **no se versionan** (pesan 160 y 250 MB). Descárgalos y colócalos en `data/`
para reproducir desde cero. La partición (`data/particion/`) y los datasets de modelado sí están incluidos, así que
se pueden ejecutar los notebooks de modelado sin los CSV originales.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Requiere Python 3.13 (versiones usadas: scikit-learn 1.9, pandas 3.0, catboost 1.2). Los modelos guardados con
`joblib` solo cargan con la misma versión de scikit-learn. El entrenamiento de CatBoost se puede repetir en GPU con
`Entrenamiento Google Colab/`.
