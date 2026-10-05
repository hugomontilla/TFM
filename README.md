# TFM — Predicción del precio de Airbnb en Nueva York

Modelo de regresión (HGB, con intervalo CQR) para recomendar el precio por noche de un anuncio nuevo. Dataset Airbnb 2008-2021.

- Resumen de resultados: [docs/resumen_ejecutivo.md](docs/resumen_ejecutivo.md)
- Estructura del proyecto: [docs/arquitectura.md](docs/arquitectura.md)
- Decisiones: [docs/bitacora_decisiones.md](docs/bitacora_decisiones.md)

Orden de los notebooks: `notebooks/0-INTRODUCCION` → `1-EXPLORACION` → `2-ANALISIS` → `3-MODELADO` → `Modelo_final_HGB`.

## Datos
`data/listings.csv` y `data/reviews.csv` no se versionan (pesan 160 y 250 MB). Colócalos en `data/` para reproducir desde cero.
La partición train/test (`data/particion/`) y los datasets de modelado sí están incluidos.

## Entorno
`python -m venv .venv && pip install -r requirements.txt`
