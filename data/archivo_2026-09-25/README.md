# Archivo: partición y experimentos del 2026-09-25 (SUPERSEDED)

Estos ficheros se generaron con la partición antigua: `StratifiedGroupKFold` estratificado solo por
`room_type`, hecho **después** del EDA con todos los datos y sobre un dataset cuyo preparador se ajustó
también con el futuro test (H-035, H-036).

El 2026-09-26 se sustituyeron por la partición fija de `data/particion/` (estrato `room_type` × `district`,
creada antes de analizar el precio). Ver `docs/plan_reestructuracion_eda.md` y la bitácora.

| Fichero | Contenido | Estado |
|---|---|---|
| `listings_ny_modelado.parquet` | Dataset de modelado 36.922 × 84, preparador ajustado con todos los datos | SUPERSEDED por `listings_ny_modelado_train.parquet` y `_test.parquet` |

No se borran: son la evidencia del cambio de procedimiento.
