# Archivo: partición y experimentos del 2026-09-25 (SUPERSEDED)

Estos ficheros se generaron con la partición antigua: `StratifiedGroupKFold` estratificado solo por
`room_type`, hecho **después** del EDA con todos los datos y sobre un dataset cuyo preparador se ajustó
también con el futuro test (H-035, H-036).

El 2026-09-26 se sustituyeron por la partición fija de `data/particion/` (estrato `room_type` × `district`,
creada antes de analizar el precio). Ver `docs/plan_reestructuracion_eda.md` y la bitácora.

| Fichero | Contenido | Estado |
|---|---|---|
| `experimentos.csv` | N0.1 (mediana global), 5 folds: RMSE en log 0,712 ± 0,013 | SUPERSEDED: se repite con los folds nuevos |
| `split_test.csv` | Partición antigua (29.537 / 7.385) | SUPERSEDED |
| `folds_cv.csv` | Folds antiguos (5 × 3) | SUPERSEDED |

No se borran: son la evidencia del cambio de procedimiento.
