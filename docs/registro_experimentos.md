# Registro de experimentos — TFM Airbnb (Nueva York)

Registro legible: una fila por experimento, con su pregunta, hipótesis, estado y decisión. Las métricas fold a
fold están en `outputs/experimentos.csv` (lo escribe `func_modelado.registrar`). El ID coincide con la columna
`experimento` de ese CSV. La fila se añade en `PLANNED` **antes** de ejecutar y nunca se borra.

**Estados:** PLANNED · RUNNING · COMPLETED · REVIEW_REQUIRED · ACCEPTED · REJECTED · SUPERSEDED

**Partición vigente:** `data/particion/` (creada el 2026-09-26; 29.586 / 7.398 anuncios válidos; tras la regla
de bloqueo, 29.538 / 7.384). Folds: `outputs/folds_cv.csv` (5 × 3, estrato `room_type` × `district`).

| ID | Fecha | Fase | Pregunta | Hipótesis | Qué cambia | Resultado (RMSE log CV, media ± sd) | Estado | Decisión | Refs |
|---|---|---|---|---|---|---|---|---|---|
| N0.1_mediana_global (partición antigua) | 2026-09-25 | 1 | ¿Funciona la infraestructura de evaluación? ¿Cuál es el suelo de error? | RMSE ≈ desviación típica del target; R² ≈ 0 | `DummyRegressor(median)`, sin variables; partición estratificada solo por `room_type`, preparador ajustado con todos los datos | 0,712 ± 0,013 | SUPERSEDED | Se repite con la partición fija | `outputs/archivo_2026-09-25/` |
| N0.1_mediana_global | 2026-09-26 | 1 | Ídem, con la partición fija | Ídem | `DummyRegressor(median)`, sin variables; 5 folds (repetición 0) | **0,711 ± 0,009** (MAE $75, mediana del error $44, MAPE 60%) | COMPLETED | Suelo de la escalera. Folds más estables que con la partición anterior | `Modelado.ipynb` §4 |
| N0.2_mediana_room_type | 2026-09-27 | 2 | ¿Cuánto explica el tipo de habitación solo? | Gran caída frente a N0.1: el `room_type` es la primera división del precio | Mediana por `room_type` | — | PLANNED | — | `Modelado.ipynb` §5 |
| N0.3_mediana_room_type_capacidad | 2026-09-27 | 2 | ¿Añadir la capacidad a la regla del anfitrión mejora? | Sí, por la escalera de precio por huésped (EDA) | Mediana por `room_type` × `accommodates` (7+) | — | PLANNED | — | §5 |
| N0.4_mediana_barrio_room_type | 2026-09-27 | 2 | ¿Cuál es el listón realista ("la mediana de mi barrio")? | Mejor que N0.2; frente a N0.3, dudoso | Mediana por barrio × `room_type`, bajando a distrito si N < 30 en el fold | — | PLANNED | — | §5 |
| L1_producto_basico | 2026-09-27 | 3 | ¿Cuánto explica el producto en un modelo aditivo? | Parecido a N0.3 | OLS con `room_type`, `accommodates`, `bedrooms` | — | PLANNED | — | §6 |
| L2_1…L2_6 | 2026-09-27 | 3 | ¿Cuánto aporta cada bloque en un modelo aditivo? | La ubicación (B1) es el mayor salto después del producto | OLS acumulando B0 → B4 | — | PLANNED | — | §6 |
| L3_interacciones | 2026-09-27 | 3 | ¿El efecto de la capacidad y de la zona depende del producto? (H-013) | Mejora pequeña pero consistente | L2 completo + `room_type` × `accommodates` y × `district` | — | PLANNED | — | §6 |
| L4_splines | 2026-09-27 | 3 | ¿Cuánto gana un lineal con no linealidad explícita? | Mejora sobre todo por lat/lon | L3 + *splines* en capacidad, distancia y coordenadas | — | PLANNED | — | §6 |
| L5_{Ridge,Lasso,ElasticNet}_alpha_* | 2026-09-27 | 3 | ¿Ayuda la regularización con ~150 columnas? | Poco: n ≫ p | Rejilla de `alpha` sobre L4 | — | PLANNED | — | §6.1 |
| F4_{ridge,hgb}_* (31 configuraciones) | 2026-09-27 | 4 | ¿Qué bloques y variables aportan, según la regla D4? | Entran B0, B1 y B2b en los dos modelos; B2a y B4, dudosos; el barrio aporta poco sobre lat/lon en HGB | Escalera por bloques, quitando un bloque cada vez, y A1-A12; Ridge L4 (`alpha = 1`) y HGB por defecto; 5 × 3 folds | — | PLANNED | — | `Modelado.ipynb` §7 |
| K1_knn_subconjunto_* | 2026-09-27 | 5 | ¿Cuánto consigue un tasador por comparables con pocas variables? | Mejor que N0.4 y peor que Ridge; k óptimo entre 20 y 50 | kNN con lat/lon, capacidad, dormitorios y `room_type`; rejilla de k y `weights` | — | PLANNED | — | §8.1 |
| K2_knn_pca*_k* | 2026-09-27 | 5 | ¿La PCA sobre todas las variables mejora al subconjunto? (propuesta del alumno) | No: la PCA prioriza la varianza de las amenities, no la ubicación | PCA (5-40 componentes) + kNN | — | PLANNED | — | §8.1 |
| T_arbol_d*_hoja* | 2026-09-27 | 5 | ¿Qué da un solo árbol y cuán inestable es? | Cerca de Ridge; la estructura cambia entre folds | Rejilla `max_depth` × `min_samples_leaf` | — | PLANNED | — | §8.2 |
| RF_base, HGB_base | 2026-09-27 | 6 | ¿Gana reducir varianza (RF) o sesgo (HGB)? | HGB gana a RF | Ensembles sin ajustar sobre las variables de HGB | — | PLANNED | — | §9.1 |
| HGB_barrio_{fuera,target,nativa} | 2026-09-27 | 6 | ¿Cómo codificar el barrio en HGB? | Empate técnico: lat/lon ya recoge casi todo | Tres codificaciones del barrio | — | PLANNED | — | §9.2 |
| T7_{ridge,elasticnet,rf,hgb}_* | 2026-09-27 | 7 | ¿Cuánto se gana ajustando los hiperparámetros? | Poco en lineal y RF; 0,005-0,01 en HGB | Búsqueda aleatoria (plan §7) | — | PLANNED | — | §10 |
| F7_hgb_ajustado_* | 2026-09-27 | 7 | ¿Se sostienen las decisiones dudosas con el HGB ajustado? | Sí en la mayoría | Pruebas dudosas repetidas con 15 folds | — | PLANNED | — | §10.1 |
| Q8_hgb_cuantilico | 2026-09-27 | 8 | ¿El intervalo del 80% cubre el 80%? | Algo menos (70%-78%), por el agrupamiento y las colas | HGB cuantílico 0,1 / 0,5 / 0,9 | — | PLANNED | — | §11.1 (`outputs/cuantiles.csv`) |
| S8_svr_* | 2026-09-27 | 8 | ¿El SVR del paper compite con los árboles? | Entre Ridge y HGB | Nystroem (1.000) + `LinearSVR` | — | PLANNED | — | §11.2 |
