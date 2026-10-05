# Registro de experimentos — TFM Airbnb (Nueva York)

Resumen de los experimentos de modelado, generado a partir de `outputs/experimentos.csv` (una fila por experimento, repetición y fold; lo escribe `func_modelado.registrar`). El nombre de la última columna coincide con la columna `experimento` de ese CSV.

**Cómo leerlo:** validación cruzada de 5 folds sobre el train (29.538 anuncios), con folds guardados en `outputs/folds_cv.csv` (estrato `room_type` × `district`, agrupados por anfitrión). Las cifras son de la primera repetición, que es la que se usa para comparar modelos. El RMSE es sobre `log(price)`; menor es mejor. Todos los experimentos usan el escenario B (con el perfil del anfitrión) salvo los marcados como escenario A. Las métricas del test están al final.

## 1. Referencias ingenuas (sin modelo)

| Experimento | Variables | RMSE log (media ± sd) | MAE | MAPE | Nombre en el registro |
|---|---|---|---|---|---|
| Mediana global | 0 | 0,711 ± 0,009 | $75 | 60% | `N0.1_mediana_global` |
| Mediana por tipo de alojamiento | 1 | 0,575 ± 0,010 | $63 | 41% | `N0.2_mediana_room_type` |
| Mediana por tipo × capacidad | 2 | 0,535 ± 0,009 | $60 | 40% | `N0.3_mediana_room_type_capacidad` |
| Mediana por barrio × tipo | 3 | 0,543 ± 0,009 | $61 | 39% | `N0.4_mediana_barrio_room_type` |

## 2. Modelos lineales

| Experimento | Variables | RMSE log (media ± sd) | MAE | MAPE | Nombre en el registro |
|---|---|---|---|---|---|
| OLS con producto básico (tipo, huéspedes, dormitorios) | 3 | 0,527 ± 0,011 | $59 | 42% | `L1_producto_basico` |
| OLS acumulando bloques: producto | 5 | 0,514 ± 0,006 | $58 | 41% | `L2_1_hasta_B0_producto` |
| ... + ubicación | 10 | 0,475 ± 0,010 | $54 | 37% | `L2_2_hasta_B1_ubicacion` |
| ... + condiciones de reserva | 13 | 0,475 ± 0,011 | $54 | 37% | `L2_3_hasta_B2a_condiciones` |
| ... + anfitrión | 23 | 0,451 ± 0,009 | $51 | 34% | `L2_4_hasta_B2b_anfitrion` |
| ... + amenities comunes | 79 | 0,437 ± 0,010 | $50 | 33% | `L2_5_hasta_B3_amenities_comunes` |
| ... + amenities de lujo | 80 | 0,435 ± 0,009 | $50 | 33% | `L2_6_hasta_B4_amenities_lujo` |
| OLS con interacciones tipo × capacidad y tipo × distrito | 79 | 0,436 ± 0,009 | $50 | 33% | `L3_interacciones` |
| OLS con interacciones y splines (numéricamente inestable) | 79 | 0,456 ± 0,036 | $104 | 82% | `L4_splines` |
| Ridge: mejor de 11 valores de alpha | 80 | 0,433 ± 0,009 | $50 | 33% | `L5_Ridge_alpha_3.2` |
| Lasso: mejor de 7 valores de alpha | 80 | 0,433 ± 0,009 | $50 | 33% | `L5_Lasso_alpha_0.00032` |
| ElasticNet: mejor de 7 valores de alpha | 80 | 0,433 ± 0,009 | $50 | 33% | `L5_ElasticNet_alpha_0.00032` |
| Ridge ajustado (71 variables): mejor de 10 candidatos | 71 | 0,439 ± 0,012 | $51 | 34% | `T7_ridge_05` |
| ElasticNet ajustado (71 variables): mejor de 10 candidatos | 71 | 0,439 ± 0,012 | $51 | 34% | `T7_elasticnet_01` |
| ElasticNet ajustado con las 79 variables | 79 | 0,434 ± 0,009 | $50 | 33% | `T7_elasticnet_01_completo` |
| Lineal ajustado, escenario A (sin perfil del anfitrión) | 67 | 0,452 ± 0,013 | $52 | 35% | `Lineal_ajustado_escenario_A` |

## 3. kNN

| Experimento | Variables | RMSE log (media ± sd) | MAE | MAPE | Nombre en el registro |
|---|---|---|---|---|---|
| kNN con 5 variables: mejor de 14 configuraciones (k y pesos) | 5 | 0,474 ± 0,012 | $54 | 37% | `K1_knn_subconjunto_k75_distance` |
| PCA + kNN: mejor de 15 | 71 | 0,497 ± 0,009 | $56 | 38% | `K2_knn_pca20_k50` |
| FAMD + kNN: mejor de 3 | 71 | 0,482 ± 0,010 | $55 | 36% | `K3_knn_famd51_k20` |
| PCAmix + kNN: mejor de 3 | 71 | 0,482 ± 0,012 | $55 | 36% | `K3_knn_pcamix51_k20` |
| PCAmix con 80% de varianza + kNN: mejor de 4 | 71 | 0,479 ± 0,011 | $55 | 36% | `K4_knn_pcamix38_k20` |

## 4. Árbol de decisión (CART)

| Experimento | Variables | RMSE log (media ± sd) | MAE | MAPE | Nombre en el registro |
|---|---|---|---|---|---|
| Árbol con prepoda: mejor de 35 combinaciones de profundidad y tamaño de hoja | 79 | 0,472 ± 0,014 | $53 | 36% | `T_arbol_d10_hoja50` |
| Árbol con poda por coste-complejidad: mejor de 25 valores | 79 | 0,482 ± 0,013 | $55 | 38% | `T_arbol_poda_ccp3.16e-04` |

## 5. Ensembles de árboles

| Experimento | Variables | RMSE log (media ± sd) | MAE | MAPE | Nombre en el registro |
|---|---|---|---|---|---|
| Random Forest, valores por defecto | 79 | 0,429 ± 0,012 | $49 | 33% | `RF_base` |
| Random Forest ajustado | 79 | 0,428 ± 0,012 | $49 | 32% | `RF_ajustado` |
| HGB, valores por defecto (barrio con TargetEncoder) | 79 | 0,423 ± 0,010 | $48 | 32% | `HGB_base` |
| HGB con barrio como categórica nativa | 79 | 0,425 ± 0,008 | $49 | 32% | `HGB_barrio_nativa` |
| HGB sin barrio | 78 | 0,426 ± 0,012 | $49 | 33% | `HGB_barrio_fuera` |
| **HGB ajustado (finalista)** | 79 | 0,419 ± 0,010 | $48 | 32% | `HGB_ajustado` |
| HGB ajustado, escenario A (sin perfil del anfitrión) | 69 | 0,434 ± 0,011 | $50 | 34% | `HGB_ajustado_escenario_A` |

Búsqueda aleatoria de Random Forest: 30 candidatos, RMSE entre 0,429 y 0,439.

Búsqueda aleatoria de HGB: 40 candidatos, RMSE entre 0,419 y 0,454.

## 6. SVR y CatBoost

| Experimento | Variables | RMSE log (media ± sd) | MAE | MAPE | Nombre en el registro |
|---|---|---|---|---|---|
| SVR con kernel Nystroem: mejor de 7 configuraciones | 79 | 0,432 ± 0,012 | $49 | 33% | `S8_svr_gamma0.005_C1.0` |
| CatBoost, valores por defecto | 79 | 0,419 ± 0,008 | $48 | 32% | `CatBoost_base` |
| Búsqueda de CatBoost | 79 | entre 0,416 y 0,423 | | | 10 candidatos (`C8_catboost_00` a `09`); el mejor es el modelo ajustado |
| **CatBoost ajustado (finalista)** | 79 | 0,416 ± 0,009 | $48 | 31% | `CatBoost_ajustado` |

Regresión cuantílica con HGB (cuantiles 0,1 y 0,9): cobertura del intervalo del 80% en validación cruzada = **72,8%**, anchura mediana $80. En el modelo final se calibra con CQR.

## 7. Selección de variables (ablación con Ridge y HGB)

Diferencia de RMSE en log al quitar cada bloque o variable frente al modelo completo (media de las 15 estimaciones: 5 folds × 3 repeticiones), en milésimas. Positivo = el modelo empeora al quitarlo, es decir, la variable aporta.

| Qué se quita | Ridge | HGB |
|---|---|---|
| Bloque producto | +111,2 | +108,7 |
| Bloque ubicación | +31,7 | +34,7 |
| Bloque condiciones de reserva | +1,0 | +2,5 |
| Bloque anfitrión | +17,2 | +15,1 |
| Bloque amenities comunes | +13,9 | +12,4 |
| Bloque amenities de lujo | -0,1 | +1,1 |
| Indicador de estudio | +2,2 | +2,3 |
| Tipo de propiedad | +3,1 | +2,0 |
| Reserva instantánea | +0,0 | +0,4 |
| Reputación de cartera | +0,4 | +1,2 |
| Tasas de respuesta y aceptación | +4,7 | +7,3 |
| Superhost | +0,2 | -0,3 |
| Identidad verificada | 0,0 | 0,0 |
| Ámbito del anfitrión | +0,4 | +1,1 |
| Antigüedad del anfitrión | +0,1 | +1,7 |
| Latitud y longitud | +2,8 | +5,2 |

La regla de decisión (fijada de antemano) y su resultado están en la bitácora. Con los hiperparámetros ajustados se repitió la comprobación en HGB (`F7_hgb_ajustado_*`): el modelo con las 79 variables gana a las versiones con menos variables, por eso los árboles usan las 79 variables.

## 8. Resultados en el test (7.384 anuncios, evaluados una vez por finalista)

| Métrica | HGB | CatBoost |
|---|---|---|
| RMSE en log | 0,424 | 0,430 |
| R² en log | 0,645 | 0,635 |
| MAE (USD) | 49,4 | 50,2 |
| Error mediano (USD) | 21,1 | 21,3 |
| MAPE | 30,9% | 31,3% |

El test se ha abierto dos veces (una por finalista) y no se ha usado para elegir. Detalle en `docs/resumen_ejecutivo.md`.
