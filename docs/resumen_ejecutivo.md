# Resumen ejecutivo — Predicción del precio de Airbnb en Nueva York

*TFM Máster en Data Science y Big Data. Documento de seguimiento para la tutora*

## 1. Qué problema resuelvo

Dado un anuncio **nuevo** en Nueva York (sus características, su ubicación y el perfil del anfitrión), predecir el
**precio por noche** y dar un **intervalo** realista, para recomendar un precio a un anfitrión. El objetivo es
`log(price)`; la predicción, `exp(ŷ)`, estima la **mediana** del precio.

- **Datos:** `listings.csv` del dataset Airbnb 2008-2021. Empecé con 10 ciudades y acoté a Nueva York (precio en
  USD, `district` completo y efectos de ubicación que cambiaban de signo entre ciudades).
- **Tamaño:** 36.922 anuncios válidos → **train 29.538 / test 7.384** (anfitriones distintos en cada lado; `particion.json` registra 29.586 / 7.398 antes de la regla de bloqueo).

## 2. Cómo evito la fuga de información

| Decisión | Motivo |
|---|---|
| Partición **antes** de mirar el precio, agrupada por `host_id` y estratificada por `room_type` × `district` | El 38,1% de los 36.922 anuncios pertenece a carteras de varios y son casi gemelos; sin agrupar, la validación se infla |
| Todo lo que aprende de los datos (amenities frecuentes, medianas, codificación del barrio) se ajusta solo con train | Evita contaminar el test |
| Sin reseñas ni variables derivadas del precio | Son posteriores a la publicación; un anuncio nuevo no las tiene |
| Test abierto **una sola vez**, tras comprobar su MD5 | La elección del modelo se hace solo con validación cruzada |
| CV de 5 folds × 3 repeticiones con folds guardados | Todas las comparaciones entre modelos son pareadas |

## 3. Variables (79)

| Bloque | Nº | Contenido |
|---|---|---|
| Producto | 5 | tipo de alojamiento, tipo de propiedad (16 + «Otros»), huéspedes, dormitorios, indicador de estudio |
| Ubicación | 5 | latitud, longitud, distancia al centro (City Hall), distrito, barrio (220) |
| Condiciones | 3 | noches mínimas, estancia mínima de 30 noches, reserva instantánea |
| Anfitrión | 10 | superhost, antigüedad, nº de anuncios en NY, ámbito de residencia, verificación, tiempo y tasa de respuesta, tasa de aceptación, reputación de cartera… |
| Amenities | 56 | binarias, presentes entre el 3,5% y el 97% de los anuncios (criterio ciego al precio) |

Fuera por diseño: reseñas, `maximum_nights` (efecto explicable por composición) y el título del anuncio (línea futura).

## 4. Modelos explorados (RMSE en log, CV de 5 folds; menor es mejor)

| Familia | Mejor configuración | RMSE log | Comentario |
|---|---|---|---|
| Referencia ingenua | mediana global | 0,711 | Suelo del error |
| | mediana por tipo × capacidad | 0,535 | Lo que haría un anfitrión con una regla simple |
| | mediana por barrio × tipo | 0,543 | Ni el barrio mejora la regla anterior |
| Lineal | OLS con producto básico | 0,527 | |
| | ElasticNet (79 variables) | 0,434 | Regularizar mejora, aunque poco; con las 71 seleccionadas, 0,439 |
| kNN | subconjunto de 5 variables (k = 75) | 0,474 | Tasador «por comparables» |
| | PCA / FAMD / PCAmix + kNN | 0,479-0,482 | Reducir dimensión no ayuda |
| Árbol CART | profundidad 10 | 0,472 | Inestable entre folds |
| SVR (Nystroem) | γ = 0,005, C = 1 | 0,432 | Parecido al lineal |
| Random Forest | ajustado | 0,428 | |
| **HGB (Histogram Gradient Boosting)** | **ajustado** | **0,419** | **Modelo final** |
| CatBoost | ajustado | 0,416 | Mejor en CV por 0,003, no concluyente |

Qué he aprendido en el camino:
- El **producto** (tipo, capacidad) explica casi todo el salto de 0,71 a 0,53; la **ubicación** (lat/lon) y el
  boosting bajan el resto hasta 0,42.
- **Seleccionar variables perjudica a los árboles ajustados** (Δ ≈ 0,005, en 5 de 5 folds): usan las 79. El lineal usa
  una selección de 71.
- Conocer al anfitrión **sí vale**: sin el perfil (escenario A) el HGB pasa de 0,419 a 0,434.

## 5. Modelo final

Elegí HGB y CatBoost (que gana 0,0031 de RMSE, a ~1 error estándar) porque ajusta unas **16 veces más rápido**
y la diferencia es menor que la variación entre folds. Es una excepción a mi propia regla de elección, tomada
después de ver la tabla de CV; la declaro como tal y reporto CatBoost como sensibilidad.

### Resultados en el test (abierto una vez)

| Métrica | Test (IC 95%) | CV |
|---|---|---|
| RMSE en log | **0,424** (0,407-0,442) | 0,419 ± 0,010 |
| R² en log | **0,645** (0,608-0,680) | 0,653 |
| MAE | **$49** (45-55) | $48 |
| Error mediano | **$21** | $21 |
| MAPE | **31%** | 32% |

El test cae dentro de CV ± 2 sd en todas las métricas: no hay señal de sobreajuste de la búsqueda de hiperparámetros.

### Intervalo del 80% (cuantiles de HGB + calibración conformal CQR)

- Sin calibrar cubre el 72% (corto); con CQR, **79%** (nominal 80%), con una anchura mediana de $93.
- **Falla en los precios altos:** cubre solo el 57,5% en el quintil más caro, y menos en Manhattan (75%), habitaciones
  de hotel (61%) y Staten Island (64%).

### Dónde se equivoca más

| Segmento | MAE | MAPE |
|---|---|---|
| Habitación privada | $25 | 29% |
| Piso entero | $69 | 32% |
| 7 o más huéspedes | $180 | 47% |
| Quintil más caro | $151 | 35% |

Los 20 peores errores son pisos enteros caros (>$1.500) infravalorados: el modelo no captura el lujo extremo.
Con la regla P99 global ($794) el RMSE en log baja a 0,389 y el MAE a $38, es decir, la cola es lo que más pesa.

### Qué determina el precio (importancia por permutación, aumento de RMSE)

Producto **0,298** ≫ ubicación **0,109** > anfitrión **0,050** ≈ amenities **0,046** ≫ condiciones **0,005**.
Variables individuales: huéspedes, tipo de propiedad, tipo de alojamiento, dormitorios y barrio.

## 6. Limitaciones que declaro

1. El perfil del anfitrión se mide en 2021, no en el momento de publicar.
2. Los precios de bloqueo ($9.999 y habitaciones > $1.000) se excluyen también del test; en producción no se conoce el precio.
3. La CQR con un solo conjunto de calibración es aproximada (anuncios correlacionados por anfitrión).
4. Las comparaciones entre modelos usan folds que comparten train: los p-valores son orientativos.
5. La permutación describe cómo usa las variables el modelo, no una relación causal.
6. Recorté la búsqueda de CatBoost de 40 a 10 candidatos viendo resultados parciales (el óptimo era plano y elegir entre muchos sobreajusta la validación). El mejor de los 10 es el modelo ajustado, que no cambia.

## 7. Dónde me gustaría su orientación

1. **¿Es defendible haber elegido HGB sobre CatBoost** por coste y empate práctico, aunque contradiga mi regla previa?
2. **La cola de precios altos** es el punto débil (cobertura 57%, MAE $151). ¿Merece un modelo específico, una
   pérdida robusta, o basta con declararlo como limitación?
3. **Comparación con la literatura:** el paper de referencia usa partición aleatoria y reseñas, no es comparable
   directamente. ¿Cómo plantearlo en la memoria?
4. **Interpretabilidad:** ahora uso permutación y PDP/ICE. ¿Compensa añadir SHAP?
5. **MLP:** lo tengo preparado y aplazado porque no espero que gane en datos tabulares. ¿Lo incluyo para completar la comparación?
6. **Estructura de la memoria**, y si debo presentar la selección de variables y el EDA con tanto detalle.

## 8. Siguiente paso

Rellenar las conclusiones del notebook final, redactar la memoria y la presentación. Aprobación del tutor antes del
**16 de octubre de 2026**.
