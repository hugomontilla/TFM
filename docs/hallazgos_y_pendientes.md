# Hallazgos y pendientes — TFM Airbnb (Nueva York)

Hallazgos de calidad de datos, riesgos metodológicos e ideas pendientes. Se conservan los
identificadores H-xxx porque los notebooks los citan. La versión completa, con el detalle del análisis
multimercado, está en el historial de git de este fichero.

**Estado:** 🟡 abierto · 🟢 cerrado · ⚪ obsoleto por el cambio a Nueva York

---

## 1. Abiertos: decidir antes de modelar

Tutor: aprobación antes del **2026-10-16**. Ya no queda ningún bloqueante del conjunto de *features*; H-023 se decide por ablación en el modelado.

| ID | Hallazgo | Qué hay que decidir |
|---|---|---|
| **H-023** | **Los `Entire place` sin `bedrooms` son estudios** (el 70% lleva *studio* en el título) y cuestan un 8%-21% menos que un piso de un dormitorio con la misma capacidad (en train, 09-26: −12% con 2 huéspedes, −10% a −27% entre 2 y 4). La imputación borra esa diferencia. Además, la fuente nunca usa `bedrooms == 0`: otros 679 estudios aparecen con 1 dormitorio. | **Aplazado (09-23):** no bloquea el modelo base. Se decidirá por ablación en el modelado (con y sin `es_estudio`, definido como nulo o *studio* en el título). Solo entra si mejora el CV. |
| **H-026** | **Los pisos enteros sin reseñas son más caros.** Con todos los datos se leyó como un 15% en los cinco distritos; **rehecho en train (09-26): un 11% (×1,11), en cuatro de los cinco** (el Bronx, ×0,98, no). | No bloquea el modelo mientras no se filtre por reseñas. Debe citarse como limitación de cualquier conclusión basada en `review_scores_*`. |
| **H-021** | **Recorte del plan de EDA.** El notebook de NY ya cubre el sesgo sin reseña, la profesionalización del anfitrión y la cardinalidad de barrios. Quedan sin hacer: `review_scores_*` multidimensional (§2.7) e `instant_bookable` (§4.2). | 🟢 **Cerrado (09-26, R6):** `review_scores_*` se recorta porque está fuera por principio. `instant_bookable` se ve con una tabla condicionada en el análisis en train (candidata A6). |
| **H-030** | **(09-25) Las ablaciones rápidas del 09-23 no son evidencia:** se hicieron fuera del repositorio, con todos los datos y sin test separado. Varias decisiones del 09-23 citan sus cifras (ubicación, `n_anuncios_ny`, `reputacion_cartera`, `minimum_nights`). | Todas esas variables vuelven a ser candidatas y se deciden con el protocolo de `docs/plan_modelado.md` (§0.1, §5). |
| **H-031** | **(09-25) `host_identity_verified` no es casi constante:** 80% `t` / 20% `f` en NY. La bitácora (09-23) la excluía por serlo; solo `host_has_profile_pic` lo es (99,7%). | Reabierta como candidata (plan, §2). |
| **H-032** | **(09-25) Criterio incoherente sobre el perfil del anfitrión:** `host_is_superhost` entra, pero las tasas de respuesta y aceptación se excluyeron por ser "actividad posterior", cuando el superhost se calcula con esa misma actividad. | Fijar el escenario de predicción (plan, D1): o el perfil entra entero o no entra. |
| **H-033** | **(09-25) Preprocesado antes del split:** la imputación de `bedrooms` usa medianas calculadas con todo el dataset, incluido el futuro test. El efecto es mínimo, pero es una fuga. | 🟢 **Cerrado (09-26, R5):** resuelto sin fuga. Las medianas se aprenden solo con el train en `PreparadorListings` (H-036). Supera el cierre del 09-25 (mantener la imputación del EDA como limitación). |
| **H-034** | **(09-25) `maximum_nights` se excluyó sin justificación de datos** ("utilidad dudosa y calendario"). El 56% deja el valor por defecto (1.125). | 🟢 **Cerrado (09-25):** fuera, con evidencia del EDA (Nivel 3.4). El ×1,09 dentro de cada producto baja a ×1,016 al condicionar por district y `estancia_min_30`, con ratio > 1 en 33 de 62 estratos: era composición. **Rehecho en train (09-26): ×1,038, 33 de 58 estratos, Manhattan ×1,10 y Queens ×0,96.** Se mantiene fuera porque el signo no es consistente, pero la evidencia es más débil: 🟡 pendiente de que el alumno lo confirme. |
| **H-035** | **(09-26) El EDA se hizo con todo el dataset, incluido el futuro test** (el split es del 09-25). Casi todas las reglas son ciegas al precio, pero algunas decisiones sobre qué variables entran se tomaron mirando el precio: `es_estudio`, la exclusión de `maximum_nights` y de `n_amenities`, y los umbrales de bloqueo. | 🟢 **Cerrado (09-26):** EDA rehecho en `Exploracion.ipynb` (sin precio) y `Analisis.ipynb` (solo train). Ninguna decisión cambia de sentido; tabla de cambios en `Analisis` §13. |
| **H-036** | **(09-26) `PreparadorListings` se ajusta con todos los anuncios, antes del split:** la ventana de amenities, los `property_type` frecuentes y las medianas de `bedrooms` se aprenden también del futuro test (`preparar_dataset_modelado`). El docstring dice lo contrario. | 🟢 **Cerrado (09-26):** `preparar_dataset_modelado(train, test, fecha)` ajusta con el train de la partición fija y exporta `listings_ny_modelado_train.parquet` y `_test.parquet`. Cambios frente al ajuste con todo: `Private room in guest suite` sale de los tipos frecuentes (no llega a 100 en train) y cambian 3 medianas de `bedrooms` (`Entire place` con 4 huéspedes: de 2 a 1). La ventana de amenities no cambia (46). |
| **H-037** | **(09-26) Split sin estratificar o estratificado por una sola variable desequilibra el test** por la concentración por anfitrión: en 50 particiones simuladas, sin estratificar hay hasta el doble de habitaciones de hotel y 4,5 puntos de desviación por distrito; con `room_type` × `district` todo queda por debajo de 0,1 puntos. Además, el plan (§3.2) dice `GroupKFold` y el código usa `StratifiedGroupKFold` por `room_type`. | 🟢 **Cerrado (09-26, R2):** partición y folds con estrato `room_type` × `district` (`particion_datos.py`, `func_modelado.generar_folds`). |
| **H-038** | **(09-26) 6 anuncios son idénticos salvo el `listing_id`.** | 🟢 **Cerrado (09-26):** 11 anuncios de dos operadores corporativos (6 y 66 anuncios en NY): unidades distintas del mismo edificio. Se conservan; al agrupar por anfitrión caen en el mismo lado. |
| **H-039** | **(09-26) `instant_bookable` se asocia a un precio más bajo, de forma consistente:** ×0,93 condicionado por producto, zona y estancia mínima, con solo 10 de 50 estratos por encima de 1 (`Analisis` §8.3). El 96,7% de las habitaciones de hotel la tienen activada. | Candidata A6 con señal a priori. Se decide en la ablación. |
| **H-040** | **(09-27) El `TargetEncoder` del barrio hace su *cross-fitting* interno por filas, no por anfitrión.** Dentro del train de cada fold, la codificación de un anuncio puede incluir el precio de su "gemelo" de cartera (23,7% de los anuncios lo tiene). `TargetEncoder.fit_transform` no admite `groups`. El efecto es pequeño (la media de un barrio se diluye entre muchos anuncios y hay suavizado), pero el modelo puede fiarse algo de más de la codificación. La validación no se ve afectada: el fold de validación nunca entra en el ajuste del codificador. | Declarar como limitación. Si el barrio resulta decisivo, alternativa: codificación propia con `GroupKFold` o la categórica nativa de HGB (Fase 6.3). |
| **H-041** | **(09-27) La OLS con *splines* (L4) es numéricamente inestable:** la matriz de diseño tiene rango incompleto (164 columnas, rango 153: la base de *splines* suma 1 y el one-hot de `room_type` × `district` repite los efectos principales). Fuera del rango del train (dos anuncios de Bellerose con longitud −73,711) predice $751.000. RMSE 0,452 ± 0,030 frente a 0,433 ± 0,009 con Ridge. | 🟢 **Cerrado (09-27):** la referencia lineal de la Fase 4 es Ridge (como ya decía el plan). Se deja registrado como argumento para la regularización. |
| **H-042** | **(09-27) Dos redundancias exactas entre candidatas:** `property_type_grp` anida `room_type` (cada tipo pertenece a un solo `room_type`, salvo "Otros"), y el nulo de `host_response_rate` coincide con `host_response_time = 'sin_dato'` (14.749 anuncios). En la OLS de inferencia hacen arbitrarios los coeficientes (habitación de hotel +0% con p = 1; nulo de respuesta +126%). | 🟢 **Cerrado (09-27):** la OLS interpretable quita `property_type_grp` y el indicador redundante. Los modelos predictivos no se ven afectados; A2 decide si `property_type_grp` aporta. |
| **H-043** | 🟢 **Superado por H-044 (10-01).** **(09-30) La Fase 4 selecciona con HGB por defecto y §10.1 reconfirma con HGB ajustado: doble trabajo.** Diseño alternativo propuesto por el alumno, pendiente de ver si es factible (coste estimado: 2-3 h de cálculo). **Tarea para reproducirlo en una sección aparte de `Modelado .ipynb` (sin borrar la actual hasta comparar):** (1) **Ridge** decide las variables de la familia lineal con la ablación actual (25 configuraciones, 15 folds, regla D4); ya está calculado. (2) **Ajustar HGB con todas las candidatas** (`COLS_B`): misma búsqueda que §10 (`buscar('T8_hgb_completo', lambda n, p: construir_hgb(n, p, COLS_B), ESPACIO_HGB, n_iter)`), con `FOLDS_MODELOS`. No es circular: los hiperparámetros no dependen de la selección. (3) **Una única ablación con ese HGB ajustado:** las mismas `configuraciones` y `pruebas` de §7, con `construir_hgb(f'T8_hgb_{nombre}', PARAMS_HGB_COMPLETO, columnas)` y `evaluar_registrado`; decidir 15 folds (`FOLDS_SELECCION`, ≈ 2 h, regla 12/15) o 5 (`FOLDS_MODELOS`, ≈ 40 min, regla 4/5, más gruesa con Δ ≈ 0,001-0,003). (4) `seleccionar('HGB')` sobre la nueva tabla da `VARS_HGB`; sustituye a la ablación con HGB por defecto y a §10.1. (5) **Comparar** el conjunto resultante con el actual: si coincide, basta con citarlo como robustez; si cambia, decidir cuál se queda y registrarlo en la bitácora. | Decidir si se hace (calendario: el test se usa el 8 oct). En la memoria, justificar el cambio de protocolo como simplificación, no como respuesta a los resultados. |
| **H-044** | **(10-01) Seleccionar variables antes perjudica a los árboles ajustados.** RF y HGB ajustados sobre las 68 variables de la selección con HGB por defecto pierden frente a los mismos hiperparámetros con las 79: Δ = 0,0045 (RF) y 0,0056 (HGB), 5/5 folds, p de Nadeau-Bengio 0,009 y 0,022 (`Modelado` §9.4). En el lineal las 79 no ganan de forma significativa (Δ = 0,0049, 4/5 folds, p = 0,12). | 🟢 **Cerrado (10-01):** árboles, ensembles, cuantílica y SVR con las 79; lineal y kNN con las 71 de Ridge (bitácora, 10-01). **Supera H-043:** se adopta su punto (2), ajustar HGB con todas las candidatas; la ablación con el HGB ajustado no se repite. |
| **H-045** | **(10-01) `prince.FAMD` transforma los datos nuevos con las proporciones del lote que recibe, no con las del ajuste** (`row_coordinates` recalcula `p` de cada modalidad). La coordenada de un anuncio depende de los demás anuncios que se transformen con él; con un solo anuncio, las modalidades ausentes dan 0/0. En el train FAMD y PCAmix coinciden exactamente (mismas componentes y autovalores). | 🟢 **Cerrado (10-01):** se implementa `func_modelado.PCAmix`, que usa las proporciones del train, y se compara con FAMD (K3 frente a K4, §8.1). Para predecir anuncios uno a uno solo vale PCAmix. |
| **H-046** | **(10-01) Importancia de variables adelantada** (D7 de `plan_modelado.md` la dejaba para la Fase 10): permutación sobre validación por bloques en todos los modelos y variable a variable en el elegido y el lineal (`Modelado` §12). | 🟡 **Pendiente: SHAP** del modelo final en la Fase 10 (requiere instalar `shap`). |
| **H-029** | **Coherencia de la documentación:** `EDA.ipynb` mezcla dos ventanas de amenities (4%-85% en código y 5%-80% en texto). Además, el heatmap de NY oculta celdas con N < 20, mientras que el criterio acordado es N < 30. **(09-26) En `Analisis` el heatmap ya usa N < 30;** queda `EDA.ipynb`. | Alinear código, texto y bitácora. |

---

## 2. Cerrados: lecciones que se aplican en Nueva York

### Lo marginal engaña: condicionar siempre por producto

| ID | Hallazgo | Consecuencia |
|---|---|---|
| **H-013** | `guests_per_bedroom` correlaciona en positivo con el precio, pero a capacidad fija la relación se invierte (−48% en NY con 6 huéspedes). | Solo se usa junto a `accommodates`. |
| **H-015** | El volumen de reseñas parece no relacionarse con el precio, pero dentro de los pisos enteros el gradiente es negativo (ρ = −0,14 en NY): mide rotación, no calidad. | `n_reviews` no es *feature* (todo anuncio nuevo tiene 0). |
| **H-017** | La correlación positiva de `minimum_nights` no se debía a `room_type`. En NY, el aparente sobreprecio era composición por district. | Ver H-025. |
| **H-022** | Los anuncios sin reseña no son aleatorios; la mayor asociación (superhost) es mecánica. | Sin *feature* de corrección. En NY se matiza con H-026. |

### El target y la fuga de información

| ID | Hallazgo | Consecuencia |
|---|---|---|
| **H-024** | Precios de bloqueo: 11 anuncios a $9.999-$10.000 y habitaciones a más de $1.000/noche. El P99 global recortaría también el lujo legítimo (el 16,7% de los pisos para 9 o más huéspedes). | Regla de exclusión: `price` ≥ $9.999 o habitación a más de $1.000 (62 anuncios). Target `log(price)`. Sensibilidad frente al P99. |
| **H-028** | La ubicación opera en tres niveles (distancia, district, barrio), pero con lat/lon en el modelo ni la distancia, ni el district, ni el barrio mejoran el RMSE (ablación rápida, 09-23). | Entran los cuatro; el barrio es provisional y se retira si no aporta en el modelado. |
| **H-025** | El 64% de los anuncios exige 30 noches (*Multiple Dwelling Law*), sin efecto sobre el precio por noche. `minimum_nights` apenas aporta en la ablación. | Se mantiene en bruto; para el modelo lineal se recorta a 365 y se le aplica `log1p`. |
| **H-027** | La reputación de cartera tiene señal marginal en NY (ρ = +0,15), pero no mejora el modelo. En cambio, el tamaño de la cartera sí mejora (0,006 de RMSE en log, consistente entre folds). | No se construye `reputacion_cartera`. Entra `n_anuncios_ny`. |
| **H-008** | `privacy_premium` se calcula con el propio precio. | No se construye. |
| **H-009 / H-002** | Las filas con `accommodates == 0` tienen todas `price == 0` y se descartaban por accidente. | Se eliminan de forma explícita (13 en NY). |
| **H-020** | El 38,1% de los 36.922 anuncios pertenece a anfitriones con varias propiedades, casi duplicados entre sí. | Split agrupado por `host_id`. |
| **H-014** | Una media de puntuación por tramo salía de un único anuncio. | Toda tabla agregada lleva su `n` y enmascara las celdas pequeñas. |

### Variables y calidad de datos

| ID | Hallazgo | Consecuencia |
|---|---|---|
| **H-004** | `maximum_nights` tiene el centinela `INT32_MAX` y el valor por defecto 1125. | Excluida. |
| **H-019** | `minimum_nights` tiene un centinela de 9999 (408 filas ≥ 365 en el global, 56 en NY). | Sin limpieza: no altera los resultados. |
| **H-010** | La media sobreestimaba la mediana hasta un 218%. | Mediana en todas las tablas. |
| **H-011** | La cola de capacidad no es homogénea. | Categorías agrupadas solo para visualizar. |
| **H-012** | `accommodates` y `bedrooms` están asociadas (ρ = 0,63 en NY). | Se conservan ambas; interpretar con SHAP. |
| **H-016** | `review_scores_location` está saturada (75% de dieces). | Gráfico retirado; las puntuaciones no discriminan precio. |
| **H-005** | `host_total_listings_count` tiene una cola extrema (máximo 2.739 en NY). | Resuelto en NY con tramos 0-1 / 2-4 / 5-20 / 20+ en la sección de superhost. |

---

## 3. Obsoletos por el cambio a Nueva York

| ID | Hallazgo original | Por qué ya no aplica |
|---|---|---|
| H-001 | `price` en diez divisas distintas. | NY está en USD. |
| H-003 | Se perdía el district al eliminar la columna. | En NY `district` está completa y se usa. |
| H-006 / H-007 | `reverse_geocoder` partía Roma en "Vaticano" y movía anuncios de Hong Kong a China. | Sin geocodificación en un solo mercado. |
| H-018 | Reputación de cartera sin señal en el global. | Reabierto en NY como H-027. |
