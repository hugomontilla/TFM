# Bitácora de decisiones — TFM Airbnb (Nueva York)

Decisiones vigentes del proyecto, tomadas por consenso entre el alumno y el tutor/analista. Alimenta la
**Documentación para analistas** de la memoria. Estructura del proyecto: `docs/arquitectura.md`. El detalle
histórico (versión de 136 líneas, con el análisis multimercado y las decisiones superadas) está en el historial de
git de este fichero.

---

## 1. Antecedentes (resumen)

El TFM empezó con las 10 ciudades (~280.000 anuncios). Casi todo lo aprendido apuntaba a lo mismo:

- **La lectura marginal engaña.** La densidad por dormitorio, el volumen de reseñas y `minimum_nights` cambiaban de
  signo al condicionar. Desde entonces cada relación se mide dentro de un mismo producto
  (`room_type` × capacidad) y con estadísticos robustos (mediana, Kruskal-Wallis, Spearman).
- **Nada que derive del precio o que no exista al publicar entra al modelo.** De ahí: no se imputa el target, no hay
  `privacy_premium`, las reseñas no son *features* y el split se agrupa por anfitrión.
- **La ciudad explicaba el 22% de la varianza del log-precio** y varios efectos cambiaban de signo entre mercados. Se
  acotó el TFM a Nueva York (09-16), lo que eliminó la conversión de divisas y la geocodificación, y recuperó
  `district` (completo en NY).

Decisiones del periodo multimercado que ya no aplican (modelo global con `city`, `reverse_geocoder`, eliminar
`district`) están descartadas y no se listan.

---

## 2. Decisiones vigentes

### Alcance y diseño experimental

| Fecha | Decisión | Por qué (y alternativas descartadas) |
|---|---|---|
| 09-16 | **Un solo mercado: Nueva York.** | Precio en USD y único mercado con `district` completo. Coste: 220 barrios con cola larga. Descartado: un modelo por ciudad; añadir París como prueba de transferencia (queda como ampliación opcional). |
| 09-26 | **Orden global → partición → train.** Cualquier decisión que mire el precio se toma solo con train. Dos notebooks: `1-EXPLORACION` (global y partición) y `2-ANALISIS` (solo lee el train). | Permite afirmar ante el tribunal que ninguna decisión que depende del precio ha visto el test. La barrera es física, no de disciplina. |
| 09-26 | **Partición agrupada por `host_id` y estratificada por `room_type` × `district`**, 20%, semilla 42, guardada en `data/particion/` con huellas MD5 (no se sobrescribe sin `--forzar`). Resultado: 29.586 / 7.398 anuncios en la partición; tras la regla de bloqueo, 29.538 / 7.384 (los que usan los modelos). | Agrupar: el 38% de los anuncios está en carteras de más de uno y el 23,7% tiene un "gemelo". Estratificar por las dos variables: en 50 particiones simuladas, sin estrato el test llegaba a tener el doble de habitaciones de hotel; con el estrato combinado la desviación queda por debajo de 0,1 puntos. La semilla sola no garantiza el resultado, por eso los ficheros son la fuente de verdad. |
| 09-25 | **CV de 5 folds × 3 repeticiones** (`StratifiedGroupKFold`, mismo estrato), guardada en `outputs/folds_cv.csv`; la primera repetición sirve para comparar modelos y ajustar hiperparámetros. | Hay que separar diferencias de ~0,005 de RMSE en log; los mismos folds hacen pareadas las comparaciones. |
| 09-23 | **Target `log(price)`.** Métricas: RMSE en log para comparar modelos; MAE/MAPE en dólares para el sponsor. | La asimetría baja de 13 (bruto) a 0,6 (log). `exp(ŷ)` estima la **mediana**, que es lo que interesa para recomendar un precio. |
| 09-25 | **Escenario de predicción: B principal** (anuncio nuevo de cualquier anfitrión, con su perfil) **y A como contraste** (sin el bloque del perfil). | La diferencia A–B mide cuánto vale conocer al anfitrión. Se descarta valorar anuncios existentes con reseñas (causalidad inversa). Limitación: el perfil se mide en 2021, no al publicar. |
| 09-25 | **El preview de ablaciones del 09-23 no cuenta como evidencia**. La selección de variables se rehace con protocolo fijado de antemano, solo con train. | Se hizo con todos los datos, sin test separado y fuera del repositorio. |

### Limpieza y preparación de datos

| Fecha | Decisión | Por qué |
|---|---|---|
| 09-09 / 09-05 | **Eliminar anuncios con `price` nulo o 0 y con `accommodates == 0`** (15 y 13 en NY). | Un precio nulo no es una observación; imputar el target fabrica filas fáciles. Los de capacidad 0 son fichas de hotel sin inventario. |
| 09-26 | **Precios de bloqueo:** fuera `price` ≥ $9.999 y habitaciones privadas/compartidas > $1.000. Umbrales fijados en train y aplicados congelados al test (48 + 14 anuncios). El P99 global ($850) queda solo para el análisis descriptivo. | El P99 recortaría el lujo legítimo (18,8% de los pisos para 9+ huéspedes). Aplicarlo al test es coherente con el objetivo (precios de mercado), pero en producción el precio no se conoce: se declara como limitación. |
| 09-26 | **`PreparadorListings` con `fit`/`transform`**, ajustado solo con train y congelado en `data/modelado/preparador.json`. Aprende la ventana de amenities, los tipos de propiedad frecuentes y las medianas de `bedrooms`. | Elimina la fuga de ajustar la imputación y la ventana de amenities con el futuro test. Reproduce el dataset y permite preparar *snapshots* nuevos. |
| 09-05 / 09-26 | **`bedrooms` se imputa con la mediana de `room_type` × `accommodates`** (solo train). **`es_estudio`** recoge los nulos informativos. | `accommodates` es el mejor predictor de dormitorios. |
| 08-28 | **No imputar `review_scores_*`.** | Sin reseñas no hay puntuación; inventarla sesga. |
| 09-09 | **`host_ambito`** (misma ciudad / país / extranjero / desconocido) sin sobrescribir `host_location`. | Captura al anfitrión absentista. |
| 09-30 | **Amenities: binarias con presencia entre 3,5% y 97% en train (56 variables), sin las de lujo.** | Criterio ciego al precio. Las de lujo tenían tan pocos anuncios (`ski-in/ski-out`: 8) que no se pueden interpretar. Descartado: top-N, selección por lift, `n_amenities` (no aporta señal propia). |
| 09-26 | **Se conservan los 6 anuncios idénticos salvo `listing_id`**. | Son unidades distintas del mismo edificio; al agrupar por anfitrión caen siempre en el mismo lado. |

### Criterios de análisis (EDA)

| Fecha | Decisión | Por qué |
|---|---|---|
| 09-05 | **Mediana en lugar de media**, y ocultar celdas con N < 30. | La media sobreestimaba la mediana un 46% típico (hasta 218%). |
| 09-04 | **Post-hoc con Mann-Whitney + Holm; tendencias con Spearman.** | Es lo visto en el máster y defendible ante el tribunal (descartados Dunn y Jonckheere-Terpstra). |
| 09-24 | **`accom_cat` (7+) y `bedrooms_cat` (5+) solo para tablas y gráficos**; el modelo usa las numéricas. | Agrupar da legibilidad pero destruye la señal de la cola que aprovechan los árboles. |
| 09-09 | **Las reseñas se analizan pero no son *features*.** El sesgo "sin reseña" se mide con `n_reviews == 0`, sin variable de corrección. | Son posteriores a la publicación (fuga). En NY los pisos enteros sin reseña son un 11% más caros: se cita como limitación. |
| 09-26 | **Resultado del EDA en train (repetido solo con train):** ninguna decisión cambia de sentido. Aparece señal en `instant_bookable` (×0,93 condicionado). | Tabla en `2-ANALISIS` sección 12. |

### Selección de variables

| Fecha | Decisión | Por qué |
|---|---|---|
| 09-27 | **Regla D4, fijada antes de ejecutar:** una variable o bloque entra si Δ = RMSE_sin − RMSE_con > 0 en ≥ 12 de 15 estimaciones pareadas **y** el Δ medio supera su desviación típica. Heurística declarada como tal; el p de Nadeau-Bengio es informativo. | Regla explicable con lo visto en el máster. Limitación: los 15 Δ no son independientes. |
| 09-23 → 09-27 | **Ubicación:** lat/lon + `dist_centro_km` (City Hall) + `district` + barrio, codificado con *target encoding* dentro del `Pipeline`. | Lat/lon es lo que más aporta; los demás niveles se mantienen por interpretabilidad. |
| 09-23 | **`n_anuncios_ny`** (otros anuncios del anfitrión en NY) entra; `host_total_listings_count` solo para perfilar. | Mejora 0,006 de RMSE en log; los operadores profesionales fijan precios de otra forma. |
| 09-23 | **`minimum_nights` en bruto**, recortada a 365 (`log1p` solo en el lineal). | Aporta poco, pero en bruto es la mejor. La regla de 30 noches (64% de anuncios) no cambia el precio por noche. |
| 09-25 / 09-26 | **`maximum_nights` fuera**. | El efecto era composición; con la evidencia rehecha en train el signo no es consistente entre estratos. **Pendiente de confirmación del alumno.** |
| 09-27 | **Sin variables del título (`name`)**; `reputacion_cartera` candidata en el escenario B con indicador de nulo. | Calendario (el título queda como línea futura); usa reseñas de otros anuncios del anfitrión, que existen al publicar. |
| 10-01 | **Variables por familia:** lineal y kNN usan las 71 de la selección con Ridge; árboles, RF, HGB, CatBoost, cuantílica y SVR usan las 79. Sin reselección por modelo. | Quitar variables a los árboles ajustados empeora (Δ ≈ 0,005, 5/5 folds); en el lineal las 79 no ganan de forma significativa. Los *wrappers* costarían miles de ajustes y exigirían validación anidada. |

### Modelado

| Fecha | Decisión | Por qué |
|---|---|---|
| 09-27 | **Escalera de complejidad:** referencias ingenuas (medianas) → lineal (OLS, Ridge/Lasso/ElasticNet) → kNN y CART → RF y HGB → opcionales: cuantílica, SVR, CatBoost. Cada búsqueda de hiperparámetros va junto a su modelo. | Cada peldaño responde a una pregunta (cuánto aporta el producto, la ubicación, la no linealidad). |
| 09-30 | **kNN con PCA: nº de componentes = el mínimo que explica ≥ 80% de la varianza**, fijado antes de mirar el RMSE; se añaden FAMD y PCAmix para datos mixtos. | PCA trata las binarias como numéricas; FAMD/PCAmix son su versión para datos mixtos. |
| 09-30 | **CatBoost como modelo opcional** con categóricas nativas, y **MLP aplazado** (código en `Entrenamiento Google Colab/`). | CatBoost aporta una tercera forma de codificar el barrio; XGBoost sería el mismo algoritmo que HGB. En datos tabulares no se espera que el MLP gane. |
| 10-01 | **Importancia de variables por permutación** sobre validación (bloques en todos los modelos; variable a variable en el elegido); SHAP solo en el modelo final. | Evita la importancia por impureza, sesgada con variables correlacionadas. |
| 10-02 | **Recorte de cómputo** (de ~8 h a ~5 h): la búsqueda de CatBoost pasa de 40 a **10 candidatos**, se quita `alpha = 1e-5` en Lasso/ElasticNet y la ablación solo se calcula con Ridge. | El óptimo de CatBoost era plano (diferencias menores que la sd entre folds) y elegir el mejor entre muchos candidatos con los mismos folds sobreajusta la validación. Los 10 primeros candidatos ya estaban calculados y el mejor (`C8_catboost_08`) es el modelo ajustado, así que el modelo final no cambia y esos `alpha` no convergían. **Se declara que se decidió viendo resultados parciales.** |
| 10-02 | **CatBoost: el resultado oficial es el de CPU**; Colab (GPU) repite la búsqueda con otros nombres como prueba de velocidad, sin mezclarse. | La GPU no reproduce exactamente la CPU (otra discretización y otro *bootstrap*). |
| 10-05 | **Parámetros comunes en `.env`** (semilla, folds, repeticiones de CV, cobertura del intervalo, réplicas del bootstrap), leídos por `scripts/config.py`. El acceso a `outputs/experimentos.csv` se concentra en `scripts/registro.py`. | Una sola definición de cada valor, en vez de repetirlo en varios scripts. Cambiar la semilla o los folds invalida el registro y los folds guardados, y obliga a recalcular todo. |
| 10-05 | **La importancia por variable de `3-MODELADO` (sección 12) deja de agrupar la ubicación bajo «coordenadas».** El grupo debía ser solo `latitude` y `longitude`, pero la regla de selección modificaba la lista compartida y le añadía `dist_centro_km`, `district` y `neighbourhood`. Corregido y reejecutado. | Con el error, la importancia de «coordenadas» en CatBoost salía de 0,108; con solo latitud y longitud es de 0,021, y el resto de variables de ubicación aparecen por separado. Los resultados de los notebooks finales no se ven afectados (ya agrupaban solo latitud y longitud). |

### Modelo final y test

| Fecha | Decisión | Por qué |
|---|---|---|
| 09-30 | **Regla de elección:** menor RMSE medio de validación (5 folds), escenario B, fijada antes de ver la tabla. Si el segundo queda a menos de 1 SE, la ventaja se declara no concluyente. | Más simple que la regla 1-SE, que solo se conserva para la poda del árbol. |
| 10-02 | **Modelo final: HGB**, no CatBoost (desviación consciente de la regla, que daba CatBoost por 0,0031 de RMSE, p = 0,097). | La ventaja es menor que la sd entre folds (~0,008) y HGB se ajusta ~16 veces más rápido. **Decisión posterior a ver la tabla de CV:** se presenta como excepción justificada, no como aplicación de la regla. CatBoost se reporta como sensibilidad. |
| 10-02 | **El test se abre una sola vez**, tras comprobar su MD5, y solo mide el modelo elegido. Los contrastes (lineal, CatBoost, N0.3) se cuentan con la CV pareada. | Elegir con el test lo invalidaría. |
| 10-02 | **Intervalo con CQR** (*conformalized quantile regression*): cuantiles 0,1 y 0,9 del propio HGB, ajustados con el train del fold 0 (80%) y calibrados con su validación; bootstrap por anfitrión para los IC. | Cobertura exacta en muestra finita, ~3 ajustes en lugar de ~13. Limitación: intercambiabilidad solo aproximada (anuncios correlacionados por anfitrión). |
| 10-02 | **Se guardan modelo, cuantiles, ficha y `predecir_precio`** en `models/`. | Limitación: `predecir_precio` necesita la cartera completa del anfitrión (`n_anuncios_ny`). |
| 10-05 | **Un único notebook final (`4-MODELO_FINAL`) con los dos finalistas**, CatBoost y HGB, con un subapartado por modelo en cada sección y una comparación al final. El test se evalúa en una sola ejecución para los dos y no se usa para elegir. CatBoost se evalúa con el `CatBoost_ajustado` vigente del registro (el mejor de los 10 candidatos). | Evita mantener dos notebooks casi idénticos. Resultado: RMSE en log de test de 0,429 (CatBoost) y 0,424 (HGB); la diferencia pareada (−0,0053, IC 95% −0,017 a +0,003) no es concluyente. |

---

## 3. Pendiente de cerrar

- Confirmar con el alumno la exclusión de `maximum_nights`.
- Tutor: aprobación antes del **2026-10-16**.
- Decidir con la tutora si se presenta un solo modelo final (HGB por coste, CatBoost por explicabilidad con SHAP) o los dos.
