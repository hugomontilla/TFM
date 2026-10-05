# Plan de modelado — TFM Airbnb (Nueva York)

**Estado (2026-09-26):** Fases 0 y 1 **rehechas** tras la reestructuración del EDA. Decisiones cerradas: D1, D2 (rev.), D3 (rev.) y D5 (rev.); **el 09-27, D4, D6, D7, D8 y D9** (sección 14).
Cuando una decisión se cierre, se anota en `bitacora_decisiones.md` y aquí se marca como cerrada.

> **09-26 — Reestructuración del EDA aplicada** (`docs/plan_reestructuracion_eda.md`). La partición es fija y
> previa al análisis del precio (`data/particion/`, estrato `room_type` × `district`); el EDA con precio se rehízo
> solo con el train (`Analisis.ipynb`); el preparador se ajusta con el train y exporta
> `listings_ny_modelado_train.parquet` y `_test.parquet`. La partición anterior y N0.1 están archivados en
> `outputs/archivo_2026-09-25/` (SUPERSEDED). Las cifras de las secciones 1 y 2 que citan el EDA original se
> mantienen como registro; las de train están en `Analisis` §13.

**Objetivo del modelado:** predecir `log(price)` de un anuncio de Nueva York con la información disponible
al publicarlo, subiendo de los modelos más simples a los más complejos. Cada escalón tiene que responder
una pregunta (¿qué señal es aditiva?, ¿cuánto aporta la no linealidad?, ¿qué variables aportan de verdad?),
y la elección final tiene que poder defenderse ante el tribunal con evidencia medida del mismo modo para
todos los modelos.

---

## 0. Reglas del juego (fijadas antes de ver ningún resultado)

### 0.1 El preview no cuenta como evidencia

Las ablaciones rápidas del 2026-09-23 se hicieron fuera del repositorio, con todos los datos y sin test
separado. **No justifican ninguna decisión.** Toda variable cuya entrada o salida dependía de esas cifras
vuelve a ser candidata y se decide en la Fase 4 con el protocolo de la sección 5.

| Decisión de la bitácora | Qué usaba del preview | Nuevo estado |
|---|---|---|
| 09-23 · Ubicación: lat/lon + `dist_centro_km` + `district` + barrio (el barrio, provisional) | RMSE en log de 0,523 a 0,463; el resto ±0,005 | Las cuatro son **candidatas** (bloque B1) |
| 09-23 · Entra `n_anuncios_ny` | Mejora de 0,006 en 4 de 5 folds | **Candidata** (bloque B2) |
| 09-23 · No se construye `reputacion_cartera` (H-027) | Mejora de 0,0002-0,0019 | **Reabierta:** candidata en el escenario B (D1) |
| 09-23 · `minimum_nights` en bruto (H-025) | 0,4655 / 0,4634 / 0,4643 | **Reabierta:** tres formas candidatas |
| 09-23 · Fuera tasas de respuesta y aceptación, verificación y foto | No usaba el preview, pero **la verificación no es casi constante** (80% / 20% en NY) | Tasas y verificación, **candidatas** en B; la foto sigue fuera (99,7% `t`) |
| 09-12 · Excluir `maximum_nights` | No usaba el preview; se justificó por "utilidad dudosa y calendario" | **Sin justificar:** candidata en forma de indicador |

Las decisiones que se apoyan en el EDA o en un principio (fuga, constancia, cardinalidad) siguen vigentes
(inventario completo en la sección 2).

### 0.2 Principios metodológicos

1. **Test intocable.** Se separa el 20% una sola vez, antes de cualquier experimento, y se evalúa una sola
   vez, con el modelo final. Nada de lo que se decide (variables, hiperparámetros, familia) mira el test.
2. **Agrupado por anfitrión.** Toda partición (test y folds) agrupa por `host_id` (H-020).
3. **Las mismas particiones para todos los modelos.** Los folds de CV se generan una vez, con semilla
   fija, y se guardan. Así las comparaciones son pareadas fold a fold.
4. **Todo lo que aprende de los datos va dentro del `Pipeline`:** codificación, escalado, imputación y
   recortes. Nada se ajusta antes del split.
5. **La regla de decisión se fija antes de mirar** (sección 5.3). No se cambia a la vista de los
   resultados.
6. **Parsimonia.** Una variable que no supera la regla sale del modelo final, aunque sea interpretable.
   Para explicar el modelo al sponsor se usan PDP y SHAP, no variables de adorno.
7. **Todo experimento queda registrado** en `outputs/experimentos.csv`, falle o no. Una tabla solo con los
   que salieron bien sería *cherry picking*.

### 0.3 Qué se conoce en el momento de predecir (decisión D1)

La regla "nada que no exista al publicar el anuncio" no basta, porque depende de **quién** publica:

| Escenario | Caso de uso | Qué variables son legítimas |
|---|---|---|
| **A. Anfitrión nuevo** (*cold start*) | Alguien sin historial publica su primer anuncio | Producto, ubicación, condiciones (`minimum_nights`, `maximum_nights`, `instant_bookable`), amenities, título |
| **B. Cualquier anfitrión** | Un anuncio nuevo de un anfitrión que puede tener historial | Lo de A + perfil del anfitrión: antigüedad, cartera, superhost, verificación, tasas de respuesta y aceptación, reputación de cartera |
| C. Valorar un anuncio existente | "¿Mi anuncio está caro?" | Lo de B + reseñas propias. **Descartado:** causalidad inversa (el precio influye en `review_scores_value`) y H-026 |

**Incoherencia que hay que resolver:** el dataset exportado incluye `host_is_superhost`, que Airbnb
calcula con la tasa de respuesta y las reseñas, pero excluye las tasas de respuesta "por ser posteriores a
la publicación". O el perfil del anfitrión entra entero (escenario B) o no entra (escenario A).

**Limitación en cualquier escenario:** el dataset es una foto de 2021. El superhost, la antigüedad y las
tasas se miden en la fecha de extracción, no el día en que se publicó cada anuncio. Hay que declararlo.

---

## 1. Fase 0 — Reconstruir el dataset de modelado

**✅ Rehecha el 2026-09-26.** `python scripts/preparacion_datos.py` ajusta el preparador solo con el
train de la partición fija y exporta `listings_ny_modelado_train.parquet` (29.538 × 84) y
`listings_ny_modelado_test.parquet` (7.384 × 84). *Versión original del 09-25, archivada en
`data/archivo_2026-09-25/`:* la sección 6 de `EDA_NY.ipynb` exportaba `listings_ny_modelado.parquet`
(36.922 × 84: 81 candidatas + `listing_id`, `host_id`, `price`) y `bloques_modelado.json` (bloques de la
sección 2). Las 76 columnas anteriores no cambian ni de filas ni de valores. Los indicadores de nulo no se
exportan: los crea el `Pipeline` (`SimpleImputer(add_indicator=True)`). Los 18 anuncios sin perfil de
anfitrión tienen `host_identity_verified = False`, igual que `host_is_superhost`, y las numéricas quedan
nulas.

**Revisión del alumno (09-25), ya aplicada:**
- **Fuera `bedrooms_raw` y `bedrooms_imputado`.** Se mantiene la imputación del EDA (D5 cerrada) y la fuga
  menor queda como limitación (H-033).
- **Fuera `max_nights_defecto`,** con evidencia en el EDA, Nivel 3.4 (H-034).

Las filas de la tabla siguiente que mencionan estas columnas quedan como registro de lo propuesto.

La exportación anterior (36.922 × 76) solo tenía lo que el preview dejó dentro; esta incluye **todas** las
candidatas. El notebook de modelado decide cuáles se usan.

**Cambios en la sección 6 de `EDA_NY.ipynb`:**

| Columna nueva | Construcción | Por qué |
|---|---|---|
| `bedrooms_raw` y `bedrooms_imputado` | `bedrooms` sin imputar y su indicador | La imputación por mediana de `room_type` × `accommodates` usa filas del test. Pasa al `Pipeline` (D5) |
| `host_identity_verified` | bool | 80% / 20% en NY: no es casi constante |
| `host_response_time` | categoría, con los nulos como `'sin_dato'` | 50% de nulos. El nulo seguramente significa "sin consultas en 30 días", así que es informativo y no se imputa |
| `host_response_rate`, `host_acceptance_rate` | numéricas en [0, 1], más un indicador de nulo | 50% y 39,5% de nulos |
| `reputacion_cartera` e indicador de nulo | *Leave-one-out* de `review_scores_rating` sobre los otros anuncios del anfitrión (código del Nivel 3.6) | Candidata en el escenario B. Solo el 33,5% de los anuncios la tiene |
| `max_nights_defecto` | `maximum_nights == 1125` | El 56% deja el valor por defecto: puede ser un indicador de anfitrión poco implicado. El valor bruto tiene centinelas (H-004) y no entra |
| `estancia_min_30` | `minimum_nights >= 30` | Forma alternativa de `minimum_nights` (H-025) |
| `host_total_listings_count` | tal cual | Alternativa a `n_anuncios_ny`; se comparan las dos |
| `name` | texto | Solo si se aprueba D8 (variables de título) |

`reputacion_cartera` se calcula antes del split sobre todo el dataset. No es fuga del target (usa
reseñas, no precios), pero sí mezcla información entre anfitriones del train y del test. Como el split
agrupa por anfitrión, todos los anuncios de un mismo anfitrión caen en el mismo lado, así que el
*leave-one-out* nunca cruza el split. Hay que comprobarlo con un `assert`.

**Comprobaciones antes de exportar:** mismo número de filas (36.922), `listing_id` único, los nulos solo
en las columnas esperadas y el tipo de cada columna documentado.

---

## 2. Inventario de variables: estado y justificación (sin preview)

| Variable original | Estado | Justificación |
|---|---|---|
| `listing_id` | Identificador | — |
| `host_id` | Solo para agrupar | Define los grupos del split |
| `name` | Opcional (D8) | Ya se usa para `es_estudio`. Palabras como *luxury* o *penthouse* existen al publicar |
| `host_since` | Candidata B → `antiguedad_host_anios` | 18 nulos, se imputan en el `Pipeline` |
| `host_location` | Candidata B → `host_ambito` | Decisión del 09-09 (EDA) |
| `host_response_time` / `_rate` / `acceptance_rate` | **Candidatas B** | Ver 0.3: fuera solo si D1 = A |
| `host_is_superhost` | Candidata B | — |
| `host_total_listings_count` | Candidata B (alternativa) | Frente a `n_anuncios_ny` |
| `host_has_profile_pic` | **Fuera** | Casi constante: 99,7% `t` (unos 110 anuncios con `f`) |
| `host_identity_verified` | **Candidata B** | 80% / 20%: la bitácora la daba por casi constante por error |
| `neighbourhood` | Candidata B1 | 220 categorías, 106 con N < 30 → codificación en la Fase 6 |
| `district` | Candidata B1 | — |
| `city` | Fuera | Constante |
| `latitude`, `longitude` | Candidatas B1 | — |
| `dist_centro_km` | Candidata B1 | Derivada (City Hall) |
| `property_type` → `property_type_grp` | Candidata B0 | Umbral de 100 anuncios: decisión de diseño, sensibilidad opcional |
| `room_type`, `accommodates`, `bedrooms` | Candidatas B0 | El EDA las da como núcleo, pero también pasan el protocolo |
| `es_estudio` | Candidata B0 (H-023) | — |
| `amenities` → 49 binarias + 11 raras | Candidatas B3 / B4 | La ventana 4%-85% es un criterio ciego al precio (EDA) |
| `n_amenities` | Fuera | Justificado por el EDA condicionado: sin efecto dentro de cada producto (09-09) |
| `price` | Target → `log(price)` | 09-23, por la asimetría (13 en bruto, 0,6 en log) |
| `minimum_nights` | Candidata B2, tres formas | Recortada a 365 en bruto / `estancia_min_30` / las dos |
| `maximum_nights` | **Fuera** | En bruto tiene centinelas (H-004). El indicador de valor por defecto no tiene un efecto consistente al condicionar por zona y estancia mínima: ×1,038 en train, ratio > 1 en 33 de 58 estratos, Queens ×0,96 (`Analisis` §8.2, H-034). Evidencia más débil que con todos los datos (×1,016): pendiente de confirmar |
| `instant_bookable` | Candidata B2 | En train, ×0,93 condicionado, 10 de 50 estratos por encima de 1 (H-039) |
| `n_anuncios_ny` | Candidata B2 | — |
| `reputacion_cartera` | Candidata B2 (escenario B) | Ver Fase 0 |
| `review_scores_*`, `n_reviews`, `sin_resena` | **Fuera por principio** | Posteriores a la publicación (todo anuncio nuevo tiene 0), causalidad inversa y H-026 |
| `guests_per_bedroom` | Fuera | Cociente de dos variables incluidas; solo tiene sentido junto a ellas (H-013) y los árboles lo aprenden |
| `privacy_premium` | Fuera por principio | Se calcula con el precio (H-008) |

**Bloques para la ablación:**

- **B0 Producto:** `room_type`, `accommodates`, `bedrooms`, `property_type_grp`, `es_estudio`
- **B1 Ubicación:** lat/lon, `dist_centro_km`, `district`, `neighbourhood`
- **B2a Condiciones del anuncio:** `minimum_nights`, `estancia_min_30`, `instant_bookable`
- **B2b Perfil del anfitrión** (solo en el escenario B): antigüedad, `n_anuncios_ny`, superhost,
  verificación, tasas, `host_ambito`, `reputacion_cartera`
- **B3 Amenities comunes** (49)
- **B4 Amenities de lujo** (11)

---

## 3. Fase 1 — Infraestructura de evaluación

**✅ Rehecha el 2026-09-26** (secciones 1-4 de `Modelado.ipynb`). Partición fija de `particion_datos.py`:
29.586 / 7.398 anuncios válidos; tras la regla de bloqueo, 29.538 / 7.384. `room_type` y `district` iguales en
train y test al 0,1%. N0.1 (mediana global): RMSE en log **0,711 ± 0,009**, MAE $75. *(La versión del 09-25 —
29.537 / 7.385, estrato solo `room_type`, KS del precio, N0.1 0,712 ± 0,013— está archivada.)*

**Notebook nuevo:** `notebooks/Modelado.ipynb`. **Funciones:** `scripts/func_modelado.py`, separado de
`func_aux.py`, que es del EDA.

### 3.1 Split train / test

- **Se hace antes de analizar el precio**, sobre los anuncios válidos (`limpiar_registros`), con
  `particion_datos.py`. `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`; se toma un fold
  (20%) como test. Los grupos son `host_id` y el estrato es **`room_type` × `district`** (R2).
- Por qué agrupar y estratificar: `Exploracion` §3 (38% de anuncios en carteras de más de uno, 23,7% con un
  "gemelo" en su cartera; simulación de 50 particiones por estrategia).
- **Comprobaciones que se enseñan en la memoria:** reparto de `room_type`, `district`, `accommodates` y de otras
  explicativas en train frente a test, y en qué lado caen los 10 anfitriones más grandes. **Sin comparar el
  precio**: mirarlo en el test no puede cambiar nada sin elegir la semilla, que es justo lo prohibido.
- Los ficheros de `data/particion/` son la fuente de verdad y no se sobrescriben sin `--forzar`; la semilla solo
  sirve para regenerarla. El estrato se eligió con `estudio_estratificacion.py`.

### 3.2 Folds de CV (solo sobre el train)

- **Selección de variables (Fase 4):** `StratifiedGroupKFold(n_splits=5, shuffle=True)` (grupos `host_id`,
  estrato `room_type` × `district`) repetido con 3 semillas (43-45), es decir, 15 estimaciones pareadas. Con
  diferencias del orden de 0,005, 5 folds no bastan para separarlas del ruido (D3).
- **Comparación de modelos y ajuste de hiperparámetros:** la primera repetición (5 folds).
- Guardados en `outputs/folds_cv.csv` (`func_modelado.generar_folds`).
- **Cuidado:** el `early_stopping` de `HistGradientBoostingRegressor` separa su validación interna **por
  fila, no por grupo**, lo que filtra casi duplicados del mismo anfitrión y para demasiado tarde. Se
  desactiva y `max_iter` se ajusta como un hiperparámetro más.

### 3.3 Función de evaluación única

`evaluar(nombre, pipeline, X, y, grupos, folds, bloques) -> dict`. Para cada fold calcula:

- **RMSE en log** (métrica principal), MAE en log y R² en log;
- **MAE en dólares, mediana del error absoluto en dólares y MAPE**, con `exp(ŷ)` (estima la mediana);
- **RMSE de entrenamiento** (sobreajuste = train frente a CV);
- **tiempo de ajuste y de predicción**.

Guarda **una fila por fold** en `outputs/experimentos.csv` (fecha, modelo, bloques, variables,
hiperparámetros, semilla, fold y métricas), para poder hacer después las comparaciones pareadas.

### 3.4 Preprocesadores reutilizables (`ColumnTransformer`)

| Tipo de variable | Lineal / kNN / SVR / MLP | Árboles |
|---|---|---|
| Categóricas pequeñas (`room_type`, `district`, `property_type_grp`, `host_ambito`, `host_response_time`) | `OneHotEncoder(drop='first')` en el lineal, `handle_unknown='ignore'` | `OrdinalEncoder` o categórica nativa en HGB |
| `neighbourhood` | `TargetEncoder` (hace el *cross-fitting* dentro del fit) | Ver Fase 6.3 |
| Numéricas | `StandardScaler`; `log1p` de `minimum_nights` recortada a 365; `n_anuncios_ny` con `log1p` | Tal cual |
| Tasas con nulos | Imputar la mediana + indicador | NaN nativo |
| Binarias | Tal cual | Tal cual |

---

## 4. Fase 2 — Nivel 0: referencias ingenuas

**Pregunta:** ¿cuánto aporta un modelo frente a la regla que usaría un anfitrión?

| Referencia | Predicción |
|---|---|
| N0.1 | Mediana global de `log(price)` (`DummyRegressor(strategy='median')`) |
| N0.2 | Mediana por `room_type` |
| N0.3 | Mediana por `room_type` × `accommodates` (capacidad agrupada en 7+) |
| N0.4 | Mediana por `neighbourhood` × `room_type`, bajando a `district` × `room_type` si la celda tiene N < 30 **en el fold de train** |

Se implementa con un estimador propio que aprende las medianas en el `fit`, para evaluarlo con el mismo
CV que el resto. N0.4 es el listón realista: todo modelo tiene que superarlo con claridad.

**Para la memoria:** "el modelo final reduce el error un X% frente a la mediana de tu barrio" es la
frase para el sponsor.

---

## 5. Fase 3 y Fase 4 — Modelo lineal y selección de variables

### 5.1 Fase 3: modelo lineal (Nivel 1)

**Pregunta:** ¿cuánta señal es aditiva y cuánto vale cada atributo en % del precio? El marco teórico es la
regresión hedónica de precios (Rosen, 1974).

| Paso | Modelo | Qué se aprende |
|---|---|---|
| L1 | OLS: `room_type` + `accommodates` + `bedrooms` | Cuánto explica el producto solo. Coeficientes como `exp(β) − 1` = % de precio |
| L2 | L1 + bloques añadidos uno a uno (5.2) | Contribución de cada bloque en un modelo aditivo |
| L3 | + interacciones `room_type` × `accommodates` y `room_type` × `district` | La lección del EDA ("lo marginal engaña", H-013) en forma de modelo: ¿el efecto de la capacidad depende del producto? |
| L4 | + `SplineTransformer` en `accommodates`, `dist_centro_km` y lat/lon | Hasta dónde llega un lineal con no linealidad explícita (casi un GAM). Responde por qué el lineal no aprovecha lat/lon en bruto |
| L5 | Ridge, Lasso y ElasticNet con todas las candidatas, con `alpha` por CV | Regularización con unas 90 variables; camino de Lasso como selección complementaria |

**Inferencia (solo en el lineal):** `statsmodels` OLS sobre todo el train, con **errores estándar
agrupados por `host_id`** (`cov_type='cluster'`). Los anuncios del mismo anfitrión no son independientes, y
sin agrupar los p-valores salen demasiado optimistas. Es la misma lógica que el split.

**Diagnósticos para el anexo del analista:** residuos frente a ajustados, QQ-plot, Breusch-Pagan
(heterocedasticidad; si sale, ya lo cubren los errores agrupados), VIF de las numéricas
(`accommodates`/`bedrooms` y `latitude`/`dist_centro_km`) y residuos sobre el mapa. Si aparecen manchas
espaciales en el mapa, el lineal no captura la ubicación.

### 5.2 Fase 4: protocolo de selección de variables

**Modelos de referencia:** se usan dos, porque una variable puede aportar en uno y no en el otro, y eso ya
es un hallazgo:

- **Ridge** con las transformaciones de L4 (lineal con no linealidad explícita);
- **HGB con hiperparámetros por defecto** (`max_iter=300`, `learning_rate=0.1`). Se usa sin ajustar para
  que la selección no dependa de un ajuste hecho con esas mismas variables (sería circular). Tras la
  Fase 7 se reconfirman las variables dudosas con el HGB ajustado.

**Procedimiento, en este orden:**

1. **Hacia delante por bloques:** B0 → +B1 → +B2a → +B2b → +B3 → +B4, con ΔRMSE en cada paso. Resultado:
   gráfico de escalera por modelo.
2. **Quitando un bloque cada vez** sobre el modelo completo: cuánto se pierde al quitar cada bloque,
   condicionado al resto.
3. **Pruebas concretas dentro de cada bloque** (lista cerrada, fijada hoy):

| ID | Prueba | Alternativas | Origen |
|---|---|---|---|
| A1 | `es_estudio` | con / sin | H-023 |
| A2 | `property_type_grp` | con / sin | 09-23 |
| A3 | Ubicación | solo lat/lon; + `dist_centro_km`; + `district`; + barrio; solo `district` + barrio (sin coordenadas) | H-028 |
| A4 | Forma de `minimum_nights` | fuera / en bruto recortada / `estancia_min_30` / las dos | H-025 |
| ~~A5~~ | ~~`max_nights_defecto`~~ | Retirada: descartada en el EDA (H-034) | — |
| A6 | `instant_bookable` | con / sin | — |
| A7 | Cartera | `n_anuncios_ny` / `host_total_listings_count` / ninguna | H-027 |
| A8 | `reputacion_cartera` + indicador de nulo | con / sin | H-027, reabierta |
| A9 | Tasas de respuesta y aceptación | con / sin (como bloque) | 0.3 |
| A10 | `host_identity_verified`, `host_is_superhost`, `host_ambito`, antigüedad | con / sin, una a una | — |
| A11 | Amenities de lujo (B4) | bloque con / sin | 09-23 |
| A12 | Amenities comunes (B3) | bloque; si entra, cribado individual con permutación | 09-09 |

Si pasan A1-A12 no se añaden pruebas nuevas. Si se añade alguna, se registra como exploratoria y no
cuenta para justificar el modelo.

### 5.3 Regla de decisión (fijada antes de ver los resultados — D4)

Para cada prueba, sobre las 15 estimaciones pareadas (5 folds × 3 semillas):
`Δᵢ = RMSE_sin − RMSE_con` (positivo = la variable ayuda).

> **Cerrada el 09-27 (decisión del alumno): consistencia + Δ medio > desviación típica**, declarada como
> heurística. El p de Nadeau-Bengio se calcula y se muestra, pero **no decide**. Propuesta original:

Una variable o bloque **entra** si cumple a la vez:

1. **Consistencia:** `Δᵢ > 0` en al menos 12 de las 15 estimaciones.
2. **Significación:** t-test pareado **corregido de Nadeau y Bengio (2003)**, unilateral, p < 0,05. La
   varianza se multiplica por `(1/15 + n_test/n_train)` = `(1/15 + 0,25)`, porque los folds comparten
   datos de entrenamiento y el t-test normal sale demasiado optimista. Se implementa en cinco líneas y está
   citado en la literatura de comparación de modelos (Bouckaert y Frank, 2004).

Si una variable entra en un modelo de referencia y no en el otro, se anota como hallazgo y entra solo en
el modelo en que aporta.

**Resultado:** tabla con ΔRMSE medio ± desviación típica, folds que mejoran, p corregido y decisión.
Es la tabla central de la selección de variables en la memoria.

---

## 6. Fase 5 y Fase 6 — Modelos no lineales (Niveles 2 y 3)

### 6.1 kNN (Nivel 2) — "el tasador por comparables"

- **Variables:** solo lat/lon, `accommodates`, `bedrooms` y `room_type` (one-hot), escaladas. Con 90
  dimensiones kNN no funciona (maldición de la dimensionalidad), así que se usa a propósito un subconjunto
  pequeño, justificado por el EDA.
- **Hiperparámetros:** `n_neighbors` de 5 a 100 y `weights` ∈ {uniform, distance}. Opcional: un peso
  mayor para lat/lon.
- **Qué aporta:** es fácil de explicar al sponsor ("tus 20 anuncios más parecidos cobran esto").

### 6.2 Árbol de decisión (Nivel 2)

- `max_depth` de 3 a 20 y `min_samples_leaf` de 5 a 200.
- **Dos usos:** un árbol de profundidad 3 para visualizar las reglas para el sponsor, y el árbol ajustado
  para la métrica.
- **Qué se enseña:** su inestabilidad (la desviación típica entre folds y la estructura que cambia de un
  fold a otro). Justifica pasar a los ensembles.

### 6.3 Random Forest y gradient boosting (Nivel 3)

- **Random Forest:** `n_estimators=500`, con `OrdinalEncoder` en las categóricas. **El error OOB no sirve**
  porque ignora los grupos por anfitrión; se usa el CV.
- **`HistGradientBoostingRegressor`:** categóricas nativas y NaN nativos. Está en scikit-learn 1.9, sin
  dependencias nuevas.
- **Experimento de codificación del barrio** (solo HGB): (a) fuera, (b) `TargetEncoder`, (c) categórica
  nativa (220 categorías, por debajo del límite de 255). Es una decisión técnica defendible en sí misma.
- **Pregunta de la memoria:** ¿gana aquí reducir varianza (RF, bagging) o reducir sesgo (boosting)?
  Hay que explicarla con el balance sesgo-varianza.

---

## 7. Fase 7 — Ajuste de hiperparámetros

- **Método:** `RandomizedSearchCV` con los folds de la sección 3.2, pasando `groups` (D7).
- **Orden:** primero variables fijadas (Fase 4) con hiperparámetros por defecto, luego el ajuste, y al
  final la reconfirmación de las variables dudosas con el modelo ajustado.

| Modelo | Espacio de búsqueda | Iteraciones |
|---|---|---|
| Ridge / Lasso / ElasticNet | `alpha` log-uniforme [1e-4, 1e2]; `l1_ratio` ∈ [0,1] | Grid de 30 |
| kNN | `n_neighbors` [5, 100]; `weights` | Grid |
| Árbol | `max_depth` [3, 20]; `min_samples_leaf` [5, 200] | Grid |
| RF | `max_features` {0,2; 0,33; 0,5; 1,0}; `min_samples_leaf` {1, 3, 5, 10, 20}; `max_depth` {None, 15, 25} | 30 |
| HGB | `learning_rate` log [0,02, 0,2]; `max_iter` [200, 2000]; `max_leaf_nodes` [15, 127]; `min_samples_leaf` [10, 200]; `l2_regularization` log [1e-3, 10]; `max_features` [0,3, 1,0] | 60 |

**Aviso para la memoria:** el RMSE de CV del mejor candidato es optimista, porque se ha elegido mirando
esos folds. Por eso existe el test.

---

## 8. Fase 8 — Modelos opcionales (Nivel 4, solo si hay tiempo — D6)

Por orden de valor:

1. **Regresión cuantílica con HGB** (`loss='quantile'`, α = 0,1 / 0,5 / 0,9). Da al sponsor un **rango de
   precio**, no un único número. Se evalúa la cobertura real del intervalo del 80% en el test y la
   *pinball loss*.
2. **SVR con kernel RBF.** Es el mejor modelo del paper de referencia. Coste O(n²): con unos 30.000
   anuncios se necesita una aproximación de Nystroem + `LinearSVR`, o submuestreo.
3. **MLP (scikit-learn).** Se espera poca ganancia en datos tabulares (Grinsztajn et al., 2022, "Why do
   tree-based models still outperform deep learning on tabular data?"). Se incluiría solo para
   demostrarlo.
4. **CatBoost / LightGBM / EBM.** Requieren instalar librerías; no se hacen salvo que sobre tiempo.

---

## 9. Fase 9 — Selección del modelo final y evaluación en el test

1. **Tabla comparativa** de todos los modelos ajustados, con los mismos 5 folds: RMSE en log (media ±
   desviación típica), MAE en dólares, diferencia train-CV, tiempo de ajuste e interpretabilidad
   (alta / media / baja).
2. **Regla de un error estándar:** se elige el modelo **más simple** cuyo RMSE medio quede dentro de un
   error estándar del mejor. Si RF y HGB empatan, gana el más rápido o el más interpretable.
3. Se reentrena con todo el train y se **evalúa el test una sola vez**. Intervalo de confianza de la
   métrica con *bootstrap* por anfitrión (se remuestrean anfitriones, no anuncios).
4. **Contraste con la literatura:** el R² en log se compara con el paper de referencia
   (`docs/Referencias/1907.12665v1.pdf`; **pendiente de verificar su cifra**), explicando que usan un
   split aleatorio y reseñas como variable.

---

## 10. Fase 10 — Interpretación y análisis de errores

| Análisis | Herramienta | Para quién |
|---|---|---|
| Importancia por bloque y por variable | Importancia por permutación sobre el test; lat/lon se permutan juntas | Analista |
| Efecto de cada variable | PDP + ICE (`PartialDependenceDisplay`) para `accommodates`, `dist_centro_km`, `minimum_nights`; PDP 2D de lat/lon como mapa | Los dos (el mapa, para el sponsor) |
| Explicación de un anuncio concreto | SHAP (D7, requiere instalar `shap`) | Sponsor: "por qué este piso vale $X" |
| Error por segmento | MAE en dólares y MAPE por `room_type`, `district`, tramo de capacidad y quintil de precio | Los dos |
| Residuos espaciales | Mapa de residuos del test | Analista |
| **Sensibilidad a la regla de exclusión (H-024)** | Métricas en el test excluyendo el P99 global frente a la regla de bloqueo, y error por tramo de precio | Analista (prometido en la bitácora) |
| Comparación lineal frente a final | Coeficientes del lineal frente a SHAP del final: ¿dónde coinciden y dónde no? | Analista |
| Coste de no conocer al anfitrión | ΔRMSE entre los escenarios A y B | Sponsor (solo si D1 = B) |

`exp(ŷ)` estima la **mediana** del precio. Si hiciera falta la media, se aplica el *smearing* de Duan
(bitácora, 09-23).

---

## 11. Qué sale de cada fase para la memoria

| Fase | Tabla o gráfico | Sección de la memoria |
|---|---|---|
| 0 | Inventario de variables (sección 2) | Analista: detalles del modelo, variables |
| 1 | Comparación train/test del split | Analista: aproximación |
| 2 | Referencias ingenuas | Sponsor: resultados ("mejora frente a la regla actual") |
| 3 | Coeficientes hedónicos (% de precio) con intervalo de confianza | Los dos: descripción del modelo |
| 4 | Escalera por bloques y tabla de decisión de variables | Analista: resultados; sponsor: qué influye en el precio |
| 5-8 | Tabla comparativa de modelos | Analista: resultados |
| 9 | Métrica en el test con intervalo de confianza | Los dos: resumen de resultados |
| 10 | PDP, mapa, SHAP, error por segmento | Los dos: resultados y recomendaciones |

---

## 12. Calendario (hoy es 2026-09-25; aprobación del tutor el 2026-10-16)

| Fechas | Trabajo | Hito |
|---|---|---|
| Vie 25 – Dom 27 sep | Cerrar D1-D9; Fase 0 (reexportación); Fase 1 (split y funciones) | Test separado y guardado |
| Lun 28 – Mar 29 sep | Fases 2 y 3 (Nivel 0 y lineal L1-L5) | Coeficientes hedónicos |
| Mié 30 sep – Jue 1 oct | Fase 4 (protocolo con Ridge y HGB por defecto) | Variables cerradas |
| Vie 2 – Sáb 3 oct | Fase 5 (kNN, árbol) y RF | — |
| Dom 4 – Mar 6 oct | HGB, codificación del barrio, Fase 7 (ajuste) y reconfirmación | — |
| Mié 7 oct | Fase 8 (solo la regresión cuantílica) | **Si hay retraso, se recorta de aquí** |
| Jue 8 oct | Fase 9: selección y **único uso del test** | Modelo final |
| Vie 9 – Sáb 10 oct | Fase 10: interpretación y errores | — |
| Dom 11 – Jue 15 oct | Redacción de la memoria (analista y sponsor), gráficos finales | — |
| Vie 16 oct | Envío al tutor | — |

**Colchón:** ninguno explícito, así que la Fase 8 es lo primero que se recorta. Si el 4 de octubre la
Fase 4 no está cerrada, se simplifica la selección a una sola semilla.

---

## 13. Riesgos y limitaciones que hay que declarar

- ~~Imputación de `bedrooms` con medianas de todo el dataset~~ y ~~decisiones del EDA con todos los datos~~:
  **resueltas el 09-26** (H-033, H-035, H-036). El preparador y todas las decisiones que miran el precio usan solo
  el train. Queda por declarar que el **análisis descriptivo sin precio** (`Exploracion`: nulos, duplicados,
  estructura geográfica) usa todos los anuncios, algo habitual y sin riesgo de fuga del target.
- **La regla de exclusión se aplica también al test:** se eliminan precios que no son de mercado. En
  producción no se conoce el precio, así que el modelo no está validado para tarifas de bloqueo.
- **Datos de una sola fecha (2021):** las variables del anfitrión se miden en la extracción, no al publicar.
- **El precio publicado no es el precio pagado:** se predice lo que pide el anfitrión, no el valor de
  mercado realizado.
- **Anuncios sin reseñas un 15% más caros (H-026):** el modelo aprende de todos los anuncios, activos o
  no, así que el precio predicho es el "precio pedido", no necesariamente uno que se alquile.
- **Nueva York está muy estudiado:** el valor del TFM está en la validación honesta y en la
  interpretación, no en batir una cifra de Kaggle.

---

## 14. Decisiones pendientes

| ID | Decisión | Alternativas | Recomendación del tutor/analista (opinión, no imposición) |
|---|---|---|---|
| ~~D1~~ ✅ | Escenario de predicción | A (anfitrión nuevo) / B (cualquier anfitrión) / los dos | **B como modelo principal y A como contraste** (el bloque B2b se quita): da al sponsor el dato de cuánto vale conocer al anfitrión |
| ~~D2~~ ✅ | Test separado | 20% estratificado por `room_type` / 20% sin estratificar / 15% | **20% estratificado y agrupado**. **Revisada el 09-26 (R2):** estrato `room_type` × `district`, partición previa al análisis del precio |
| ~~D3~~ ✅ | CV para la selección | 5 folds / 5 × 3 semillas / 10 folds | **5 × 3** para la selección; 5 folds para el ajuste |
| ~~D4~~ ✅ | Regla de "la variable aporta" | Consistencia sola (≥ 12/15) / consistencia + Nadeau-Bengio / umbral fijo de ΔRMSE | **Consistencia + Nadeau-Bengio**. Si no se ha visto en el máster, consistencia + ΔRMSE medio > desviación típica, declarando que es heurística. **Cerrada 09-27: consistencia ≥ 12/15 + Δ medio > sd (heurística); N-B solo informativo** |
| ~~D5~~ | Imputación de `bedrooms` | Mantener la del EDA / dentro del `Pipeline` | ~~Cerrada (09-25): se mantiene la del EDA~~. **Revisada el 09-26 (R5): medianas aprendidas solo con el train en `PreparadorListings`**; sin fuga |
| ~~D6~~ ✅ | Modelos opcionales | Cuantílica / SVR / MLP / CatBoost | **Solo la cuantílica**; SVR si sobra un día. **Cerrada 09-27: cuantílica + SVR** |
| ~~D7~~ ✅ | Dependencias nuevas | Ninguna (`RandomizedSearchCV`, PDP y permutación) / + `shap` / + `optuna` | **Solo `shap`**, para explicar anuncios concretos al sponsor. `optuna` no hace falta con 37k filas. **Cerrada 09-27: ninguna** (sin SHAP; permutación + PDP/ICE) |
| ~~D8~~ ✅ | Variables del título (`name`) | No / 5-10 palabras clave fijadas de antemano / TF-IDF + Ridge | **Palabras clave pre-especificadas** (*luxury*, *penthouse*, *loft*, *cozy*, *view*...) como prueba A13 exploratoria, o no hacerlo si el calendario aprieta. **Cerrada 09-27: no** |
| ~~D9~~ ✅ | `reputacion_cartera` | Candidata con indicador de nulo / fuera por principio (usa reseñas) | **Candidata en el escenario B**: usa reseñas de **otros** anuncios, que existen al publicar. Queda como limitación que se miden en 2021. **Cerrada 09-27: candidata (A8)** |
