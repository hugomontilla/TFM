# Arquitectura del proyecto — TFM Airbnb (Nueva York)

Objetivo: predecir el precio por noche (`log(price)`) de un anuncio nuevo de Airbnb en Nueva York, con un intervalo
de precio, y explicar qué lo determina. Fuente: `listings.csv` (y `reviews.csv` solo para fechar el snapshot).

## 1. Vista de conjunto

```
data/listings.csv ──► 0-INTRODUCCION ──► decisión: acotar a Nueva York
        │
        ▼
 1-EXPLORACION  (global, sin mirar el precio)
        │  particion_datos.py ──► data/particion/  (train / test fijos, MD5)
        ▼
 2-ANALISIS     (solo train, con el precio)
        │  preparacion_datos.py (PreparadorListings.fit(train))
        ▼
 data/modelado/{train,test}.parquet + preparador.json
        │
        ▼
 3-MODELADO     (solo train, CV 5×3 agrupada)  ──►  outputs/experimentos.csv  (registro)
        │  elige modelo por RMSE medio de validación
        ▼
 4-MODELO_FINAL   (CatBoost y HGB; el test se abre UNA vez)
        │
        ▼
 models/ (modelo, cuantiles, ficha)  +  outputs/resultados_test*.csv
```

La idea que ordena todo: **una barrera física contra la fuga de información**. El test se separa antes de mirar
el precio; los notebooks 2 y 3 solo leen el train; el test solo lo abre el notebook del modelo final.

## 2. Capas

### 2.1 Datos (`data/`)

| Fichero | Qué es |
|---|---|
| `listings.csv`, `reviews.csv` | Datos en bruto (10 ciudades; se usa solo NY) |
| `particion/` | **Fuente de verdad del split** (se crea una vez, antes de mirar el precio): `listings_{train,test}.csv` con las filas en bruto y `particion.json` (semilla, recuentos y MD5, lo escribe `particion_datos.py`). No se sobrescribe sin `--forzar` |
| `modelado/train.parquet`, `test.parquet` | Dataset de modelado (79 candidatas + target), derivado de los CSV de la partición. Se regenera con `python scripts/preparacion_datos.py` |
| `modelado/preparador.json` | Lo aprendido por `PreparadorListings.fit(train)`: ventana de amenities, tipos de propiedad frecuentes, medianas de `bedrooms`. De él salen también los bloques de variables (B0 producto, B1 ubicación, B2 anfitrión, B3 amenities…) |
| `archivo_2026-09-25/` | Evidencia de la partición antigua (SUPERSEDED); no se usa |

### 2.2 Código (`scripts/`)

| Módulo | Responsabilidad | Lo usan |
|---|---|---|
| `config.py` | Parámetros comunes (semilla, folds, cobertura del intervalo…), leídos de `.env` en la raíz del proyecto | todos |
| `registro.py` | Único acceso a `outputs/experimentos.csv`: guardar, leer un experimento y recuperar un valor del registro | 3, final |
| `particion_datos.py` | Crea y carga la partición fija (agrupada por `host_id`, estratificada por `room_type` × `district`) y escribe `particion.json` | 1, 2, 3, final |
| `estudio_estratificacion.py` | Simula 50 particiones para justificar el estrato | 1 |
| `preparacion_datos.py` | Limpieza y *feature engineering*. `PreparadorListings` (`fit`/`transform`) sin fuga; regla de precios de bloqueo; validación del dataset | 2, 3, final |
| `func_aux.py` | Estadística del EDA: Welch-ANOVA, η², Cramér V, lift condicionado por estrato, contrastes por grupo | 1, 2 |
| `estilo_graficos.py` | Estilo único de gráficos (paleta, mapas de color, figuras comunes) | todos |
| `func_modelado.py` | Núcleo del modelado: folds guardados, `Experimento`, `evaluar`/`registrar` (caché en el CSV), preprocesadores por familia, `RegresorCatBoost`, `MedianaPorGrupo`, PCAmix/FAMD, búsqueda de hiperparámetros, importancia por permutación, comparación pareada | 3, final |
| `func_final.py` | Modelo final: `ModeloFinal` (un finalista de principio a fin: configuración desde el registro, ajuste, CQR, test con bootstrap por anfitrión, intervalo, errores, importancia, guardado y ficha), `predecir_precio` y la diferencia pareada entre modelos | final |

### 2.3 Notebooks (`notebooks/`)

| Notebook | Entrada | Qué hace | Salida |
|---|---|---|---|
| `0-INTRODUCCION` | CSV en bruto | Compara las 10 ciudades y decide acotar a Nueva York | Decisión de alcance |
| `1-EXPLORACION` | `listings.csv` (NY) | Duplicados, nulos, filas no válidas, ubicación, anfitriones; **diseña y crea la partición** | `data/particion/` |
| `2-ANALISIS` | train | EDA con el precio: distribución, precios de bloqueo, producto, capacidad, amenities, ubicación, estancia, anfitrión | Qué variables pasan al modelado |
| `3-MODELADO` | train | Referencias ingenuas → lineal → selección de variables → kNN/CART → RF/HGB → cuantílica/SVR/CatBoost → tabla comparativa e importancia | `outputs/experimentos.csv`, importancias |
| `4-MODELO_FINAL` | train + **test (una vez)** | Para **CatBoost y HGB** (un subapartado por modelo en cada sección): reentrena con todo el train, calibra el intervalo (CQR), evalúa en el test, analiza errores, interpreta y guarda. Termina con una comparación entre los dos | `models/`, `outputs/resultados_test*.csv` |

### 2.4 Resultados (`outputs/`, `models/`)

- `outputs/experimentos.csv`: **registro de experimentos**, una fila por experimento × repetición × fold (513
  experimentos). Es a la vez caché (un experimento ya registrado no se reejecuta) y fuente de hiperparámetros del
  modelo final. `docs/registro_experimentos.md` es su versión legible.
- `outputs/folds_cv.csv`: folds 5×3 guardados; todos los modelos comparten folds, así que las comparaciones son
  pareadas.
- `models/`: para cada finalista, el modelo y sus cuantiles (`.joblib`) y su ficha (`ficha_modelo_catboost.json` para CatBoost,
  `ficha_modelo_hgb.json` para HGB: hiperparámetros, 79 variables, métricas de CV y test, MD5 de la partición, versiones).
- `Entrenamiento Google Colab/`: copia mínima del código y datos de train para entrenar en GPU (CatBoost, MLP).

### 2.5 Documentación y roles (`docs/`, `.claude/skills/`, `.agents/skills/`)

- `bitacora_decisiones.md`: decisiones vigentes y por qué. `hallazgos_y_pendientes.md`: riesgos metodológicos y pendientes.
  `registro_experimentos.md`: experimentos. `Planes/`: planes de modelado, mejoras y modelo final. `INDICACIONES/`:
  enunciado del TFM, descripción del dataset y plantilla de portada. `Referencias/`: paper de referencia.
- Skills de roles `tfm-director`, `tfm-ml-engineer`, `tfm-analista`, `tfm-juzgado` en `.claude/skills/`; skills
  genéricas (grill-me, teach…) en `.agents/skills/`.

## 3. Invariantes de diseño

1. **El test se abre una vez**, en el notebook final, tras comprobar su MD5 contra `particion.json`.
2. **Todo lo que aprende de los datos se ajusta con el train**: el preparador, los *encodings*, la imputación. El
   preparador congela su estado en `data/modelado/preparador.json`.
3. **Ninguna variable posterior a la publicación ni derivada del precio** entra al modelo (reseñas, `privacy_premium`).
4. **Split y CV agrupados por `host_id`**: el 38,1% de los 36.922 anuncios está en carteras de más de uno (`1-EXPLORACION`, §3) y son casi gemelos.
5. **Experimentos registrados antes de ejecutarse**, con folds fijos y comparación pareada; las reglas de decisión
   se fijan antes de ver resultados.
6. **Una sola definición** de cada cosa: parámetros comunes en `.env`, espacio de búsqueda en `func_modelado.py`,
   hiperparámetros de los modelos finales en el registro, partición en `data/particion/`.

## 4. Estado actual

- Finalistas: **CatBoost** (CV 0,416 ± 0,009) y **HGB** (CV 0,419 ± 0,010); CatBoost gana en CV por muy poco y HGB se ajusta
  unas 16 veces más rápido. Los resultados en el test están en `docs/resumen_ejecutivo.md`; qué modelo se presenta queda por decidir.
- El MLP está preparado en la carpeta de Colab y aplazado.
- Pendiente: redacción de la memoria y cerrar los pendientes de `hallazgos_y_pendientes.md`.
