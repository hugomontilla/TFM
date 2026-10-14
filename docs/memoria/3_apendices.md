# Apéndices

> Dirigidos a los ingenieros y analistas que quieran reproducir, revisar o
> desplegar el trabajo. El repositorio completo acompaña a esta memoria.

---

## A. Mapa del repositorio

Repositorio del proyecto: [https://github.com/hugomontilla/TFM](https://github.com/hugomontilla/TFM)

```
data/        listings.csv y reviews.csv (no versionados); particion/ (train y test fijos con MD5);
             modelado/ (parquet preparados y parámetros del preparador)
scripts/     código reutilizable (limpieza, variables, partición, modelado, modelo final, gráficos)
notebooks/   análisis y modelado, en orden numérico (0 a 4)
models/      modelos finalistas, cuantiles del intervalo y fichas
outputs/     registro de experimentos, folds de CV, importancias y resultados en test
docs/        memoria, bitácora de decisiones, hallazgos, registro de experimentos, planes y enunciado
.env         parámetros comunes (semilla, folds, cobertura del intervalo, réplicas del bootstrap)
```

## B. Notebooks (orden de ejecución)

| Notebook | Entrada | Qué hace | Salida |
|---|---|---|---|
| `0-INTRODUCCION` | `listings.csv` | Compara las 10 ciudades y justifica acotar a Nueva York | Decisión de alcance |
| `1-EXPLORACION` | `listings.csv` (NY) | Calidad de datos sin mirar el precio: duplicados, nulos, filas no válidas, ubicación, anfitriones. Diseña y crea la partición | `data/particion/` |
| `2-ANALISIS` | train | Análisis exploratorio del precio, solo con el train | Variables candidatas y reglas de limpieza |
| `3-MODELADO` | train | Referencias, lineal, selección de variables, kNN, árbol, ensembles, SVR, CatBoost, comparación e importancia | `outputs/experimentos.csv`, importancias |
| `4-MODELO_FINAL` | train + test (una vez) | Reentrena los dos finalistas, calibra el intervalo, evalúa en test, analiza errores, interpreta, guarda y compara | `models/`, `outputs/resultados_test_*.csv` |

Los notebooks 3 y 4 se pueden ejecutar sin los CSV originales: la partición y los datasets de modelado están
versionados. Los experimentos ya registrados no se recalculan (el registro hace de caché).

## C. Módulos de `scripts/`

| Módulo | Responsabilidad |
|---|---|
| `config.py` | Lee los parámetros comunes de `.env` |
| `particion_datos.py` | Crea y carga la partición fija (agrupada por `host_id`, estratificada por `room_type` × `district`); escribe `particion.json` con las huellas MD5; no sobrescribe sin `--forzar` |
| `estudio_estratificacion.py` | Simula 50 particiones para justificar el estrato |
| `preparacion_datos.py` | `PreparadorListings` (`fit`/`transform`): limpieza, regla de precios de bloqueo y construcción de las 79 variables. Es la única fuente de verdad de las variables. Se ejecuta sin argumentos para regenerar `data/modelado/` |
| `func_aux.py` | Estadística del EDA (Welch-ANOVA, η², V de Cramér, lift por estrato, contrastes por grupo) |
| `func_modelado.py` | Núcleo del modelado: folds guardados, `evaluar`, preprocesadores por familia, `RegresorCatBoost`, `MedianaPorGrupo`, PCAmix/FAMD, búsqueda de hiperparámetros e importancia por permutación |
| `registro.py` | Único acceso a `outputs/experimentos.csv` (escritura atómica, sustituye un experimento si se reejecuta) |
| `func_final.py` | `ModeloFinal` (configuración desde el registro, ajuste, CQR, test con *bootstrap* por anfitrión, errores, importancia, guardado), `predecir_precio` y la diferencia pareada entre modelos |
| `estilo_graficos.py` | Estilo único de gráficos: un color fijo por categoría en toda la memoria |
| `extraer_figuras.py` | Extrae a PNG las figuras ya renderizadas de los notebooks, sin volver a ejecutarlos |

## D. Reproducción

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
python scripts/preparacion_datos.py   # solo si se parte de los CSV originales
```

- **Versiones usadas:** Python 3.13, scikit-learn 1.9, pandas 3.0, CatBoost 1.2 (registradas en
  `models/ficha_modelo_*.json`). Los modelos `joblib` solo cargan con la misma versión de scikit-learn y CatBoost.
- **Datos no versionados:** `data/listings.csv` (160 MB) y `data/reviews.csv` (250 MB), del conjunto público de
  Airbnb 2008-2021 (licencia CC0).
- **Aleatoriedad controlada:** `SEMILLA=42` en `.env`. Cambiar la semilla o el número de folds invalida el registro de
  experimentos y los folds guardados.
- **Integridad de la partición:** `particion.json` guarda el MD5 de `listings_train.csv` y `listings_test.csv`; el
  notebook final comprueba el del test antes de cargarlo.
- **GPU:** `notebooks/Entrenamiento Google Colab/` contiene una copia mínima para repetir la búsqueda de CatBoost en
  GPU; el resultado oficial es el de CPU.

## E. Uso del modelo guardado

Para cada finalista hay tres ficheros en `models/`: el modelo (`modelo_final_<modelo>.joblib`), los cuantiles con su
corrección CQR (`cuantiles_<modelo>.joblib`) y la ficha (`ficha_modelo_<modelo>.json`, con hiperparámetros, las 79
variables, métricas de CV y test, MD5 de la partición y versiones).

```python
# Desde la raíz del repositorio, con scripts/ en el sys.path (como hacen los notebooks)
import joblib, json
from preparacion_datos import PreparadorListings, RUTA_PREPARADOR, cargar_listings_ny
from func_final import predecir_precio

preparador = PreparadorListings.cargar(RUTA_PREPARADOR)
modelo = joblib.load('models/modelo_final_catboost.joblib')
intervalo = joblib.load('models/cuantiles_catboost.joblib')
columnas = json.load(open('models/ficha_modelo_catboost.json', encoding='utf-8'))['variables']

anuncios = cargar_listings_ny('data/listings.csv')        # filas en bruto con el formato de listings.csv
precios = predecir_precio(anuncios, preparador, modelo,
                          intervalo['cuantiles'], intervalo['correccion_cqr'], columnas)
# -> listing_id, precio_mediano, precio_bajo, precio_alto (USD por noche)
```

**Atención:** `n_anuncios_ny` y `reputacion_cartera` se calculan sobre las filas que se pasan. Para un anfitrión con
varios anuncios hay que pasar su cartera completa en Nueva York, no solo el anuncio nuevo.

## F. Documentación de trazabilidad

| Documento | Qué contiene |
|---|---|
| `docs/bitacora_decisiones.md` | Cada decisión vigente, con fecha, alternativas descartadas y motivo |
| `docs/hallazgos_y_pendientes.md` | Hallazgos de calidad de datos y riesgos metodológicos (identificadores H-xxx citados en los notebooks) |
| `docs/registro_experimentos.md` | Versión legible de `outputs/experimentos.csv`, por familia de modelos |
| `docs/arquitectura.md` | Cómo está construido el proyecto e invariantes de diseño |
| `docs/Planes/` | Planes de modelado y de modelo final con los que se trabajó |

## G. Tablas complementarias

### G.1 Ablación por bloques y variables

Diferencia de RMSE en log al quitar cada bloque o variable frente al modelo completo (media de 15 estimaciones), en
milésimas. Positivo = el modelo empeora al quitarlo. Fuente: `docs/registro_experimentos.md`, §7.

| Qué se quita | Ridge | HGB |
|---|---|---|
| Bloque producto | +111,2 | +108,7 |
| Bloque ubicación | +31,7 | +34,7 |
| Bloque anfitrión | +17,2 | +15,1 |
| Bloque amenities comunes | +13,9 | +12,4 |
| Bloque condiciones de reserva | +1,0 | +2,5 |
| Tasas de respuesta y aceptación | +4,7 | +7,3 |
| Latitud y longitud | +2,8 | +5,2 |
| Tipo de propiedad | +3,1 | +2,0 |
| Indicador de estudio | +2,2 | +2,3 |
| Antigüedad del anfitrión | +0,1 | +1,7 |
| Reputación de cartera | +0,4 | +1,2 |
| Ámbito del anfitrión | +0,4 | +1,1 |
| Reserva instantánea | +0,0 | +0,4 |
| Superhost | +0,2 | −0,3 |
| Identidad verificada | 0,0 | 0,0 |

### G.2 Importancia por permutación de cada bloque en todos los modelos (validación)

Aumento del RMSE en log al permutar el bloque. Fuente: `outputs/importancia_bloques.csv`.

| Modelo | Producto | Ubicación | Condiciones | Anfitrión | Amenities |
|---|---|---|---|---|---|
| CatBoost | 0,307 | 0,107 | 0,008 | 0,055 | 0,042 |
| HGB | 0,309 | 0,102 | 0,007 | 0,053 | 0,038 |
| Random Forest | 0,308 | 0,098 | 0,002 | 0,045 | 0,030 |
| SVR | 0,267 | 0,086 | 0,004 | 0,051 | 0,061 |
| Lineal (Lasso) | 0,301 | 0,092 | 0,004 | 0,053 | 0,036 |
| Árbol (prepoda) | 0,320 | 0,094 | 0,002 | 0,048 | 0,018 |
| kNN (K1) | 0,311 | 0,107 | — | — | — |
| kNN con PCAmix | 0,227 | 0,042 | 0,000 | 0,021 | 0,060 |
