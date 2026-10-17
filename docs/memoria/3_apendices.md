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
docs/        memoria, bitácora de decisiones, hallazgos, registro de experimentos, referencias y enunciado
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
| `Entrenamiento Google Colab/Modelado_MLP_colab` | train (copia) | Red neuronal (MLP) en GPU con los mismos folds: búsqueda de 60 candidatos, semillas y comparación. Guardado sin salidas; no hace falta reejecutarlo | `outputs/experimentos_mlp.csv` |

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
- **GPU:** `notebooks/Entrenamiento Google Colab/` contiene una copia mínima del proyecto (train, folds, scripts) para
  entrenar la red neuronal en Google Colab con GPU. Sus resultados se copian a `outputs/experimentos_mlp.csv`. Los
  modelos oficiales (CatBoost y HGB) se entrenaron en CPU.

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

---

## H. Referencias

Referencias básicas donde se explican las técnicas y los modelos empleados. Entre corchetes, dónde se usan en la
memoria.

### H.1 Documentos de referencia

- Rezazadeh Kalehbasti, P., Nikolenko, L. y Rezaei, H. (2019). Airbnb price prediction using machine learning and
  sentiment analysis. *arXiv:1907.12665*. Copia en el repositorio: `docs/Referencias/1907.12665v1.pdf`. [Antecedente
  del problema: predicción del precio de Airbnb con regresión lineal, árboles, SVR y redes]
 

### H.2 Contrastes estadísticos del análisis exploratorio

- **Kruskal-Wallis:** [§5.1]
  Kruskal, W. H. y Wallis, W. A. (1952). Use of ranks in one-criterion variance analysis. *Journal of the American
  Statistical Association*, 47(260), 583–621.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/Kruskal%E2%80%93Wallis_test).
- **Mann-Whitney:** versión de Kruskal-Wallis para dos grupos. [§5.1]
  Mann, H. B. y Whitney, D. R. (1947). On a test of whether one of two random variables is stochastically larger
  than the other. *The Annals of Mathematical Statistics*, 18(1), 50–60.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test).
- **Corrección de Holm:** [§5.1]
  Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*,
  6(2), 65–70.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method).
- **ANOVA de Welch:** ANOVA que no supone varianzas iguales. [§5.1]
  Welch, B. L. (1951). On the comparison of several mean values: an alternative approach. *Biometrika*, 38(3/4),
  330–336.
- **Correlación de Spearman:** [§5.1]
  Spearman, C. (1904). The proof and measurement of association between two things. *The American Journal of
  Psychology*, 15(1), 72–101.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient).

### H.3 Comparación de modelos e incertidumbre

- **Validación agrupada (fuga entre anuncios del mismo anfitrión):** [§3.3]
  Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., *et al.* (2017).
  Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*,
  40(8), 913–929.
  Kapoor, S. y Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science.
  *Patterns*, 4(9), 100804.
- **t de Student corregida de Nadeau-Bengio:** comparación pareada de modelos sobre los mismos *folds*. [§6.1]
  Nadeau, C. y Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52(3), 239–281.
- **TOST (prueba de equivalencia):** [§5.3]
  Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing
  the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/Equivalence_test).
- **Bootstrap:** intervalos de confianza en test. [§5.4]
  Efron, B. y Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)).

### H.4 Modelos

- **Manual de referencia:** todos los modelos de la escalera.
  Hastie, T., Tibshirani, R. y Friedman, J. (2009). *The Elements of Statistical Learning* (2.ª ed.). Springer.
- **Regresión hedónica:** lectura de los coeficientes. [§5.2]
  Rosen, S. (1974). Hedonic prices and implicit markets: product differentiation in pure competition. *Journal of
  Political Economy*, 82(1), 34–55.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/Hedonic_regression).
- **Ridge:** [§5.2]
  Hoerl, A. E. y Kennard, R. W. (1970). Ridge regression: biased estimation for nonorthogonal problems.
  *Technometrics*, 12(1), 55–67.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/Ridge_regression).
- **Lasso:** modelo lineal elegido. [§5.2]
  Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. *Journal of the Royal Statistical
  Society, Series B*, 58(1), 267–288.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/Lasso_(statistics)).
- **ElasticNet:** [§5.2]
  Zou, H. y Hastie, T. (2005). Regularization and variable selection via the elastic net. *Journal of the Royal
  Statistical Society, Series B*, 67(2), 301–320.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/Elastic_net_regularization).
- **kNN (k vecinos más cercanos):** [§5.2]
  Cover, T. y Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory*,
  13(1), 21–27.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm).
- **PCAmix:** reducción de datos mixtos para el kNN. [§5.2]
  Chavent, M., Kuentz-Simonet, V., Labenne, A. y Saracco, J. (2014). Multivariate analysis of mixed data: the R
  package PCAmixdata. *arXiv:1411.4911*.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/Factor_analysis_of_mixed_data).
- **Árbol de regresión (CART):** [§5.2]
  Breiman, L., Friedman, J. H., Olshen, R. A. y Stone, C. J. (1984). *Classification and Regression Trees*.
  Wadsworth.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/Decision_tree_learning).
- **Random Forest e importancia por permutación:** [§5.2, §6.1]
  Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/Random_forest).
- ***Gradient boosting*:** base de HGB y CatBoost. [§5.2]
  Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *The Annals of Statistics*,
  29(5), 1189–1232.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/Gradient_boosting).
- **HGB (*HistGradientBoosting*):** el de scikit-learn, inspirado en LightGBM. [§5.2]
  Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. y Liu, T.-Y. (2017). LightGBM: a highly efficient
  gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/LightGBM).
- **CatBoost:** modelo final. [§4, §5.3]
  Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V. y Gulin, A. (2018). CatBoost: unbiased boosting with
  categorical features. *Advances in Neural Information Processing Systems*, 31.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/CatBoost).
- **SVR (regresión de vectores soporte):** [§5.2]
  Drucker, H., Burges, C. J. C., Kaufman, L., Smola, A. y Vapnik, V. (1997). Support vector regression machines.
  *Advances in Neural Information Processing Systems*, 9, 155–161.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/Support_vector_machine).
- **Aproximación de Nyström:** aproximación del núcleo de la SVR. [§3.6]
  Williams, C. K. I. y Seeger, M. (2001). Using the Nyström method to speed up kernel machines. *Advances in Neural
  Information Processing Systems*, 13, 682–688.
- **Codificación por la variable objetivo (*target encoding*):** `TargetEncoder` del barrio en HGB. [§6.4]
  Micci-Barreca, D. (2001). A preprocessing scheme for high-cardinality categorical attributes in classification
  and prediction problems. *ACM SIGKDD Explorations*, 3(1), 27–32.
- **Red neuronal (perceptrón multicapa, MLP):** entrenamiento por retropropagación del error. [§5.2]
  Rumelhart, D. E., Hinton, G. E. y Williams, R. J. (1986). Learning representations by back-propagating errors.
  *Nature*, 323(6088), 533–536.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/Multilayer_perceptron).
- **AdamW:** optimizador de la red, con *weight decay* desacoplado. [§5.2]
  Loshchilov, I. y Hutter, F. (2019). Decoupled weight decay regularization. *International Conference on Learning
  Representations (ICLR)*.
- **Árboles frente a redes en datos tabulares:** por qué se esperaba que el MLP no ganara a los *boosting*. [§5.2]
  Grinsztajn, L., Oyallon, E. y Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on
  typical tabular data? *Advances in Neural Information Processing Systems*, 35 (Datasets and Benchmarks Track).

### H.5 Rango de precio e interpretabilidad

- **Regresión cuantílica:** modelos cuantílicos. [§5.5]
  Koenker, R. y Bassett, G. (1978). Regression quantiles. *Econometrica*, 46(1), 33–50.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/Quantile_regression).
- **CQR (regresión cuantílica conformalizada):** calibración del intervalo. [§5.5]
  Romano, Y., Patterson, E. y Candès, E. (2019). Conformalized quantile regression. *Advances in Neural Information
  Processing Systems*, 32.
  Lectura: [Wikipedia](https://en.wikipedia.org/wiki/Conformal_prediction).
- **SHAP:** [§6.1]
  Lundberg, S. M. y Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural
  Information Processing Systems*, 30.
- **TreeSHAP:** SHAP para árboles, el que usa CatBoost. [§6.1]
  Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., *et al.* (2020). From local
  explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56–67.

