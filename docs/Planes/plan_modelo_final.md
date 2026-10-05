# Plan: notebook del modelo final y evaluación en el test (Fases 9 y 10)

**Estado (2026-10-02):** CatBoost elegido (RMSE en log de CV 0,4160; segundo, HGB, 0,4191). F1-F7 resueltas con el
alumno (ver «Resolución» abajo); el notebook `Modelo_final.ipynb` está planteado y **sin ejecutar**. Solo queda
rellenar las conclusiones tras ejecutarlo.

**Notebook:** `notebooks/Modelo_final.ipynb`. **Funciones nuevas:** `scripts/func_final.py`. `RegresorCatBoost` gana
el argumento `loss_function` (por defecto `'RMSE'`; el repr de los modelos ya registrados no cambia).

---

## 0. Reglas del juego (antes de abrir el test)

1. **El test se abre una sola vez.** Todo lo que se vaya a medir en el test (modelos, métricas, segmentos,
   análisis) se fija en este documento **antes** de ejecutarlo. Nada de lo que salga en el test puede cambiar la
   elección del modelo, los hiperparámetros ni las variables.
2. **La elección del modelo es la de `Modelado .ipynb` §11:** menor RMSE medio de validación en 5 folds, escenario
   B (bitácora, 09-30).
3. **Los hiperparámetros, las variables y la codificación se leen del registro** (`outputs/experimentos.csv`) o de
   un fichero que escriba `Modelado .ipynb`, no se copian a mano.
4. **Integridad del test:** antes de cargarlo se comprueba su MD5 contra `data/particion/particion.json`
   (`6efbad2d…`). Si no coincide, se para.

---

## 1. Decisiones pendientes (se cierran con el alumno)

| ID | Decisión | Alternativas | Recomendación del analista |
|---|---|---|---|
| **F1** | Qué modelos se evalúan en el test | (A) Solo el elegido. (B) El elegido + contrastes **fijados ahora**: el lineal y la mejor referencia ingenua (N0.3). (B') Como B, más el segundo de §11 si queda a menos de 1 SE. (C) Varios y elegir en el test: **descartada**, invalida el test | **B**. B' solo si el segundo está a menos de 1 SE, y declarando que **no** cambia la elección aunque gane en el test |
| **F2** | Fila informativa del lineal con 79 variables (L5) | Incluirla en §11 y en el test como "no elegido"; no incluirla | Incluirla en §11 (ya está en el registro). En el test, solo si F1 = B |
| **F3** | Calibración del intervalo de precio | (a) Cuantílica tal cual (cobertura de CV 72,8%). (b) **CQR** (*conformalized quantile regression*, Romano et al., 2019), calibrada con las predicciones fuera de fold de la CV. (c) Sin intervalo | **(b)**: ensancha el intervalo lo justo para cubrir el 80% y se calibra sin apartar más datos |
| **F4** | SHAP | Instalar `shap` (supera D7) para el modelo final; solo PDP/ICE + permutación | Instalar `shap`: es la forma de explicar "por qué este piso vale $X" al sponsor. Con CatBoost también vale su SHAP nativo |
| **F5** | Escenario A en el test | Solo en CV (bitácora, 09-30); también en el test | Solo en CV, como está decidido |
| **F6** | Qué se guarda para usar el modelo después | Solo el `Pipeline` (`joblib`); `Pipeline` + preparador + ficha; además, una función de predicción | `Pipeline` + preparador + ficha + función `predecir_precio` (sección 4) |

### Resolución (2026-10-02, decisiones del alumno)

| ID | Resuelto | Consecuencia en el plan |
|---|---|---|
| F1 | **A: solo CatBoost en el test** (la recomendación era B') | Sin contrastes en el test. Lineal, HGB y N0.3 se comparan con la CV pareada (§1 del notebook). La ventaja sobre HGB (0,0031 ± 0,0029, 5/5 folds) queda a ~1 SE y se declara no concluyente |
| F2 | Lineal-79 solo en §11 de `Modelado` | Como F1 = A, no entra en el test |
| F3 | **CQR con un solo conjunto de calibración** (ajustado tras AUD-001) | Cuantiles ajustados con el train del fold 0 y calibrados con su validación; son los cuantiles finales. Sustituye a la CQR cruzada de 5 folds (13 ajustes → 3) |
| F4 | **SHAP nativo de CatBoost** | Sin instalar `shap`; gráficos propios (|SHAP| medio, dependencia y 3 anuncios elegidos con regla fijada) |
| F5 | Escenario A solo en CV | Sin cambios |
| F6 | Modelo + preparador + ficha + `predecir_precio` | `models/` con `.joblib` y `.cbm` nativo |
| **F7 (nueva)** | **Cuantiles de CatBoost** (0,1 y 0,9, mismos hiperparámetros) en lugar de HGB (Q8) | Se ajustan 2 modelos (80% del train, fold 0). Q8 queda como referencia (cobertura 72,8%) |

Detalle abierto: `predecir_precio` calcula `n_anuncios_ny` sobre las filas que recibe; para un anfitrión con varios
anuncios hay que pasar su cartera completa.

---

## 2. Índice del notebook

### 0. Introducción
Qué se hace, qué modelos entran (F1) y la regla de que el test se abre una vez.

### 1. Carga
- Train de modelado y bloques (`cargar_modelado('train')`).
- Configuración del modelo elegido y de los contrastes: hiperparámetros y variables leídos del registro.
- Comprobación del MD5 del test. **El test no se carga hasta la sección 4.**

### 2. Reentrenamiento con todo el train
- El elegido (y los contrastes de F1) sobre los 29.538 anuncios del train, con sus variables (79 o 71).
- Tiempo de ajuste. Comprobación: el RMSE en train coincide con el orden de magnitud de la CV.
- Cuantílica 0,1 / 0,5 / 0,9 con los hiperparámetros del elegido (si F3 ≠ c).
- **Calibración CQR (F3 = b):** con las predicciones fuera de fold de los 5 folds de la CV (agrupados por
  anfitrión), se calcula cuánto se quedan cortos los cuantiles y la corrección que lleva la cobertura al 80%.

### 3. Guardado del modelo (detalle en la sección 4 de este plan)
`models/modelo_final.joblib`, `models/cuantiles.joblib`, `models/ficha_modelo.json`.

### 4. Evaluación en el test (única)
- Carga del test, preparado con el preparador ajustado en el train (`listings_ny_modelado_test.parquet`).
- **Métricas:** RMSE, MAE y R² en log; MAE y mediana del error en dólares; MAPE. Las mismas que en la CV.
- **Intervalo de confianza con *bootstrap* por anfitrión** (se remuestrean anfitriones, no anuncios; 2.000
  réplicas): para cada métrica y para la diferencia pareada con cada contraste.
- **CV frente a test:** si el test sale fuera del rango de la CV (media ± 2 desviaciones), se discute: sobreajuste
  a la validación por la búsqueda de hiperparámetros o diferencia de composición.
- **Contrastes (F1):** mejora del elegido sobre el lineal y sobre N0.3, en % de RMSE y en dólares.
- **Comparación con la literatura:** R² en log frente al paper de referencia (`docs/Referencias/1907.12665v1.pdf`;
  **pendiente de verificar su cifra**), explicando que usa split aleatorio y reseñas como variable.

### 5. Intervalo de precio en el test
- Cobertura real del intervalo, sin calibrar y calibrado (CQR), global y por `room_type`, `district` y quintil de
  precio. Anchura mediana en dólares.

### 6. Análisis de errores
- Error por segmento: `room_type`, `district`, tramo de capacidad y quintil de precio (MAE en dólares y MAPE).
  Toda tabla lleva su `n` y se enmascaran las celdas con N < 30 (H-014).
- **Mapa de residuos del test**, como en la OLS (§5.3 de `Modelado`): ¿quedan manchas espaciales?
- **Los 20 peores errores:** qué tienen en común (¿lujo no visible en las variables? ¿precios de bloqueo que se
  escaparon de la regla?).
- **Sensibilidad a la regla de exclusión (H-024):** métricas excluyendo el P99 global frente a la regla de
  bloqueo. Está prometido en la bitácora.

### 7. Interpretación
- **Importancia por permutación en el test**, por bloques y por variable (lat/lon juntas), frente a la de CV
  (`Modelado` §12). ¿Se mantiene el orden?
- **PDP + ICE** de `accommodates`, `dist_centro_km` y `minimum_nights`; **PDP 2D de lat/lon** como mapa de precio.
- **SHAP (F4):** resumen global (*beeswarm*) y explicación de 2-3 anuncios concretos elegidos de antemano
  (uno típico, uno caro bien predicho y uno de los peores errores).
- **Lineal frente a final:** coeficientes hedónicos (`Modelado` §5.2) frente a SHAP medio. ¿Dónde coinciden y
  dónde no (no linealidades, interacciones)?

### 8. Uso del modelo guardado
- Se carga `modelo_final.joblib` en limpio y se predice un anuncio de ejemplo **en bruto** (una fila de
  `listings.csv`) con `predecir_precio`: precio mediano y el intervalo calibrado.
- Comprobación: la predicción coincide con la del modelo en memoria.

### 9. Conclusiones y recomendaciones
- Para el analista: resultado en el test con su intervalo, qué funciona y dónde falla.
- Para el sponsor: "un anuncio nuevo se predice con un error típico de $X", qué influye en el precio y el rango.
- `exp(ŷ)` estima la **mediana** del precio. Si se necesita la media, *smearing* de Duan (bitácora, 09-23).
- Limitaciones (sección 13 de `plan_modelado.md`) y líneas futuras.

---

## 3. Funciones nuevas

| Función | Qué hace |
|---|---|
| `bootstrap_por_anfitrion(y, pred, grupos, metrica, n, semilla)` | Intervalo de confianza de una métrica (o de una diferencia pareada) remuestreando anfitriones |
| `calibrar_cqr(pred_bajo, pred_alto, y, cobertura)` | Corrección conformal de los cuantiles con predicciones fuera de fold |
| `predecir_precio(anuncio_bruto, preparador, modelo, cuantiles, correccion)` | De una fila de `listings.csv` a precio mediano e intervalo en dólares |
| `error_por_segmento(datos, pred, columnas, min_n=30)` | Tabla de MAE y MAPE por segmento con su `n` |

---

## 4. Guardar el modelo para usarlo después

- **`joblib.dump` del `Pipeline` entero** (preprocesado + estimador): el `TargetEncoder` del barrio, las
  imputaciones y la codificación viajan con el modelo. `func_modelado.py` ya está preparado: las transformaciones
  son funciones con nombre, no lambdas, para que se puedan guardar.
- **El preparador** (`PreparadorListings`, ya guardado en `data/preparador_ny.json`) convierte un anuncio en bruto
  en las variables del modelo. Cadena completa: anuncio bruto → preparador → `Pipeline` → `exp(ŷ)`.
- **Ficha del modelo** (`models/ficha_modelo.json`): modelo y hiperparámetros, variables, versiones de Python,
  `scikit-learn`, `catboost` y `pandas`, métricas de CV y de test con su intervalo, corrección CQR, fecha y MD5 del
  train. Un `.joblib` solo se carga con seguridad con las mismas versiones: se fijan en `requirements.txt`.
- Si el final es CatBoost, el envoltorio `RegresorCatBoost` se guarda igual con `joblib`. Además, como respaldo,
  `save_model` nativo de CatBoost (más estable entre versiones).
- `models/` se añade a `.gitignore` si los ficheros pesan mucho (un RF de 500 árboles puede ocupar cientos de MB).

---

## 5. Coste estimado

| Parte | Tiempo de cálculo | Trabajo |
|---|---|---|
| Reentrenar elegido + contrastes + cuantílica | 1-5 min (HGB) / ~2 min en CPU (CatBoost, con todo el train) | Bajo. La búsqueda oficial de CatBoost se hizo en CPU (bitácora, 10-02): se reentrena en CPU. La GPU de Colab solo se usa como prueba de velocidad |
| Bootstrap (2.000 réplicas) | < 1 min (solo predicciones) | Bajo |
| CQR | Segundos (reutiliza predicciones fuera de fold; hay que generarlas: 3 × 5 ajustes) | Medio |
| Análisis de errores, mapa y sensibilidad H-024 | Minutos | Medio |
| PDP/ICE, permutación | Minutos | Bajo |
| SHAP | Minutos con `TreeExplainer` | Medio (instalación y gráficos) |
| Guardado y `predecir_precio` | Segundos | Bajo-medio |

Total: **2-3 días de trabajo**, con poco cálculo. Encaja antes de la aprobación del tutor (2026-10-16).

---

## 6. Verificación

- El MD5 del test coincide antes de cargarlo, y el test solo se carga una vez en todo el notebook.
- Los hiperparámetros y las variables del modelo reentrenado coinciden con los de `Modelado` §11 (se comprueba
  el `repr` contra el registro).
- Las métricas del test se calculan con la misma función `metricas` que la CV.
- `modelo_final.joblib` se carga en una sesión limpia y predice igual que el modelo en memoria.
- Cada sección cierra con `#### Conclusiones:` con números reales.
