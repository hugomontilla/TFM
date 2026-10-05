# Plan: variables, ajuste de hiperparámetros dentro de cada modelo, kNN con PCAmix/FAMD y gráfico final

## Contexto
`notebooks/Modelado .ipynb` tiene problemas detectados por el alumno:
1. §10 (ajuste) va antes que §11 (cuantílica, SVR, CatBoost) y separado de la definición de cada modelo.
2. §10.1 concluye que el modelo completo (79 variables) gana a la selección en RF y HGB (Δ ≈ 0,005; 5/5 folds; p = 0,009 y 0,022). Esas familias se ajustaron sobre un conjunto (68, `VARS_HGB`) elegido con Ridge y HGB por defecto. En el lineal el completo no gana de forma significativa (p = 0,12).
3. El gráfico comparativo final (§12, celda con `ax.barh(orden['modelo'], ...)`) solapa las etiquetas de RMSE con las barras de error y solo colorea de azul el mejor modelo.
4. kNN con PCA (K2) rinde peor que K1 y PCA trata mal las binarias y categóricas.

## Decisiones (cerradas con el alumno)

**D1. Variables por familia**
| Familia | Variables | Motivo |
|---|---|---|
| Lineal (Ridge, ElasticNet) | `VARS_RIDGE` (71) | La selección se hizo con Ridge y el completo no gana de forma significativa |
| Ensembles y árboles (RF, HGB, CatBoost, CART), cuantílica, SVR | `COLS_B` (79) | Eligen por sí mismos qué usan; seleccionar previamente perjudica (§10.1) |
| kNN | `VARS_RIDGE` (71) | Las distancias sí sufren con variables inútiles |

**D2. `VARS_HGB` (68) desaparece de todo el notebook.** Se sustituye por `COLS_B` en CART (§8.2), RF y HGB por defecto (§9.1), codificación del barrio (§9.2), valor por defecto de `columnas` en `construir_rf`/`construir_hgb`/`construir_catboost`, cuantílica (§11.1), SVR (§11.2), CatBoost (§11.3), escenario A (`VARS_HGB_A` → `COLS_B` sin el bloque del anfitrión) y la comprobación de §7.3. **Se borra la selección con HGB de §7.3**; la evidencia queda en §10.1 y en la bitácora. Se conserva la selección con Ridge.

**D3. Ajuste de hiperparámetros dentro de la definición de cada modelo**, no en una sección aparte.
- RF y HGB: la búsqueda pasa a §9 junto a su versión por defecto, con 79 variables. Hay que mover también `buscar()` (hoy en §10). Se mantienen los nombres `RF_ajustado` y `HGB_ajustado`.
- Lineal: se **conserva `T7_ridge`/`T7_elasticnet`** (alpha y l1_ratio sobre las 71), reubicado junto al lineal tras §7.3, y se **compara con las 79** de L5 (§6) con los mismos folds. `MEJOR_RIDGE` sale de esa comparación; revisar su uso en §10.1 y §12 (etiqueta "L4 + selección, ajustado").
- SVR, CatBoost, cuantílica: ya tienen búsqueda propia; CatBoost pasa a 79.
- §10.1 queda como constancia del resultado (la selección perjudica a RF y HGB). Se quita de `comprobar_seleccion` lo que ya no haga falta; el SVR se sigue comprobando.

**D4. kNN.** Se mantiene en la comparativa (no se elimina) y **no se fija umbral de cercanía**: se calcula y se compara con el resto.
- Entrada: las 71 variables de Ridge.
- **PCA (K2) se sustituye por PCAmix y por FAMD**, y se comparan entre sí, con K1 y con el mejor modelo, en pareado con los mismos folds. El número de componentes se fija con el mismo criterio que K2 (mínimo que explica ≥ 80 % de la varianza), salvo que se decida otra cosa.
- `prince` ya está instalado (el alumno). Pendiente de comprobar: `prince` implementa FAMD, pero no está claro que tenga PCAmix (en R existe `PCAmixdata`). Verificar qué clases ofrece la versión instalada; si no hay PCAmix, implementarlo a mano o usar `rpy2`. Añadir `prince` a `requirements.txt` con la versión instalada.
- Otros métodos considerados y descartados por coste/defensa: búsquedas tipo forward/backward/genéticos/SA/VNS (wrappers con miles de evaluaciones y validación anidada); filtros y búsqueda por bloques quedan como alternativa si PCAmix/FAMD no mejoran.

**D5. Importancia de variables.**
- **Permutación sobre validación** en todos los modelos (también el lineal). Por **bloques** (`data/bloques_modelado.json`) en todos y en 1-2 folds; **variable a variable** solo en el modelo ganador y en el lineal, reentrenando una vez. Medir el tiempo antes; si es caro, bajar a 1 fold o a 3 repeticiones.
- Importancia por impureza (reducción de varianza, MDI) solo para CART, como ilustración de su sesgo.
- Lineal: variables que anulan Lasso y ElasticNet y en cuáles coinciden (comprobar antes qué hay: §6.1 cuenta lo que anula Lasso y existe `outputs/estabilidad_elasticnet.csv`).
- **SHAP** solo para el modelo final en la Fase 10.

## Cambios

### 1. Notebook
- Aplicar D1-D4 según arriba. Renumerar encabezados y referencias cruzadas (`§10.1`, `§10.2`, `§11.2`, comentarios "10.2 para el SVR" y "10.1 para CatBoost") y revisar qué celdas de §12 dependen de las variables anteriores.
- **Reejecutar** con 79 variables: §8.2 (CART), §9.1, §9.2, RF (30 combinaciones), HGB (60), CatBoost (40), cuantílica y SVR. Reejecutar el lineal con 71 (T7) y kNN con PCAmix/FAMD. Todo se guarda en `outputs/experimentos.csv`; los experimentos antiguos con 68 quedan como histórico y §12 lee los nuevos.
- Estimar el tiempo total antes de lanzar.

### 1b. Celdas de conclusión (obligatorio en todo lo que se añada)
Cada bloque nuevo o modificado del notebook debe cerrar con una celda Markdown `#### Conclusiones:` (como las que ya hay en el notebook) que explique **qué se concluye y por qué**, con los números reales del resultado, no con texto genérico. Se escribe tras ejecutar, no antes. Como mínimo, una conclusión en:
- Ajuste dentro de cada modelo: RF, HGB, CatBoost, lineal (T7 con 71 frente a L5 con 79), con la mejora respecto a la versión por defecto y su comparación pareada.
- §10.1: la selección perjudica a RF y HGB y por qué se usan las 79.
- kNN con PCAmix y FAMD: número de componentes, qué recogen, RMSE frente a K1, K2 y al mejor modelo, y si merece la pena.
- Importancia por permutación (bloques y variable a variable): qué bloques y variables pesan, si coinciden entre familias y con lo que anulan Lasso/ElasticNet; en CART, el sesgo de la impureza frente a la permutación.
- Reejecución de §8.2, §9.1, §9.2, cuantílica y SVR con 79 variables: si cambia algo respecto a las 68.
- Gráfico final (§12): qué modelos quedan en cabeza y cuáles son indistinguibles dentro de 1 SE.
Cada conclusión debe distinguir lo observado de lo interpretado, y señalar qué se descarta o queda pendiente.

### 2. Gráfico comparativo (§12)
En la celda con `ax.barh(orden['modelo'], orden['rmse_log_val'], xerr=orden['se'], ...)`:
- Colorear de azul (`COLOR_BASE`) los **tres modelos con menor RMSE**; el resto en `COLOR_SECUNDARIO`. Hoy solo se colorea `mejor['modelo']`.
- Etiquetas fuera de la barra de error: `x = valor + se + margen`, o columna fija a la derecha con `ax.text(..., transform=ax.get_yaxis_transform())`.
- Ampliar `set_xlim` por la derecha y ajustar el texto "mejor + 1 SE" para que no se pise.
- Título: "en azul, los tres de menor RMSE de validación".
- Si K2 se sustituye, mantener en el gráfico la mejor variante de kNN y, como mucho, la mejor de PCAmix/FAMD.

### 3. Documentación
- `docs/bitacora_decisiones.md`: nueva fila (fecha de hoy) con D1-D5. Alternativas descartadas: reseleccionar por modelo; mantener 68; borrar la selección de Ridge; wrappers para kNN. Anotar que supera la fila "Comprobación de la selección en cada modelo ajustado (§10.2)" y la herencia del conjunto de HGB por CatBoost.
- `docs/hallazgos_y_pendientes.md`: registrar el hallazgo de §10.1, la importancia por permutación hecha ya (adelanta D7 de `plan_modelado.md`) y SHAP pendiente en la Fase 10.

## Verificación
- Reejecutar el notebook de arriba abajo con el kernel del proyecto (folds y partición guardados en `data/particion/` y `outputs/folds_cv.csv`).
- `outputs/experimentos.csv` contiene los nuevos experimentos y §12 lee la tabla sin errores.
- `MEJOR_RIDGE` y las variables que usaba §10 siguen definidas; no queda `VARS_HGB` (buscar con grep) ni referencias a la numeración antigua.
- Gráfico final: etiquetas legibles, tres barras azules y línea de "mejor + 1 SE" visible.
- La comparación pareada de §12 usa los mismos folds que el resto.
- Cada bloque nuevo tiene su celda `#### Conclusiones:` con números reales (1b).
