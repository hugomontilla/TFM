# Plan de refactorización: ficheros de partición y de modelado

Estado: **ejecutado (2026-10-05) con la opción A: se mantiene el parquet.** Estructura final:

```
data/
  particion/   listings_train.csv, listings_test.csv, particion.json
  modelado/    train.parquet, test.parquet, preparador.json
```

Comprobado tras el cambio: los parquet y el preparador se regeneran idénticos desde los CSV (`assert_frame_equal`,
preparar train y test tarda ~4,5 s), los MD5 de la partición no cambian, y `cargar_modelado` devuelve los mismos
bloques. `particion.json` se recalcula con `python scripts/particion_datos.py --metadatos`.

## 1. Problema

Seis cosas hablan de lo mismo: la partición train/test y su versión preparada para modelar.

| Fichero | Qué contiene | Quién lo escribe | Quién lo lee |
|---|---|---|---|
| `data/particion/listings_train.csv`, `listings_test.csv` | Filas en bruto de `listings.csv` (33 columnas) | `particion_datos.py` | `2-ANALISIS`, el preparador, el MD5 de los notebooks finales |
| `data/particion/particion.csv` | `listing_id`, `host_id`, `conjunto` | `particion_datos.py` | Nadie (`1-EXPLORACION` define la ruta y no la usa) |
| `data/particion/particion.json` | Semilla, recuentos y MD5 | **Ningún script** | Los notebooks finales, para comprobar el test |
| `data/preparador_ny.json` | Lo aprendido con el train: amenities, tipos de propiedad, medianas | `preparacion_datos.entrenar()` | `2-ANALISIS`, notebooks finales |
| `data/bloques_modelado.json` | Bloques de variables | `entrenar()` | `cargar_modelado()` |
| `data/listings_ny_modelado_{train,test}.parquet` | Tabla preparada (83 columnas) | `entrenar()` | `cargar_modelado()` (`3-MODELADO` y finales) |
| ~~`data/variables_seleccionadas.json`~~ | Código muerto en el flujo principal | | **Borrado** |

Problemas de lógica detectados:
1. `particion.json` no lo genera ningún script: el MD5 que se comprueba no tiene origen reproducible.
2. `particion.csv` y `bloques_modelado.json` son derivables de otros ficheros.
3. La ficha del modelo guarda `md5_train_parquet` y `md5_test_parquet`, pero ahí va el MD5 de los CSV de la
   partición, no de los parquet.

## 2. Decisiones ya tomadas

- Borrar `data/variables_seleccionadas.json` (hecho).
- Eliminar `particion.csv` (redundante).
- Generar `particion.json` desde el script de partición, para que su origen sea reproducible.
- Quitar `bloques_modelado.json`: los bloques se obtienen de `PreparadorListings.bloques_`.
- Renombrar `md5_*_parquet` a `md5_*_particion` en la ficha del modelo.

## 3. Decisión pendiente: ¿parquet o solo CSV?

Idea del alumno: si se tienen los CSV de train y test, el parquet se puede generar a partir de ellos, así que
basta con usar los CSV (en `2-ANALISIS` y también en `3-MODELADO`, que hoy lee el parquet).

| | A. Mantener parquet | B. Solo CSV |
|---|---|---|
| Ficheros de datos | CSV + 2 parquet + preparador | CSV + preparador (o ni preparador) |
| Peso en el repo | CSV 22 MB + parquet 2,7 MB | 22 MB |
| `3-MODELADO` | Lee el parquet (instantáneo) | Aplica `preparador.transform` al CSV cada vez que carga: hay que medir cuánto tarda |
| Fuente de verdad | Dos copias que pueden desfasarse | Una sola |
| Tipos de columna | El parquet conserva los tipos (categóricas, booleanas) | Se recalculan en `transform`, que ya los fija |
| Barrera del test | Igual | Igual (el test sigue en su propio fichero) |

Nota sobre el peso: el parquet pesa **menos** que el CSV (2,7 MB frente a 22 MB), porque comprime y no arrastra
las columnas de texto. Lo que gana B no es peso sino tener una sola fuente.

Variante B': ni siquiera guardar `preparador_ny.json`; ajustar `PreparadorListings.fit(train)` al cargar. Es
determinista, pero cada notebook paga el ajuste y se pierde el JSON legible que se cita en la memoria.

**Pendiente de medir antes de decidir:** tiempo de `transform` sobre el train, tiempo de `fit`, y comprobar que
`transform` reproduce exactamente el parquet actual (`assert_frame_equal`).

Recomendación provisional: **B** si el `transform` tarda unos segundos; **A** si tarda minutos, porque
`3-MODELADO` carga los datos en varios puntos.

## 4. Estructura objetivo

Con A:
```
data/
  particion/   listings_train.csv, listings_test.csv, particion.json
  modelado/    train.parquet, test.parquet, preparador.json
```
Con B:
```
data/
  particion/   listings_train.csv, listings_test.csv, particion.json, preparador.json
```
En ambos casos desaparecen `particion.csv`, `bloques_modelado.json` y `variables_seleccionadas.json`.

## 5. Cambios por fichero

**Scripts**
| Fichero | Cambio |
|---|---|
| `particion_datos.py` | Quitar `manifiesto` de `RUTAS` y de `crear_particion`; escribir `particion.json` (semilla, método, recuentos, MD5, versiones) |
| `preparacion_datos.py` | Rutas del preparador (y de los parquet si A) a su carpeta nueva; `entrenar()` deja de escribir `bloques_modelado.json`. Con B, nueva función que devuelve el dataset de modelado a partir del CSV |
| `func_modelado.py` | `cargar_modelado()` obtiene los bloques de `PreparadorListings.cargar(...).bloques_`; con B, prepara desde el CSV en vez de leer el parquet |
| `func_final.py` | Claves de la ficha: `md5_train_particion`, `md5_test_particion` |

**Notebooks** (ninguno hay que reejecutarlo si las rutas se mueven y los resultados no cambian)
| Notebook | Cambio |
|---|---|
| `1-EXPLORACION` | Quitar `RUTA_MANIFIESTO` y el import de `RUTAS`; corregir la tabla de texto de la sección 3 |
| `2-ANALISIS` | Nada en el código (usa constantes de ruta) |
| `3-MODELADO` | Texto de la cabecera que cita `listings_ny_modelado_train.parquet` y `bloques_modelado.json` |
| `4-MODELO_FINAL`, `Modelo_final_HGB` | Nada en el código: el MD5 sigue leyendo `particion/listings_test.csv` y `particion.json` |

**Otros:** `models/ficha_*.json` (claves nuevas), README, `docs/arquitectura.md`, bitácora y el `.gitignore`.
La carpeta de Colab trae copias de estos scripts; se actualiza o se elimina (decisión aparte).

## 6. Orden de ejecución

1. Medir tiempos y comprobar equivalencia (sección 3). Decidir A o B.
2. Hacer que `crear_particion` escriba `particion.json`. Comprobar que los MD5 coinciden con los actuales
   (`33b8368667c428a1014afefdc838056e` train, `6efbad2d0dd977cb41d7feb0ee114f38` test).
3. Quitar el manifiesto y `bloques_modelado.json`.
4. Mover ficheros con `git mv` y ajustar las rutas en los scripts.
5. Actualizar los notebooks (solo texto y el import de `1-EXPLORACION`).
6. Renombrar las claves MD5 de la ficha y regenerar `models/ficha_*.json`.
7. Actualizar documentación.

## 7. Comprobaciones

- El dataset de modelado de train y de test es idéntico al actual (`assert_frame_equal`).
- Los MD5 de la partición no cambian.
- `1-EXPLORACION` (celdas de importación) y la primera celda de `3-MODELADO` funcionan con las rutas nuevas.
- `predecir_precio` sigue devolviendo lo mismo con el modelo guardado (celda «Uso del modelo guardado»).
- `git grep` no encuentra rutas antiguas.

## 8. Riesgos y vuelta atrás

- Todo el trabajo va en una rama o commit aparte; el historial anterior sigue en `backup-historial-2026-10-05`.
- Riesgo principal: cambiar el contenido del dataset de modelado sin darse cuenta. Lo cubre el
  `assert_frame_equal` del paso 1 y de la sección 7.
