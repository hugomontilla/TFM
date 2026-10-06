# Contexto común a todos los roles del TFM

Este fichero lo leen todas las skills `tfm-*`. Resume lo que es fijo en el proyecto y dónde está cada
cosa. Lo que cambia (estado de las decisiones, fase actual) **no se copia aquí**: se lee siempre de su
fuente.

## Capas de instrucciones (de más general a más concreta)

1. **Skill TFM base** (`.agents/skills/TFM/SKILL.md`, también disponible como
   `tfm-airbnb-analista-profesor`): doble rol analista objetivo + profesor socrático, estructura de la
   memoria (analista / sponsor / apéndices), formato `#### Conclusiones:`, estilo de código. **Siempre
   activa**; los roles trabajan encima de ella, no la sustituyen.
2. **`docs/AGENTS_TFM.md`**: sistema de roles, protocolo de activación, reglas de experimentación.
3. **`docs/plan_modelado.md`**: hoja de ruta vigente (fases 0-10, protocolo A1-A12, decisiones D1-D9).
   Si algo de AGENTS_TFM choca con el plan, **manda el plan** (es posterior y está consensuado) y se avisa
   al alumno del choque.

## Roles vigentes y registros (acordado el 2026-09-26, recogido en AGENTS_TFM)

- **Cuatro roles:** `tfm-director`, `tfm-ml-engineer` (con Feature Engineer), `tfm-analista` (Data
  Scientist + Analista de resultados) y `tfm-juzgado` (con Tribunal de defensa). Cuándo usar cada uno:
  `tfm-director/SKILL.md`.
- **Solo Nueva York.** Lo que AGENTS_TFM plantea "por ciudad" se hace **por `district`** y, si hay muestra,
  por barrio.
- **Decisiones:** `docs/bitacora_decisiones.md` + `D1-D9` del plan para las pendientes (no hay DEC-xxx).
- **Experimentos:** `outputs/experimentos.csv` (numérico, una fila por fold) + `docs/registro_experimentos.md`
  (pregunta, hipótesis, estado, decisión). IDs del plan: `N0.x`, `L1-L5`, `A1-A12`; nombres descriptivos
  para el resto.
- **No se reorganizan carpetas.** Solo se crean `docs/auditorias/` y `docs/defensa/` cuando el Juzgado las
  necesite. No hay `project_log.md`.

## Mapa de ficheros

| Qué | Dónde |
|---|---|
| Datos brutos | `data/listings.csv`, `data/reviews.csv` |
| Dataset de modelado | `data/modelado/{train,test}.parquet`; los bloques salen de `PreparadorListings.bloques_` |
| Preparación de datos (única fuente de verdad de las *features*) | `scripts/preparacion_datos.py` (`PreparadorListings`; se ejecuta sin argumentos); parámetros en `data/modelado/preparador.json` |
| Funciones de modelado | `scripts/func_modelado.py` (`cargar_modelado`, `generar_folds`, `evaluar`, `registrar`, `resumir`); partición en `scripts/particion_datos.py` |
| Funciones del EDA | `notebooks/func_aux.py` |
| Notebooks | `notebooks/EDA_NY.ipynb`, `notebooks/Modelado.ipynb` (`EDA.ipynb` = 10 ciudades, solo evidencia) |
| Partición y folds guardados | `data/particion/` (train, test y manifiesto), `outputs/folds_cv.csv` |
| Resultados por fold | `outputs/experimentos.csv` |
| Decisiones | `docs/bitacora_decisiones.md` |
| Hallazgos y pendientes (H-xxx) | `docs/hallazgos_y_pendientes.md` — **los IDs H-xxx son estables**, los notebooks los citan |
| Guía oficial del máster | `docs/Guia/indicaciones.pdf` |
| Paper de referencia | `docs/Referencias/1907.12665v1.pdf` |

## Reglas que ningún rol puede saltarse

- **El test es intocable.** Se evalúa una sola vez, con el modelo final (Fase 9). Ningún rol mira métricas
  de test para decidir nada antes de eso.
- **Toda partición agrupa por `host_id`** (H-020). Se usan los folds guardados, no se regeneran.
- **Todo lo que aprende de los datos va dentro del `Pipeline`.** Excepciones ya declaradas como limitación:
  imputación de `bedrooms` (H-033) y decisiones del EDA con todo el dataset (H-035).
- **El preview del 09-23 no es evidencia** (H-030). Nunca se citan sus cifras como justificación.
- **Toda *feature* nueva se implementa en `preparacion_datos.py`** y se añade a `PreparadorListings.bloques_`, no
  solo en un notebook.
- **Todo experimento se registra, salga bien o mal.** Nunca se borra un resultado.
- **La regla de decisión se fija antes de mirar** (plan §5.3, D4). No se cambia la métrica principal
  (RMSE en log) a la vista de los resultados.
- **Asociación ≠ causalidad.** Importancia, SHAP o coeficientes son asociación predictiva.
- No inventar resultados, fuentes ni ejecuciones. Si no se ha ejecutado, se dice.

## Fechas

Aprobación del tutor **antes del 2026-10-16**, entrega el **2026-10-23**, defensa en la primera quincena
de noviembre de 2026. El calendario por fases está en `docs/plan_modelado.md` §12. Si una tarea no cabe,
se recorta por el orden de prioridad de AGENTS_TFM §43 (primero se cae la Fase 8).

## Formatos compartidos

**Antes de una tarea sustantiva** (AGENTS_TFM §5):

```text
ROL A ACTIVAR
Objetivo:
Resultado esperado:
Mi recomendación:
Alternativas:
Preguntas para el usuario:
Rango de acción:
Riesgos:
```

**Al terminar:**

```text
ROL UTILIZADO
Objetivo solicitado:
Trabajo realizado:
Resultado:
Interpretación:
Decisiones:
Supuestos:
Limitaciones:
Siguiente decisión propuesta:
¿Debe pasar al Juzgado? Sí/No
```

**No hace falta pedir confirmación** (AGENTS_TFM §4.2) para: ejecutar un script o experimento ya aprobado,
regenerar un informe, actualizar logs con resultados ya obtenidos, ejecutar tests o comprobar el pipeline.
**Sí hace falta** ante cualquier decisión metodológica nueva.

**Conclusiones:** siempre con `#### Conclusiones:` y viñetas con el concepto clave en negrita (skill TFM
base, §6).

**Código:** PEP8, nombres claros, comentarios naturales que explican el porqué, sin numerar y sin
cabeceras decorativas (skill TFM base, §7).
