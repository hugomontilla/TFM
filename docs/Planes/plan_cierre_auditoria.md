# Plan: cerrar los hallazgos de AUD-002 (binarias y coherencia)

**Origen:** `docs/auditorias/AUD-002_binarias_y_coherencia.md`. **Plazo:** aprobación del tutor antes del 2026-10-16;
entrega el 2026-10-23. Actualizado el 2026-10-06 (Fases 1 y 2 cerradas). La parte documental (sin reentrenar nada) debe quedar lista el 10-09 y
el experimento opcional, si se hace, el 10-12.

**Regla:** este plan no cambia ningún resultado ya registrado. Solo se alinean documentos, se añaden diagnósticos sobre
train y, como mucho, un experimento nuevo registrado en `outputs/experimentos.csv`. El test no se toca.

## Hallazgo → acción

| Hallazgo | Acción | Entregable | Dónde |
|---|---|---|---|
| 6 (bloqueante) ¿Cuál es el modelo final? | Ver Fase 1 | Fila en la bitácora y fichas coherentes | bitácora, README, resumen, fichas |
| 7 Cifras incoherentes | Ver Fase 2 | Una sola cifra por dato, generada desde fichero | docs |
| 1 Nº de binarias | Ver Fase 3 | Tabla de binarias por familia | resumen y memoria |
| 2 Colinealidad entre amenities | Ver Fase 3 | Diagnóstico + limitación escrita | notebook 3, memoria |
| 3 kNN dominado por binarias | Ver Fase 4 (opcional) | Experimento K5 o reformular como hipótesis | notebook 3 |
| 4 Escalado de binarias en el lineal | Ver Fase 3 | Párrafo de justificación | notebook 3, memoria |
| 5 Efecto pequeño y difuso de amenities | Ver Fase 3 | Conclusión a nivel de bloque | notebook 4, memoria |
| 8 Elección HGB frente a CatBoost | Ver Fase 1 | Decisión explícita con sensibilidad | bitácora |
| 9 Cola de precios | Ver Fase 5 | Alcance declarado | memoria, resumen |
| 10 «No significativo» frente a «equivalente» | Ver Fase 5 | Reformulación, o TOST | notebook 3 |

## Fase 1 — Decidir el modelo final (bloqueante; cerrada 2026-10-06)

Hechos de partida, que hay que verificar leyendo: `plan_modelo_final.md` (2026-10-02) dice que el elegido por la regla
fijada de antemano es **CatBoost** (CV 0,4160 frente a 0,4191 de HGB). El resumen ejecutivo y el README presentan **HGB**
como final, y `models/ficha_modelo_catboost.json` es de CatBoost.

**Decisión del alumno (2026-10-05): se mantienen los dos como finalistas** (CatBoost y HGB) por empate práctico y
coste de HGB. Es la opción **C** de abajo. Se concreta así:

- **C. Dos finalistas con modelo de referencia declarado.** CatBoost es el modelo de referencia y el que se entrega
  (`modelo_final_catboost.joblib`), porque lo elige la regla fijada de antemano (menor RMSE de CV, plan 10-02). HGB se presenta
  como alternativa equivalente y más rápida. Condiciones: (1) escribir en la bitácora la regla y quién es el de referencia;
  (2) demostrar el «empate» con una comparación pareada CatBoost frente a HGB en los mismos folds, con un margen de
  equivalencia (TOST, por ejemplo ±0,005 de RMSE en log) **fijado antes de calcularlo**; (3) corregir el resumen §5, que
  hoy llama «final» a HGB; (4) en la defensa, la respuesta a «¿cuál usarías?» es CatBoost, y a «¿por qué no el más
  rápido?» es que la diferencia de coste no justifica abandonar la regla.

Las opciones A y B quedan como alternativas descartadas:

1. Decidir con el director **una** de estas opciones y escribirla en `bitacora_decisiones.md` con fecha:
   - **A. Final = CatBoost** (cumple la regla fijada; HGB queda como alternativa rápida). Es la opción que no exige
     justificar ninguna excepción. Cambia README y resumen, no los modelos.
   - **B. Final = HGB** (excepción declarada por coste, 16 veces más rápido). Exige exponer la excepción como tal y
     reportar CatBoost como sensibilidad.
   - Mi recomendación: **A**, porque el tribunal ataca las excepciones a la regla propia y la diferencia de test (0,424
     frente a 0,429) es menor que el IC del 95%. El test no puede usarse para decidir, y este caso lo parece.
2. **Hecho (10-06):** artefactos renombrados con el sufijo del modelo (`sufijo='_catboost'` en `4-MODELO_FINAL`):
   `modelo_final_catboost.joblib`, `cuantiles_catboost.joblib`, `ficha_modelo_catboost.json` y `resultados_test_catboost.csv`,
   más los de HGB con `_hgb` (sensibilidad). Comprobado: el notebook guarda los dos modelos, las dos fichas comparten el
   MD5 de la partición y tienen 79 variables. Comprobado el 10-06: los cuatro `.joblib` cargan con el entorno `.venv`.
3. Alinear README §Resultados, resumen §5 y `4-MODELO_FINAL` (título, conclusiones) con la decisión.

**TOST (10-06, margen ±0,005 fijado antes):** Δ(CatBoost − HGB) = −0,0029, IC 90% [−0,0042; −0,0016], p = 0,012: equivalentes en la práctica; CatBoost gana en 5/5 folds (t pareado p = 0,009). Folds no independientes: p orientativos. Escrito en bitácora (10-06), README, resumen §5 y notebook 4.

**Criterio de cierre:** cumplido; ninguna frase del repo llama «final» a más de un modelo.

## Fase 2 — Una sola fuente para las cifras (cerrada 2026-10-06)

No hace falta un script nuevo: las corridas ya existen y las cifras salen de ficheros que ya están (`particion.json`,
`data/modelado/*.parquet`, `models/ficha_modelo_*.json`, `outputs/experimentos.csv`, `1-EXPLORACION` §3). Solo había que
unificar los textos. Resultado de la comprobación:

- **Tamaños:** no son dos particiones, son dos etapas de la misma. `particion.json`: 29.586 / 7.398 (antes de la regla de
  bloqueo); parquet: 29.538 / 7.384 (los que usan los modelos; 36.922 en total). README y resumen lo dicen ahora.
- **Carteras:** 38,1% (36.922 anuncios; 38,0% solo en train; 0 anfitriones compartidos entre train y test). El «46%» del
  informe no existe como cifra de carteras (el 46 del repo es la ventana de amenities y la sobreestimación de la media
  de la bitácora 09-05). Fijado 38,1% con su base en arquitectura, resumen y H-020.
- **«Gemelos»:** 23,7% (`1-EXPLORACION`, clave host + tipo + capacidad + dormitorios + barrio del dato bruto). Sobre los
  parquet da 24,2% porque `neighbourhood` está ya preparado; se cita siempre el 23,7% con su origen.
- **Variables:** ElasticNet 79 → 0,434; con las 71 → 0,439; Ridge/OLS lineal 71; árboles y fichas de CatBoost y HGB: 79
  (comprobado en las fichas). La etiqueta «Ridge / ElasticNet (79)» del resumen pasa a «ElasticNet (79)». Las filas de 80
  variables ya están marcadas como histórico en el registro.

**Criterio de cierre:** cumplido; el mismo `grep` ya no devuelve cifras contradictorias.

## Fase 3 — Binarias: medir, escribir y declarar (1 día, sin reentrenar)

1. **Tabla de binarias por familia** (hallazgo 1), generada de los ficheros: cuántas entran en lineal (71), árboles (79),
   kNN, SVR. Va al resumen y a la memoria.
2. **[Hecho 10-06, H-047]** **Diagnóstico de colinealidad** (hallazgo 2), una celda en `3-MODELADO` solo con train: pares con |r| > 0,5,
   componentes para el 80% y los grupos de gemelas (cocina, lavandería, seguridad). Conclusión: se interpretan **bloques,
   no amenities sueltos**. Se añadió la reducción de duplicados (|r| > 0,8, 71 → 65) como decisión de diseño para el kNN con FAMD/PCAmix (el lineal la compara y la regla D1 elige las 79); coste ≈ 0,0003 de RMSE (`K6_*`, notebook 3 §7.0b, §7.3b y §7.5).
3. **Escalado** (hallazgo 4): un párrafo que diga qué se hace (se escalan también las binarias) y su consecuencia (la
   penalización de Ridge pesa distinto según la prevalencia). La sensibilidad ya existente (71 frente a 79, Δ = 0,004) sirve de respaldo.
4. **Efecto de los amenities** (hallazgo 5): en `4-MODELO_FINAL` y la memoria, presentar el bloque (0,027 en CatBoost) y
   decir que 38 de 56 aportan casi nada. No decir «el gimnasio sube el precio»: es asociación y proxy de edificio.
5. Añadir las tres limitaciones del informe a la sección de limitaciones.

**Criterio de cierre:** la memoria no contiene ninguna frase que interprete un amenity aislado como causa o como clave.

## Fase 4 — kNN y las binarias (hallazgo 3; opcional, 0,5 día; decisión 10-06: se descarta K5 y se reformula como hipótesis)

Decisión del director: ¿merece un experimento? Es barato, pero kNN no es un finalista.

- **Opción mínima (recomendada si falta tiempo):** reformular la conclusión del notebook como hipótesis: «el peor
  resultado de FAMD y PCAmix es compatible con que las binarias dominen la distancia; no se ha aislado».
- **Opción completa:** un único experimento registrado `K5_knn_subconjunto_mas_amenities` (las 5 variables de K1 más el
  bloque de amenities, con los mismos folds). Hipótesis fijada antes: si las binarias dominan la distancia, K5 empeora
  claramente frente a K1 (Δ de RMSE > 0,005, en al menos 4 de 5 folds). Se registra salga como salga.

## Fase 5 — Redacción defensiva (1 día)

1. **Cola de precios** (hallazgo 9): el alcance del modelo es «mediana del precio de anuncios típicos». En la memoria, la cobertura
   del 57,5% del quintil superior y el RMSE con P99 van juntos en una sola tabla, no repartidos.
2. **«No significativo» frente a «equivalente»** (hallazgo 10): recorrer las conclusiones del notebook 3 que digan «igual» o
   «no cambia» y reformular. Si hay tiempo, añadir una prueba de equivalencia (TOST con margen de 0,005 de RMSE fijado
   antes) solo para la comparación 71 frente a 79.
3. Actualizar `hallazgos_y_pendientes.md`: nuevos H-xxx para los hallazgos bloqueantes e importantes (6, 7, 2, 3, 8, 9).

## Fase 6 — Reauditoría (0,5 día)

El Juzgado repasa el resultado: `grep` de cifras, coherencia de las fichas, checklist y veredicto en un `AUD-003`.
Pasa a **APROBADO** solo si la Fase 1 y la 2 están cerradas.

## Calendario

| Día | Fase |
|---|---|
| 10-06 | 1 y 2 ✅ |
| 10-07 / 10-08 | 3 y 5 |
| 10-09 | 4 (opcional) y 6 |
| 10-10 a 10-15 | Margen, envío al tutor |

## Riesgos

- Elegir la Fase 1 B obliga a defender una excepción; elegir A obliga a reescribir las secciones del README y el resumen.
- El recálculo de porcentajes (Fase 2) puede dar una cifra tercera; no es un fallo, solo hay que citar la base.
- El experimento K5 (Fase 4) no debe usarse para cambiar ninguna decisión ya cerrada: kNN no es finalista.
