# AUD-002 — Uso de variables binarias y coherencia general del trabajo
Fecha: 2026-10-05 · Objeto: las binarias (amenities y flags) en cada familia de modelos, más una revisión de coherencia entre README, resumen, bitácora y ficheros de modelo.
Revisado: `models/ficha_modelo*.json`, `data/modelado/train.parquet` (solo train), `notebooks/3-MODELADO.ipynb` (celdas 42-52, 77), `outputs/importancia_*.csv`, `docs/*.md`.
Nota de independencia: no he ejecutado ni modificado ningún experimento. Los diagnósticos de colinealidad se calculan sobre train y no tocan el test.

## Veredicto: REQUIERE MODIFICACIÓN (documental) y APROBADO CON OBSERVACIONES (técnico)

Las binarias no invalidan el modelo final. Sí hay contradicciones entre documentos que un tribunal detectaría en cinco minutos.

## Hallazgos

| # | Gravedad | Hallazgo | Evidencia | Qué hacer |
|---|---|---|---|---|
| 1 | importante | **No existen "46 binarias"** en ningún modelo. Hay 61 en el modelo final (56 amenities + 5 flags) y 57 en el conjunto de 71 (Ridge, kNN). El 46 solo coincide con el nº de pares de amenities con \|r\| > 0,5. Hay que fijar la cifra. | `ficha_modelo.json`; train: 61 binarias de 79 | Una tabla en la memoria: nº de binarias por familia |
| 2 | importante | **Colinealidad fuerte dentro de los amenities**: washer/dryer r = 0,96; stove/oven 0,89; refrigerator con stove, oven, dishes, microwave 0,81-0,84. 46 pares con \|r\| > 0,5 y 109 con \|r\| > 0,3. Son un solo factor "cocina equipada". Hacen falta 31 componentes para explicar el 80% de la varianza de los 56 amenities. | Cálculo sobre train | No afecta a la predicción de árboles ni de Ridge. **Sí invalida leer coeficientes o importancias individuales** (se reparten entre gemelas). Declararlo |
| 3 | importante | **kNN**: con `StandardScaler` cada binaria pesa lo mismo que `latitude` en la distancia. En el conjunto de 71, unas 57 de 71 variables son binarias, así que ~80% de la distancia es ruido de amenities. Coincide con el resultado: K1 (5 variables) 0,474 gana a FAMD/PCAmix con 71 (0,482), 0/5 folds. | Celdas 49-52: Δ = -0,008, p = 0,97 | Es la explicación del resultado; hoy el notebook la atribuye a "dimensionalidad". Escribirla como hipótesis, sin afirmarla |
| 4 | menor | **Ridge/ElasticNet**: `StandardScaler` sobre una binaria con p = 3,5% la multiplica por ~5, así que la penalización la trata de forma distinta a una binaria con p = 50%. No es un error, pero «hemos escalado» no basta como defensa: cambia qué se penaliza. | `func_modelado.py:486-494` | Citar que se escalan también las binarias y por qué es aceptable (ElasticNet ≈ Ridge con 80 vs 71 variables, Δ = 0,004) |
| 5 | menor | **El impacto de las binarias es real pero pequeño y repartido**: permutación en CatBoost, suma de las 56 = 0,027; la mayor (`am_gym`) 0,005; 38 de 56 aportan < 0,0005 cada una. Ablación del bloque: +12-14 milésimas en RMSE. | `importancia_variables.csv`; registro §7 | Las 56 juntas aportan algo; ninguna aislada es interpretable. Conclusión honesta: señal difusa, no hay "amenities clave" salvo gym/elevator/dishwasher |
| 6 | **bloqueante** | **¿Cuál es el modelo final?** README y resumen dicen HGB (test 0,424). `models/ficha_modelo.json` (modificado hoy) describe CatBoost (test 0,429). `importancia_*.csv` solo existen para CatBoost y Lineal; los cuantiles/CQR de las dos fichas difieren (0,0675 frente a 0,0657). | Fichas, README, resumen §5 | Decidir uno y alinear todo. Hoy hay dos finalistas sin decisión escrita |
| 7 | importante | **Cifras incoherentes**: tamaño de partición 29.586/7.398 (bitácora) frente a 29.538/7.384 (README, resumen); anuncios en carteras 38% (resumen, arquitectura) frente a 46% (H-020); Ridge "79 variables" (resumen §4) frente a 71 (registro, notebook) y "80" (registro L5). | grep en `docs/` | Una sola fuente por cifra (`particion.json`) |
| 8 | importante | **Elección de HGB sobre CatBoost "tras ver la tabla de CV" y contra la regla propia** (resumen §5). El propio documento lo declara, pero es el punto que más ataca el tribunal. | resumen §5 | Pre-registrar nada nuevo: presentarlo como sensibilidad y exponer ambos resultados |
| 9 | importante | **Cola de precios**: cobertura 57,5% del intervalo en el quintil más caro; con P99 el RMSE baja de 0,424 a 0,389. El modelo no sirve para el segmento donde un anfitrión más se equivoca. | resumen §5 | Declarar el alcance: mediana de anuncios típicos, no lujo |
| 10 | menor | Los p-valores Nadeau-Bengio con 12/15 folds (p = 0,085) se leen como "no significativo". No es lo mismo que "equivalente". Sin prueba de equivalencia (TOST), «el completo y el seleccionado son lo mismo» es un exceso. | celda 43 | Reformular: «no se detecta diferencia» |

## Checklist
```text
[x] Pregunta definida
[x] No hay fuga conocida (declarada: H-033, H-035)
[x] Validación agrupada por host_id, folds guardados
[ ] Comparación justa: kNN usa conjuntos distintos (5 frente a 71) y se concluye "dimensionalidad" sin aislarla
[ ] Reproducible con una sola identidad de modelo final (hallazgo 6)
[ ] La interpretación no excede la evidencia: importancias individuales de amenities correlacionados (hallazgo 2)
[ ] Cifras coherentes entre documentos (hallazgo 7)
[x] Limitaciones documentadas (casi todas)
```

## Preguntas probables del tribunal
1. *¿Por qué 56 amenities si 38 no aportan casi nada?* Porque la selección ciega al precio fue una decisión pre-registrada; los árboles pierden 0,005 al seleccionar (5/5 folds). Coste: ruido en la interpretación. Se acepta.
2. *¿No hay multicolinealidad?* Sí, entre gemelas (washer/dryer 0,96). No afecta a la predicción, solo a leer coeficientes; no se interpretan.
3. *¿Por qué kNN con 71 rinde peor que con 5?* Hipótesis: 80% de la distancia procede de binarias de baja señal. No lo he aislado.
4. *¿Cuál es vuestro modelo final?* **Hoy no hay una respuesta única en el repo.**

## Limitaciones que deben aparecer en la memoria
- Amenities colineales: la importancia por variable no es interpretable, solo la del bloque.
- Escalado de binarias en kNN y en el lineal: afecta a la distancia y a la penalización.
- Listados declarados por el anfitrión en 2021, no en la fecha de publicación.
- La señal de los amenities puede ser proxy de calidad o tipo de edificio, no efecto causal.
