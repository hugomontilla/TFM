# Resumen ejecutivo — Predicción del precio de Airbnb en Nueva York

*TFM Máster en Data Science y Big Data. Documento de seguimiento para la tutora. Actualizado el 6 de octubre de 2026.*

## 1. Qué problema resuelvo

Dado un anuncio **nuevo** en Nueva York (características, ubicación y perfil del anfitrión), predecir el **precio por noche** y dar un **intervalo del 80%**, para recomendar un precio a un anfitrión. El objetivo es `log(price)`; `exp(ŷ)` estima la **mediana** del precio.

Datos: `listings.csv` del Airbnb 2008-2021, acotado a Nueva York. **36.922 anuncios** → train 29.538 / test 7.384, con anfitriones distintos a cada lado.

## 2. Cómo evito la fuga de información

- **Partición antes de mirar el precio**, agrupada por anfitrión y estratificada por tipo y distrito. El 38,1% de los anuncios pertenece a carteras de varios y son casi gemelos: sin agrupar, la validación se infla.
- Todo lo que se aprende de los datos (codificación del barrio, medianas, amenities frecuentes) se ajusta **solo con train**.
- Sin reseñas ni variables derivadas del precio: un anuncio nuevo no las tiene.
- **Test abierto una sola vez**, tras comprobar su MD5. Todo lo demás se decide con validación cruzada (5 folds, folds guardados y comparaciones pareadas).

## 3. Qué variables entran y por qué

Parto de **79 variables**: producto (5), ubicación (5), condiciones de reserva (3), perfil del anfitrión (10) y 56 amenities binarias.

1. **Ablación por bloques** con reglas fijadas antes de mirar: el producto y la ubicación son lo que más pesa; las condiciones de reserva casi nada. Para el lineal sobran 8 variables (71); para los árboles, quitar variables **empeora** el error (Δ ≈ 0,005 en 5 de 5 folds), así que usan las 79.
2. **Colinealidad de los amenities.** Con 46 pares de |r| > 0,5, lavandería y cocina son casi duplicados (washer–dryer 0,96; VIF máximo 12,3). Es un problema para el lineal (coeficientes) y para el kNN (distancia), no para los árboles. Regla escrita antes de calcular: con |r| > 0,8 se conserva una variable por grupo (71 → 65, quitando 6). Efecto: ningún par por encima de 0,8 y VIF máximo de 2,9. Se aplica al kNN con FAMD/PCAmix; para el lineal la regla de selección sigue prefiriendo las 79. El coste de precisión de quitar los duplicados es de unas 0,3 milésimas de RMSE.
3. Consecuencia para la memoria: **se interpretan bloques de variables, no amenities sueltos**; ningún amenity aislado se lee como causa del precio.

## 4. Qué modelo (RMSE en log, CV de 5 folds; menor es mejor)

| Peldaño | Mejor configuración | RMSE | Qué aprendo |
|---|---|---|---|
| Referencia ingenua | mediana por tipo × capacidad | 0,535 | Lo que haría un anfitrión con una regla simple |
| Lineal | Lasso con las 79 (con las 65: 0,439) | 0,434 | El producto y la ubicación explican casi todo; regularizar mejora poco |
| kNN | 5 variables elegidas (K1) | 0,474 | FAMD/PCAmix con 65 variables empeoran (0,481): es compatible con que amenities y anfitrión dominen la distancia, pero no lo he aislado |
| Árbol CART | profundidad 10 | 0,472 | Inestable entre folds |
| SVR (Nystroem) | γ = 0,005, C = 1 | 0,432 | Parecido al lineal |
| Random Forest | ajustado | 0,428 | |
| HGB | ajustado | 0,419 | |
| **CatBoost** | **ajustado** | **0,416** | **Menor RMSE de CV** |

Lectura: el salto grande (0,71 → 0,53) lo da el **producto**; la **ubicación** y los modelos de boosting bajan hasta 0,42. Las no linealidades y las interacciones valen un 4% frente al lineal. **Conocer al anfitrión sí vale:** sin su perfil, el RMSE sube un 3,7% (HGB) y un 4,0% (lineal).

## 5. Elección del modelo final

- **Regla fijada antes de ver la tabla:** menor RMSE medio de CV. Gana **CatBoost** (0,4162 frente a 0,4191 de HGB), que es el modelo de referencia y el que se entrega.
- **HGB como alternativa equivalente y más rápida** (unas 10-20 veces, según se mida). Para no llamar «empate» a algo sin demostrar, se hizo un contraste de equivalencia (TOST pareado, margen de ±0,005 fijado antes de calcular): la diferencia de −0,0029 queda dentro del margen (IC 90% [−0,0042; −0,0016], p = 0,012). Aun así CatBoost gana en los 5 folds, así que la diferencia es pequeña pero consistente.

### Resultados en el test (abierto una vez)

| Métrica | CatBoost (IC 95%) | HGB (IC 95%) |
|---|---|---|
| RMSE en log | **0,429** (0,408-0,451) | 0,424 (0,407-0,442) |
| R² en log | **0,636** | 0,645 |
| MAE | **$49,9** | $49,4 |
| Error mediano | **$21** | $21 |
| MAPE | **31%** | 31% |

El test cae dentro de CV ± 2 desviaciones en todas las métricas (sin señal de sobreajuste de la búsqueda). En test gana HGB por 0,005, dentro del IC y en sentido contrario al de CV: **el test no se usa para elegir**.

## 6. Dónde falla y alcance declarado

El modelo estima la **mediana del precio de anuncios típicos**. Se equivoca en la cola alta, y las cifras buenas y malas van juntas:

| | CatBoost |
|---|---|
| Cobertura del intervalo del 80%, global (con CQR) | 77,9% |
| Cobertura en el **quintil más caro** | **55,9%** |
| Cobertura en Manhattan | 73,7% |
| MAE: habitación privada / piso entero | $25 / $70 |
| MAE: 7 o más huéspedes / quintil más caro | $172 / $154 |
| RMSE en log, todo el test / sin los precios sobre el P99 ($794, 81 anuncios) | 0,429 / 0,395 |

El P99 es solo un análisis de sensibilidad posterior (no se usa para entrenar ni elegir); la regla de exclusión real quita precios ≥ $9.999 y habitaciones > $1.000 (62 anuncios). Los 20 peores errores son pisos enteros caros infravalorados: el modelo no captura el lujo extremo con las variables disponibles.

**Qué determina el precio** (importancia por permutación, aumento del RMSE): producto 0,30 ≫ ubicación 0,11 > anfitrión ≈ amenities 0,05 ≫ condiciones de reserva 0,004. Describe cómo usa las variables el modelo, no una relación causal.

## 7. Limitaciones

1. El perfil del anfitrión se mide en 2021, no en el momento de publicar.
2. Los precios de bloqueo se excluyen también del test; en producción el precio no se conoce.
3. La CQR con un solo conjunto de calibración es aproximada (anuncios correlacionados por anfitrión).
4. Las comparaciones entre modelos usan folds que comparten train: los p-valores son orientativos. «No se detecta diferencia» no significa «equivalente»; solo CatBoost frente a HGB está demostrado con TOST.
5. Los amenities son colineales: no se interpretan de uno en uno.
6. Decisiones tomadas viendo resultados parciales, declaradas: recorte de la búsqueda de CatBoost de 40 a 10 candidatos, y umbral de colinealidad de 0,9 a 0,8 (antes de calcular ningún RMSE).

## 8. Dónde me gustaría su orientación

1. **¿Es defendible presentar CatBoost como modelo de referencia y HGB como alternativa equivalente?**
2. **La cola de precios altos** es el punto débil (cobertura 56% en el quintil caro). ¿Merece un modelo específico o basta con declararla como limitación?
3. **Interpretabilidad:** hoy uso permutación, PDP/ICE y SHAP en CatBoost. ¿Es suficiente?
4. **Estructura de la memoria**, y cuánto detalle dar a la selección de variables y al EDA.

## 9. Siguiente paso

Cerrar la redacción defensiva (alcance de la cola de precios y pruebas de equivalencia), redactar la memoria y preparar la presentación. Aprobación antes del **16 de octubre de 2026**; entrega el 23 de octubre.
