# Predicción del precio de Airbnb en Nueva York


## 1. RESUMEN

La idea del proyecto se basa en una premisa: si alguien va a publicar un anuncio nuevo en Airbnb en Nueva York, ¿qué precio por noche debería poner? Con lo que se sabe del alojamiento, de su ubicación y del anfitrión (informacion obligatoria en airbnb), el modelo propone un precio y, además, un rango en el que el precio real debería caer aproximadamente.

Los datos salen del conjunto público de Airbnb 2008-2021. En este trabajo nos centramos solo en Nueva York, lo que deja **36.922 anuncios**.

## 2. DECISIONES

- He trabajado con el logaritmo del precio en lugar del precio directamente, porque los precios tienen una cola muy larga (hay pocos anuncios carísimos que lo distorsionan todo). Al deshacer el logaritmo, lo que obtengo es una estimación del precio *mediano* de un anuncio parecido.
- Separé entrenamiento y test **antes** de mirar los precios, y lo hice por anfitrión: ningún anfitrión aparece en los dos lados. Esto es importante porque el 38% de los anuncios pertenecen a anfitriones con varios pisos casi idénticos, si no los separamos, el modelo memoriza la respuesta y los resultados salen inflados.
- No he usado reseñas ni nada calculado a partir del precio, porque un anuncio nuevo todavía no tiene esa información.
- Todas las decisiones las tomé con validación cruzada (5 particiones, siempre las mismas). El test solo se usa en el notebook final.
- Quité del test los precios absurdos (de 9.999 $ o más), pero con un anuncio real nuevo no sabríamos de antemano si es uno de esos casos.

## 3. VARIABLES

Empecé con **79 variables**, que agrupé en cinco bloques: el propio alojamiento (tipo, capacidad, habitaciones…), la ubicación, las condiciones de reserva, el perfil del anfitrión y 56 servicios o *amenities* (wifi, cocina, lavadora, etc.), estas ultimas las sacamos de la columna amenities original buscando aquellas que aparecieran entre el 3,5%-85%.

Para ver qué aportaba cada bloque fui quitándolos de uno en uno. Lo que más pesa, con diferencia, es el tipo de alojamiento y su ubicación, por otro lado las condiciones de reserva apenas cambian nada. 

También me encontré con que muchos amenities van casi siempre juntos (quien tiene lavadora casi siempre tiene secadora, alta correlacion). Para algunos modelos esto es un problema, así que en esos casos dejé solo una variable de cada grupo. 

## 4. MODELOS

Fuimos de los más simple a los más complejo. La medida que uso para compararlos es el RMSE sobre el logaritmo del precio: cuanto más bajo, mejor. 

| Modelo | Error (RMSE) | Comentario |
|---|---|---|
| Mediana por tipo y capacidad | 0,535 | Lo que haría un anfitrión fijándose en anuncios parecidos |
| Regresión lineal (Lasso) | 0,434 | Ya captura la mayor parte de lo que se puede explicar |
| Vecinos más cercanos (kNN) | 0,474 | Funciona mejor con pocas variables, se probo tambien tecnicas de reduccion de dimensionalidad |
| CART | 0,472 | Bastante inestable según la partición |
| SVR (Nyosterm) | 0,432 | Muy parecido a la regresión lineal |
| Random Forest | 0,428 | |
| Gradient Boosting (HGB) | 0,419 | Gradient boosting por histograma|
| **CatBoost** | **0,416** | **El mejor en validación cruzada y el mas pesado** |


Mi conclusion es que la mayor mejora viene de conocer bien el alojamiento, la ubicación y los modelos de boosting afinan algo más, pero los modelos complejos solo mejoran sustancialmente  a la regresión lineal. Un dato interesante es que la información del anfitrión sí ayuda: sin ella el error sube alrededor de un 4%.

## 5. MODELO FINAL

Antes de ver los resultados decidí que me quedaría con el modelo de menor error en validación cruzada, y ese fue **CatBoost**. HGB quedó muy cerca y es entre 10 y 20 veces más rápido. Hice una prueba estadística para comprobar que la diferencia entre ambos es realmente pequeña (menor que el margen de 0,005 que fijé de antemano), y lo es, aunque CatBoost gana en las cinco particiones. Por eso presento CatBoost como modelo principal y HGB como alternativa práctica.

Estos son los resultados en el test, es decir, en anuncios que el modelo no había visto nunca:

| | CatBoost | HGB |
|---|---|---|
| Error (RMSE en log) | 0,429 | 0,424 |
| Varianza explicada (R²) | 0,64 | 0,65 |
| Error medio en dólares | 50 $ | 49 $ |
| Error típico (mediana) en dólares | 21 $ | 21 $ |
| Error medio en porcentaje | 31% | 31% |

Los resultados son coherentes con lo visto en validación cruzada, así que no parece que el modelo esté sobreajustado. Dicho de forma sencilla: para un anuncio típico, el modelo se equivoca en unos 20 dólares. El modelo funciona bien con anuncios "normales", pero le cuesta mucho con los caros. Me parece importante dejarlo claro:

| Situación | Resultado |
|---|---|
| El rango propuesto acierta (objetivo: 80%) | 78% en general |
| … en el 20% de anuncios más caros | solo 56% |
| … en Manhattan | 74% |
| Error medio: habitación privada / piso entero | 25 $ / 70 $ |
| Error medio: 7 o más huéspedes / anuncios más caros | 172 $ / 154 $ |

Los 20 peores errores son pisos enteros de lujo a los que el modelo pone un precio demasiado bajo. Mi interpretación es que, con la información disponible en el anuncio, no hay forma de distinguir bien un piso de lujo de uno simplemente grande. Si quito el 1% de precios más altos, el error baja de 0,429 a 0,395, lo que confirma que gran parte del error se concentra ahí.

## 6. ORDEN VARIABLES

Midiendo cuánto empeora el modelo al "desordenar" cada bloque de variables, el orden es claro: el tipo de alojamiento es lo más importante con diferencia, después la ubicación, luego el anfitrión y los amenities (con un peso similar) y, por último, las condiciones de reserva, que prácticamente no influyen. Esto describe cómo usa el modelo la información, no quiere decir que una cosa *cause* la otra.

## 7. DUDAS

1. ¿Te parece defendible presentar CatBoost como modelo principal y HGB como alternativa equivalente?
2. Los anuncios caros son el punto débil. ¿Valdría la pena un modelo específico para ellos, o es suficiente con explicarlo como limitación? Como hemos estudiado en general valores tan altos como 9999$ son precios de bloqueo.
3. Para interpretar el modelo uso importancia por permutación, gráficos PDP/ICE y SHAP. ¿Crees que es suficiente?
4. ¿Cómo debería organizar la memoria, y cuánto espacio dedicar al análisis exploratorio y a la selección de variables?

