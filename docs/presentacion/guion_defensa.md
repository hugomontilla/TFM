# Guion de la defensa del TFM

**Predicción del precio por noche de un anuncio nuevo de Airbnb en Nueva York**

Duración objetivo: **15 minutos** (el guion suma unos 18:20 leído a ritmo tranquilo: hay que recortar unos 3 minutos en el ensayo).
Tono: mezcla de sponsor y analista. Se cuenta la historia y las decisiones; las cifras se enseñan en pantalla y solo se
dicen en voz alta las cuatro o cinco que importan (**21$**, **22%**, **8 de cada 10**, **el lujo**).

Estructura del documento:

1. Guion diapositiva a diapositiva (qué se ve y qué se dice)
2. Decisiones clave de la bitácora: qué, por qué y cómo defenderlas
3. Banco de preguntas probables del tribunal
4. Apéndice: glosario de todo lo usado (técnicas, modelos, herramientas)
5. Lo que te falta para la defensa

---

## 1. Guion diapositiva a diapositiva

> **Cifras que hay que decir de memoria:** 36.922 anuncios válidos · 29.538 train / 7.384 test · 79 variables en 5 bloques · error típico **21$** · **22%** mejor que la regla ingenua · rango del 80% que cubre el **77,9%** · en el 20% más caro el rango cubre solo el **56%** · CatBoost 0,416 en CV y 0,429 en test.

### Diapositiva 1 — Portada · 0:45

**En pantalla:** título, subtítulo, nombre, máster, tutora, fecha.

> "Buenos días, mi nombre es Hugo Montilla Anta y hoy les vengo a presentar mi TFM. Este aborda uno de los problemas clásicos de la Ciencia de Datos, y que hemos trabajado a lo largo de este curso: la predicción de precios. En concreto, la pregunta que responde es: **¿cuánto debería cobrar por noche alguien que publica un anuncio nuevo en Airbnb en Nueva York?** Un anuncio nuevo todavía no tiene reseñas ni historial propio, así que la respuesta tiene que salir únicamente de lo que se sabe en el momento de publicarlo. Lo que he desarrollado es un tasador automatico que te puede dar un precio de guia al subir un nuevo anuncio.
> En estas diapositivas vamos a ver el razonamiento que hemos seguido, las restricciones que nos hemos encontrado por el camino y los resultados que hemos obtenido, incluido dónde falla nuestro modelo."

*Si preguntan:*
- **¿Qué entendéis por "anuncio nuevo"?** Uno que todavía no tiene reseñas, puntuaciones ni historial de reservas: solo lo que el anfitrión declara al publicarlo (tipo, capacidad, ubicación, servicios) y su propio perfil.
- **¿Qué aporta frente a las herramientas de precio de la propia plataforma?** Esas herramientas usan demanda y calendario, datos que no tenemos. Nuestro caso es el arranque en frío: un precio de referencia con su rango y una explicación, solo con datos públicos del anuncio.
- **¿Hay trabajos previos?** Sí, Rezazadeh Kalehbasti et al. (2019), también en Nueva York. Ver la sección 3: usan reseñas y parten por filas, así que su cifra no es comparable.

### Diapositiva 2 — Un cambio de alcance: de 10 ciudades a Nueva York · 1:15

**En pantalla:** Figura 1 de analistas (elasticidad a la capacidad por ciudad); ×17 entre medianas; 84% vs 39%;
37.012 anuncios.

> "El dataset traía unos 280k anuncios repartidos de forma desigual entre diez ciudades. La primera decisión importante que tomamos fue **acotar el estudio a una única ciudad, Nueva York**, y lo hicimos por dos razones principales. 
> Primera: el precio viene en la moneda local de cada ciudad sin que el fichero lo documente; la mediana va de 65 en Roma a 1.100 en Bangkok, y al aplicar un tipo de cambio aproximado la diferencia baja a ×3, así que lo que estábamos viendo era la divisa.
> Segunda, y más importante: **el mercado no funciona igual en cada ciudad**. Este gráfico que vemos mide cuánto sube el precio al doblar la capacidad, una medida a la que la divisa no afecta, y va de 1,5 veces en Bangkok a 2,2 en Ciudad del Cabo. Un único modelo para las diez ciudades habría necesitado interacciones de la ciudad con casi todas las variables.
> Elegimos Nueva York porque es la única con el distrito completo, la que sitúa los anuncios con más detalle, con 220 barrios, y tiene muestra de sobra: unos 37k anuncios, todos en dólares."

*Si preguntan:*
- **¿Cómo se calcula esa "elasticidad"?** Es la pendiente de una recta entre `log(price)` y `log(accommodates)` en cada ciudad. Cambiar de divisa multiplica todos los precios por una constante: la recta sube o baja, pero no cambia su inclinación.
- **¿Por qué no convertir las divisas y seguir con las diez ciudades?** Haría falta el tipo de cambio de la fecha del volcado, que no consta, y tampoco resolvería las diferencias de poder adquisitivo. Además, el problema de fondo no era la divisa, sino que el mercado se comporta distinto en cada ciudad.
- **¿Por qué no un modelo con la ciudad como variable, o uno por ciudad?** La ciudad explicaba el 22% de la varianza del log-precio y varios efectos cambiaban de signo entre ciudades: un único modelo necesitaría interacciones de la ciudad con casi todo. Uno por ciudad serían diez proyectos, y en nueve de ellas falta `district`.
- **¿Esas diferencias entre ciudades no son solo composición del inventario?** En parte sí, y así lo declaramos: son descriptivas, sin contraste formal. Para la decisión basta con que el orden de magnitud cambie.
- **¿Por qué Nueva York y no París, que tiene más anuncios?** Es la única con `district` completo, la que tiene más
  detalle de ubicación (220 barrios) y todo en dólares. Muestra le sobra: 37.012 anuncios.

### Diapositiva 3 — Qué queremos predecir y con qué información · 1:05

**En pantalla:** "se conoce / aún no existe / dos escenarios"; los tres objetivos.

> "Con el alcance definido, nos marcamos tres objetivos: dar un **precio** de referencia, acompañarlo de un **rango** que contenga el precio real en 8 de cada 10 casos, y **explicar** qué es lo que determina ese precio.
> La restricción que lo ordena todo es usar **solo la información que existe al publicar el anuncio**. Por eso no usamos reseñas ni puntuaciones: un anuncio nuevo todavía no las tiene, y un modelo que dependiera de ellas no serviría para este caso; sería una fuga de información.
> Ahora bien, que el anuncio no tenga historial no significa que no lo tenga su anfitrión. Por eso definimos dos escenarios: el principal, un anuncio nuevo de un anfitrión del que conocemos su perfil (su antigüedad, cuántos anuncios gestiona o la reputación de sus otros anuncios); y uno de contraste, un anfitrión que publica por primera vez. La diferencia entre ambos nos dice cuánto vale conocer al anfitrión."

*Si preguntan:*
- **¿Por qué un rango del 80% y no del 95%?** Para un anfitrión, un rango del 95% sería tan ancho que no le ayudaría a decidir. El 80% equilibra utilidad y fiabilidad, y lo fijamos como objetivo antes de entrenar.
- **`reputacion_cartera` usa reseñas, ¿no es fuga?** No: usa las reseñas de los **otros** anuncios del anfitrión, que ya existen cuando publica el nuevo. Las del propio anuncio nunca entran.
- **¿El perfil del anfitrión no está medido después de publicar?** Sí, en la foto de 2021; lo declaramos como limitación. Por eso existe el escenario A sin ese bloque: el error solo sube un 4%, así que la conclusión no depende de ello.
- **¿Por qué no predecir también el precio de anuncios que ya existen, con sus reseñas?** Por causalidad inversa: el precio influye en las reseñas que recibe un anuncio. Es otro problema, que dejamos como línea futura.

### Diapositiva 4 — Predecir el logaritmo del precio · 1:20

**En pantalla:** Figura 2 de analistas (distribución del precio en dólares y en logaritmo); asimetría 21,1 → 0,7; tres
motivos.

> "Una decision importante fue predecir el **logaritmo del precio** y no el precio bruto. Como se ve en el gráfico, el precio tiene una cola derecha muy larga: la mayoría de anuncios se concentra en precios moderados, pero hay pisos de lujo de varios miles de dólares (10000 el maximo). Su asimetría en logaritmo baja muchisimo (de 21 a 0.7) y la distribución es casi simétrica (normalidad).
> Tomamos esta decisión por tres motivos. Primero, si predecimos el precio en bruto, unos pocos anuncios de lujo dominan el error. 
> Segundo, en logaritmo el error pasa a ser **relativo**: equivocarse en 30 dólares no es lo mismo en un piso de 60 que en uno de 600.
> Tercero, el precio se comporta de forma **multiplicativa**: cada dormitorio adicional sube el precio en torno a un 16%, no una cantidad fija, y en logaritmo esa relación se vuelve aditiva, que es lo que mejor aprenden los modelos.
> Esta decisión tiene una consecuencia que conviene conocer: al deshacer el logaritmo obtenemos la **mediana** del precio de un anuncio con esas características. Para recomendar un precio es justo lo que buscamos, porque la mediana no se deja arrastrar por el lujo. Y aunque comparamos los modelos con el error en logaritmo, los resultados para el negocio los damos siempre en dólares."

*Si preguntan:*
- **¿Por qué el logaritmo y no otra transformación, como Box-Cox?** El logaritmo ya deja la asimetría en 0,7, y es el único que da una lectura directa en porcentaje (cada dormitorio, +16%). Una Box-Cox afinaría un poco la forma, a cambio de perder esa interpretación.
- **¿Cómo se interpreta un RMSE de 0,43 en logaritmo?** Como un error relativo: e^0,43 ≈ 1,5, es decir, un error "típico cuadrático" de un factor 1,5 por encima o por debajo. Por eso al negocio le damos la mediana del error en dólares (21$).
- **¿Y el sesgo al deshacer el logaritmo?** `exp(ŷ)` estima la mediana, no la media; y la mediana es justo lo que queremos recomendar. Ver la sección 3 (*smearing* de Duan).
- **¿De dónde sale el 16% por dormitorio?** Es el coeficiente de la OLS hedónica con errores agrupados por anfitrión, a igual tipo, capacidad y ubicación. Es una asociación condicionada, no un efecto causal.
- **¿Por qué no optimizar directamente el error en dólares?** Unos pocos pisos de lujo dominarían la pérdida y el modelo sacrificaría el resto de anuncios por acertarlos.

### Diapositiva 5 — Los datos y su limpieza · 2:00

**En pantalla:** embudo 37.012 → 36.922 (precio nulo o 0, capacidad 0, precios de bloqueo); 5 bloques de variables con nº de variables; equipamiento: de 562 servicios en texto libre a 56 binarias.

> "Durante la limpieza y el tratamiento de valores nos hemos encontrado valores faltantes, en algunos casos tenía sentido, como la puntuación de un anuncio sin reseñas, sin embargo en otros hacia falta imputar el valor como en el numero de habitaciones. También encontramos valores nulos en campos como el precio o el número de huéspedes.
> Eliminamos lo que no es una observación válida, como los precios nulos, y los **precios de bloqueo**: anuncios a 9.999 dólares que suponemos que los anfitriones usan para cerrar fechas. En un principio nos planteamos **recortar por percentile**, pero lo descartamos porque eliminaba  el lujo real, que el modelo tiene que aprender.
> La variable **amenities** viene como una lista de texto libre por anuncio, con 562 servicios distintos. Aqui he definido variables binarias por ammenitie, para elegir cuales usamos nos quedamos con los que aparecen en entre el 3,5% y el 97% de los anuncios del train, porque los que tiene casi todo el mundo o casi nadie apenas aportan información. Así salen 56 variables binarias de esta variable original."
> También construimos algunas variables nuevas, como si el anuncio es un estudio (la desarrollamos a la hora de la imputacion de), la distancia al centro o la reputación del resto de anuncios del anfitrión.
> Tras este tratamiento nos quedamos con 36.922 anuncios y 79 variables, que podemos agrupar en cinco bloques: producto, ubicación, condiciones de reserva, anfitrión y equipamiento. 

*Si preguntan:* la ventana de amenities la aprende el `PreparadorListings` solo con el train. Los servicios de lujo se quedaron fuera porque tenían muy pocos anuncios para interpretarlos (*ski-in/ski-out*: 8). Para el kNN y la búsqueda del lineal quitamos además los servicios casi duplicados (|r| > 0,8: lavadora–secadora y una cadena de 6 de cocina): de 71 a 65 variables y el VIF máximo de 12,3 a 2,9. Al controlar por producto, el efecto de los servicios sobre el precio
baja un 42%; los que más se asocian a precios altos son de edificio y vivienda de gama alta (gimnasio, chimenea), con primas de alrededor del 20% en la OLS.

### Diapositiva 6 — Un diseño que impide engañarse · 1:30

**En pantalla:** esquema exploración → partición → train 80% / test 20% sellado con MD5; 38% en carteras de varios
pisos, 23,7% con un gemelo, 0 cambios al rehacer el EDA.

> Un error de predicción solo es creíble si se mide sobre datos que nunca se han visto. Una de las partes mas importante en todo trabajo de ciencia de datos es la particion del conjunto de datos para obtener resultados realistas y evitar el sobreajuyste. En nuestro caso, **apartamos el 20% de los anuncios antes de mirar el precio**, lo guardamos con una huella MD5, que garantiza que el fichero no se ha modificado, y solo lo abrimos una vez, al final del trabajo para evaluar nuestro modelo final. 
> Un detalle esencial es que la partición está **agrupada por anfitrión**. Nos encontramos con que el 38% de los anuncios pertenece a anfitriones con varios pisos, y que casi uno de cada cuatro tiene un gemelo en la misma cartera: mismo tipo, capacidad, dormitorios y barrio. Si uno cae en entrenamiento y su gemelo en el test, el modelo prácticamente se sabe la respuesta y el error sale optimista. Agrupando, siempre evaluamos con anfitriones que el modelo no ha visto. También estratificamos por tipo de alojamiento y distrito (StratifiedGroupKFold nos permite hacer ambas): antes de tomar la decision simulamos 50 particiones y, sin estratificar, el test podía llegar a tener el doble de habitaciones de hotel de las esperadas, que como es una de las variables que mas influyen en el precio.

*Si preguntan:*
- **¿Por qué 80/20?** Con 36.922 anuncios, un test de 7.384 da intervalos estrechos (RMSE de CatBoost 0,408-0,451) y deja casi 30.000 anuncios para entrenar y validar.
- **¿Qué es exactamente un "gemelo" y por qué es fuga?** Dos anuncios del mismo anfitrión con igual tipo, capacidad, dormitorios y barrio: suelen tener casi el mismo precio. Si uno está en train y otro en test, el modelo "recuerda" el precio en vez de predecirlo.
- **¿Por qué no una partición temporal?** Tenemos una única foto (febrero de 2021). El caso de uso es un anuncio de un anfitrión que el modelo no conoce, y eso lo imita la agrupación por anfitrión. Validar en otra ventana temporal es una línea futura.
- **¿Para qué el MD5, si ya hay semilla?** La semilla no garantiza la misma partición si cambia una versión de librería; la huella sí garantiza que el fichero del test es el mismo. Ver D3.
- **¿Qué significa "0 cambios al rehacer el EDA"?** Que el primer EDA lo hicimos con todos los datos, nos dimos cuenta de que así el test influía en nuestras decisiones, y lo rehicimos solo con el train: ninguna decisión cambió de sentido. Contarlo así, de frente: es un error que detectamos y corregimos.
- **¿La exploración de calidad (`1-EXPLORACION`) usa todo el dataset?** Sí, pero sin mirar el precio: duplicados, nulos y tipos. Todo lo que mira el precio se hace solo con el train.


### Diapositiva 7 — Lo que dicen los datos antes de modelar · 1:17

**En pantalla:** mapa hexagonal del precio (Figura 3); 140$ piso entero / 60$ habitación; Manhattan 120$ / Bronx 67$;
−50% de precio por plaza.

> "Con el train ya separado, hicimos el análisis exploratorio. Usamos medianas y contrastes no paramétricos, porque el precio es muy asimétrico y la media se deja arrastrar por los anuncios caros (el log(precio) se usa a la hora de modelar en el analisis de datos seguimos utilizando el precio bruto). Nos quedamos con tres mensajes:
> El primero: **el tipo de alojamiento es lo que más separa el precio**. Un piso entero cuesta más del doble que una habitación privada, y entre los dos suman el 97% de la oferta.
> El segundo: **la ubicación funciona en tres niveles**, el distrito, el barrio y la distancia al centro, y, como se ve en el mapa, el gradiente se mantiene incluso dentro de cada distrito. 
> El tercero: **hay economía de escala**, y depende de dónde. Un huésped extra vale 36 dólares en Manhattan y 20 en el Bronx: las variables no suman por separado, se combinan.
> Más allá de las cifras, esta fase nos dejó un hábito de trabajo: medir cada relación **dentro del mismo tipo de producto**. La comparación a simple vista nos engañó varias veces. Por ejemplo, los pisos con más huéspedes por dormitorio parecían más caros, pero a igual capacidad, más dormitorios es siempre más precio,

*Si preguntan:*
- **¿Por qué Kruskal-Wallis y no ANOVA?** El precio en dólares es muy asimétrico y con varianzas distintas entre grupos, y Kruskal-Wallis no supone normalidad. Aun así lo contrastamos con ANOVA clásico y de Welch: las tres dicen lo mismo.
- **Hacéis muchos contrastes, ¿no salen falsos positivos?** Por eso los *post-hoc* de Mann-Whitney llevan corrección de Holm. Con casi 30.000 anuncios casi todo es "significativo", así que nos fijamos en el tamaño del efecto (ε², η²) y no en el p-valor.
- **¿Por qué ocultáis las celdas con menos de 30 anuncios?** Una mediana con muy pocos anuncios es inestable: un par de pisos caros la mueven mucho.
- **¿Cómo sabéis que el barrio añade algo al distrito?** Con el η² sobre `log(price)`: el distrito explica 0,084 y el barrio 0,190; y dentro de los pisos enteros, 0,023 frente a 0,096.
- **¿Desde dónde medís la distancia al centro?** Desde City Hall, un punto de referencia externo fijado de antemano, no derivado del precio.
- **¿Qué implica la economía de escala para el modelo?** Que un huésped extra vale 36$ en Manhattan y 20$ en el Bronx: capacidad y ubicación interactúan. Un lineal no lo capta salvo que se le añada la interacción, y los árboles sí. Es una de las razones por las que ganan los *boosting*.
- **¿Encontrasteis algo más?** Señales menores: la reserva inmediata se asocia a un precio un 7% más bajo a igual producto y zona, los anuncios inactivos piden un 18% más, y los superhosts no cobran más.

### Diapositiva 8 — Cómo entrenamos y comparamos los modelos · 1:15

**En pantalla:** a la izquierda, las cinco familias de modelos de simple a complejo; a la derecha, el protocolo
(validación cruzada de 5 folds guardada, pipeline, búsqueda aleatoria de hiperparámetros, reglas fijadas de antemano).

> Los modelos que hemos usado de menor a mayor complejidad son: reglas ingenuas, que predicen la mediana de un grupo de anuncios parecidos; modelos lineales, desde la regresión clásica hasta sus versiones regularizadas (lasso, ridge y elasticnet); vecinos más cercanos (KNN) y árboles de decisión (CART), Random Forest y dos modelos de *gradient boosting* ya que es de las tecnicas mas potentes, HGB (Histogram GB) y CatBoost; y, por último, una SVR y una red neuronal (MLP, multilayer perceptron).
> Para evaluarlos usamos **validación cruzada de 5 folds**, agrupada por anfitrión y estratificada igual que la partición, y guardamos los folds en un fichero. Así todos los modelos se evalúan con exactamente los mismos datos y podemos compararlos de forma pareada, fold a fold.
>Hacemos la busqueda hiperparametrica sobre esos mismos folds. La metrica que utilizamos para elegir el mejor modelo es el RSMLE (Root Squared Mean Logarithmic Error)

*Si preguntan:* Random Forest es *bagging* (*bootstrap aggregating*): cada árbol se entrena sobre una muestrabootstrap de los datos y sus predicciones se promedian; HGB y CatBoost son *boosting*: árboles pequeños en secuencia, cada uno corrigiendo los errores de los anteriores. Búsqueda aleatoria y no en rejilla porque, con muchos hiperparámetros, explora mejor con el mismo número de pruebas. Candidatos: HGB 40, CatBoost 10 (bajó de 40 por cómputo, viendo que el óptimo era plano), red neuronal 60. Parada temprana de CatBoost: 65 iteraciones sin mejora sobre un 10% del train del fold. kNN con PCA: componentes que explican al menos el 80% de la varianza. Árbol con prepoda y regla 1-SE.

- **¿Por qué dos *gradient boosting*? ¿Tanto difieren?** Los elegimos porque difieren en **cómo tratan las
  categóricas**, sobre todo el barrio (220 categorías). HGB crece los árboles hoja a hoja (máximo 15 hojas), como LightGBM, y necesita codificar el barrio antes con `TargetEncoder`, cuyo *cross-fitting* es por filas, no por anfitrión. CatBoost usa árboles simétricos (la misma pregunta en todo un nivel, profundidad 7), que regularizan  predicen muy rápido, y codifica las categóricas él mismo con *ordered target statistics*, sin que el precio de un anuncio entre en su propia codificación. A cambio, tarda unas 15 veces más en entrenar (325 s frente a 21 s).
- **¿Y en la práctica?** Casi no difieren: 0,416 frente a 0,419 en CV (equivalentes según el TOST), el orden se invierte en el test sin que la diferencia sea concluyente, y los 20 peores errores son prácticamente los mismos anuncios. Que dos mecanismos distintos acaben en el mismo error y fallen en los mismos anuncios indica que **el límite está en los datos, no en el algoritmo**: lo que falta es información que distinga el lujo.
- **¿Por qué no XGBoost o LightGBM?** Usan el mismo algoritmo que HGB; no habrían aportado una forma distinta de aprender (D16).

### Diapositiva 9 — La preparación de cada modelo · 1:30

**En pantalla:** tabla modelo / preparación de los datos / tratamiento del barrio; debajo, el empate de las tres codificaciones del barrio en HGB.

| Modelo | Preparación de los datos | El barrio (220 categorías) |
|---|---|---|
| Lineales y SVR | One-hot, log de conteos, imputación con indicador de nulo, escalado | Target encoding |
| kNN | 5 variables escaladas, o reducción con FAMD / PCAmix | Fuera (coordenadas) o target encoding |
| Árbol y Random Forest | Categóricas a enteros, nulos nativos, sin escalar | Código entero |
| HGB | Categóricas y nulos nativos, sin escalar | Target encoding |
| CatBoost | Sin preprocesado: categóricas nativas | Ordered target statistics |
| Red neuronal | Como los lineales | Embedding de 16 dimensiones |

> "Cada familia de modelos necesita los datos preparados de una forma distinta, y toda esa preparación va **dentro del pipeline**.
> Los modelos lineales, la SVR y el kNN trabajan con coeficientes o con distancias, así que necesitan todo en números y en la misma escala: pasamos las categóricas a variables binarias, aplicamos logaritmo a los conteos, imputamos con la mediana añadiendo un indicador de que el dato faltaba y escalamos. 
> Los árboles, en cambio, no necesitan escalado y manejan los nulos por sí mismos, así que su preparación es mínima.
> El caso más interesante es **el barrio**: 220 categorías, la mitad con menos de 30 anuncios. Pasarlo a binarias crearía 220 columnas casi vacías, así que lo tratamos de forma distinta según el modelo. En los lineales y en HGB usamos **target encoding**: cada barrio se sustituye por el precio medio de sus anuncios, calculado dentro del fold para no usar el propio precio. CatBoost lo hace de forma nativa con ***ordered target statistics***, que codifica cada anuncio solo con los anuncios anteriores en un orden aleatorio. 
> En la red neuronal, cada barrio aprende su propio vector, un ***embedding***. 
> Y en el kNN más sencillo lo dejamos fuera y usamos directamente las coordenadas.

*Si preguntan:* el `TargetEncoder` hace *cross-fitting* interno de 5 particiones, pero por filas, no por anfitrión
(efecto pequeño, declarado como limitación). RMSE en HGB: target encoding 0,423, nativa 0,425, sin barrio 0,426. El kNN
de 5 variables usa latitud, longitud, capacidad, dormitorios y tipo de alojamiento. En la red, los barrios con muy
pocos anuncios comparten un índice y los no vistos en el train del fold van a otro aparte.

### Diapositiva 10 — Los resultados de la escalera · 1:10

**En pantalla:** barras horizontales con el RMSE de validación de cada modelo, de la mediana global a CatBoost.

> "Este es el resultado de la escalera, medido con el error de validación cruzada: cuanto más corta la barra, mejor. Nuestra vara de medir es la segunda barra, predecir la mediana de los anuncios del mismo tipo y capacidad, que es lo que haría hoy un anfitrión; todo lo que la mejore es valor que aporta el modelo.
> Del modelo lineal sacamos una primera lectura: al ir añadiendo bloques de variables, el producto explica lo básico, **la ubicación es el mayor salto**, las condiciones de reserva no aportan nada y el anfitrión y el equipamiento aportan algo, pero poco.
> La conclusión más interesante es que **el modelo lineal ya recoge la mayor parte de la señal** y queda a solo un 4% del mejor. Lo que añaden los *boosting*, HGB y CatBoost, son no linealidades, como la ubicación fina o los pisos más grandes, y eso basta para que sean los mejores. La red neuronal no consigue superarlos."

*Si preguntan:*
- **¿Por qué la referencia es la mediana por tipo × capacidad y no por barrio × tipo, que tiene menos error en dólares (26$ frente a 29$)?** Porque elegimos con la métrica fijada, el RMSE en log, y ahí gana tipo × capacidad (0,535 frente a 0,543). Además, es la regla que aplicaría un anfitrión ("pisos como el mío").
- **Si el lineal queda a un 4% del mejor, ¿merece la pena un modelo más complejo?** En CV, CatBoost gana al lineal en todos los folds, y la mejora se concentra en los pisos grandes y en la ubicación fina, que es donde más se equivoca la regla. Si lo que se prioriza es la interpretabilidad, el lineal es una alternativa honesta, y así lo decimos.
- **¿Por qué el árbol de decisión va tan mal?** Un solo árbol predice valores constantes por tramos y es inestable; con la prepoda que necesita para no sobreajustar, se queda corto. Por eso lo usamos como peldaño intermedio y no como candidato.
- **Si las condiciones de reserva no aportan, ¿por qué siguen en el modelo?** La selección solo se aplica al lineal y al kNN. En los árboles, quitar variables empeoraba el resultado, así que usan las 79 (D12).
- **¿Qué significa "un 4%"?** Es la diferencia relativa de RMSE en CV: 0,434 del Lasso frente a 0,416 de CatBoost (un 4,3%).

### Diapositiva 11 — Cómo elegimos el modelo final · 1:00

**En pantalla:** CatBoost 0,416 vs HGB 0,419 en CV; "regla fijada el 30-09: menor RMSE medio de CV"; "TOST ±0,005:
equivalentes".

> "Elegimos a nuestro modelo final como el modelo con menor error medio en validación cruzada. Ganó **CatBoost**, por poco pero de forma consistente, porque gana en los cinco folds. Como la diferencia con HGB es menor que un error estándar, hicimos además una prueba de **equivalencia**, el TOST y concluye que ambos son equivalentes en la práctica. Por eso presentamos dos finalistas: **CatBoost, que es el modelo que entregamos**, y **HGB como alternativa equivalente, unas 15 veces más rápida** de entrenar, por si el coste de cálculo fuera importante. Fijar la regla de antemano nos protege de acabar eligiendo lo que más nos convenga después de ver los resultados."

*Si preguntan:*
- **¿Por qué no combinar CatBoost y HGB en un *ensemble*?** Son equivalentes y fallan en los mismos anuncios (los 20 peores errores son prácticamente los mismos), así que la ganancia esperada es menor que nuestro margen de equivalencia. Además no estaba en el plan fijado de antemano. Queda como línea futura.
- **¿Qué habríais hecho si el TOST no hubiera concluido equivalencia?** Entregar CatBoost igualmente, porque lo dice la regla. El TOST solo decide si HGB puede presentarse como alternativa equivalente.
- **¿Ganar en 5 de 5 folds es suficiente?** Por sí solo no; por eso lo acompañamos del contraste pareado con la corrección de Nadeau-Bengio (p = 0,040) y del TOST.
- **¿Llegasteis a plantear elegir HGB por ser más rápido?** Sí, se propuso y se retiró porque contradecía la regla. Contarlo (D21): demuestra que la regla se respetó.
- **¿Por qué ±0,005 de margen?** Ver la sección 3.

### Diapositiva 12 — Resultados en el test · 1:25

**En pantalla:** 21$ de mediana del error, −22% frente a la regla ingenua, 77,9% de cobertura; Figura S3 del sponsor
(29$ → 21$).

> "Una vez obtenido los dos finalistas pasamos a predecir sobre los datos tests. En un anuncio típico, **el precio sugerido se desvía unos 21 dólares** del real. Frente a la regla de 'anuncios parecidos' (referencia ingenua), el error se reduce un **22%**. Además, el resultado del test es coherente con la validación cruzada, así que no hay señal de que nos hayamos sobreajustado a ella.
> El rango del 80% lo calibramos con una técnica llamada **regresión cuantílica conformal**: sin calibrar, el rango solo cubría el 72% de los precios; calibrado, **cubre el 78% en el test**, con el 80% dentro de su intervalo de confianza.
> Para saber cuánto podemos fiarnos de estas cifras, calculamos sus intervalos de confianza con **bootstrap**: remuestreamos 2.000 veces los anfitriones del test, no los anuncios sueltos, porque los anuncios de un mismo anfitrión se parecen entre sí y tratarlos como independientes daría una falsa precisión
> Finalmente sobre los datos test, HGB sale ligeramente mejor que CatBoost, pero la diferencia no es concluyente"

*Si preguntan:*
- **El 22%, ¿es en test o en validación?** **En CV**: RMSE 0,416 frente a 0,535 de la regla ingenua. Cuidado, porque en esta diapositiva se presenta junto a las cifras del test; o se dice "en validación" o se calcula la regla
  ingenua sobre el test.
- **¿Por qué dais la mediana del error (21$) y no el MAE (50$)?** El MAE lo arrastran unos pocos pisos caros con errores grandes. La mediana describe el anuncio típico, que es lo que verá un anfitrión. Las dos cifras están en la memoria.
- **¿Cómo funciona la CQR, en dos frases?** Entrenamos dos modelos para los cuantiles 10% y 90% y miramos en un conjunto de calibración cuánto se salen los precios reales del rango. Ensanchamos el rango justo lo necesario para que cubra el 80%, sin suponer ninguna distribución.
- **El rango cubre el 77,9%, no el 80%: ¿falla la calibración?** No de forma concluyente: el intervalo de confianza (72,3-81,5) contiene el 80%. Declaramos dos limitaciones: un solo conjunto de calibración (el fold 0) e intercambiabilidad solo aproximada, porque los anuncios de un mismo anfitrión se parecen.
- **¿Por qué un *bootstrap* por anfitrión y no por anuncio?** Los anuncios de un mismo anfitrión no son independientes; remuestrearlos sueltos daría intervalos demasiado estrechos.
- **¿Por qué el test sale algo peor que la CV?** Es lo esperable: el RMSE de CV del ganador es algo optimista, porque se elige sobre esos mismos folds. La diferencia (0,013) está dentro de la variabilidad.

### Diapositiva 13 — Qué mueve el precio · 1:00

**En pantalla:** Figura S2 del sponsor (cuánto sube el error si se oculta cada bloque); +4% sin el anfitrión.

> "Para entender qué mueve el precio, medimos cuánto empeora el modelo si le ocultamos cada tipo de información. Si le quitamos el tipo y el tamaño del alojamiento, su error sube un 73%; la ubicación, un 25%; el anfitrión y el equipamiento, alrededor de un 10%; y las condiciones de reserva, casi nada. El orden es el mismo en **todas** las familias de modelos, y SHAP lo confirma en el modelo final.
> Conocer al anfitrión ayuda, pero poco: sin su perfil, el error solo sube un 4%, así que la herramienta también sirve para quien publica por primera vez.
> Y un aviso importante: esto describe **cómo razona el modelo, no relaciones causales**. Que los pisos con gimnasio sean más caros no significa que instalar un gimnasio vaya a subir el precio."

*Si preguntan:*
- **¿Por qué la importancia por permutación y no la del propio árbol?** La de impureza premia las variables continuas
  con muchos cortes: `n_anuncios_ny` era la 4.ª por impureza y la 9.ª por permutación. La permutación se mide en
  validación, que es lo que importa para generalizar (D17).
- **¿Por qué por bloques y no variable a variable?** Las variables de ubicación están muy correlacionadas: si
  permutamos solo la latitud, la longitud y la distancia compensan, y la importancia se reparte y parece pequeña.
- **¿Qué diferencia hay entre SHAP y la permutación?** La permutación mide cuánto empeora el error global sin una
  variable. SHAP reparte cada predicción individual entre sus variables. Coinciden en el orden: tipo de alojamiento y
  capacidad, en cabeza.
- **¿El 4% de no conocer al anfitrión es sólido?** Sale igual en dos familias distintas: +3,7% en HGB y +4,0% en el
  lineal.
- **¿Entonces SHAP dice qué hacer para subir el precio?** No. Ver la sección 3: describe cómo razona el modelo, no
  relaciones causales.

### Diapositiva 14 — Dónde no fiarse: el lujo · 0:55

**En pantalla:** Figura S4 del sponsor (cobertura por quintil de precio); el caso de 6.500$ predicho en 460$.

> "Tan importante como saber cuánto acierta el modelo es saber dónde falla. En los precios intermedios el rango acierta en más de 8 de cada 10 casos, pero **en el 20% más caro solo en algo más de la mitad**. Cuando analizamos los 20 peores errores, nos encontramos con que todos son pisos de lujo a los que el modelo pone un precio demasiado bajo.
> El caso más extremo es un piso de 6.500 dólares que se predice en 460: para el modelo es un piso de siete plazas y tres dormitorios como cualquier otro, porque **nada en los datos delata el lujo**: no hay superficie, ni fotos, ni descripción. No es un problema del algoritmo, sino de la información disponible."

*Si preguntan:*
- **¿Cuánto pesa la cola en el error?** Sin los precios por encima del percentil 99 del train (794$, 81 anuncios), el
  RMSE baja de 0,429 a 0,395 y el MAE de 50$ a 39$: la cola explica unos 11$ del MAE.
- **Entonces, ¿por qué no quitasteis esos pisos?** Porque son lujo legítimo, no errores. Recortar por el P99 eliminaba
  el 18,8% de los pisos para 9 o más huéspedes, y el modelo tiene que saber que existen (D7).
- **¿Por qué no un modelo específico para el lujo, o dar más peso a esos anuncios?** Porque el problema es de
  información: sin superficie, fotos ni descripción, un piso de lujo y uno grande normal son iguales para cualquier
  modelo, y darles más peso solo empeoraría el resto.
- **¿La baja cobertura en el quintil caro no indica un rango mal calibrado?** La CQR garantiza la cobertura media, no
  en cada segmento. Además, el quintil se define con el precio real, lo que penaliza por construcción los extremos.
- **¿Hay otros segmentos débiles?** Sí, las habitaciones de hotel: un MAE de 175$, pero son solo 57 anuncios en el
  test. Y Manhattan, donde el rango cubre el 73,7%.

### Diapositiva 15 — Recomendaciones · 1:05

**En pantalla:** tres recomendaciones de uso + siguiente paso.

> "A partir de todo esto, hacemos tres recomendaciones. Primera: usar la herramienta como **sugerencia de precio al publicar**, mostrando el precio y el rango, nunca como un precio automático; la decisión final es del anfitrión. Podria ser una herramienta automatica de AriBNB.
> Segunda: **marcar para revisión humana** los anuncios grandes o de gama alta, y aquellos cuyo precio queda fuera del rango, que pueden ser un error, un precio de bloqueo o un lujo que el modelo no ve. Tercera: **reentrenarla periódicamente**, porque aprendió de una foto del mercado de febrero de 2021, en plena pandemia.
> En nuestra opinión, el siguiente paso con más impacto no es un modelo más complejo, sino **mejores datos**: superficie, fotos (crear una rnn y sacar datos de estas) y texto de la descripción, que es justo lo que falta para distinguir el lujo.
> Todo el trabajo es reproducible: el repositorio, la partición con su huella, el registro de experimentos y una función que, a partir de un anuncio en bruto, devuelve su precio y su rango."

*Si preguntan:*
- **¿Por qué no fijar el precio automáticamente?** El modelo falla en el lujo, no ve la demanda ni el calendario, y
  sus relaciones no son causales. Debe ser una sugerencia, y la decisión, del anfitrión.
- **¿Qué pasa con un anfitrión que publica por primera vez?** Es el escenario A: se predice sin su perfil, y el error
  sube solo un 4%. En la función actual, la cartera del anfitrión se calcula con las filas que recibe; en producción,
  habría que sacarla de la base de datos de anfitriones.
- **¿Cada cuánto reentrenaríais?** No lo hemos medido, porque solo tenemos una foto. Proponemos vigilar la cobertura
  del rango por segmento y reentrenar si baja del 75%, y en cualquier caso con cada instantánea nueva.
- **¿Cómo comprobaríais que la herramienta aporta valor real?** Con una prueba A/B: anfitriones nuevos con y sin
  sugerencia, comparando la ocupación y el tiempo hasta la primera reserva. El modelo solo mide el acierto frente al
  precio publicado, no el beneficio del anfitrión.
- **¿Afecta la normativa de 30 noches mínimas?** Explica que el 64% de los anuncios las exija. No cambia el precio por
  noche, pero define un mercado de estancia media. Ver la sección 5.

### Diapositiva 16 — Cierre · 0:20

> "En resumen: hemos construido una herramienta que mejora un 22% la referencia que usa hoy un anfitrión, con un rango que cumple lo prometido y un diagnóstico claro de sus límites. Muchas gracias; quedo a su disposición para las preguntas."

### Consejos de ritmo

- Si vas tarde al llegar a la diapositiva 10 (más de 11:10), acorta la 7 y la 13: son las que más se solapan.
- Estilo: hablar siempre en plural ("decidimos", "nos encontramos con"), salvo al presentarte y al despedirte.
- Las diapositivas 6, 8, 9, 11 y 14 son las que más valora un tribunal técnico (rigor y honestidad): no las recortes.
- No leas las tablas: señala con el puntero la fila de CatBoost y la de la regla ingenua y sigue.

---

## 2. Decisiones clave de la bitácora

Formato: **qué decidí · por qué · qué descarté · respuesta corta si me lo preguntan**.
Fuente: `docs/bitacora_decisiones.md`.

### 2.1 Alcance y diseño

**D1. Solo Nueva York (16-09).**
- *Por qué:* divisas mezcladas y no documentadas; el mercado se comporta distinto en cada ciudad (elasticidad 0,55 a
  1,12; habitación = 84% del piso en Bangkok y 39% en Sídney); la ciudad explicaba el 22% de la varianza del
  log-precio y varios efectos cambiaban de signo entre ciudades.
- *Descartado:* un modelo global con `city`; un modelo por ciudad; convertir divisas (no hay fecha del volcado ni
  resuelve el poder adquisitivo).
- *Respuesta corta:* "Un modelo global habría aprendido una media de mercados distintos y sería peor en todos. Nueva
  York tenía USD, `district` completo y muestra suficiente."

**D2. Barrera física contra la fuga: global → partición → train (26-09).**
- *Por qué:* cualquier decisión que mire el precio se toma solo con train. El EDA se hizo primero con todo y lo rehice
  (hallazgo H-035).
- *Respuesta corta:* "Puedo afirmar que ninguna decisión dependiente del precio ha visto el test, porque los notebooks
  de análisis y modelado no pueden leerlo."

**D3. Partición agrupada por `host_id`, estratificada por `room_type` × `district`, 20%, semilla 42, MD5 (26-09).**
- *Por qué agrupar:* 38% de anuncios en carteras múltiples, 23,7% con gemelo → con partición por filas el error sale
  optimista (Roberts et al., 2017; Kapoor y Narayanan, 2023).
- *Por qué estratificar por las dos:* simulación de 50 particiones; sin estrato, hasta el doble de hoteles en test;
  con estrato, desviación < 0,1 puntos en las 18 celdas.
- *Por qué MD5:* la semilla sola no garantiza el mismo resultado si cambia una versión de librería; los ficheros son la
  fuente de verdad.

**D4. CV de 5 folds × 3 repeticiones, guardada (25-09).**
- *Por qué:* hay que separar diferencias de ~0,005 de RMSE; los mismos folds hacen las comparaciones pareadas. Las 15
  estimaciones para seleccionar variables; la primera repetición (5) para comparar modelos y ajustar hiperparámetros
  (por coste).

**D5. Target `log(price)` (23-09).**
- *Por qué:* asimetría de 21,1 a 0,7; error relativo; `exp(ŷ)` estima la **mediana**, que es lo que se quiere
  recomendar.
- *Descartado:* predecir el precio en bruto (lo dominarían los anuncios caros).

**D6. Escenario B principal, A de contraste (25-09).**
- *Por qué:* la diferencia mide el valor de conocer al anfitrión (≈ 4%). Se descartó valorar anuncios existentes con
  reseñas por causalidad inversa (el precio influye en las reseñas).
- *Limitación declarada:* el perfil del anfitrión está medido en 2021, no en el momento de publicar.

### 2.2 Limpieza y variables

**D7. Precios de bloqueo fuera; no se recorta por P99 (26-09).**
- *Regla:* fuera `price` ≥ 9.999$ y habitaciones > 1.000$; umbrales fijados en train y aplicados al test (48 + 14).
- *Por qué no P99 (850$):* quitaba el 18,8% de los pisos para 9+ huéspedes, lujo legítimo.
- *Limitación:* en producción el precio no se conoce, así que el modelo no está pensado para tarifas de bloqueo.

**D8. `PreparadorListings` con `fit`/`transform` solo en train (26-09).**
- Aprende la ventana de amenities, los tipos de propiedad frecuentes y las medianas para imputar dormitorios.
- *Respuesta corta:* "Igual que un `StandardScaler`: se ajusta en train y se aplica congelado al test y a datos
  nuevos."

**D9. `bedrooms`: imputación por mediana de `room_type` × `accommodates` + variable `es_estudio`.**
- *Por qué:* el 70% de los pisos enteros sin dormitorios dice *studio* en el título y cuestan un 12% menos que un
  piso de un dormitorio de igual capacidad: el nulo es informativo.

**D10. Amenities: 56 binarias con prevalencia entre 3,5% y 97% (30-09).**
- *Por qué:* criterio **ciego al precio** (no mira el target, no hay fuga). Por encima del 97% o por debajo del 3,5%
  casi no varían.
- *Descartado:* top-N, selección por *lift* (mira el precio), contar amenities (`n_amenities`, no aporta señal propia),
  amenities de lujo (8 anuncios con *ski-in/ski-out*: no interpretables).

**D11. Reseñas: se analizan, no son variables (09-09).**
- *Por qué:* son posteriores a la publicación (fuga). Sesgo medido: pisos enteros sin reseña un 11% más caros.
- *Matiz:* `reputacion_cartera` sí usa reseñas, pero de **otros** anuncios del anfitrión, que existen al publicar.

**D12. Regla de selección de variables fijada antes de ejecutar (27-09).**
- *Regla:* entra si Δ = RMSE_sin − RMSE_con > 0 en ≥ 12 de 15 estimaciones **y** Δ medio > su desviación típica.
- *Matiz:* es una heurística, declarada como tal; el p de Nadeau-Bengio es informativo.
- *Aplicación:* la selección solo decide en el lineal y el kNN; los árboles usan las 79 (quitar variables les
  empeoraba).

**D13. Amenities casi duplicados (71 → 65) para kNN y lineal (06-10).**
- *Por qué:* colinealidad (|r| > 0,8: lavadora–secadora y una cadena de cocina). VIF máximo de 12,3 a 2,9.
- *Matiz que hay que declarar:* el umbral pasó de 0,9 a 0,8 el mismo día, tras ver la estructura de correlaciones pero
  **antes** de calcular ningún RMSE.

### 2.3 Modelado

**D14. Escalera de complejidad (27-09).** Cada peldaño responde a una pregunta. Ver diapositivas 8, 9 y 10.

**D15. kNN con PCA: nº de componentes = mínimo que explica ≥ 80% de la varianza, fijado antes; FAMD y PCAmix para
datos mixtos (30-09).**

**D16. CatBoost como modelo adicional y MLP aplazado a Colab (30-09).**
- *Por qué CatBoost y no XGBoost/LightGBM:* XGBoost/LightGBM son el mismo algoritmo que HGB; CatBoost aporta una
  forma distinta de tratar las categóricas (*ordered target statistics*).

**D17. Importancia por permutación en validación, no por impureza (01-10).**
- *Por qué:* la impureza premia variables continuas con muchos cortes (`n_anuncios_ny`: 4.ª por impureza, 9.ª por
  permutación). Por bloques porque las variables de ubicación están correlacionadas y su importancia se reparte.

**D18. Recorte de cómputo: CatBoost de 40 a 10 candidatos (02-10).**
- *Declarado:* se decidió viendo resultados parciales. Defensa: el óptimo ya era plano (8 mejores en 0,002, menos que
  la sd entre folds), y elegir entre muchos candidatos con los mismos folds sobreajusta la validación.

**D19. Resultado oficial de CatBoost en CPU (02-10).** La GPU no reproduce exactamente la CPU.

### 2.4 Modelo final y test

**D20. Regla de elección: menor RMSE medio de CV, escenario B (30-09, fijada antes de ver la tabla).**

**D21. Dos finalistas: CatBoost de referencia y entregado, HGB alternativa (06-10).**
- *Historia que debes conocer:* el 02-10 se propuso HGB "por excepción a la regla" por ser más rápido. **Se retiró**
  porque contradecía la regla fijada de antemano. Si te lo preguntan, cuéntalo: demuestra que la regla se respetó
  aunque hubo tentación de saltársela.
- *Respuesta corta:* "¿Cuál usarías? CatBoost, porque lo dice la regla. ¿Por qué no el rápido? Porque la diferencia de
  coste no justifica abandonar una regla fijada antes de ver los resultados; si el cliente prioriza coste, HGB es
  equivalente."

**D22. El test se abre una vez, tras comprobar MD5, y no elige (02-10).**

**D23. Intervalo con CQR (02-10).**
- Cuantiles 0,1 y 0,9 entrenados con el train del fold 0 (80%) y calibrados con su validación (20%).
- *Por qué:* garantía de cobertura en muestra finita sin suponer una distribución; pocos ajustes.
- *Limitación:* un solo conjunto de calibración; intercambiabilidad aproximada (anuncios del mismo anfitrión).

**D24. Modelo, cuantiles, ficha y `predecir_precio` guardados (02-10).**
- *Limitación:* `predecir_precio` necesita la cartera completa del anfitrión (`n_anuncios_ny`, `reputacion_cartera`).

---

## 3. Banco de preguntas probables del tribunal

### Sobre el planteamiento

**¿Por qué solo Nueva York? ¿No pierde generalidad?**
Ver D1. Añade: "La generalización a otra ciudad es una línea futura: validar la transferencia a otra ciudad con precio
en USD. Prefiero un modelo bueno en un mercado que uno mediocre en diez."

**¿Por qué no usaste las reseñas, si seguramente mejorarían el modelo?**
"Porque el caso de uso es un anuncio nuevo, que no tiene reseñas. Usarlas sería fuga: información que no existe en el
momento de predecir. Además hay causalidad inversa: el precio influye en las reseñas. El artículo de referencia sí las
usa (análisis de sentimiento), y por eso su cifra no es comparable con la mía."

**¿Cómo te comparas con la literatura?**
"El antecedente es Rezazadeh Kalehbasti et al. (2019), también en Nueva York: su mejor modelo, una SVR, obtiene un
R² de 0,69 y un MSE de 0,147 en log (RMSE ≈ 0,38). Yo obtengo R² 0,64 y RMSE 0,43 en test. Pero no son comparables:
ellos usan reseñas y su análisis de sentimiento, y parten por filas, sin agrupar por anfitrión, lo que infla el
resultado si hay gemelos. Mi cifra es la de un caso de uso más difícil y medida de forma más conservadora."

**¿Un R² de 0,64 es bueno?**
"Para un precio sin información de calidad (sin superficie, fotos ni descripción), sí: el lineal ya explica el 63% y
los *boosting* añaden poco más, lo que sugiere que estamos cerca del techo de lo que estas variables permiten. El
error restante se concentra en el lujo, que las variables no ven. La cifra útil para el negocio es la mediana del
error: 21$."

### Sobre la metodología

**¿Por qué agrupar por anfitrión?** Ver D3. Clave: gemelos → optimismo.

**¿Cómo sabes que no hay fuga?**
"Cuatro barreras: el test se separa antes de mirar el precio; los notebooks de análisis y modelado no lo leen; todo lo
que aprende de los datos (imputación, codificación, escalado) va dentro del `Pipeline` de cada fold o en el
`PreparadorListings` ajustado solo con train; y no uso nada posterior a la publicación. Además el test sale solo
ligeramente peor que la CV, que es lo esperable sin fuga."

**¿No hay fuga temporal en el perfil del anfitrión o en `reputacion_cartera`?**
"Sí, es una limitación que declaro: el perfil está medido en la foto de 2021, no en el momento exacto de publicar. Por
eso existe el escenario A sin ese bloque: el error solo sube un 4%, así que la conclusión no depende de ello. En
producción habría que tomar el perfil en el momento de publicar."

**¿Por qué 5 folds y no 10? ¿Por qué 3 repeticiones?**
"5 es un compromiso entre sesgo y coste con 29.500 anuncios (cada fold de validación tiene ~5.900). Las 3
repeticiones reducen la varianza de la estimación para la selección de variables, donde las diferencias son de
milésimas. Para comparar modelos y ajustar uso solo la primera repetición por coste."

**¿Por qué la t de Nadeau-Bengio y no una t normal?**
"Porque los folds comparten datos de entrenamiento: las diferencias entre folds no son independientes y la t clásica
subestima la varianza, dando p demasiado pequeños. Nadeau-Bengio corrige esa varianza. Aun así, la conclusión de
equivalencia se mantiene con las dos."

**¿Qué es el TOST y por qué lo usaste?**
"Un contraste normal solo dice si puedo rechazar que son iguales; no rechazar no significa que sean iguales. El TOST
invierte la pregunta: compruebo si la diferencia está dentro de un margen que considero irrelevante (±0,005 de RMSE en
log, fijado antes de calcular). Concluye que lo está: CatBoost y HGB son equivalentes en la práctica."

**¿Por qué ±0,005 como margen?**
"Es del orden de la desviación entre folds y equivale a menos de un 1,2% de diferencia relativa en el error: en
dólares, céntimos sobre una mediana de 21$. Se fijó antes de calcular y está fechado en la bitácora."

**¿Por qué no recortaste outliers por percentil?** Ver D7.

**¿Por qué mediana y no media al deshacer el logaritmo? ¿No hay sesgo de retransformación?**
"`exp(ŷ)` estima la mediana condicional, no la media. Para recomendar un precio la mediana es lo que quiero: el precio
'típico' de un anuncio así, robusto a la cola. Si quisiera la media habría que corregir (por ejemplo con el *smearing*
de Duan), pero no es el objetivo."

**¿Por qué Lasso como lineal si empataba con Ridge?**
"Con muchas más filas que variables, Ridge, Lasso y ElasticNet empatan en precisión; la regularización estabiliza los
coeficientes. Elegí el que dio el menor RMSE en la búsqueda, que fue Lasso, por una diferencia mínima."

**¿Por qué el kNN va tan mal?**
"El kNN sufre con muchas variables mezcladas (binarias, numéricas, categóricas): la distancia pierde sentido. Por eso
probé reducir con PCA, FAMD y PCAmix, y aun así queda lejos. Es esperable: no aprende qué variables importan, las
pondera todas por igual en la distancia."

**¿Por qué la red neuronal no ganó?**
"Es lo esperado en datos tabulares de tamaño medio (Grinsztajn et al., 2022): los árboles manejan mejor variables
heterogéneas, umbrales bruscos y variables irrelevantes. Hice una búsqueda de 60 candidatos y repetí con varias
semillas para no elegir por suerte; pierde con los dos *boosting* en los 5 folds. Detalle: usa 71 variables, no 79,
pero en la SVR pasar a 79 mejoró 0,004, menos que la distancia con HGB, así que la conclusión no cambia."

**¿Por qué CatBoost tiene RMSE 0,318 en train y 0,416 en CV? ¿Sobreajusta?**
"Se ajusta más al train, pero eso no es sobreajuste dañino si generaliza: el error en validación y en test es igual o
mejor que el de los demás. Lo controlan la parada temprana y la regularización L2. Lo declaro como inconveniente frente
a HGB."

**¿Por qué en test HGB es mejor y entregas CatBoost?** Ver D21 y D22. "Elegir con el test lo invalidaría como medida
independiente. Además la diferencia en test (−0,005, IC 95% de −0,017 a +0,003) incluye el cero."

**¿El RMSE de CV del ganador no es optimista?**
"Sí, un poco: se elige el mejor candidato sobre los mismos folds en los que se mide. Por eso existe el test, y la
diferencia CV–test (0,013 en CatBoost) es pequeña y está dentro de la variabilidad."

**¿Por qué recortaste la búsqueda de CatBoost viendo los resultados?** Ver D18. Reconócelo sin rodeos: está declarado.

**¿Cómo codificaste el barrio con 220 categorías?**
"Probé tres opciones en HGB: *target encoding* (dentro del pipeline, para que no haya fuga), categórica nativa y sin
barrio. Ninguna diferencia pasa la regla: con coordenadas, distancia y distrito, el barrio apenas añade información.
CatBoost usa *ordered target statistics*, que codifica cada anuncio solo con los anteriores en un orden aleatorio, así
el precio de un anuncio no entra en su propia codificación."

**¿Por qué no hay mayor cobertura en el quintil caro? ¿Está mal calibrado el intervalo?**
"La CQR garantiza la cobertura media, no en cada segmento. Además el quintil se define con el precio real, lo que
favorece mecánicamente peor cobertura en los extremos. El problema de fondo es de información: los pisos de lujo se
parecen a los grandes normales. Por eso recomiendo revisión humana en ese segmento."

**¿Los coeficientes o SHAP son causales?**
"No. Describen cómo usa el modelo cada variable o una asociación condicionada. Que el gimnasio se asocie a un 20% más
de precio refleja que está en edificios de gama alta, no que instalarlo suba el precio."

**¿Por qué errores agrupados por anfitrión en la OLS?**
"Los anuncios del mismo anfitrión no son independientes. Agrupar hace los errores estándar 1,5 veces mayores y evita
declarar significativos siete efectos que no lo son."

**¿Por qué excluiste `maximum_nights`?**
"Su efecto marginal era composición: al condicionar por producto, el signo no era consistente entre estratos. No aporta
información estable."

### Sobre negocio y despliegue

**¿Cómo se pondría en producción?**
"La cadena completa está empaquetada en `predecir_precio`: anuncio en bruto → preparador → modelo → precio y rango.
Habría que servirla en un entorno con las versiones congeladas, calcular la cartera del anfitrión desde la base de
datos, reentrenar periódicamente y vigilar la cobertura del intervalo por segmento (alerta si baja del 75%)."

**¿Qué harías con más tiempo?**
"Primero, datos que distingan el lujo: superficie, fotos, texto de la descripción. Después, el perfil del anfitrión en
el momento de publicar, otra ventana temporal y probar la transferencia a otra ciudad."

**¿Qué pasa con la pandemia?**
"La foto es de febrero de 2021, con la demanda alterada. El modelo aprende la relación entre características y precio
en ese momento; antes de usarlo hay que reentrenarlo con datos recientes."

**¿Cuánto tarda?**
"Reentrenar con todo el train: CatBoost ~325 s, HGB ~21 s. Predecir es inmediato. El plan completo de experimentos
costó unas 5 horas de cómputo."

### Preguntas "de pillar" (contradicciones a tener resueltas)

Antes de la defensa, revisa estas discrepancias entre documentos para que no te pillen:

| Dato | Memoria (lo que vale) | Bitácora | Qué decir |
|---|---|---|---|
| Asimetría del precio | 21,1 → 0,7 (train) | 13 → 0,6 | La de la memoria está calculada en el train final, tras la limpieza; la de la bitácora es de una versión anterior. |
| Anuncios idénticos salvo el id | 11 | 6 | `1-EXPLORACION` imprime "Filas idénticas salvo el listing_id: 11". Di 11; el 6 de la bitácora es de una versión anterior (o cuenta grupos). |
| HGB más rápido | ~15 veces | ~16 veces | Di "unas 15 veces". |
| Partición | 29.538 / 7.384 | 29.586 / 7.398 | La bitácora da ambas: la segunda pareja es antes de quitar los precios de bloqueo. |

---

## 4. Apéndice: glosario de todo lo usado

Para cada elemento: **qué es** y **para qué lo usé**. Pensado para responder "¿y esto qué hace?" en una o dos frases.

### 4.1 Métricas y conceptos básicos

- **RMSE en log:** raíz del error cuadrático medio sobre `log(price)`. Penaliza más los errores grandes. Como es en
  log, un RMSE de 0,43 equivale aproximadamente a equivocarse en un factor de e^0,43 ≈ 1,5 en un error "típico
  cuadrático". Lo usé para comparar modelos.
- **MAE:** error absoluto medio en dólares. Lo arrastran los anuncios caros (50$).
- **Mediana del error absoluto:** el error de un anuncio típico (21$). Es la cifra para el sponsor.
- **R²:** proporción de la varianza del log-precio que explica el modelo (0,64 en test).
- **Validación cruzada (CV):** dividir el train en *k* partes, entrenar con *k*−1 y validar con la restante, rotando.
  Da una estimación del error con su variabilidad.
- **Fuga de información (*data leakage*):** que el modelo use, al entrenar o evaluar, información que no tendría al
  predecir de verdad. Hace que el error parezca menor de lo que es.
- **Elasticidad:** pendiente en escala log-log; cuánto varía el precio en % al variar la capacidad en %. No le afecta
  la divisa.
- **Huella MD5:** un código que resume el contenido de un fichero; si cambia un byte, cambia la huella. Lo usé para
  garantizar que el test no se modificó.

### 4.2 Antecedente (H.1)

- **Rezazadeh Kalehbasti, Nikolenko y Rezaei (2019).** Predicción del precio de Airbnb en Nueva York con regresión
  lineal, Ridge, *gradient boosting*, k-means + Ridge, SVR y redes, con selección de variables por Lasso y análisis de
  sentimiento de reseñas. Su mejor modelo (SVR) logra R² 0,69 en test. Lo usé como antecedente del problema; mi
  enfoque difiere en que no uso reseñas y valido agrupando por anfitrión.

### 4.3 Contrastes estadísticos del EDA (H.2)

- **Kruskal-Wallis:** contraste no paramétrico que compara la distribución de una variable en varios grupos usando
  rangos en vez de valores. No supone normalidad. Lo usé para ver si el precio difiere entre tipos de alojamiento o
  distritos; su tamaño de efecto es **ε²** (proporción de variación en rangos explicada por el grupo; 0,41 para el
  tipo de alojamiento).
- **Mann-Whitney:** la versión para dos grupos. Lo usé como *post-hoc* tras Kruskal-Wallis, para ver qué pares de
  grupos difieren.
- **Corrección de Holm:** al hacer muchos contrastes a la vez aumenta la probabilidad de algún falso positivo. Holm
  ajusta los p-valores de forma escalonada; es menos conservadora que Bonferroni pero controla igual el error.
- **ANOVA de Welch:** ANOVA que no supone varianzas iguales entre grupos. Su tamaño de efecto es **η²** (proporción de
  varianza explicada; 0,19 para el barrio).
- **Correlación de Spearman:** correlación sobre rangos; mide relaciones monótonas aunque no sean lineales y es robusta
  a valores extremos. La usé para tendencias (por ejemplo, densidad por dormitorio ρ = +0,38).
- **V de Cramér:** asociación entre dos variables categóricas (0 = ninguna, 1 = total). Para estudiar relaciones entre
  categóricas en el EDA.

### 4.4 Comparación de modelos e incertidumbre (H.3)

- **Validación agrupada (Roberts et al., 2017; Kapoor y Narayanan, 2023):** cuando los datos tienen estructura
  (espacial, temporal, jerárquica), las observaciones de un mismo grupo deben ir juntas al partir. Roberts lo demuestra
  en ecología; Kapoor y Narayanan documentan la fuga como causa de la "crisis de reproducibilidad" en ciencia con ML.
  Lo apliqué agrupando por anfitrión (`StratifiedGroupKFold`).
- **t corregida de Nadeau-Bengio:** t pareada para comparar dos modelos sobre los mismos folds, que infla la varianza
  para compensar que los conjuntos de entrenamiento se solapan. La usé para dar p-valores honestos en las comparaciones.
- **TOST (*two one-sided tests*):** contraste de **equivalencia**: dos contrastes unilaterales que comprueban que la
  diferencia no es mayor que +margen ni menor que −margen. Si ambos rechazan, los modelos son equivalentes. Lo usé
  para CatBoost vs HGB con margen ±0,005.
- ***Bootstrap*:** remuestrear con reemplazo para estimar la variabilidad de una métrica. Lo hice remuestreando
  **anfitriones** (no anuncios), 2.000 réplicas, para los intervalos de confianza en test.

### 4.5 Modelos (H.4)

- ***The Elements of Statistical Learning* (Hastie, Tibshirani, Friedman):** manual de referencia de todos los modelos
  de la escalera.
- **Regresión hedónica (Rosen, 1974):** idea de que el precio de un bien es la suma de lo que vale cada una de sus
  características. Justifica leer los coeficientes de la OLS en log como "prima" de cada atributo (habitación privada
  −40%, cada dormitorio +16%).
- **OLS con errores agrupados:** regresión lineal clásica (`statsmodels`), con errores estándar robustos a que los
  anuncios de un mismo anfitrión estén correlacionados. Para interpretar coeficientes con inferencia correcta.
- **Ridge:** regresión lineal con penalización L2 (suma de coeficientes al cuadrado). Encoge todos los coeficientes y
  estabiliza el modelo cuando hay colinealidad.
- **Lasso:** penalización L1 (suma de valores absolutos). Puede llevar coeficientes exactamente a cero, así que además
  selecciona variables. Es el lineal elegido (0,434).
- **ElasticNet:** mezcla de Ridge y Lasso. Útil con grupos de variables correlacionadas.
- ***Splines*:** funciones por tramos para captar no linealidades en un modelo lineal. Sin penalizar hacían la OLS
  inestable, por eso no se usaron.
- **kNN (k vecinos más cercanos):** predice con la media de los *k* anuncios más parecidos según una distancia. Sencillo
  e intuitivo ("anuncios parecidos"), pero sufre con muchas variables.
- **PCA:** reduce variables numéricas a componentes no correlacionadas que retienen la mayor varianza.
- **FAMD / PCAmix:** la versión de PCA para datos **mixtos** (numéricas + categóricas). FAMD con la librería `prince`;
  PCAmix con implementación propia. Para que la distancia del kNN tenga sentido.
- **Árbol de regresión (CART):** divide los datos con preguntas sí/no ("¿es piso entero?") y predice la media de cada
  hoja. Interpretable pero inestable. Lo usé con prepoda (limitar profundidad/tamaño de hoja) y la regla 1-SE.
- **Random Forest:** muchos árboles entrenados sobre muestras *bootstrap* y subconjuntos aleatorios de variables; se
  promedian. Reduce la varianza del árbol individual.
- **Importancia por permutación:** desordenar una variable (o un bloque) en validación y medir cuánto empeora el error.
  Si empeora mucho, el modelo depende de ella. La usé por bloques en todos los modelos.
- ***Gradient boosting* (Friedman, 2001):** árboles pequeños en secuencia, cada uno corrigiendo los errores (el
  gradiente de la pérdida) de los anteriores. Es la base de HGB y CatBoost.
- **HGB (*HistGradientBoostingRegressor*):** *gradient boosting* de scikit-learn que agrupa las variables continuas en
  histogramas (intervalos) para ser mucho más rápido; inspirado en **LightGBM**. Finalista alternativo.
- **CatBoost:** *gradient boosting* de Yandex con **árboles simétricos** (la misma pregunta en todo un nivel, lo que
  regulariza y acelera la predicción) y tratamiento nativo de categóricas con ***ordered target statistics*** (codifica
  cada categoría con el precio medio de los anuncios *anteriores* en una permutación aleatoria, evitando la fuga del
  target encoding ingenuo). Modelo final.
- **Parada temprana (*early stopping*):** dejar de añadir árboles cuando el error en un conjunto de validación interno
  (10% del train del fold) deja de mejorar durante 65 iteraciones. Evita sobreajuste.
- **SVR (regresión de vectores soporte):** busca una función que deje la mayoría de puntos dentro de un "tubo" de
  anchura ε, penalizando solo los que quedan fuera. Con kernel capta no linealidades.
- **Aproximación de Nyström:** aproxima el kernel de la SVR con una muestra de puntos, para que escale a 30.000 filas
  (el kernel exacto sería cuadrático en memoria).
- ***Target encoding* (`TargetEncoder`):** sustituye cada categoría (barrio) por la media del target en esa categoría,
  calculada con validación cruzada interna para no usar el propio anuncio. Útil con muchas categorías. Matiz: su
  *cross-fitting* interno es por filas, no por anfitrión (limitación pequeña, declarada).
- **Red neuronal MLP (perceptrón multicapa):** capas de neuronas totalmente conectadas, entrenadas por
  **retropropagación** del error. La mía: 256-128-64 neuronas, activación **SiLU** (una versión suave de ReLU),
  ***dropout*** 0,2 (apaga neuronas al azar al entrenar para regularizar) y un ***embedding*** de 16 dimensiones para el
  barrio (aprende un vector por barrio en vez de 220 columnas binarias). Entrenada con PyTorch en Colab.
- **AdamW:** optimizador de la red; variante de Adam con ***weight decay*** desacoplado (penaliza los pesos grandes, como
  Ridge, sin interferir con la adaptación del paso de Adam).
- **Grinsztajn et al. (2022):** estudio que muestra que en datos tabulares de tamaño medio los modelos de árboles
  siguen superando a las redes. Por eso se esperaba que la MLP no ganara.

### 4.6 Rango de precio e interpretabilidad (H.5)

- **Regresión cuantílica (Koenker y Bassett, 1978):** en vez de predecir la media, predice un cuantil (aquí el 10% y
  el 90%) minimizando la pérdida *pinball*. Da los extremos del rango.
- **CQR (regresión cuantílica conformalizada; Romano, Patterson y Candès, 2019):** corrige los cuantiles con un
  conjunto de calibración: mira cuánto se salen los precios reales del rango y lo ensancha lo justo. Garantiza la
  cobertura en promedio sin suponer ninguna distribución (solo que los datos sean intercambiables). Pasó la cobertura
  del 72% al 78%.
- **SHAP (Lundberg y Lee, 2017):** reparte la predicción de cada anuncio entre sus variables según los valores de
  Shapley de teoría de juegos: cuánto "aporta" cada variable respecto a la predicción media. Permite explicar
  predicciones individuales y, promediando, la importancia global.
- **TreeSHAP (Lundberg et al., 2020):** algoritmo que calcula SHAP de forma exacta y rápida en modelos de árboles. Es
  el que trae CatBoost de forma nativa.

### 4.7 Herramientas y librerías

- **Python 3.13:** lenguaje del proyecto.
- **pandas 3.0:** manipulación de tablas (DataFrames).
- **pyarrow:** lectura y escritura de ficheros **Parquet** (formato columnar, comprimido y con tipos), usado para los
  datasets de modelado.
- **scikit-learn 1.9:** modelos (Ridge, Lasso, kNN, árbol, RF, HGB, SVR), `Pipeline`, `StratifiedGroupKFold`,
  `TargetEncoder`, `Nystroem`, importancia por permutación.
  - **`Pipeline`:** encadena preprocesado y modelo para que todo se ajuste solo con los datos de entrenamiento de cada
    fold. Es la principal defensa contra la fuga.
  - **`StratifiedGroupKFold`:** partición en folds que mantiene juntos los grupos (anfitriones) y equilibra un estrato.
- **CatBoost 1.2:** librería del modelo final, con SHAP nativo.
- **statsmodels:** OLS con errores estándar agrupados y contrastes.
- **prince:** FAMD para datos mixtos.
- **matplotlib y seaborn:** gráficos (con un estilo común en `estilo_graficos.py`).
- **PyTorch:** red neuronal.
- **Google Colab:** entorno en la nube con GPU gratuita, usado solo para la MLP.
- **joblib:** guardar y cargar los modelos entrenados (`.joblib`). Requiere las mismas versiones de librerías.
- **`.env` + `config.py`:** parámetros comunes (semilla 42, folds, cobertura, réplicas) en un solo sitio.
- **Git / GitHub:** control de versiones; el repositorio es el apéndice de la memoria.

---

## 5. Lo que te falta para la defensa

Lo que veo pendiente, por orden de importancia:

1. **Las diapositivas.** Este guion define 14; hay que hacerlas. Casi todas las figuras ya existen en
   `docs/memoria/figuras/` (las `nb*` y las `S2`–`S4` del sponsor). Faltan dos esquemas sencillos: el diseño anti-fuga
   (diapositiva 6) y el embudo de datos (diapositiva 5).
2. **Ensayar cronometrado al menos tres veces**, en voz alta. El texto de cada diapositiva está escrito para leerlo
   a ~130 palabras por minuto; en la práctica se tarda más. Si en el primer ensayo pasas de 16 minutos, recorta las
   diapositivas 7 y 11.
3. **Tener presentes las discrepancias** de la tabla de la sección 3. Si el tribunal ha leído la memoria y la
   bitácora, es justo lo que pregunta.
4. **Diapositivas de reserva** (después del cierre, solo si preguntan): Tabla 1 completa, ablación por bloques
   (apéndice G.1), hiperparámetros elegidos y el óptimo plano (Figura 10), diagnósticos de la OLS (Figura 6), mapa de
   residuos (Figura 7). Enseñar un gráfico al responder da mucha seguridad.
5. **Una demo opcional** (30 s o como reserva): ejecutar `predecir_precio` sobre un anuncio y mostrar precio y rango.
   Si la haces, llévala grabada o en captura por si falla el equipo.
6. **Repasar la normativa de Nueva York** (*Multiple Dwelling Law*, mínimo de 30 noches): la memoria la cita y un
   tribunal curioso puede preguntar. Respuesta: explica que el 64% de anuncios exija 30 noches; no cambia el precio por
   noche pero define un mercado de estancia media.
7. **Conocer el estado de `maximum_nights`**: la bitácora la deja "pendiente de confirmación del alumno". Ten clara la
   respuesta (sección 3) y, si quieres, cierra esa línea en la bitácora.
8. **Practicar con el simulacro del tribunal**: la skill `tfm-juzgado` tiene un modo de defensa que hace preguntas y
   busca contradicciones. Úsala cuando tengas las diapositivas.
