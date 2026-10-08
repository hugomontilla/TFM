# Indicaciones para la redacción de la memoria del TFM

> Transcripción fiel de [indicaciones.pdf](indicaciones.pdf) (Máster en Data Science y Big Data, Universidad de Sevilla, versión del 12 de enero de 2026).
> Las secciones 1–6 reproducen el original. La sección **7. Lectura operativa** es un añadido nuestro (no figura en el PDF) para usar este documento como checklist al redactar la memoria.

---

## 1. Observaciones iniciales

El siguiente documento es una breve guía para la elaboración de la memoria del Trabajo Fin de Máster (TFM). Pretende adaptar la normativa general de TFM elaborada por el Centro de Formación Permanente (véase http://www.cfp.us.es/area-de-profesores-us/directrices-trabajo-fin-de-master) a las particularidades de este máster. Concretamente, se centra en **detalles específicos** para el Máster en Data Science y Big Data referentes a la documentación a entregar como memoria del TFM, y recoge respuestas a algunas dudas que han planteado compañeros de otras ediciones. **Se ruega la lectura atenta de este documento.**

## 2. Contexto del TFM dentro del Máster y objetivo de la memoria

En este máster, el TFM representa la elaboración de un proyecto completo de Data Science, y como tal, la memoria debe reflejar tanto las actividades realizadas por el alumno como la documentación final (los *deliverables* o entregables) que se deberían presentar tanto a los responsables de datos de la empresa cliente (que denominaremos *analistas*) como al ejecutivo que planteó el proyecto (que denominaremos *sponsor*).

Desde este punto de vista, la comisión académica desea que la documentación ofrezca información adecuada sobre la actividad que el alumno ha realizado durante la ejecución del proyecto. Aunque parezca secundario desde el punto de vista científico, es importante considerar ambas facetas (sponsor y analista) porque el trabajo debe ser comunicado de manera eficiente al cliente, tanto técnicamente como a alto nivel. El objetivo, pues, es **convencer a la audiencia (analistas y ejecutivos) de los resultados del TFM**. Aunque en este Máster no es el caso, hay que tener en cuenta que la decisión de desplegar el modelo/tecnología en la empresa-cliente depende de esta fase. Por tanto, **los beneficios que aporten los resultados del TFM deben quedar bien reflejados en la documentación**.

Aunque la audiencia de un proyecto de Data Science dentro de la empresa-cliente es variada (sponsor del proyecto, el analista de negocio, el ingeniero de datos y el científico de datos), la comisión académica ha decidido que en la memoria del TFM aparezcan dos partes diferenciadas: la documentación para el **sponsor** (el que ha aportado el dataset, que hace el papel de cliente-ejecutivo) y para el **analista** (papel que será representado por **los profesores del tribunal** que corresponda).

Idealmente, un proyecto real de Data Science debería mantener una documentación actualizada en cada momento. En nuestro caso, debido a su carácter académico, posiblemente la memoria será redactada una vez finalizado prácticamente el trabajo. Aun siendo así, es aconsejable que **la memoria refleje las actividades realizadas por el alumno durante el desarrollo del proyecto** (en la documentación para los analistas).

### 2.1. Características generales de la documentación

Como ya hemos comentado, la documentación **para los TFM de tipo 1 o 2** constará de dos partes bien diferenciadas, junto con apéndices técnicos relacionados con el código y/o especificaciones técnicas (recuérdese que en estos casos el TFM simula la realización de un encargo por un cliente):

- La parte de **presentación al sponsor** de los resultados debe contener descripciones a nivel ejecutivo del modelo (es decir, no con detalles técnicos que no entenderían), características, resultados, etc. Para ello será necesario usar técnicas de visualización de la información y de los datos para mejorar su explicación.
- La parte dedicada a la **presentación a los analistas** debe describir los detalles técnicos (usando gráficos más especializados). Tras haber realizado un proyecto, es crucial que la audiencia técnica entienda las características del análisis realizado, cómo se han elegido los modelos y técnicas, y cómo son los resultados obtenidos (**incluyendo también los modelos seleccionados que no han dado frutos, explicando su no adecuación o inconvenientes**). Si es posible, sería aconsejable añadir comentarios acerca de la **eficiencia de ejecución** (experimental).
- El **código** se aporta como apéndice y debe estar descrito para los ingenieros de la empresa, es decir, puede ser documentación técnica (por ejemplo, sobre cómo implementarlo, si fuera necesario).

**Los TFM de tipo 3, debido a sus características especiales, seguirán la estructura que el tutor indique en cada caso, aunque siempre se puede usar la expuesta para los restantes tipos de TFM.**

#### Cuadro 1: Características de las dos partes principales de la documentación

| Sección | Para el sponsor | Para el analista |
|---|---|---|
| Objetivos del proyecto | Resumen de objetivos | Resumen de objetivos |
| Principales logros obtenidos | Resumen de los logros | Resumen de los logros |
| Aproximación elegida | Descripción a alto nivel. Caracterización del problema como un problema de... | • Descripción a alto nivel. Caracterización del problema como un problema de...<br>• Detalles relevantes sobre las técnicas de modelado y tecnología empleadas |
| Descripción del modelo | Descripción general | Descripción general |
| Resultados principales (soportados por datos) | Descripción (incluyendo gráficas y diagramas) de los principales resultados. Visualmente bien diseñado | Descripción detallada. Gráficos detallados (para analistas) usando librerías estándar R/Python |
| Detalles del modelo | No es necesario, o muy general | • Descripción con detalle del modelo, el código implementado<br>• Descripción de las variables utilizadas y su importancia en el análisis<br>• Tecnologías utilizadas<br>• Análisis de la ejecución: ventajas e inconvenientes<br>• Discusión de la utilidad de las variables y su poder predictivo si es el caso |
| Recomendaciones | Sección centrada en el impacto esperado de la adopción del modelo (según tu opinión). Discusión de los potenciales beneficios y riesgos | Recomendaciones similares pero con detalles técnicos de implementación y su bondad y/o riesgos |

## 3. Estructura general del TFM y su contenido

La estructura general de la memoria debe ser la siguiente:

1. Documentación para analistas
2. Documentación para el ejecutivo
3. Apéndices

La estructura específica de las dos primeras partes es similar aunque, como hemos comentado, su profundidad puede ser distinta para algunas secciones (pues las audiencias lo son) ya que el grado de especificación técnica será diferente.

En el cuadro 1 se resumen las secciones principales de las dos partes (**que deberían aparecer en el mismo orden en cada una**).

Comentamos a continuación con un poco de detalle cada sección:

### 3.1. Objetivos del proyecto

Esta sección es, en general, la misma en ambos documentos. El objetivo es que el autor del TFM comparta con el lector los objetivos que se pretenden conseguir con el proyecto, por lo que es importante que sean descritos con claridad.

Posiblemente coincidan con los objetivos originales que propuso el sponsor, pero también es importante que **si alguno se ha cambiado (después de analizar el dataset o de entrevistarse con el sponsor) quede constancia en la memoria del TFM**.

El resumen de esta sección sería, aproximadamente, como sigue:

- *Situación*: un resumen de la situación inicial del problema a resolver.
- *Complicación*: justificar la necesidad/interés de abordar este problema con técnicas de Data Science (incluido algún ejemplo si se cree conveniente).
- *Objetivos*.

### 3.2. Resumen de los resultados principales obtenidos

Esta sección es importante, tanto para la documentación para analistas como para los sponsors.

En el caso de los sponsors aún más si cabe porque la descripción de resultados es lo que, posiblemente, se van a leer con más detalle. El lenguaje debería ser **claro y conciso**. Un **resumen ejecutivo** será siempre bien recibido por los sponsors que están muy ocupados, y es una oportunidad de evangelizar sobre la bondad de los resultados obtenidos con las metodologías utilizadas.

### 3.3. Aproximación elegida

Aquí debe explicarse la metodología empleada, incluyendo fundamentalmente:

- La(s) fase(s) de extracción, transformación y carga de datos que se hayan realizado.
- Descripción de las entrevistas con expertos, sponsors, y otros agentes de la organización cliente.
- Metodología aplicada a los datos (general para ambos, más detallada para el analista).
- Descripción de la solución.
- Detalles relevantes sobre las técnicas de modelado y tecnología empleadas en el documento para el analista.
- Comentarios adicionales sobre las hipótesis de trabajo que se han adoptado, y por qué razón.
- Restricciones tecnológicas, científicas o legales encontradas, así como sus implicaciones en la realización del proyecto.
- Descripción de los datasets que se han utilizado para enriquecer el aportado por el cliente (y forma de acceso a éste, en su caso).

### 3.4. Descripción del modelo

Como se comenta en la tabla, basta una descripción general. Debe ser **convincente y legible por el sponsor** porque debe trasladar la información a otros integrantes de su organización. En el caso de los analistas ocurre algo similar, lo que ocurre es que debe comunicarlo a una audiencia más experta, por lo que podemos ofrecer más detalles. En la Fig. 1 aparece un pequeño guion, usando un ejemplo, de lo que se debe incluir en esta sección.

#### Figura 1: Descripción esquemática (en formato de transparencia) del modelo

> **Model Description**
>
> - **Overview of Basic Methodology:** predict the likelihood of churn for each customer. Identify customers with a greater probability for churn then compare with actual churn outcomes to train the algorithm and enable predictions for existing customers.
> - **Model:** Logistic regression model
> - **Dependent variable:** Binary variable, of churn/no churn
> - **Scope:**
>   - 500,000 Yoyodyne bank customers, based on churn within a 150 day period after 1/31/2011
>   - 500,000 Customers with all churners through 6/30/11, plus a random sample of 45,000 accounts
>   - All selected customers were Active, Suspended or Pending as of 2011-01-31
>   - Call History detail data extracted from Call Data Record Warehouse for customers from 1/31/11 to 6/30/11
> - **Sampling**
>   - Training sample: 50,000 subscribers
>   - Testing sample: 100,000 subscribers
> - **The model developed has predictive power at least as good as the bank's current churn model**
>   - We created a baseline model without social networking variables and the bank's marketing analytics team verified that the predictive power was at least as good as the current model
>   - Social networking variables were added to the model and that further increased its predictive power

### 3.5. Principales resultados y conclusiones soportados por datos

Aquí se deben identificar los resultados clave, basados en datos y en los resultados obtenidos con el modelo diseñado. En ambos casos es importante trasladarlos con claridad, por lo que **el aspecto visual e informativo es fundamental**.

En el documento para el sponsor es importante el aspecto de diseño y el uso de técnicas de visualización claras y no difícilmente interpretables, mientras que en el de analistas se pueden usar librerías de gráficos de R/Python y acompañarlos con resultados matemáticos complejos. Podríamos decir que lo estudiado en la parte de **visualización de datos** correspondería al documento para los analistas mientras lo estudiado en **visualización de la información** (con software de diseño no específico de Data Science) correspondería al del sponsor.

### 3.6. Detalles del modelo

En esta sección (**que solo aparecerá en la documentación para el analista**) se incluyen detalles técnicos del modelo que se consideren relevantes y que se describieron en el cuadro Tabla 1.

### 3.7. Recomendaciones finales: uso futuro, potenciales aplicaciones, etc.

En esta sección se expondrán un conjunto de recomendaciones —**incluido cómo desplegar el modelo en la organización**— que se concluyen del proyecto. A título de ejemplo, las recomendaciones pueden ser algo como:

- *Concluimos que si el tiempo empleado por el cliente para decidirse es mayor que X, entonces se debería ofrecer un descuento (porque con la tarifa Y la probabilidad de compras del servicio es mayor)* (recomendación para sponsor).
- *Para mejorar la eficiencia del modelo, sería aconsejable que el sistema de la organización realizara imputación de datos de acuerdo a una distribución normal en la variable edad o descartara los individuos sin valor en esa variable* (recomendación para el analista).

Para el sponsor hay que centrarse en recomendaciones que tengan **impacto en el negocio**, y así le permita difundir la bondad en el resto de la organización. Para el analista se deben añadir además recomendaciones acerca de **cómo la adopción del modelo cambiará sus sistemas** (ampliando o sustituyendo ciertos módulos y/o actividades).

### 3.8. Apéndice: código

Explicación del código si se aporta, código documentado.

## 4. Formato del TFM

El TFM se presentará en formato digital. Se informará más adelante del procedimiento para la entrega.

## 5. Fechas de entrega y defensa

Se establecen como **fechas límite de entrega y de defensa** de los TFM las siguientes:

- **Aprobación del tutor para la entrega**: antes del viernes 16 de octubre de 2026.
- **Entrega**: viernes 23 de octubre de 2026.
- **Defensa**: en la primera quincena de noviembre de 2026, a determinar por el tribunal.

Si termina antes el TFM y desea defenderlo antes de esas fechas contacte con el coordinador para ajustar los plazos.

## 6. Importante: asignación de tutores

### 6.1. El papel del tutor

El tutor asignado velará por el cumplimiento de los objetivos científicos y la correcta redacción y presentación de la memoria (por ejemplo, comprobando que respeta las indicaciones sobre su estructura ofrecidas en esta guía). Es decir, se encargará de la supervisión de la memoria que realiza el alumno, mientras que **para cualquier duda acerca de una técnica/teoría/herramienta concreta que pretenda aplicar el alumno se deberá dirigir al profesor que impartió el correspondiente módulo en clase**. Todos los profesores del claustro tienen horas de tutorías para el TFM.

**Su visto bueno es condición imprescindible para depositar la memoria del TFM.**

### 6.2. Asignación de tutores

La asignación de tutores para los distintos alumnos se publicará en la carpeta de TFM de la Enseñanza Virtual.

---

## 7. Lectura operativa (añadido nuestro, no forma parte del PDF)

Checklist derivado del texto anterior, para usarlo como arnés al redactar. Cada punto remite a la sección que lo exige.

### Estructura obligatoria de la memoria (§3)

1. **Documentación para analistas** (tribunal; técnica).
2. **Documentación para el sponsor / ejecutivo** (no técnica).
3. **Apéndices** (código documentado; el repositorio y los notebooks).

Las partes 1 y 2 llevan **las mismas secciones y en el mismo orden** (Cuadro 1):

| # | Sección | Sponsor | Analista |
|---|---|---|---|
| 1 | Objetivos del proyecto | Situación → Complicación → Objetivos; constancia de objetivos cambiados | Igual |
| 2 | Principales logros / resumen de resultados | Resumen ejecutivo, claro y conciso; "evangelizar" | Resumen de logros |
| 3 | Aproximación elegida | Alto nivel: "es un problema de…" | + ETL, metodología, técnicas, tecnología, hipótesis de trabajo y por qué, restricciones, datasets externos |
| 4 | Descripción del modelo | General, convincente, que el sponsor pueda reenviar (guion Fig. 1) | General con más detalle (guion Fig. 1) |
| 5 | Resultados principales soportados por datos | Visualización de la **información**, bien diseñada y fácil de leer | Visualización de **datos** con librerías Python, resultados matemáticos |
| 6 | Detalles del modelo | No aparece o muy general | Modelo y código, variables e importancia, tecnologías, análisis de ejecución (ventajas/inconvenientes), utilidad y poder predictivo de las variables |
| 7 | Recomendaciones | Impacto en negocio, beneficios y riesgos (opinión propia) | + implementación técnica, despliegue, cómo cambia los sistemas, bondad/riesgos |

### Requisitos fáciles de olvidar

- **Analistas:** incluir los **modelos que no funcionaron** y por qué (§2.1).
- **Analistas:** comentarios de **eficiencia de ejecución** (tiempos) si es posible (§2.1, Cuadro 1).
- **Analistas:** reflejar las **actividades realizadas durante el proyecto**, no solo el resultado final (§2) → bitácora, registro de experimentos.
- **Ambos:** dejar constancia de los **objetivos que cambiaron** y por qué (§3.1) → en nuestro caso, la restricción a Nueva York.
- **Ambos:** **hipótesis de trabajo** y **restricciones** (tecnológicas, científicas, legales) (§3.3).
- **Ambos:** recomendaciones que incluyan **cómo desplegar el modelo** (§3.7).
- **Sponsor:** nada técnico que no entendería; los beneficios deben quedar claros (§2, §2.1).
- **Apéndice de código:** documentado para ingenieros, explicando cómo implementarlo (§2.1, §3.8).

### Fechas (§5)

- Visto bueno del tutor: **antes del viernes 16/10/2026** (requisito para depositar).
- Entrega: **viernes 23/10/2026**.
- Defensa: primera quincena de noviembre de 2026.
