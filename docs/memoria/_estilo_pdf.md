# Estilo del PDF de la memoria

Tomado de la plantilla oficial [../Enunciado_TFM/Plantilla portada TFM.docx](../Enunciado_TFM/Plantilla%20portada%20TFM.docx)
y aplicado a todas las páginas. La implementación está en [estilo/memoria.css](estilo/memoria.css); el PDF se genera con
`python scripts/generar_pdf_memoria.py` y sale en `docs/memoria/pdf/memoria_TFM.pdf`.

## Lo que fija la plantilla

| Elemento | Valor en la plantilla |
|---|---|
| Página | Carta (8,5 × 11 in), márgenes de 1 in |
| Fuente | Arial en todo el documento |
| Texto base | 11 pt, interlineado 1,15 (`line="276"`) |
| "Máster en Data Science y Big Data" | 24 pt, negro, centrado |
| "Trabajo de Fin de Máster" | 30 pt, granate `#7d283b`, centrado |
| Título principal | 36 pt, `#222222`, centrado (aquí 32 pt para que quepa en dos líneas) |
| Subtítulo | 30 pt, `#222222`, centrado (aquí 20 pt) |
| Nombre y apellidos | 18 pt, alineado a la derecha |
| Fecha | 14 pt, alineada a la derecha |
| Marca de agua | Escudo de la US en gris, 468 pt de ancho, centrado en la página |
| Logos | Escudo de la US en color (abajo a la izquierda), logotipo CFP a su derecha, "UNIVERSIDAD DE SEVILLA" bajo el nombre del máster |

Colores de los logos: granate del escudo `#84113c`, dorado del escudo `#fdb628`.

## Cómo se extiende al resto de páginas

| Elemento | Estilo |
|---|---|
| Cabecera | 8 pt; a la izquierda "Máster en Data Science y Big Data · Universidad de Sevilla" en gris, a la derecha el nombre de la parte en granate |
| Pie | Número de página centrado, 9 pt, granate |
| Título de parte (H1) | 26 pt, granate, subrayado dorado de 3 pt; cada parte empieza en página nueva |
| Sección (H2) | 16 pt, granate, línea fina debajo |
| Subsección (H3) | 12,5 pt, negrita, `#222222` |
| Párrafos | 11 pt justificados; viñetas alineadas a la izquierda con marcador granate |
| Tablas | 9 pt; cabecera granate con texto blanco, filas alternas en rosa muy claro |
| Figuras | Centradas, máx. 4,6 in de alto; pie en cursiva, 9 pt, gris |
| Notas (blockquote) | Barra dorada a la izquierda, fondo crema, 10 pt |
| Código | Consolas; en línea en granate, en bloque sobre gris claro con barra granate |

Orden del documento: portada → índice (con número de página) → documentación para analistas → documentación para el
sponsor → apéndices.

## Datos de la portada

Título, subtítulo, autor y fecha están en el diccionario `PORTADA` de `scripts/generar_pdf_memoria.py`.

## Requisitos

- Microsoft Edge (imprime el HTML a PDF en modo headless).
- `markdown-it-py` (ya en el entorno).
- `pymupdf`, opcional, para poner los números de página en el índice: `pip install pymupdf`.
