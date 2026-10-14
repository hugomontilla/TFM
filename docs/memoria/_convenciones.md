# Convenciones de la memoria

Lo leen las skills `tfm-memoria-analistas` y `tfm-memoria-sponsor` antes de escribir. Lo que exige el máster está en
[../Enunciado_TFM/indicaciones.md](../Enunciado_TFM/indicaciones.md) (§7 es el checklist); aquí solo va **cómo** lo
escribimos nosotros.

## Ficheros

| Fichero | Contenido |
|---|---|
| `1_analistas.md` | Documentación para analistas (tribunal). Va primero, como exige el máster |
| `2_sponsor.md` | Documentación para el sponsor (ejecutivo) |
| `3_apendices.md` | Apéndices: guía del repositorio y del código para ingenieros, reproducción, tablas largas |
| `figuras/notebooks/` | Figuras extraídas de los notebooks con `python scripts/extraer_figuras.py` (no se editan a mano; ver `_inventario.md`) |
| `figuras/sponsor/` | Figuras rediseñadas para el sponsor, generadas por código a partir de `outputs/` y `models/` |

Las partes 1 y 2 tienen **las mismas siete secciones, en el mismo orden** (Cuadro 1 del enunciado). Los títulos de
sección son idénticos en ambos ficheros para que el lector pueda saltar de uno a otro.

1. Objetivos del proyecto
2. Principales logros obtenidos
3. Aproximación elegida
4. Descripción del modelo
5. Resultados principales soportados por datos
6. Detalles del modelo (en el sponsor: ausente o un párrafo)
7. Recomendaciones

## Imágenes

- Sintaxis: `![Texto alternativo que dice la conclusión](figuras/notebooks/nb3_c094.png)` y debajo una línea en
  cursiva `*Figura N. Título-conclusión. Fuente: notebook 3-MODELADO, §11.*`
- Numeración de figuras y tablas **independiente en cada parte** (Figura 1, 2… en analistas; Figura 1, 2… en sponsor).
- Ninguna figura sin una frase en el texto que diga qué hay que ver en ella.
- No se reentrena nada ni se vuelven a ejecutar los notebooks para conseguir una figura. Si hace falta una figura
  nueva, se dibuja con `scripts/estilo_graficos.py` a partir de lo ya guardado en `outputs/` y `models/`.

## Una sola fuente para cada cifra

Las dos partes cuentan la misma historia con distinto nivel de detalle: **una cifra que aparece en las dos debe ser
idéntica** (mismo redondeo). Cada cifra se toma de su fuente, no de memoria ni de un borrador anterior:

| Cifra | Fuente |
|---|---|
| Métricas en test (RMSE, R², MAE, cobertura) | `outputs/resultados_test_catboost.csv`, `outputs/resultados_test_hgb.csv` |
| Métricas de CV por modelo y fold | `outputs/experimentos.csv` (+ `docs/registro_experimentos.md`) |
| Importancia de variables y bloques | `outputs/importancia_variables.csv`, `outputs/importancia_bloques.csv` |
| Intervalos / cuantiles | `outputs/cuantiles.csv`, `models/ficha_modelo_*.json` |
| Tamaños de muestra, partición | `data/particion/particion.json`, `README.md` |
| Decisiones y su porqué | `docs/bitacora_decisiones.md`, `docs/hallazgos_y_pendientes.md` |

Si una cifra del borrador (`docs/resumen_ejecutivo.md`, `README.md`) no coincide con su fuente, manda la fuente y se
avisa al alumno.

## Estilo común

- Español, frases cortas, tono cordial y no mecánico. Impersonal por defecto ("se decidió", "el modelo realiza");
  primera persona solo para explicar una decisión de diseño o un hallazgo propio.
- Sin bloque `#### Conclusiones:` al final de cada sección (decisión del alumno, 2026-10-09): las ideas clave ya se
  dicen a lo largo de cada apartado y el bloque las repetía.
- El PDF sigue el estilo de [_estilo_pdf.md](_estilo_pdf.md); se genera con `python scripts/generar_pdf_memoria.py`.
- Asociación ≠ causalidad: importancia, SHAP o coeficientes describen cómo predice el modelo, no qué causa el precio.
- Las limitaciones se dicen, no se esconden: los precios altos son el punto débil y aparece en las dos partes.
- Storytelling según [.agents/skills/storytelling/SKILL.md](../../.agents/skills/storytelling/SKILL.md):
  contexto → idea → implicación; títulos de figura que dicen la conclusión, no el contenido.

## Flujo de trabajo

Sección a sección: propuesta de guion y figuras → visto bueno del alumno → redacción → revisión del alumno. En la
parte de analistas, `tfm-juzgado` revisa cada sección terminada antes de darla por cerrada.
