"""Extrae las figuras ya renderizadas de los notebooks a PNG, sin volver a ejecutarlos.

La memoria (docs/memoria/) se escribe en Markdown y enlaza estas imagenes. Ademas genera un inventario con la
seccion y el titulo de cada figura, para elegir cuales van a la memoria sin abrir los notebooks uno a uno.

Uso: python scripts/extraer_figuras.py
"""
import base64
import json
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOKS = RAIZ / 'notebooks'
DESTINO = RAIZ / 'docs' / 'memoria' / 'figuras' / 'notebooks'

PATRON_TITULO = re.compile(r"""(?:set_title|title|suptitle)\(\s*[rf]?['"]([^'"]+)['"]""")


def titulos_en_codigo(fuente: str) -> str:
    return ' / '.join(dict.fromkeys(PATRON_TITULO.findall(fuente)))


def extraer(notebook: Path) -> list[dict]:
    celdas = json.loads(notebook.read_text(encoding='utf-8'))['cells']
    prefijo = notebook.stem.split('-')[0]
    seccion, figuras = '', []
    for indice, celda in enumerate(celdas):
        fuente = ''.join(celda['source'])
        if celda['cell_type'] == 'markdown':
            # Nos quedamos con el ultimo encabezado visto: es la seccion a la que pertenece la figura
            encabezados = [l.lstrip('#').strip() for l in fuente.splitlines() if l.startswith('#')]
            encabezados = [e for e in encabezados if not e.startswith('Conclusiones')]
            seccion = encabezados[-1] if encabezados else seccion
            continue
        imagenes = [o['data']['image/png'] for o in celda.get('outputs', []) if 'image/png' in o.get('data', {})]
        for n, imagen in enumerate(imagenes):
            nombre = f'nb{prefijo}_c{indice:03d}' + (f'_{n}' if len(imagenes) > 1 else '') + '.png'
            (DESTINO / nombre).write_bytes(base64.b64decode(imagen))
            figuras.append({'fichero': nombre, 'notebook': notebook.name, 'celda': indice,
                            'seccion': seccion, 'titulo': titulos_en_codigo(fuente)})
    return figuras


def escribir_inventario(figuras: list[dict]) -> None:
    filas = ['# Inventario de figuras de los notebooks', '',
             'Generado por `scripts/extraer_figuras.py`. No editar a mano: se sobrescribe.', '',
             '| Figura | Notebook | Celda | Sección | Título en el código |', '|---|---|---|---|---|']
    filas += [f"| [{f['fichero']}]({f['fichero']}) | {f['notebook']} | {f['celda']} | {f['seccion']} | {f['titulo']} |"
              for f in figuras]
    (DESTINO / '_inventario.md').write_text('\n'.join(filas) + '\n', encoding='utf-8')


if __name__ == '__main__':
    DESTINO.mkdir(parents=True, exist_ok=True)
    figuras = [f for nb in sorted(NOTEBOOKS.glob('*.ipynb')) for f in extraer(nb)]
    escribir_inventario(figuras)
    print(f'{len(figuras)} figuras en {DESTINO.relative_to(RAIZ)}')
