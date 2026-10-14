"""Genera el PDF de la memoria a partir de docs/memoria/*.md con el estilo de la plantilla de portada del máster.

Uso, desde la raíz del repositorio:
    python scripts/generar_pdf_memoria.py

Une portada, índice, documentación para analistas, para el sponsor y apéndices en un HTML con
docs/memoria/estilo/memoria.css y lo imprime a PDF con Microsoft Edge en modo headless. Si PyMuPDF está instalado,
hace una segunda pasada para poner los números de página en el índice. El estilo está descrito en
docs/memoria/_estilo_pdf.md.
"""
import html
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path

from markdown_it import MarkdownIt

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
DIR_MEMORIA = RAIZ_PROYECTO / 'docs' / 'memoria'
DIR_ESTILO = DIR_MEMORIA / 'estilo'
PDF_SALIDA = DIR_MEMORIA / 'pdf' / 'memoria_TFM.pdf'

PORTADA = {
    'titulo': 'Predicción del precio de Airbnb en Nueva York',
    'subtitulo': 'Un precio de partida para cada anuncio nuevo',
    'autor': 'Hugo Montilla',
    'fecha': 'Octubre de 2026',
}
PARTES = [
    ('analistas', '1_analistas.md'),
    ('sponsor', '2_sponsor.md'),
    ('apendices', '3_apendices.md'),
]
RUTAS_EDGE = [
    Path(r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'),
    Path(r'C:\Program Files\Microsoft\Edge\Application\msedge.exe'),
]


def _id_seguro(texto):
    """Identificador HTML a partir del texto de un título."""
    sin_tildes = unicodedata.normalize('NFKD', texto).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z0-9]+', '-', sin_tildes.lower()).strip('-')


def _parte_a_html(clave, texto_md, entradas_indice):
    """HTML de una parte; añade un id a cada título y registra los de nivel 1 y 2 en `entradas_indice`."""
    md = MarkdownIt('commonmark', {'html': True}).enable('table')
    tokens = md.parse(texto_md)
    for i, token in enumerate(tokens):
        if token.type != 'heading_open':
            continue
        texto = tokens[i + 1].content
        ancla = f'{clave}-{_id_seguro(texto)}'
        token.attrSet('id', ancla)
        nivel = int(token.tag[1])
        if nivel <= 2:
            entradas_indice.append((nivel, re.sub(r'[`*]', '', texto), ancla))
    cuerpo = md.renderer.render(tokens, md.options, {})
    return f'<section class="parte {clave}">\n{_agrupar_figuras(cuerpo)}</section>\n'


def _agrupar_figuras(cuerpo):
    """Une cada imagen con su pie en un <figure>, quita cabeceras de tabla vacías y marca los títulos de tabla."""
    patron_figura = re.compile(r'<p><img ([^>]*)></p>\s*<p><em>(Figura .*?)</em></p>', re.S)
    cuerpo = patron_figura.sub(r'<figure><img \1><figcaption>\2</figcaption></figure>', cuerpo)
    cuerpo = re.sub(r'<thead>\s*<tr>(?:\s*<th>\s*</th>)+\s*</tr>\s*</thead>', '', cuerpo)
    return re.sub(r'<p><em>(Tabla \d+\..*?)</em></p>', r'<p class="tabla-titulo">\1</p>', cuerpo, flags=re.S)


def _portada_html():
    datos = {k: html.escape(v) for k, v in PORTADA.items()}
    return f"""<section class="portada">
  <img class="marca-agua" src="estilo/marca_agua_us.png" alt="">
  <div class="master">Máster en Data Science y Big Data</div>
  <img class="us-texto" src="estilo/us_texto.png" alt="Universidad de Sevilla">
  <div class="bloque-titulo">
    <p class="tfm">Trabajo de Fin de Máster</p>
    <p class="titulo">{datos['titulo']}</p>
    <p class="subtitulo">{datos['subtitulo']}</p>
  </div>
  <div class="autor">{datos['autor']}</div>
  <div class="fecha">{datos['fecha']}</div>
  <img class="logo-us" src="estilo/logo_us.png" alt="">
  <img class="logo-cfp" src="estilo/logo_cfp.png" alt="Centro de Formación Permanente">
</section>
"""


def _indice_html(entradas, paginas):
    filas = []
    for (nivel, texto, ancla), pagina in zip(entradas, paginas):
        filas.append(f'<li class="n{nivel}"><a href="#{ancla}">{html.escape(texto)}</a>'
                     f'<span class="relleno"></span><span class="pag">{pagina}</span></li>')
    return '<section class="indice"><h1>Índice</h1><ol>\n' + '\n'.join(filas) + '\n</ol></section>\n'


def _documento_html(entradas, paginas, cuerpo_partes):
    css = (DIR_ESTILO / 'memoria.css').read_text(encoding='utf-8')
    base = DIR_MEMORIA.as_uri() + '/'
    return (f'<!doctype html><html lang="es"><head><meta charset="utf-8"><base href="{base}">'
            f'<title>Memoria TFM</title><style>{css}</style></head><body>\n'
            f'{_portada_html()}{_indice_html(entradas, paginas)}{cuerpo_partes}</body></html>')


def _ruta_edge():
    for ruta in RUTAS_EDGE:
        if ruta.exists():
            return ruta
    sys.exit('No se encuentra Microsoft Edge; ajusta RUTAS_EDGE en este script.')


def _imprimir_pdf(ruta_html, ruta_pdf, dir_perfil):
    orden = [str(_ruta_edge()), '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
             f'--user-data-dir={dir_perfil}', f'--print-to-pdf={ruta_pdf}', ruta_html.as_uri()]
    subprocess.run(orden, check=True, capture_output=True, timeout=180)
    if not ruta_pdf.exists():
        sys.exit('Edge no generó el PDF.')


def _paginas_del_indice(ruta_pdf, n_entradas):
    """Página de destino de cada enlace del índice, leída del PDF de la primera pasada (None sin PyMuPDF)."""
    try:
        import pymupdf
    except ImportError:
        return None
    destinos = []
    with pymupdf.open(ruta_pdf) as pdf:
        for pagina in pdf:
            enlaces = [e for e in pagina.get_links()
                       if e.get('kind') in (pymupdf.LINK_GOTO, pymupdf.LINK_NAMED) and e.get('page', -1) >= 0]
            enlaces.sort(key=lambda e: (e['from'].y0, e['from'].x0))
            destinos.extend(e['page'] + 1 for e in enlaces)
            if len(destinos) >= n_entradas:
                break
    return destinos[:n_entradas] if len(destinos) >= n_entradas else None


def main():
    entradas = []
    cuerpo = ''.join(_parte_a_html(clave, (DIR_MEMORIA / fichero).read_text(encoding='utf-8'), entradas)
                     for clave, fichero in PARTES)
    PDF_SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ruta_html = tmp / 'memoria.html'
        ruta_html.write_text(_documento_html(entradas, ['00'] * len(entradas), cuerpo), encoding='utf-8')
        _imprimir_pdf(ruta_html, tmp / 'pasada1.pdf', tmp / 'perfil')
        paginas = _paginas_del_indice(tmp / 'pasada1.pdf', len(entradas))
        if paginas is None:
            print('Aviso: sin PyMuPDF el índice sale sin números de página (pip install pymupdf).')
            paginas = [''] * len(entradas)
        ruta_html.write_text(_documento_html(entradas, paginas, cuerpo), encoding='utf-8')
        _imprimir_pdf(ruta_html, tmp / 'final.pdf', tmp / 'perfil')
        shutil.copy(tmp / 'final.pdf', PDF_SALIDA)
    print(f'PDF generado: {PDF_SALIDA.relative_to(RAIZ_PROYECTO)}')


if __name__ == '__main__':
    main()
