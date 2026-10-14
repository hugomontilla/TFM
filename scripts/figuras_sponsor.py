"""Figuras rediseñadas para la documentación del sponsor (docs/memoria/figuras/sponsor/).

Uso, desde la raíz del repositorio:
    python scripts/figuras_sponsor.py

No reentrena nada: lee lo guardado en outputs/ y, para la cobertura por quintil, las cifras de la salida de
4-MODELO_FINAL §5 (no se guardaron en un CSV). Gráficos de barras horizontales con una barra destacada y el resto en
gris, cada barra con su valor escrito, para un lector sin formación técnica.

Usan el estilo del PDF de la memoria (docs/memoria/_estilo_pdf.md: Arial, granate de la portada, fondo blanco) en
lugar del azul de los notebooks, para que las figuras del sponsor se lean como parte del mismo documento.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from estilo_graficos import aplicar_estilo

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
DIR_SALIDA = RAIZ_PROYECTO / 'docs' / 'memoria' / 'figuras' / 'sponsor'

MODELO_FINAL = 'CatBoost (ajustado)'
NOMBRES_BLOQUE = {
    'B0_producto': 'El alojamiento\n(tipo, tamaño)',
    'B1_ubicacion': 'La ubicación',
    'B2b_anfitrion': 'El anfitrión',
    'B3_amenities_comunes': 'El equipamiento',
    'B2a_condiciones': 'Las condiciones\nde reserva',
}
EXPERIMENTO_REGLA = 'N0.3_mediana_room_type_capacidad'
EXPERIMENTO_MODELO = 'CatBoost_ajustado'
# Cobertura del rango CQR de CatBoost en el test por quintil del precio real (4-MODELO_FINAL, §5)
COBERTURA_QUINTIL = {
    'El 20% más barato': 0.771,
    'Segundo 20%': 0.840,
    'Tercer 20%': 0.852,
    'Cuarto 20%': 0.875,
    'El 20% más caro': 0.559,
}
COBERTURA_OBJETIVO = 0.80

# Estilo del PDF (docs/memoria/estilo/memoria.css)
COLOR_FOCO = '#7d283b'       # granate de "Trabajo de Fin de Máster"
COLOR_CONTEXTO = '#c4b6b9'   # gris cálido para las barras que no se destacan
TEXTO = '#222222'
TEXTO_SUAVE = '#5f5f5f'
FONDO = '#ffffff'


def _numero_es(valor, decimales=0):
    """Número con coma decimal, como en el texto de la memoria."""
    return f'{valor:.{decimales}f}'.replace('.', ',')


def _ejes(ax, titulo, xlabel):
    """Título a la izquierda, sin ejes ni rejilla: los valores van escritos en cada barra."""
    ax.set_facecolor(FONDO)
    ax.set_title(titulo, loc='left', fontsize=15, color=TEXTO, pad=14)
    ax.set_xlabel(xlabel, fontsize=12, color=TEXTO_SUAVE)
    ax.spines[:].set_visible(False)
    ax.set_xticks([])
    ax.tick_params(axis='y', length=0)


def _barras(ax, etiquetas, valores, destacada, formato):
    """Barras horizontales, la `destacada` en color y el resto en gris, con su valor al final de cada barra."""
    colores = [COLOR_FOCO if e == destacada else COLOR_CONTEXTO for e in etiquetas]
    posiciones = range(len(etiquetas))[::-1]
    ax.barh(list(posiciones), valores, color=colores, height=0.62)
    ax.set_yticks(list(posiciones), etiquetas, fontsize=12, color=TEXTO)
    margen = max(valores) * 0.015
    for y, valor, etiqueta in zip(posiciones, valores, etiquetas):
        peso = 'bold' if etiqueta == destacada else 'normal'
        ax.text(valor + margen, y, formato(valor), va='center', fontsize=12, fontweight=peso, zorder=3,
                color=COLOR_FOCO if etiqueta == destacada else TEXTO,
                bbox={'facecolor': FONDO, 'edgecolor': 'none', 'pad': 1.5})


def figura_peso_informacion():
    """S2: cuánto empeora el error del modelo final si se le oculta cada tipo de información."""
    datos = pd.read_csv(RAIZ_PROYECTO / 'outputs' / 'importancia_bloques.csv')
    datos = datos[datos['modelo'] == MODELO_FINAL]
    aumento_pct = (datos['aumento_rmse'] / datos['rmse_base'] * 100).groupby(datos['grupo']).mean()
    aumento_pct = aumento_pct.reindex(list(NOMBRES_BLOQUE)).rename(NOMBRES_BLOQUE)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    _ejes(ax, 'El tipo y el tamaño del alojamiento pesan más que todo lo demás junto',
          'Cuánto aumenta el error si la herramienta no conoce esa información')
    _barras(ax, list(aumento_pct.index), aumento_pct.to_numpy(), aumento_pct.index[0],
            lambda v: f'+{_numero_es(v)}%')
    ax.set_xlim(0, aumento_pct.max() * 1.15)
    return fig


def figura_error_tipico():
    """S3: error típico (mediana del error en $) de la regla de anuncios parecidos frente a la herramienta."""
    experimentos = pd.read_csv(RAIZ_PROYECTO / 'outputs' / 'experimentos.csv')
    experimentos = experimentos[experimentos['repeticion'] == 0]
    mediana = experimentos.groupby('experimento')['medae_usd_val'].mean()
    etiquetas = ['Anuncios parecidos\n(mismo tipo y tamaño)', 'La herramienta']
    valores = [mediana[EXPERIMENTO_REGLA], mediana[EXPERIMENTO_MODELO]]

    fig, ax = plt.subplots(figsize=(10, 2.8))
    _ejes(ax, f'La herramienta reduce el error típico de {_numero_es(valores[0])}$ '
              f'a {_numero_es(valores[1], 1)}$ por noche',
          'Desviación típica del precio sugerido frente al real ($ por noche)')
    _barras(ax, etiquetas, valores, etiquetas[1], lambda v: f'{_numero_es(v, 1)}$')
    ax.set_xlim(0, max(valores) * 1.15)
    return fig


def figura_cobertura_quintil():
    """S4: cuántas veces el precio real cae dentro del rango, por quinto de precio, con el objetivo del 80%."""
    etiquetas = list(COBERTURA_QUINTIL)
    valores = [v * 100 for v in COBERTURA_QUINTIL.values()]

    fig, ax = plt.subplots(figsize=(10, 4.4))
    _ejes(ax, 'El rango acierta en la mayoría de anuncios, pero no en los de lujo',
          'Anuncios cuyo precio real cae dentro del rango propuesto')
    _barras(ax, etiquetas, valores, etiquetas[-1], lambda v: f'{_numero_es(v)}%')
    objetivo = COBERTURA_OBJETIVO * 100
    ax.axvline(objetivo, color=TEXTO, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.text(objetivo, len(etiquetas) - 0.45, f'Objetivo: {_numero_es(objetivo)}%', ha='center', va='bottom',
            fontsize=11, color=TEXTO, zorder=3, bbox={'facecolor': FONDO, 'edgecolor': 'none', 'pad': 1.5})
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(etiquetas) - 0.1)
    return fig


def main():
    aplicar_estilo()
    plt.rcParams.update({
        'text.parse_math': False,  # los "$" de los importes son texto, no fórmulas
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'figure.facecolor': FONDO, 'savefig.facecolor': FONDO,
    })
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    figuras = {
        'S2_peso_informacion.png': figura_peso_informacion,
        'S3_error_tipico.png': figura_error_tipico,
        'S4_cobertura_quintil.png': figura_cobertura_quintil,
    }
    for nombre, dibujar in figuras.items():
        fig = dibujar()
        fig.savefig(DIR_SALIDA / nombre, facecolor=FONDO)
        plt.close(fig)
        print(f'Guardada {DIR_SALIDA.relative_to(RAIZ_PROYECTO) / nombre}')


if __name__ == '__main__':
    main()
