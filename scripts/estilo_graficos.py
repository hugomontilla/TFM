"""Estilo unico de graficos del TFM: tokens de color, mapas categoria -> color y figuras comunes.

Regla de uso: ningun notebook define colores propios. Si una categoria aparece en un grafico,
su color sale de aqui; asi 'Manhattan' o 'Entire place' se reconocen en toda la memoria.
Los tamanos priorizan la legibilidad (fuentes de 11-15 pt, trazos de 3 pt) sobre el ahorro de espacio.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Tokens de superficie y texto ---
SURFACE, INK, INK_MUTED = '#fcfcfb', '#0b0b0b', '#6b6963'
GRID = '#e1e0d9'

# --- Roles semanticos (el color significa algo, no es una categoria) ---
COLOR_FOCO, COLOR_CONTEXTO = '#2a6f97', '#9aa5b1'        # destacar una barra frente al resto
COLOR_BASE, COLOR_SECUNDARIO = COLOR_FOCO, COLOR_CONTEXTO  # serie principal y su comparador
COLOR_ENCARECE, COLOR_ABARATA = '#e34948', '#2a78d6'       # efecto sobre el precio: rojo sube, azul baja
COLOR_MEDIA, COLOR_MEDIANA = '#e34948', INK                # lineas verticales sobre histogramas

# --- Categoricas: un color fijo por categoria ---
COLOR_DISTRICT = {'Manhattan': '#2a78d6', 'Brooklyn': '#eb6834', 'Queens': '#1baf7a',
                  'Bronx': '#eda100', 'Staten Island': '#e87ba4'}
COLOR_ROOM_TYPE = {'Entire place': '#0e7c86', 'Private room': '#c0488f',
                   'Hotel room': '#8f63d2', 'Shared room': '#9a8a3c'}
COLOR_SPLIT = {'train': '#2a6f97', 'val': '#7fb2d3', 'test': '#eb6834'}
# Una variable, un color: cuando dos paneles muestran variables distintas, cada una conserva el suyo
COLOR_VARIABLE = {'price': '#2a78d6', 'price_log': '#eb6834', 'accommodates': '#2a6f97',
                  'bedrooms': '#c0488f', 'bathrooms': '#9a8a3c'}

ORDEN_DISTRICT = list(COLOR_DISTRICT)

# --- Mapas continuos ---
CMAP_ORDINAL = 'Blues'          # categorias ordenadas (capacidad, tramos)

COLORES = {'district': COLOR_DISTRICT, 'room_type': COLOR_ROOM_TYPE, 'split': COLOR_SPLIT,
           'variable': COLOR_VARIABLE}


def paleta(col, categorias=None):
    """Mapa categoria -> color de `col`, acotado y ordenado segun `categorias` si se indican.

    Sirve directamente como `palette=` de seaborn.
    """
    mapa = COLORES[col]
    return {c: mapa[c] for c in (categorias if categorias is not None else mapa)}


def paleta_ordinal(n, cmap=CMAP_ORDINAL):
    """`n` colores de claro a oscuro para categorias ordenadas (capacidad, numero de dormitorios...)."""
    return [tuple(c) for c in plt.get_cmap(cmap)(np.linspace(0.35, 0.95, n))]


def aplicar_estilo():
    """Fija los rcParams globales; llamar una vez al inicio de cada notebook."""
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'savefig.dpi': 200, 'savefig.bbox': 'tight', 'figure.dpi': 100,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.edgecolor': INK_MUTED,
        'axes.labelcolor': INK_MUTED, 'axes.labelsize': 12, 'axes.titlesize': 15,
        'axes.titleweight': 'bold', 'axes.titlelocation': 'left', 'axes.titlecolor': INK,
        'axes.titlepad': 14, 'axes.axisbelow': True,
        'axes.prop_cycle': plt.cycler(color=[COLOR_VARIABLE[v] for v in COLOR_VARIABLE]),
        'xtick.color': INK_MUTED, 'ytick.color': INK_MUTED, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'grid.color': GRID, 'grid.linewidth': 0.9,
        'legend.frameon': False, 'legend.fontsize': 11, 'legend.title_fontsize': 11,
        'lines.linewidth': 3, 'lines.markersize': 8,
    })


def estilo(eje, titulo='', xlabel='', ylabel='', rejilla=None):
    """Lo que comparten todas las figuras; cada celda sigue dibujando lo suyo.

    rejilla: 'y' o 'x' para una rejilla tenue en ese eje; None la deja sin rejilla.
    """
    eje.set_facecolor(SURFACE)
    eje.set_title(titulo, loc='left', fontsize=15, color=INK, fontweight='bold', pad=14)
    eje.set_xlabel(xlabel, fontsize=12, color=INK_MUTED)
    eje.set_ylabel(ylabel, fontsize=12, color=INK_MUTED)
    eje.spines[['top', 'right']].set_visible(False)
    eje.tick_params(colors=INK_MUTED, labelsize=11)
    if rejilla:
        getattr(eje, f'{rejilla}axis').grid(True, color=GRID, linewidth=0.9)
        eje.set_axisbelow(True)
    return eje


def grafico_reparto_pie(df_, col, colores, titulo, min_pct_etiqueta=5.0):
    """Tarta del reparto de anuncios por categoria, en el orden de `colores`.

    Las porciones por debajo de `min_pct_etiqueta` no admiten etiqueta dentro; la leyenda lleva todos los valores.
    """
    orden = list(colores)
    conteo = df_[col].value_counts().loc[orden]
    reparto = conteo / conteo.sum() * 100

    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor=SURFACE, subplot_kw={'aspect': 'equal'})
    ax.set_facecolor(SURFACE)
    cunas, _ = ax.pie(reparto, colors=list(colores.values()), startangle=90, counterclock=False,
                      wedgeprops={'edgecolor': SURFACE, 'linewidth': 2})

    for cuna, pct in zip(cunas, reparto.values):
        if pct < min_pct_etiqueta:
            continue
        ang = np.radians((cuna.theta1 + cuna.theta2) / 2)
        ax.text(0.60 * np.cos(ang), 0.60 * np.sin(ang), f'{pct:.1f}%', ha='center', va='center',
                fontsize=13, fontweight='bold', color=SURFACE)

    ax.legend(cunas, [f'{c} (n = {n:,})' for c, n in conteo.items()], title=col,
              loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=12, title_fontsize=12)
    ax.set_title(titulo, fontsize=15, color=INK, fontweight='bold', loc='center', x=0, pad=18)
    plt.tight_layout()
    plt.show()


def grafico_barras_apiladas(df_, col_x, col_apilada, orden_x, colores, min_pct_etiqueta=5.0):
    """Barras apiladas de `col_apilada` dentro de cada categoria de `col_x`: recuento absoluto y % por barra.

    Devuelve la tabla de recuentos para poder citar las cifras en las conclusiones.
    """
    orden_apilada = list(colores)
    conteo = pd.crosstab(df_[col_x], df_[col_apilada]).reindex(index=orden_x, columns=orden_apilada, fill_value=0)
    pct = conteo.div(conteo.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(17, 6.5), facecolor=SURFACE)
    for ax, tabla, ylabel in [(axes[0], conteo, 'Numero de anuncios'), (axes[1], pct, f'% dentro de cada {col_x}')]:
        ax.set_facecolor(SURFACE)
        base = np.zeros(len(tabla))
        for categoria in orden_apilada:
            valores = tabla[categoria].to_numpy()
            ax.bar(tabla.index, valores, bottom=base, color=colores[categoria], label=categoria,
                   edgecolor=SURFACE, linewidth=1)
            if tabla is pct:
                for x, (v, b) in enumerate(zip(valores, base)):
                    if v >= min_pct_etiqueta:
                        ax.text(x, b + v / 2, f'{v:.0f}%', ha='center', va='center', fontsize=11, color=SURFACE,
                                fontweight='bold')
            base += valores
        ax.set_ylabel(ylabel, color=INK, fontsize=12)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(colors=INK_MUTED, labelsize=11)

    for x, total in enumerate(conteo.sum(axis=1)):
        axes[0].text(x, total, f'{total:,}', ha='center', va='bottom', fontsize=11, color=INK_MUTED)
    axes[1].set_ylim(0, 100)

    axes[0].set_title(f'{col_apilada} por {col_x}: recuento', fontsize=14, color=INK, fontweight='bold', loc='left')
    axes[1].set_title(f'{col_apilada} por {col_x}: composicion (%)', fontsize=14, color=INK, fontweight='bold',
                      loc='left')
    axes[0].legend(title=col_apilada, loc='upper right', frameon=False, fontsize=11)
    plt.tight_layout()
    plt.show()
    return conteo
