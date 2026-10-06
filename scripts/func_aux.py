from scipy.stats import kruskal, mannwhitneyu, spearmanr, chi2_contingency
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from IPython.display import display

from config import SEMILLA
from preparacion_datos import haversine_km

from estilo_graficos import (SURFACE, INK, INK_MUTED, GRID, estilo,
                              grafico_reparto_pie, grafico_barras_apiladas)


def diagnostico_normalidad_shapiro(grupos, n_sub=5000, n_repeticiones=20, semilla=SEMILLA):
    """Normalidad por grupo combinando Shapiro sobre submuestras con medidas de tamano de efecto."""
    rng = np.random.default_rng(semilla)
    filas = []
    for nombre, serie in grupos.items():
        x = np.asarray(serie, dtype=float)
        rechazos = sum(
            stats.shapiro(rng.choice(x, size=min(n_sub, len(x)), replace=False)).pvalue < 0.05
            for _ in range(n_repeticiones)
        )
        muestra_qq = rng.choice(x, size=min(n_sub, len(x)), replace=False)
        r_qq = stats.probplot(muestra_qq, dist='norm')[1][2]
        filas.append({
            'grupo': nombre,
            'n': len(x),
            'asimetria': round(float(stats.skew(x)), 3), # en [0.5,1] asimetria moderada
            'curtosis': round(float(stats.kurtosis(x)), 3), # >0 colas mas pesadas y pico mas agudo
            'shapiro_rechaza_pct': round(rechazos / n_repeticiones * 100, 1), # rechazo de normalidad
            'r_filliben': round(float(r_qq), 4), # Cercano a 1 indica parecido al grafico Q-Q
        })
    return pd.DataFrame(filas)


def welch_anova(grupos):
    """ANOVA de Welch: relaja el supuesto de varianzas iguales que ANOVA clasico si exige."""
    n = np.array([len(g) for g in grupos])
    medias = np.array([np.mean(g) for g in grupos])
    varianzas = np.array([np.var(g, ddof=1) for g in grupos])
    pesos = n / varianzas
    peso_total = pesos.sum()
    media_ponderada = (pesos * medias).sum() / peso_total
    k = len(grupos)
    lam = ((1 - pesos / peso_total) ** 2 / (n - 1)).sum()
    numerador = (pesos * (medias - media_ponderada) ** 2).sum() / (k - 1)
    f = numerador / (1 + 2 * (k - 2) / (k ** 2 - 1) * lam)
    gl2 = (k ** 2 - 1) / (3 * lam)
    return f, k - 1, gl2, stats.f.sf(f, k - 1, gl2)


def eta_cuadrado(grupos):
    """Varianza explicada por el factor. Comparable con el epsilon2 del Kruskal-Wallis."""
    todos = np.concatenate(grupos)
    media_global = todos.mean()
    ss_entre = sum(len(g) * (np.mean(g) - media_global) ** 2 for g in grupos)
    return ss_entre / ((todos - media_global) ** 2).sum()


def cramers_v(tabla):
    """Asociacion entre dos categoricas, normalizada entre 0 (independencia) y 1 (asociacion total)."""
    chi2 = chi2_contingency(tabla)[0]
    filas, columnas = tabla.shape
    return np.sqrt((chi2 / tabla.values.sum()) / min(filas - 1, columnas - 1))


def ratio_medianas(datos, col_binaria, col_valor='price'):
    """Mediana de col_valor con col_binaria / sin ella (x1 = sin efecto)."""
    return datos.loc[datos[col_binaria], col_valor].median() / datos.loc[~datos[col_binaria], col_valor].median()


def ratios_por_estrato(datos, col_binaria, estratos, min_por_lado=30, col_valor='price'):
    """ratio_medianas dentro de cada estrato que tenga al menos min_por_lado anuncios con y sin col_binaria."""
    ratios = []
    for _, grupo in datos.groupby(estratos, observed=True):
        n_con = grupo[col_binaria].sum()
        if min(n_con, len(grupo) - n_con) >= min_por_lado:
            ratios.append(ratio_medianas(grupo, col_binaria, col_valor))
    return pd.Series(ratios, dtype=float)


def tabla_lift(datos, cols_binarias, estratos, min_por_lado, min_estratos):
    """Lift marginal y condicionado (mediana de los ratios por estrato) de las columnas am_* con evidencia suficiente."""
    filas = []
    for col in cols_binarias:
        ratios = ratios_por_estrato(datos, col, estratos, min_por_lado)
        if len(ratios) >= min_estratos:
            filas.append({
                'amenity': col.removeprefix('am_'),
                'presencia_pct': round(datos[col].mean() * 100, 2),
                'lift_marginal': round(ratio_medianas(datos, col), 3),
                'lift_condicionado': round(ratios.median(), 3),
                'estratos': len(ratios),
            })
    tabla = pd.DataFrame(filas, columns=['amenity', 'presencia_pct', 'lift_marginal', 'lift_condicionado', 'estratos'])
    return tabla.sort_values('lift_condicionado', ascending=False).reset_index(drop=True)


def contraste_grupos(df_, col_grupo, orden, col_valor='price_log'):
    """Protocolo completo: normalidad, homocedasticidad, ANOVA/Welch/Kruskal y post-hoc de Holm."""
    grupos = {g: df_.loc[df_[col_grupo] == g, col_valor].dropna() for g in orden}
    print(f'Normalidad de {col_valor} dentro de cada {col_grupo}:')
    display(diagnostico_normalidad_shapiro(grupos))
    print('='*80)

    valores = [g.to_numpy() for g in grupos.values()]
    h, p_kw = kruskal(*valores)
    k, n_tot = len(valores), sum(len(v) for v in valores)
    bf = stats.levene(*valores, center='median')
    f_c, p_c = stats.f_oneway(*valores)
    f_w, gl1, gl2, p_w = welch_anova(valores)

    if bf.pvalue > 0.05:
        print(f'\nSe acepta la homocedasticidad mediante Brown-Forsythe: W = {bf.statistic:.1f}, p = {bf.pvalue:.2e}')
    else:
        print(f'\nSe rechaza homocedasticidad mediante Brown-Forsythe: W = {bf.statistic:.1f}, p = {bf.pvalue:.2e}\n')
    print('='*80)

    print('\nMismo contraste con los tres tests:')
    print(f'  -ANOVA clasico    F = {f_c:9.1f}   p = {p_c:.2e}   eta2     = {eta_cuadrado(valores):.4f}')
    print(f'  -ANOVA de Welch   F = {f_w:9.1f}   p = {p_w:.2e}   gl = ({gl1}, {gl2:.0f})')
    print(f'  -Kruskal-Wallis   H = {h:9.1f}   p = {p_kw:.2e}   epsilon2 = {(h - k + 1) / (n_tot - k):.4f}\n')
    print('='*80)

    pares = list(combinations(orden, 2))
    p_raw = [mannwhitneyu(grupos[a], grupos[b], alternative='two-sided').pvalue for a, b in pares]
    _, p_adj, _, _ = multipletests(p_raw, method='holm')
    posthoc = pd.DataFrame({
        'grupo_a': [a for a, b in pares],
        'grupo_b': [b for a, b in pares],
        'mediana_a': [df_.loc[df_[col_grupo] == a, 'price'].median() for a, b in pares],
        'mediana_b': [df_.loc[df_[col_grupo] == b, 'price'].median() for a, b in pares],
        'p_ajustado': p_adj,
        'significativo_5pct': p_adj < 0.05,
    })
    print('\nPost-hoc de Mann-Whitney con correccion de Holm:')
    display(posthoc)
    return posthoc
