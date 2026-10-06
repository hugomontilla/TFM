"""Funciones del modelo final (4-MODELO_FINAL.ipynb): hiperparametros desde el registro, intervalo de confianza por
anfitrion, calibracion CQR del intervalo de precio, error por segmento, prediccion de un anuncio en bruto y el
recorrido completo de un modelo finalista (ModeloFinal).

Convenciones: y, pred, cuantiles estan en log(price); los precios en dolares se obtienen con exp.
"""
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.pipeline import Pipeline, make_pipeline

from particion_datos import md5_fichero
from registro import valor_unico
from config import COBERTURA_NOMINAL, N_REPLICAS_BOOTSTRAP
from func_modelado import (BLOQUES_ESCENARIO, RegresorCatBoost, SEMILLA, TARGET, cobertura_intervalo, columnas_de,
                           importancia_permutacion, leer_experimento, metricas, preprocesador_arboles)

# Cuantiles del intervalo centrado en la cobertura nominal (0,8 -> 0,1 y 0,9)
CUANTIL_BAJO = round((1 - COBERTURA_NOMINAL) / 2, 4)
CUANTIL_ALTO = round(1 - CUANTIL_BAJO, 4)
DESCRIPCION_CATBOOST = ('RegresorCatBoost (RMSE sobre log(price)) + dos cuantiles CatBoost (80% del train) calibrados '
                        'con CQR en el fold 0 de calibracion')
DESCRIPCION_HGB = ('Pipeline(TargetEncoder del barrio + HistGradientBoostingRegressor, error cuadratico sobre log(price)) '
                   '+ dos HistGradientBoostingRegressor de perdida cuantil (80% del train) calibrados con CQR en el '
                   'fold 0 de calibracion')


def descripcion_registrada(nombre: str, ruta_registro: Path) -> str:
    """Descripcion (repr del modelo) que guarda el registro de experimentos para un experimento."""
    return valor_unico(nombre, ruta_registro, 'modelo')


def hiperparametros_registrados(nombre: str, ruta_registro: Path) -> dict:
    """Hiperparametros de un RegresorCatBoost leidos de la descripcion que guarda el registro de experimentos."""
    modelo = eval(descripcion_registrada(nombre, ruta_registro),
                  {'RegresorCatBoost': RegresorCatBoost, 'np': np})  # fichero propio, no externo
    return {k: v for k, v in modelo.get_params().items() if k != 'loss_function'}


def columnas_registradas(nombre: str, ruta_registro: Path) -> list[str]:
    """Variables que uso un experimento del registro, en el orden con que se entreno."""
    return valor_unico(nombre, ruta_registro, 'variables').split(';')


def cuantil_catboost(alfa: float, hiperparametros: dict) -> RegresorCatBoost:
    """CatBoost con perdida de cuantil `alfa` y los mismos hiperparametros que el modelo elegido."""
    return RegresorCatBoost(loss_function=f'Quantile:alpha={alfa}', **hiperparametros)


def hiperparametros_hgb_registrados(nombre: str, ruta_registro: Path) -> dict:
    """Hiperparametros del HistGradientBoostingRegressor de un pipeline leidos del registro de experimentos.

    El repr del pipeline abrevia la lista de variables, pero el del HGB va completo: se evalua solo ese trozo.
    """
    descripcion = descripcion_registrada(nombre, ruta_registro)
    inicio = descripcion.index('HistGradientBoostingRegressor(')
    fin = descripcion.index(')', descripcion.index('random_state', inicio)) + 1
    modelo = eval(descripcion[inicio:fin], {'HistGradientBoostingRegressor': HistGradientBoostingRegressor,
                                           'np': np})  # fichero propio, no externo
    return {k: v for k, v in modelo.get_params().items() if k not in ('loss', 'quantile')}


def construir_hgb_final(columnas: list[str], hiperparametros: dict, cuantil: float | None = None) -> Pipeline:
    """Pipeline del HGB elegido (mismo preprocesador que la CV); con cuantil, pierde pinball de ese cuantil."""
    perdida = {} if cuantil is None else {'loss': 'quantile', 'quantile': cuantil}
    hgb = HistGradientBoostingRegressor(**{**hiperparametros, 'early_stopping': False, 'random_state': SEMILLA,
                                           **perdida})
    return make_pipeline(preprocesador_arboles(columnas), hgb)


def calibrar_cqr(pred_bajo: np.ndarray, pred_alto: np.ndarray, y: np.ndarray,
                 cobertura: float = COBERTURA_NOMINAL) -> float:
    """Correccion de la regresion cuantilica conformal (Romano et al., 2019), en unidades de log(price).

    Se suma al cuantil alto y se resta al bajo. Puntuacion de cada anuncio: cuanto se sale del intervalo
    (negativa si cae dentro). Las predicciones deben ser fuera de fold: no pueden venir del modelo que vio y.
    """
    puntuacion = np.maximum(pred_bajo - y, y - pred_alto)
    n = len(y)
    nivel = min(1.0, np.ceil((n + 1) * cobertura) / n)
    return float(np.quantile(puntuacion, nivel, method='higher'))


def bootstrap_por_anfitrion(y: np.ndarray, pred: np.ndarray, anfitriones: np.ndarray,
                            metricas_fn: Callable[[np.ndarray, np.ndarray], dict[str, float]],
                            n: int = N_REPLICAS_BOOTSTRAP, semilla: int = SEMILLA) -> pd.DataFrame:
    """Replicas bootstrap de varias metricas a la vez, remuestreando anfitriones (no anuncios) con todos sus anuncios.

    Los anuncios de un anfitrion no son independientes, asi que remuestrear filas daria intervalos demasiado
    estrechos. metricas_fn(y, pred) devuelve {nombre: valor}; se evalua una vez por replica. Devuelve una fila por
    replica; el intervalo al 95% de cada columna son sus percentiles 2,5 y 97,5 (resumen_bootstrap).
    """
    codigos, anfitriones_unicos = pd.factorize(anfitriones)
    filas_por_anfitrion = [np.flatnonzero(codigos == i) for i in range(len(anfitriones_unicos))]
    azar = np.random.default_rng(semilla)
    replicas = []
    for _ in range(n):
        elegidos = azar.integers(0, len(filas_por_anfitrion), len(filas_por_anfitrion))
        filas = np.concatenate([filas_por_anfitrion[j] for j in elegidos])
        replicas.append(metricas_fn(y[filas], pred[filas]))
    return pd.DataFrame(replicas)


def resumen_bootstrap(replicas: np.ndarray, valor: float) -> dict[str, float]:
    """Valor observado con su intervalo de confianza al 95% (percentiles 2,5 y 97,5 de las replicas)."""
    bajo, alto = np.percentile(replicas, [2.5, 97.5])
    return {'valor': valor, 'ic95_bajo': bajo, 'ic95_alto': alto}


def error_por_segmento(datos: pd.DataFrame, pred_log: np.ndarray, columnas: list[str],
                       min_n: int = 30) -> pd.DataFrame:
    """MAE (USD), mediana del error (USD), MAPE y n por segmento; las celdas con n < min_n quedan en NaN."""
    tabla = datos[columnas].copy()
    precio, precio_pred = np.exp(datos[TARGET].to_numpy()), np.exp(pred_log)
    tabla['error_abs'] = np.abs(precio - precio_pred)
    tabla['error_rel'] = tabla['error_abs'] / precio
    resumen = tabla.groupby(columnas, observed=True).agg(
        n=('error_abs', 'size'), mae_usd=('error_abs', 'mean'), mediana_usd=('error_abs', 'median'),
        mape=('error_rel', 'mean'))
    resumen.loc[resumen['n'] < min_n, ['mae_usd', 'mediana_usd', 'mape']] = np.nan
    return resumen


def predecir_precio(anuncios_brutos: pd.DataFrame, preparador, modelo: BaseEstimator,
                    cuantiles: dict[float, BaseEstimator], correccion: float,
                    columnas: list[str]) -> pd.DataFrame:
    """De filas de listings.csv a precio mediano e intervalo (USD por noche).

    ATENCION: n_anuncios_ny y reputacion_cartera se calculan sobre las filas que se pasan (decision del
    09-26): para un anfitrion con mas anuncios hay que pasar su cartera completa en NY, no solo el anuncio.
    """
    # Un anuncio nuevo no tiene precio: solo se exige capacidad, no limpiar_registros
    validos = anuncios_brutos[anuncios_brutos['accommodates'] > 0].reset_index(drop=True)
    preparados = preparador.transform(validos)
    X = preparados[columnas]
    bajo = cuantiles[CUANTIL_BAJO].predict(X) - correccion
    alto = cuantiles[CUANTIL_ALTO].predict(X) + correccion
    return pd.DataFrame({
        'listing_id': preparados['listing_id'].to_numpy(),
        'precio_mediano': np.exp(modelo.predict(X)),
        'precio_bajo': np.exp(bajo), 'precio_alto': np.exp(alto)})


def ficha_modelo(hiperparametros: dict, columnas: list[str], metricas_cv: dict, metricas_test: dict,
                 correccion_cqr: float, md5_train: str, md5_test: str, descripcion: str = DESCRIPCION_CATBOOST,
                 librerias: tuple[str, ...] = ('catboost', 'scikit-learn', 'pandas')) -> dict:
    """Ficha del modelo final: hiperparametros, variables, metricas de CV y test, MD5 de la particion y versiones de las librerias."""
    from datetime import date
    from importlib.metadata import version
    return {
        'modelo': descripcion,
        'hiperparametros': {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                            for k, v in hiperparametros.items()},
        'variables': columnas, 'escenario': 'B',
        'correccion_cqr_log': correccion_cqr, 'cobertura_nominal': COBERTURA_NOMINAL,
        'metricas_cv': metricas_cv, 'metricas_test': metricas_test,
        'md5_train_particion': md5_train, 'md5_test_particion': md5_test,
        'versiones': {'python': platform.python_version(), **{libreria: version(libreria) for libreria in librerias}},
        'fecha': date.today().isoformat(),
        'nota': 'exp(prediccion) estima la MEDIANA del precio, no la media.'}


def guardar_ficha(ficha: dict, ruta: Path) -> None:
    """Escribe la ficha del modelo en un JSON legible."""
    Path(ruta).write_text(json.dumps(ficha, indent=2, ensure_ascii=False), encoding='utf-8')


# --- Un modelo finalista de principio a fin -----------------------------------------------------------------------
# El notebook final trata igual a los dos finalistas (CatBoost y HGB): cada paso es una funcion que recibe un
# ModeloFinal y guarda en el lo que calcula, para poder compararlos al final.

METRICAS_PRINCIPALES = ['rmse_log_val', 'mae_log_val', 'r2_log_val', 'mae_usd_val', 'medae_usd_val', 'mape_val']


@dataclass
class ModeloFinal:
    """Un modelo finalista y todo lo que se calcula sobre el a lo largo del notebook final.

    nombre: etiqueta ('CatBoost', 'HGB'). sufijo: se anade a los ficheros guardados ('_catboost' para CatBoost, '_hgb' para HGB).
    construir(columnas) y construir_cuantil(columnas, alfa) devuelven el estimador y el cuantil sin ajustar.
    Los campos de abajo (modelo, cuantiles, correccion...) los van rellenando las funciones de esta seccion.
    """
    nombre: str
    experimento: str
    hiperparametros: dict
    columnas: list
    metricas_cv: dict
    metricas_cv_sd: dict
    construir: Callable
    construir_cuantil: Callable
    descripcion: str
    librerias: tuple
    sufijo: str = ''
    modelo: BaseEstimator | None = None
    cuantiles: dict | None = None
    correccion: float | None = None
    t_ajuste: float | None = None
    rmse_train: float | None = None
    pred_test: np.ndarray | None = None
    metricas_test: dict | None = None
    tabla_test: pd.DataFrame | None = None
    intervalos: dict | None = None


def configurar_modelo(nombre: str, experimento: str, hiperparametros: dict, construir: Callable,
                      construir_cuantil: Callable, descripcion: str, librerias: tuple, sufijo: str,
                      ruta_registro: Path, folds: list, bloques: dict) -> ModeloFinal:
    """Crea el ModeloFinal con sus variables y sus metricas de CV leidas del registro (no se copian a mano)."""
    columnas = columnas_registradas(experimento, ruta_registro)
    assert set(columnas) == set(columnas_de(bloques, BLOQUES_ESCENARIO['B'])), \
        f'Las variables de {experimento} no son las del escenario B'
    cv = leer_experimento(experimento, ruta_registro, folds)
    return ModeloFinal(nombre, experimento, hiperparametros, columnas,
                       {c: float(cv[c].mean()) for c in METRICAS_PRINCIPALES},
                       {c: float(cv[c].std()) for c in METRICAS_PRINCIPALES},
                       construir, construir_cuantil, descripcion, librerias, sufijo)


def ajustar_modelo(m: ModeloFinal, train: pd.DataFrame) -> None:
    """Entrena el modelo con todo el train y anota el tiempo y el RMSE en train."""
    inicio = perf_counter()
    m.modelo = m.construir(m.columnas).fit(train[m.columnas], train[TARGET])
    m.t_ajuste = perf_counter() - inicio
    m.rmse_train = metricas(train[TARGET], m.modelo.predict(train[m.columnas]))['rmse_log_val']


def calibrar_modelo(m: ModeloFinal, train: pd.DataFrame, fold) -> dict:
    """Ajusta los cuantiles con el train del fold dado y calcula la correccion CQR con su validacion.

    Devuelve la cobertura en el conjunto de calibracion sin calibrar y con CQR (con CQR es la nominal por construccion).
    """
    X, y = train[m.columnas], train[TARGET]
    X_cal, y_cal = X.iloc[fold.validacion], y.iloc[fold.validacion].to_numpy()
    m.cuantiles = {alfa: m.construir_cuantil(m.columnas, alfa).fit(X.iloc[fold.train], y.iloc[fold.train])
                   for alfa in (CUANTIL_BAJO, CUANTIL_ALTO)}
    bajo, alto = (m.cuantiles[a].predict(X_cal) for a in (CUANTIL_BAJO, CUANTIL_ALTO))
    m.correccion = calibrar_cqr(bajo, alto, y_cal)
    return {'n_calibracion': len(y_cal), 'cobertura_sin_calibrar': cobertura_intervalo(y_cal, bajo, alto),
            'cobertura_con_cqr': cobertura_intervalo(y_cal, bajo - m.correccion, alto + m.correccion),
            'correccion_log': m.correccion}


def guardar_modelo(m: ModeloFinal, directorio: Path) -> dict[str, str]:
    """Guarda con joblib el modelo y sus cuantiles (con la correccion CQR) y devuelve el tamano de cada fichero."""
    directorio.mkdir(exist_ok=True)
    rutas = [directorio / f'modelo_final{m.sufijo}.joblib', directorio / f'cuantiles{m.sufijo}.joblib']
    joblib.dump(m.modelo, rutas[0])
    joblib.dump({'cuantiles': m.cuantiles, 'correccion_cqr': m.correccion}, rutas[1])
    return {p.name: f'{p.stat().st_size / 1e6:.1f} MB' for p in rutas}


def evaluar_en_test(m: ModeloFinal, test: pd.DataFrame) -> pd.DataFrame:
    """Metricas del modelo en el test con IC al 95% por bootstrap de anfitriones, frente a la media de la CV.

    Guarda la prediccion y la tabla en m. 'dentro_de_cv_2sd' indica si el valor del test cae a menos de 2 sd de la CV.
    """
    y = test[TARGET].to_numpy()
    m.pred_test = m.modelo.predict(test[m.columnas])
    m.metricas_test = metricas(y, m.pred_test)
    replicas = bootstrap_por_anfitrion(y, m.pred_test, test['host_id'].to_numpy(), metricas)
    filas = []
    for nombre in METRICAS_PRINCIPALES:
        ic = resumen_bootstrap(replicas[nombre].to_numpy(), m.metricas_test[nombre])
        filas.append({'metrica': nombre, 'test': ic['valor'], 'ic95_bajo': ic['ic95_bajo'], 'ic95_alto': ic['ic95_alto'],
                      'cv_media': m.metricas_cv[nombre], 'cv_sd': m.metricas_cv_sd[nombre],
                      'dentro_de_cv_2sd': abs(ic['valor'] - m.metricas_cv[nombre]) <= 2 * m.metricas_cv_sd[nombre]})
    m.tabla_test = pd.DataFrame(filas).set_index('metrica')
    return m.tabla_test


def evaluar_intervalo_test(m: ModeloFinal, test: pd.DataFrame) -> pd.DataFrame:
    """Cobertura real del intervalo sin calibrar y con CQR en el test (con IC por bootstrap) y su anchura mediana en USD."""
    y, X = test[TARGET].to_numpy(), test[m.columnas]
    bajo, alto = (m.cuantiles[a].predict(X) for a in (CUANTIL_BAJO, CUANTIL_ALTO))
    m.intervalos = {'sin calibrar': (bajo, alto), 'CQR': (bajo - m.correccion, alto + m.correccion)}
    filas = []
    for nombre, (b, a) in m.intervalos.items():
        replicas = bootstrap_por_anfitrion(y, np.column_stack([b, a]), test['host_id'].to_numpy(),
                                           lambda yy, p: {'cobertura': cobertura_intervalo(yy, p[:, 0], p[:, 1])})
        ic = resumen_bootstrap(replicas['cobertura'].to_numpy(), cobertura_intervalo(y, b, a))
        filas.append({'intervalo': nombre, 'cobertura': ic['valor'], 'ic95_bajo': ic['ic95_bajo'],
                      'ic95_alto': ic['ic95_alto'], 'anchura_mediana_usd': np.median(np.exp(a) - np.exp(b))})
    return pd.DataFrame(filas).set_index('intervalo')


def cobertura_por_segmento(m: ModeloFinal, test: pd.DataFrame, columna: str, min_n: int = 30) -> pd.DataFrame:
    """Cobertura del intervalo CQR y anchura mediana (USD) por segmento del test; se omiten los de menos de min_n."""
    y = test[TARGET].to_numpy()
    bajo, alto = m.intervalos['CQR']
    tabla = pd.DataFrame({columna: test[columna].to_numpy(), 'cubierto': (y >= bajo) & (y <= alto),
                          'anchura_usd': np.exp(alto) - np.exp(bajo)})
    resumen = tabla.groupby(columna, observed=True).agg(n=('cubierto', 'size'), cobertura_cqr=('cubierto', 'mean'),
                                                        anchura_mediana_usd=('anchura_usd', 'median'))
    return resumen[resumen['n'] >= min_n]


def peores_errores(test: pd.DataFrame, pred_log: np.ndarray, n: int = 20) -> pd.DataFrame:
    """Los n anuncios del test con mayor error absoluto en USD, con su precio real y predicho."""
    tabla = test[['listing_id', 'room_type', 'district', 'accommodates', 'bedrooms', 'price']].copy()
    tabla['pred_usd'] = np.exp(pred_log)
    tabla['error_abs_usd'] = (tabla['price'] - tabla['pred_usd']).abs()
    return tabla.nlargest(n, 'error_abs_usd')


def mapa_residuos(test: pd.DataFrame, pred_log: np.ndarray, titulo: str = '') -> None:
    """Mapa de latitud y longitud coloreado por el residuo en log y por el error absoluto en USD."""
    residuo = test[TARGET].to_numpy() - pred_log
    error_usd = np.abs(test['price'].to_numpy() - np.exp(pred_log))
    fig, ejes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    for eje, (nombre, serie, mapa) in zip(ejes, [('Residuo (log)', residuo, 'coolwarm'),
                                                 ('Error absoluto (USD)', error_usd, 'viridis')]):
        limite = np.percentile(np.abs(serie), 98)
        grafico = eje.scatter(test['longitude'], test['latitude'], c=serie, s=4, alpha=0.6, cmap=mapa,
                              vmin=-limite if mapa == 'coolwarm' else 0, vmax=limite)
        eje.set_title(f'{nombre}{" - " + titulo if titulo else ""}', loc='left')
        eje.set_aspect(1 / np.cos(np.radians(40.7)))
        eje.spines[['top', 'right']].set_visible(False)
        plt.colorbar(grafico, ax=eje, shrink=0.8)
    plt.tight_layout()
    plt.show()


def sensibilidad_p99(train: pd.DataFrame, test: pd.DataFrame, pred_log: np.ndarray) -> pd.DataFrame:
    """Metricas con la regla de bloqueo usada frente a quitar los precios por encima del P99 del TRAIN (no usa el test)."""
    y = test[TARGET].to_numpy()
    corte = train['price'].quantile(0.99)
    sin_p99 = (test['price'] <= corte).to_numpy()
    filas = {'regla de bloqueo (la usada)': metricas(y, pred_log),
             f'sin P99 global (>{corte:.0f} USD)': metricas(y[sin_p99], pred_log[sin_p99])}
    return pd.DataFrame(filas).T.round(4).assign(n=[len(test), int(sin_p99.sum())])


def importancia_modelo(m: ModeloFinal, test: pd.DataFrame, bloques: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Importancia por permutacion en el test, por bloque y por variable (latitud y longitud juntas).

    Devuelve (resumen por bloque, resumen por variable), con la media y la desviacion del aumento de RMSE.
    """
    grupos_bloque = {b: [c for c in cols if c in m.columnas] for b, cols in bloques.items()}
    grupos_bloque = {b: c for b, c in grupos_bloque.items() if c}
    grupos_variable = {c: [c] for c in m.columnas if c not in ('latitude', 'longitude')}
    grupos_variable['latitude + longitude'] = ['latitude', 'longitude']
    X, y = test[m.columnas], test[TARGET]

    def resumen(tabla):
        return tabla.groupby('grupo')['aumento_rmse'].agg(['mean', 'std']).sort_values('mean', ascending=False)

    return (resumen(importancia_permutacion(m.modelo, X, y, grupos_bloque)),
            resumen(importancia_permutacion(m.modelo, X, y, grupos_variable)))


def grafico_pdp(m: ModeloFinal, test: pd.DataFrame, variables: list[str], color_linea: str, color_ice: str) -> None:
    """Dependencia parcial (PDP) y curvas ICE de las variables indicadas, calculadas sobre el test."""
    X = test[m.columnas].astype({v: 'float64' for v in variables})   # sklearn no admite enteros en PDP
    fig, ejes = plt.subplots(1, len(variables), figsize=(5 * len(variables), 4))
    PartialDependenceDisplay.from_estimator(m.modelo, X, variables, kind='both', subsample=300, random_state=SEMILLA,
                                            ax=ejes, ice_lines_kw={'color': color_ice, 'alpha': 0.2},
                                            pd_line_kw={'color': color_linea, 'lw': 3})
    plt.tight_layout()
    plt.show()


def valores_shap(modelo: RegresorCatBoost, X: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Valores SHAP nativos de CatBoost: contribucion de cada variable a cada prediccion (en log) y valor base."""
    from catboost import Pool
    pool = Pool(modelo._preparar(X), cat_features=modelo.categoricas_)
    bruto = modelo.modelo_.get_feature_importance(pool, type='ShapValues')
    return bruto[:, :-1], bruto[:, -1][0]


def comprobar_modelo_guardado(m: ModeloFinal, train: pd.DataFrame, brutos: pd.DataFrame, preparador,
                              directorio: Path) -> pd.DataFrame:
    """Carga el modelo desde disco, predice la cartera de un anfitrion en bruto y comprueba que coincide con el de memoria."""
    modelo = joblib.load(directorio / f'modelo_final{m.sufijo}.joblib')
    guardado = joblib.load(directorio / f'cuantiles{m.sufijo}.joblib')
    anfitrion = train.loc[train['n_anuncios_ny'].between(2, 3), 'host_id'].iloc[0]
    cartera = brutos[brutos['host_id'] == anfitrion]
    resultado = predecir_precio(cartera, preparador, modelo, guardado['cuantiles'], guardado['correccion_cqr'],
                                m.columnas)
    real = cartera.set_index('listing_id')['price'].reindex(resultado['listing_id']).to_numpy()
    en_train = resultado[resultado['listing_id'].isin(train['listing_id'])]   # los excluidos del dataset no estan
    en_memoria = m.modelo.predict(train.set_index('listing_id').loc[en_train['listing_id'], m.columnas])
    assert np.allclose(np.exp(en_memoria), en_train['precio_mediano'], rtol=1e-6), 'El modelo cargado predice distinto'
    return resultado.assign(precio_real=real)


def guardar_ficha_modelo(m: ModeloFinal, particion: dict, directorio: Path) -> dict:
    """Escribe la ficha del modelo (hiperparametros, variables, metricas de CV y test, MD5 y versiones) y la devuelve."""
    ficha = ficha_modelo(m.hiperparametros, m.columnas, {'media': m.metricas_cv, 'sd': m.metricas_cv_sd},
                         {k: float(v) for k, v in m.metricas_test.items()}, m.correccion,
                         particion['md5']['train'], particion['md5']['test'], m.descripcion, m.librerias)
    guardar_ficha(ficha, directorio / f'ficha_modelo{m.sufijo}.json')
    return ficha


def diferencia_pareada_bootstrap(a: ModeloFinal, b: ModeloFinal, test: pd.DataFrame,
                                 metrica: str = 'rmse_log_val') -> dict[str, float]:
    """Diferencia (a - b) en una metrica del test con IC al 95% por bootstrap de anfitriones, con los mismos anfitriones.

    Remuestrear los mismos anfitriones para los dos modelos hace la comparacion pareada: el intervalo de la
    diferencia es mas estrecho que el de cada metrica por separado.
    """
    y = test[TARGET].to_numpy()
    pred = np.column_stack([a.pred_test, b.pred_test])

    def diferencia(yy, p):
        return {'dif': metricas(yy, p[:, 0])[metrica] - metricas(yy, p[:, 1])[metrica]}

    replicas = bootstrap_por_anfitrion(y, pred, test['host_id'].to_numpy(), diferencia)
    resumen = resumen_bootstrap(replicas['dif'].to_numpy(), a.metricas_test[metrica] - b.metricas_test[metrica])
    resumen['replicas_a_favor_de_a'] = float((replicas['dif'] < 0).mean())
    return resumen


def anuncios_ilustrativos(test: pd.DataFrame, pred_log: np.ndarray) -> dict[str, int]:
    """Posicion en el test de tres anuncios elegidos con una regla fijada antes de mirar su explicacion.

    'tipico': error absoluto mas cercano a la mediana. 'caro bien predicho': el de menor error relativo entre el 5%
    mas caro. 'peor error': el de mayor error absoluto en USD.
    """
    precio = test['price'].to_numpy()
    error = np.abs(precio - np.exp(pred_log))
    caros = np.flatnonzero(precio >= np.quantile(precio, 0.95))
    return {'tipico': int(np.abs(error - np.median(error)).argmin()),
            'caro bien predicho': int(caros[(error[caros] / precio[caros]).argmin()]),
            'peor error': int(error.argmax())}
