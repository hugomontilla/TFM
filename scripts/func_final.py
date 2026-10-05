"""Funciones del modelo final (4-MODELO_FINAL.ipynb y Modelo_final_HGB.ipynb): hiperparametros desde el registro,
intervalo de confianza por anfitrion, calibracion CQR del intervalo de precio, error por segmento y prediccion de un
anuncio en bruto.

Convenciones: y, pred, cuantiles estan en log(price); los precios en dolares se obtienen con exp.
"""
import hashlib
import json
import platform
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline, make_pipeline

from func_modelado import (RegresorCatBoost, SEMILLA, TARGET, cobertura_intervalo, metricas,
                           preprocesador_arboles)

COBERTURA_NOMINAL = 0.8
CUANTIL_BAJO, CUANTIL_ALTO = 0.1, 0.9
N_REPLICAS_BOOTSTRAP = 2000
DESCRIPCION_CATBOOST = ('RegresorCatBoost (RMSE sobre log(price)) + dos cuantiles CatBoost (80% del train) calibrados '
                        'con CQR en el fold 0 de calibracion')
DESCRIPCION_HGB = ('Pipeline(TargetEncoder del barrio + HistGradientBoostingRegressor, error cuadratico sobre log(price)) '
                   '+ dos HistGradientBoostingRegressor de perdida cuantil (80% del train) calibrados con CQR en el '
                   'fold 0 de calibracion')


def md5_fichero(ruta: Path) -> str:
    return hashlib.md5(Path(ruta).read_bytes()).hexdigest()


def _unico_del_registro(nombre: str, ruta_registro: Path, columna: str) -> str:
    """Valor de `columna` para un experimento del registro; falla si sus filas no coinciden en ese valor."""
    registro = pd.read_csv(ruta_registro, usecols=['experimento', columna])
    valores = registro.loc[registro['experimento'] == nombre, columna].unique()
    assert len(valores) == 1, f"'{nombre}' tiene {len(valores)} valores distintos de '{columna}' en el registro"
    return valores[0]


def descripcion_registrada(nombre: str, ruta_registro: Path) -> str:
    """Descripcion (repr del modelo) que guarda el registro de experimentos para un experimento."""
    return _unico_del_registro(nombre, ruta_registro, 'modelo')


def hiperparametros_registrados(nombre: str, ruta_registro: Path) -> dict:
    """Hiperparametros de un RegresorCatBoost leidos de la descripcion que guarda el registro de experimentos."""
    modelo = eval(descripcion_registrada(nombre, ruta_registro),
                  {'RegresorCatBoost': RegresorCatBoost, 'np': np})  # fichero propio, no externo
    return {k: v for k, v in modelo.get_params().items() if k != 'loss_function'}


def columnas_registradas(nombre: str, ruta_registro: Path) -> list[str]:
    return _unico_del_registro(nombre, ruta_registro, 'variables').split(';')


def cuantil_catboost(alfa: float, hiperparametros: dict) -> RegresorCatBoost:
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
    from datetime import date
    from importlib.metadata import version
    return {
        'modelo': descripcion,
        'hiperparametros': {k: (float(v) if isinstance(v, (np.floating, float)) else v)
                            for k, v in hiperparametros.items()},
        'variables': columnas, 'escenario': 'B',
        'correccion_cqr_log': correccion_cqr, 'cobertura_nominal': COBERTURA_NOMINAL,
        'metricas_cv': metricas_cv, 'metricas_test': metricas_test,
        'md5_train_parquet': md5_train, 'md5_test_parquet': md5_test,
        'versiones': {'python': platform.python_version(), **{libreria: version(libreria) for libreria in librerias}},
        'fecha': date.today().isoformat(),
        'nota': 'exp(prediccion) estima la MEDIANA del precio, no la media.'}


def guardar_ficha(ficha: dict, ruta: Path) -> None:
    Path(ruta).write_text(json.dumps(ficha, indent=2, ensure_ascii=False), encoding='utf-8')
