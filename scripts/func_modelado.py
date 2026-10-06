"""Funciones de la fase de modelado: folds de CV, evaluacion y registro de experimentos, referencias ingenuas,
preprocesadores por familia de modelo y comparaciones pareadas.

Convenciones de los argumentos que se repiten:
    datos: DataFrame de modelado (cargar_modelado), con la columna TARGET y las variables de los bloques.
    folds: lista de Fold (cargar_folds); indices posicionales sobre las filas de datos.
    bloques: {nombre del bloque: [columnas]}, los da PreparadorListings.bloques_.
"""
import os
import re
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error, mean_pinball_loss,
                             median_absolute_error, r2_score, root_mean_squared_error)
from sklearn.model_selection import KFold, ParameterSampler, StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder, OrdinalEncoder, SplineTransformer,
                                   StandardScaler, TargetEncoder)
from threadpoolctl import threadpool_limits

from config import N_FOLDS, N_REPETICIONES_CV, SEMILLA
from particion_datos import COL_GRUPO, estrato
from registro import leer_experimento, registrar
from preparacion_datos import RUTA_PREPARADOR, RUTAS_MODELADO, PreparadorListings

TARGET = 'log_price'

# Busqueda de CatBoost (Modelado, seccion 10.3 y Busqueda_CatBoost_colab.ipynb): una sola definicion para los dos notebooks.
# 10 candidatos (ParameterSampler con semilla fija da los mismos primeros)
ESPACIO_CATBOOST = {
    'learning_rate': loguniform(0.04, 0.2),
    'depth': randint(4, 9),
    'l2_leaf_reg': loguniform(3, 100),
}

FIJOS_CATBOOST = dict(iterations=3000, early_stopping_rounds=65, random_strength=1)
N_ITER_CATBOOST = 10

BLOQUES_ESCENARIO_A = ['B0_producto', 'B1_ubicacion', 'B2a_condiciones', 'B3_amenities_comunes']
# El escenario B solo añade el perfil del anfitrion: la diferencia entre ambos mide lo que vale conocerlo
BLOQUES_ESCENARIO = {
    'A': BLOQUES_ESCENARIO_A,
    'B': BLOQUES_ESCENARIO_A + ['B2b_anfitrion'],
}


@dataclass(frozen=True)
class Fold:
    """Una particion de la CV. Inmutable para que ningun experimento pueda alterarla.

    Atributos:
        repeticion: repeticion del esquema repetido (0 .. n_repeticiones - 1); cada una usa otra semilla.
        fold: numero de fold dentro de la repeticion (0 .. N_FOLDS - 1).
        train: posiciones (iloc) de las filas de entrenamiento.
        validacion: posiciones (iloc) de las filas de validacion.
    """
    repeticion: int
    fold: int
    train: np.ndarray
    validacion: np.ndarray


@dataclass(frozen=True)
class Experimento:
    """Un modelo sin ajustar y las columnas que ve; el nombre identifica sus filas en el registro.

    Atributos:
        nombre: identificador unico en el registro de experimentos.
        modelo: estimador o Pipeline de scikit-learn sin ajustar; se clona en cada fold.
        columnas: variables de datos que recibe el modelo.
        escenario: 'A' (sin perfil del anfitrion) o 'B' (con el).
    """
    nombre: str
    modelo: BaseEstimator
    columnas: list[str] = field(default_factory=list)
    escenario: Literal['A', 'B'] = 'B'


def cargar_modelado(conjunto: Literal['train', 'test'] = 'train') -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Dataset de modelado y bloques de variables. El test solo se carga una vez, con el modelo final.

    Devuelve (datos con la columna TARGET = log(price), {bloque: [columnas]}).
    """
    datos = pd.read_parquet(RUTAS_MODELADO[conjunto])
    datos[TARGET] = np.log(datos['price'])
    return datos, PreparadorListings.cargar(RUTA_PREPARADOR).bloques_


def columnas_de(bloques: dict[str, list[str]], nombres_bloque: list[str]) -> list[str]:
    """Columnas de los bloques indicados, en el orden de nombres_bloque."""
    return [col for nombre in nombres_bloque for col in bloques[nombre]]

# La particion se genera antes de empezar con el modelado dejandola fija y solo cargando su imagen en tabla

def generar_folds(datos: pd.DataFrame, n_repeticiones: int) -> pd.DataFrame:
    """Tabla (listing_id, repeticion, fold) con el fold de validacion de cada anuncio en cada repeticion.

    Folds agrupados por anfitrion y estratificados como la particion; cada repeticion usa otra semilla.
    No usa el precio: solo room_type, district y host_id. Se guarda por listing_id y no por posicion para
    que no dependa del orden de las filas.
    """
    tablas = []
    for repeticion in range(n_repeticiones):
        divisor = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA + 1 + repeticion)
        fold = np.empty(len(datos), dtype=int)
        for numero, (_, idx_validacion) in enumerate(divisor.split(datos, estrato(datos), datos[COL_GRUPO])):
            fold[idx_validacion] = numero
        tablas.append(pd.DataFrame({'listing_id': datos['listing_id'].to_numpy(), 'repeticion': repeticion,
                                    'fold': fold}))
    return pd.concat(tablas, ignore_index=True)


def folds_desde_tabla(tabla: pd.DataFrame, ids: np.ndarray) -> list[Fold]:
    """Convierte la tabla de generar_folds en Fold con posiciones sobre las filas cuyo listing_id es ids.

    Falla si en alguna repeticion la tabla no cubre exactamente los anuncios de ids 
    (han cambiado desde que se hizo la particion)
    """
    posiciones = pd.Series(np.arange(len(ids)), index=ids)
    todas = np.arange(len(ids))
    folds = []
    for repeticion, tabla_rep in tabla.groupby('repeticion'):
        if set(tabla_rep['listing_id']) != set(ids) or len(tabla_rep) != len(ids):
            raise ValueError(f'Los folds guardados de la repeticion {repeticion} no cubren los anuncios de los datos')
        for fold, tabla_fold in tabla_rep.groupby('fold'):
            validacion = np.sort(posiciones.loc[tabla_fold['listing_id']].to_numpy())
            folds.append(Fold(int(repeticion), int(fold), np.setdiff1d(todas, validacion), validacion))
    return folds


def cargar_folds(datos: pd.DataFrame, ruta: Path, n_repeticiones: int) -> list[Fold]:
    """Folds de CV con el CSV guardado como fuente de verdad: los lee si existe y solo los genera si no.

    Asi una version distinta de scikit-learn o un cambio de orden de las filas no altera los folds en silencio.
    Para regenerarlos hay que borrar el CSV a mano (y recalcular el registro de experimentos).
    """
    if not ruta.exists():
        generar_folds(datos, n_repeticiones).to_csv(ruta, index=False)
    folds = folds_desde_tabla(pd.read_csv(ruta), datos['listing_id'].to_numpy())
    guardadas = 1 + max(f.repeticion for f in folds)
    if guardadas != n_repeticiones:
        raise ValueError(f'{ruta.name} tiene {guardadas} repeticiones y se piden {n_repeticiones}')
    return folds


def metricas(y_log: pd.Series | np.ndarray, pred_log: np.ndarray) -> dict[str, float]:
    """Metricas en log para comparar modelos y en dolares para el sponsor; exp(pred) estima la mediana.

    y_log, pred_log: valores reales y predichos de log(price).
    """
    precio, precio_pred = np.exp(y_log), np.exp(pred_log)
    return {
        'rmse_log_val': root_mean_squared_error(y_log, pred_log),
        'mae_log_val': mean_absolute_error(y_log, pred_log),
        'r2_log_val': r2_score(y_log, pred_log),
        'mae_usd_val': mean_absolute_error(precio, precio_pred),
        'medae_usd_val': median_absolute_error(precio, precio_pred),
        'mape_val': mean_absolute_percentage_error(precio, precio_pred),
    }


def cobertura_intervalo(y: np.ndarray, bajo: np.ndarray, alto: np.ndarray) -> float:
    """Fraccion de valores reales que caen dentro del intervalo [bajo, alto]."""
    return float(np.mean((y >= bajo) & (y <= alto)))


def _sin_direcciones(texto: str) -> str:
    """Quita las direcciones de memoria de un repr, que cambian en cada sesion."""
    # El repr de una funcion incluye su direccion de memoria, que cambia en cada sesion. Se usa en
    # descripcion_modelo (al escribir el registro) y en evaluar_registrado (al leerlo, por las filas antiguas)
    return re.sub(r' at 0x[0-9A-Fa-f]+', '', texto)


def descripcion_modelo(modelo: BaseEstimator) -> str:
    """repr completo en una linea: el repr por defecto recorta los pipelines largos con '...'."""
    return _sin_direcciones(' '.join(modelo.__repr__(N_CHAR_MAX=100_000).split()))


def _evaluar_fold(experimento: Experimento, X: pd.DataFrame, y: pd.Series, f: Fold, hilos: int | None,
                  paralelo: int) -> dict:
    """Ajusta un clon del modelo en un fold y devuelve su fila de metricas.

    hilos: tope de hilos de BLAS/OpenMP dentro del ajuste (None: sin tope). Va a nivel de modulo para que joblib
    pueda enviarla a otro proceso.
    """
    with nullcontext() if hilos is None else threadpool_limits(limits=hilos):
        # Creamos una copia nueva sin entrenar del modelo
        modelo = clone(experimento.modelo)
        # Cronometramos el tiempo que toma entrenar
        inicio = perf_counter()
        modelo.fit(X.iloc[f.train], y.iloc[f.train])
        t_ajuste = perf_counter() - inicio

        # Cronometramos el tiempo que toma predecir
        inicio = perf_counter()
        pred_validacion = modelo.predict(X.iloc[f.validacion])
        t_prediccion = perf_counter() - inicio
        # El error en train contra el de validacion es la medida de sobreajuste de la tabla comparativa
        rmse_train = root_mean_squared_error(y.iloc[f.train], modelo.predict(X.iloc[f.train]))

    return {
        'experimento': experimento.nombre,
        'escenario': experimento.escenario,
        'repeticion': f.repeticion,
        'fold': f.fold,
        **metricas(y.iloc[f.validacion], pred_validacion),
        'rmse_log_train': rmse_train,
        't_ajuste_s': t_ajuste,
        't_prediccion_s': t_prediccion,
        'n_variables': len(experimento.columnas),
        'variables': ';'.join(experimento.columnas),
        'modelo': descripcion_modelo(experimento.modelo),
        'paralelo': paralelo,
    }


def evaluar(experimento: Experimento, datos: pd.DataFrame, folds: list[Fold], paralelo: int = 1) -> pd.DataFrame:
    """Ajusta un clon del modelo en cada fold y devuelve una fila de metricas por fold.

    paralelo: folds que se ajustan a la vez, cada uno en su proceso. Con mas de 1, cada proceso limita sus hilos a
    nucleos // paralelo para que no compitan entre si. Solo compensa en modelos que no usan ya todos los nucleos
    (lineales con coordenadas, HGB, arbol, SVR); no en RF (n_jobs=-1) ni CatBoost. Los resultados son los mismos,
    pero t_ajuste_s se mide con la maquina cargada: los tiempos finales de la comparativa se miden con paralelo=1.
    """
    X = datos[experimento.columnas]
    y = datos[TARGET]
    if paralelo <= 1 or len(folds) == 1:
        return pd.DataFrame([_evaluar_fold(experimento, X, y, f, None, 1) for f in folds])
    procesos = min(paralelo, len(folds))
    hilos = max(1, (os.cpu_count() or 1) // procesos)
    # Los procesos hijos importan este modulo al recibir el experimento: tiene que estar en su sys.path
    os.environ['PYTHONPATH'] = os.pathsep.join([str(Path(__file__).resolve().parent), os.environ.get('PYTHONPATH', '')])
    filas = Parallel(n_jobs=procesos)(delayed(_evaluar_fold)(experimento, X, y, f, hilos, procesos) for f in folds)
    return pd.DataFrame(filas)


def resumir(resultados: pd.DataFrame) -> pd.DataFrame:
    """Media y desviacion tipica entre folds de las metricas principales, por experimento.

    Devuelve una fila por experimento con columnas '<metrica>_mean' y '<metrica>_std'.
    """
    columnas = ['rmse_log_val', 'r2_log_val', 'mae_usd_val', 'medae_usd_val', 'mape_val', 'rmse_log_train', 't_ajuste_s']
    resumen = resultados.groupby('experimento', sort=False)[columnas].agg(['mean', 'std'])
    resumen.columns = [f'{metrica}_{estadistico}' for metrica, estadistico in resumen.columns]
    return resumen


def evaluar_registrado(experimento: Experimento, datos: pd.DataFrame, folds: list[Fold], ruta: Path,
                       recalcular: bool = False, paralelo: int = 1) -> pd.DataFrame:
    """Evalua y registra, o reutiliza las filas del registro si el experimento ya se ejecuto igual.

    Solo reutiliza si coinciden el modelo, las variables y los folds: cambiar cualquiera obliga a recalcular.
    Evita repetir la Fase 4 (mas de una hora) cada vez que se ejecuta el notebook.
    recalcular: True fuerza la evaluacion aunque haya filas validas en el registro.
    paralelo: folds en paralelo (ver evaluar); no forma parte de lo que se compara para reutilizar filas.
    """
    if ruta.exists() and not recalcular:
        previos = pd.read_csv(ruta)
        previos = previos[previos['experimento'] == experimento.nombre]
        mismo_modelo = (previos['modelo'].map(_sin_direcciones) == descripcion_modelo(experimento.modelo)).all()
        mismas_variables = (previos['variables'].fillna('') == ';'.join(experimento.columnas)).all()
        mismos_folds = set(zip(previos['repeticion'], previos['fold'])) == {(f.repeticion, f.fold) for f in folds}
        if len(previos) and mismo_modelo and mismas_variables and mismos_folds:
            return previos.reset_index(drop=True)
    resultados = evaluar(experimento, datos, folds, paralelo)
    registrar(resultados, ruta)
    return resultados


def buscar_hiperparametros(prefijo: str, construir, espacio: dict, n_iter: int, datos: pd.DataFrame,
                           folds: list[Fold], ruta: Path, paralelo: int = 1, solo_leer: bool = False,
                           paciencia: int | None = None, margen: float = 0.0) -> pd.DataFrame:
    """Busqueda aleatoria con ParameterSampler: cada candidato se evalua y registra como un experimento.

    prefijo: los candidatos se llaman '<prefijo>_00', '<prefijo>_01'... La semilla es fija: con otro n_iter los
        primeros candidatos son los mismos y se reutilizan del registro.
    construir: funcion (nombre, params) -> Experimento.
    espacio: distribuciones o listas de ParameterSampler.
    solo_leer: no entrena: lee los candidatos del registro con leer_experimento y comprueba que los hiperparametros
        enteros coinciden con los de la descripcion guardada (que ParameterSampler da lo mismo que cuando se entreno).
        Para saber los valores por defecto necesita construir (nombre, params) -> Experimento.
    paciencia: parada por meseta. Si los ultimos `paciencia` candidatos seguidos no mejoran el mejor RMSE en mas de
        `margen`, la busqueda se detiene antes de n_iter. None (por defecto) = evalua los n_iter. Como la semilla es
        fija, el orden de candidatos y el punto de parada son reproducibles.
    Devuelve una fila por candidato, ordenada de menor a mayor RMSE en log de validacion.
    """
    filas = []
    mejor, sin_mejora = np.inf, 0
    for i, params in enumerate(ParameterSampler(espacio, n_iter=n_iter, random_state=SEMILLA)):
        if solo_leer:
            res = leer_experimento(f'{prefijo}_{i:02d}', ruta, folds)
            # El repr guardado omite los hiperparametros que valen su valor por defecto: ausente = valor por defecto
            defectos = construir(prefijo, {}).modelo.get_params() if construir else {}
            for clave, valor in params.items():
                if not isinstance(valor, (int, np.integer)):
                    continue
                descripcion = res['modelo'].iloc[0]
                if f'{clave}=' in descripcion:
                    coincide = f'{clave}={valor}' in descripcion
                else:
                    coincide = clave not in defectos or defectos[clave] == valor
                if not coincide:
                    raise ValueError(f"{prefijo}_{i:02d}: {clave}={valor} no coincide con el registro; el espacio o la "
                                     f"semilla de la busqueda han cambiado")
        else:
            res = evaluar_registrado(construir(f'{prefijo}_{i:02d}', params), datos, folds, ruta, paralelo=paralelo)
        filas.append({**params, 'experimento': f'{prefijo}_{i:02d}', 'rmse_log_val': res['rmse_log_val'].mean(),
                      'rmse_log_val_sd': res['rmse_log_val'].std(), 'rmse_log_train': res['rmse_log_train'].mean(),
                      't_ajuste_s': res['t_ajuste_s'].mean()})
        rmse = filas[-1]['rmse_log_val']
        sin_mejora = 0 if rmse < mejor - margen else sin_mejora + 1
        mejor = min(mejor, rmse)
        if paciencia is not None and sin_mejora >= paciencia:
            print(f'{prefijo}: parada por meseta tras {i + 1} de {n_iter} candidatos '
                  f'({paciencia} seguidos sin mejorar el mejor en mas de {margen})')
            break
    return pd.DataFrame(filas).sort_values('rmse_log_val')



class MedianaPorGrupo(RegressorMixin, BaseEstimator):
    """Referencia ingenua: mediana del target por celda, bajando al nivel siguiente si la celda tiene menos de min_n.

    Argumentos:
        niveles: tuplas de columnas que definen las celdas, del nivel mas fino al mas grueso,
            p. ej. (('neighbourhood', 'room_type'), ('district', 'room_type')). Si ningun nivel tiene la
            celda, se usa la mediana global.
        min_n: anuncios minimos en el train para usar la mediana de una celda.
        tope_capacidad: agrupa accommodates en 'tope o mas', como haria un anfitrion al buscar comparables;
            None no agrupa.
    """

    def __init__(self, niveles: tuple[tuple[str, ...], ...] = (('room_type',),), min_n: int = 1,
                 tope_capacidad: int | None = None):
        """Guarda los niveles de agrupacion, el minimo de anuncios por celda y el tope de capacidad."""
        self.niveles = niveles
        self.min_n = min_n
        self.tope_capacidad = tope_capacidad

    def _celdas(self, X: pd.DataFrame, nivel: tuple[str, ...]) -> pd.Series:
        """Etiqueta de celda de cada fila para un nivel (concatena las columnas del nivel)."""
        partes = []
        for col in nivel:
            valores = X[col]
            if col == 'accommodates' and self.tope_capacidad is not None:
                valores = valores.clip(upper=self.tope_capacidad)
            partes.append(valores.astype(str))
        clave = partes[0]
        for parte in partes[1:]:
            clave = clave + ' | ' + parte
        return clave

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> 'MedianaPorGrupo':
        """Calcula la mediana del target por celda en cada nivel y la mediana global."""
        y = pd.Series(np.asarray(y), index=X.index)
        self.mediana_global_ = float(y.median())
        self.tablas_ = []
        for nivel in self.niveles:
            celdas = y.groupby(self._celdas(X, nivel)).agg(['median', 'size'])
            self.tablas_.append(celdas.loc[celdas['size'] >= self.min_n, 'median'])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice con la celda mas fina que tenga muestra suficiente y baja de nivel si no la hay."""
        pred = pd.Series(self.mediana_global_, index=X.index)
        # Del nivel mas grueso al mas fino: cada nivel sobrescribe donde tiene una celda con muestra suficiente
        for nivel, tabla in reversed(list(zip(self.niveles, self.tablas_))):
            encontrados = self._celdas(X, nivel).map(tabla)
            pred = encontrados.fillna(pred)
        return pred.to_numpy()


# Tipos de columna de los preprocesadores (plan, seccion 3.4)
CATEGORICAS = ['room_type', 'district', 'property_type_grp', 'host_ambito', 'host_response_time']
COL_BARRIO = 'neighbourhood'
RECORTE_ESTANCIA = 365
CONTEOS = ['n_anuncios_ny', 'host_total_listings_count']
CON_NULOS = ['antiguedad_host_anios', 'host_response_rate', 'host_acceptance_rate', 'reputacion_cartera']
CON_SPLINES = ['accommodates', 'dist_centro_km', 'latitude', 'longitude']
TIPOS_NO_REFERENCIA = ['Private room', 'Shared room', 'Hotel room']
INTERACCION_DISTRITO = 'room_type_x_district'


# Funciones para FunctionTransformer. Van a nivel de modulo y con nombre, no como lambdas, por dos motivos:
# el repr de una lambda no muestra su contenido (la cache de evaluar_registrado no veria un cambio) y
# pickle no puede guardar lambdas ni funciones locales (joblib.dump del modelo final fallaria).

def _log_estancia(x: np.ndarray | pd.DataFrame) -> np.ndarray:
    """log1p de las noches minimas recortadas a un ano (rama 'estancia' del preprocesador lineal)."""
    # Rama 'estancia' de preprocesador_lineal, solo para minimum_nights. El recorte a un ano neutraliza los
    # centinelas y el log comprime la cola; los arboles la reciben en bruto
    return np.log1p(np.clip(np.asarray(x, dtype=float), None, RECORTE_ESTANCIA))


def _a_float(x: np.ndarray | pd.DataFrame) -> np.ndarray:
    """Convierte las columnas restantes del preprocesador lineal a un array float homogeneo."""
    # Rama 'resto' de preprocesador_lineal: casi todas sus columnas son bool (amenities y flags) mezcladas con
    # numericas, y se pasan a un array float homogeneo antes del StandardScaler
    return np.asarray(x, dtype=float)


def anadir_interacciones(X: pd.DataFrame) -> pd.DataFrame:
    """Interacciones del modelo L3: la capacidad y el distrito cambian de efecto segun el tipo de habitacion.

    No aprende nada de los datos (las categorias de room_type son fijas), asi que no hay riesgo de fuga.
    Requiere room_type en X. Devuelve una copia de X con las interacciones añadidas al final.
    Es la unica definicion de las interacciones: preprocesador_lineal la aplica a una tabla vacia para
    saber que columnas nuevas va a recibir.
    """
    X = X.copy()
    tipos = X['room_type'].astype(str)
    if 'accommodates' in X:
        for tipo in TIPOS_NO_REFERENCIA:
            X[f'accommodates_x_{tipo}'] = X['accommodates'] * (tipos == tipo)
    if 'district' in X:
        X[INTERACCION_DISTRITO] = tipos + ' | ' + X['district'].astype(str)
    return X


def _acepta_cv_como_divisor() -> bool:
    """True si esta version de scikit-learn admite un divisor de folds en TargetEncoder(cv=...)."""
    # Las versiones antiguas de scikit-learn (p. ej. la de Colab) solo admiten un entero en TargetEncoder(cv=...)
    try:
        TargetEncoder(cv=KFold(2)).fit(pd.DataFrame({'x': ['a', 'b', 'a', 'b']}), [1.0, 2.0, 1.0, 2.0])
        return True
    except Exception:
        return False


CV_BARRIO_ADMITE_DIVISOR = _acepta_cv_como_divisor()


def _codificador_barrio() -> TargetEncoder:
    """TargetEncoder del barrio con la configuracion comun a los preprocesadores lineal y de arboles."""
    # Un TargetEncoder nuevo para cada preprocesador que codifica neighbourhood (preprocesador_lineal y
    # preprocesador_arboles con barrio='target'); asi los dos usan la misma configuracion.
    # cross-fitting por filas, no por anfitrion: fuga menor entre anuncios gemelos del train
    if CV_BARRIO_ADMITE_DIVISOR:
        return TargetEncoder(cv=KFold(N_FOLDS, shuffle=True, random_state=SEMILLA))
    # Con un objetivo continuo, TargetEncoder usa internamente KFold(cv, shuffle, random_state): el mismo reparto
    return TargetEncoder(cv=N_FOLDS, shuffle=True, random_state=SEMILLA)


def preprocesador_lineal(columnas: list[str], splines: bool = False,
                         interacciones: bool = False) -> ColumnTransformer | Pipeline:
    """ColumnTransformer para Ridge, kNN y SVR: one-hot, target encoding del barrio, log de conteos, imputacion y escalado.

    columnas: variables del experimento; cada una va a su rama segun las listas de tipos de columna.
    splines: sustituye el escalado de CON_SPLINES por B-splines cubicos (8 nodos).
    interacciones: añade las de anadir_interacciones; solo tiene efecto si room_type esta en columnas.
    Devuelve un Pipeline (interacciones + ColumnTransformer) si hay interacciones y si no el ColumnTransformer.
    """
    columnas = list(columnas)
    interacciones = interacciones and 'room_type' in columnas
    if interacciones:
        # Las columnas que tendra X tras anadir_interacciones, sin necesitar los datos
        columnas = list(anadir_interacciones(pd.DataFrame(columns=columnas)).columns)
    categoricas = [c for c in columnas if c in CATEGORICAS + [INTERACCION_DISTRITO]]
    en_splines = [c for c in columnas if splines and c in CON_SPLINES]
    conteos = [c for c in columnas if c in CONTEOS]
    con_nulos = [c for c in columnas if c in CON_NULOS]
    barrio = [c for c in columnas if c == COL_BARRIO]
    estancia = [c for c in columnas if c == 'minimum_nights']
    especiales = set(categoricas + en_splines + conteos + con_nulos + barrio + estancia)
    resto = [c for c in columnas if c not in especiales]

    ramas = [
        ('categoricas', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categoricas),
        ('barrio', make_pipeline(_codificador_barrio(), StandardScaler()), barrio),
        ('estancia', make_pipeline(FunctionTransformer(_log_estancia, feature_names_out='one-to-one'),
                                   StandardScaler()), estancia),
        ('conteos', make_pipeline(SimpleImputer(strategy='median'),
                                  FunctionTransformer(np.log1p, feature_names_out='one-to-one'), StandardScaler()),
         conteos),
        ('con_nulos', make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()), con_nulos),
        ('splines', SplineTransformer(n_knots=8, degree=3), en_splines),
        ('resto', make_pipeline(FunctionTransformer(_a_float, feature_names_out='one-to-one'), StandardScaler()), resto),
    ]
    transformador = ColumnTransformer([rama for rama in ramas if rama[2]])
    if interacciones:
        return make_pipeline(FunctionTransformer(anadir_interacciones), transformador)
    return transformador


def preprocesador_arboles(columnas: list[str], barrio: Literal['target', 'nativa'] = 'target') -> ColumnTransformer:
    """Para HGB: categoricas y NaN nativos.

    barrio: 'target' (TargetEncoder dentro del fold) o 'nativa' (categorica de HGB).
    Para dejarlo fuera se quita de las columnas del experimento.
    """
    columnas = list(columnas)
    ramas = []
    if COL_BARRIO in columnas and barrio == 'target':
        ramas.append(('barrio', _codificador_barrio(), [COL_BARRIO]))
        columnas.remove(COL_BARRIO)
    ramas.append(('resto', 'passthrough', columnas))
    return ColumnTransformer(ramas, verbose_feature_names_out=False).set_output(transform='pandas')


def preprocesador_ordinal(columnas: list[str]) -> ColumnTransformer:
    """Para el arbol y el Random Forest: categoricas a enteros y NaN nativos (scikit-learn >= 1.4)."""
    categoricas = [c for c in columnas if c in CATEGORICAS + [COL_BARRIO]]
    resto = [c for c in columnas if c not in categoricas]
    return ColumnTransformer([
        ('categoricas', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan,
                                       encoded_missing_value=np.nan), categoricas),
        ('resto', 'passthrough', resto),
    ], verbose_feature_names_out=False).set_output(transform='pandas')


class RegresorCatBoost(RegressorMixin, BaseEstimator):
    """CatBoostRegressor con la interfaz de scikit-learn, sin preprocesador delante.

    Las columnas categoricas de X (dtype category u object) van como cat_features nativas: CatBoost las codifica
    con ordered target statistics (cada fila solo con las que la preceden en una permutacion aleatoria, sin su
    propio precio). El resto, NaN incluidos, pasa en bruto como en HGB.
    Existe por dos motivos: el repr de CatBoostRegressor no muestra sus hiperparametros (la cache de
    evaluar_registrado no veria un cambio) y no sigue del todo la API de estimadores de scikit-learn.

    Argumentos: los de CatBoostRegressor con el mismo nombre; learning_rate=None lo elige CatBoost segun
    iterations y el tamaño de los datos. random_state es su random_seed. task_type='GPU' lo ajusta en la tarjeta
    grafica: no da exactamente los mismos resultados que en CPU, asi que una busqueda se hace entera en un
    dispositivo. Aviso: con el bootstrap por defecto en CPU (MVS) bagging_temperature se ignora; solo actua con
    'Bayesian'.
    early_stopping_rounds: si no es None, fit aparta validacion_fraccion del train que recibe (en el CV, del train
    del fold, nunca del fold de validacion), para al no mejorar en ese trozo y se queda con la mejor iteracion;
    iterations pasa a ser un tope. Con None entrena todas las iteraciones, como antes.
    loss_function: 'RMSE' (media de log(price)) o 'Quantile:alpha=0.1' para un cuantil (intervalo de precio).
    """

    def __init__(self, iterations: int = 1000, learning_rate: float | None = None, depth: int = 6,
                 l2_leaf_reg: float = 3.0, random_strength: float = 1.0, bagging_temperature: float = 1.0,
                 random_state: int = SEMILLA, task_type: Literal['CPU', 'GPU'] = 'CPU',
                 early_stopping_rounds: int | None = None, validacion_fraccion: float = 0.1,
                 loss_function: str = 'RMSE'):
        """Guarda los hiperparametros del CatBoost; no entrena nada hasta fit."""
        self.loss_function = loss_function
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_strength = random_strength
        self.bagging_temperature = bagging_temperature
        self.random_state = random_state
        self.task_type = task_type
        self.early_stopping_rounds = early_stopping_rounds
        self.validacion_fraccion = validacion_fraccion

    def _preparar(self, X: pd.DataFrame) -> pd.DataFrame:
        """Pasa las categoricas a texto, como exige CatBoost."""
        # CatBoost exige categoricas de tipo str o int; los NaN pasan a la categoria 'nan'
        X = X.copy()
        for col in self.categoricas_:
            X[col] = X[col].astype(str)
        return X

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> 'RegresorCatBoost':
        """Detecta las categoricas y entrena un CatBoostRegressor nativo."""
        from catboost import CatBoostRegressor  # dependencia solo de este modelo
        self.categoricas_ = list(X.select_dtypes(include=['category', 'object']).columns)
        self.modelo_ = CatBoostRegressor(
            loss_function=self.loss_function, iterations=self.iterations, learning_rate=self.learning_rate, depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg, random_strength=self.random_strength,
            bagging_temperature=self.bagging_temperature, random_seed=self.random_state, thread_count=-1,
            task_type=self.task_type, verbose=False, allow_writing_files=False)
        X, y = self._preparar(X), np.asarray(y)
        if self.early_stopping_rounds is None:
            self.modelo_.fit(X, y, cat_features=self.categoricas_)
            return self
        X_ent, X_val, y_ent, y_val = train_test_split(
            X, y, test_size=self.validacion_fraccion, random_state=self.random_state)
        self.modelo_.fit(X_ent, y_ent, cat_features=self.categoricas_, eval_set=(X_val, y_val),
                         early_stopping_rounds=self.early_stopping_rounds, use_best_model=True)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predice log(price) con el CatBoost entrenado."""
        return self.modelo_.predict(self._preparar(X))


class PreparadorMixto(BaseEstimator, TransformerMixin):
    """Tabla mixta para FAMD y PCAmix: numericas en float y categoricas (binarias incluidas) como texto.

    Mismo tratamiento que preprocesador_lineal en lo que no es codificacion: barrio con TargetEncoder (una
    columna numerica), minimum_nights recortada y en log, conteos en log, mediana en las variables con nulos y
    su indicador de nulo. La diferencia es que las binarias y las categoricas no se pasan a one-hot escalado:
    se dejan como categorias para que FAMD/PCAmix les den su propio peso (1 / proporcion de la modalidad).
    columnas: variables del experimento.
    """

    def __init__(self, columnas: list[str]):
        """Guarda las columnas con las que se va a trabajar."""
        self.columnas = columnas

    def _partes(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Separa las columnas en numericas y categoricas (el barrio va aparte)."""
        categoricas = [c for c in self.columnas
                       if c in CATEGORICAS or X[c].dtype == bool or isinstance(X[c].dtype, pd.CategoricalDtype)]
        categoricas = [c for c in categoricas if c != COL_BARRIO]
        numericas = [c for c in self.columnas if c not in categoricas and c != COL_BARRIO]
        return numericas, categoricas

    def _tabla(self, X: pd.DataFrame, barrio: np.ndarray | None) -> pd.DataFrame:
        """Construye la tabla mixta: numericas (con indicador de nulo), categoricas y barrio codificado."""
        partes = {}
        for col in self.numericas_:
            valores = X[col].astype(float)
            if col in self.medianas_:
                partes[f'{col}_nulo'] = valores.isna().map({True: 'si', False: 'no'})
                valores = valores.fillna(self.medianas_[col])
            if col == 'minimum_nights':
                valores = pd.Series(_log_estancia(valores), index=X.index)
            elif col in CONTEOS:
                valores = np.log1p(valores)
            partes[col] = valores
        for col in self.categoricas_:
            partes[col] = X[col].astype(str)
        if barrio is not None:
            partes[COL_BARRIO] = pd.Series(np.ravel(barrio), index=X.index)
        # Los indicadores de nulo son texto, asi que FAMD y PCAmix los tratan como categoricas
        return pd.DataFrame(partes, index=X.index)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> 'PreparadorMixto':
        """Ajusta el preparador con X e y."""
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | np.ndarray, **_) -> pd.DataFrame:
        """Ajusta y transforma el train; el barrio se codifica con cross-fitting para no usar su propio precio."""
        self.numericas_, self.categoricas_ = self._partes(X)
        self.medianas_ = {c: float(X[c].median()) for c in self.numericas_ if X[c].isna().any()}
        barrio = None
        if COL_BARRIO in self.columnas:
            self.codificador_ = _codificador_barrio()
            # fit_transform hace el cross-fitting interno: el barrio de un anuncio no usa su propio precio
            barrio = self.codificador_.fit_transform(X[[COL_BARRIO]], y)
        return self._tabla(X, barrio)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforma datos nuevos con lo aprendido en fit."""
        barrio = self.codificador_.transform(X[[COL_BARRIO]]) if COL_BARRIO in self.columnas else None
        return self._tabla(X, barrio)


class FAMDPrince(BaseEstimator, TransformerMixin):
    """FAMD de la libreria prince (Pages, 2004) con la interfaz de scikit-learn, sobre la salida de PreparadorMixto.

    Las columnas float son numericas (se estandarizan) y el resto, categoricas (one-hot centrado y dividido por
    la raiz de la proporcion de cada modalidad). Devuelve las coordenadas de las n_components primeras
    componentes. Aviso: prince recalcula las proporciones de las modalidades con los datos que transforma, no
    con las del ajuste, asi que la transformacion de un anuncio depende del resto del lote (ver PCAmix).
    """

    def __init__(self, n_components: int = 2, random_state: int = SEMILLA):
        """Guarda el numero de componentes y la semilla."""
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y=None) -> 'FAMDPrince':
        """Ajusta el FAMD de prince, que admite variables numericas y categoricas."""
        import prince  # dependencia solo de este modelo
        self.famd_ = prince.FAMD(n_components=self.n_components, random_state=self.random_state,
                                 handle_unknown='ignore')
        self.famd_.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Coordenadas de cada fila en las componentes del FAMD."""
        return self.famd_.row_coordinates(X).to_numpy()


class PCAmix(BaseEstimator, TransformerMixin):
    """PCAmix (Chavent et al., 2014), sobre la salida de PreparadorMixto: equivale a FAMD.

    Numericas estandarizadas e indicadores de cada modalidad centrados y divididos por la raiz de su
    proporcion; despues, descomposicion en valores singulares. A diferencia de FAMDPrince, la transformacion de
    datos nuevos usa las medias, desviaciones y proporciones del ajuste, como cualquier transformador de
    scikit-learn. n_components: None guarda todas las componentes (para ver la varianza explicada).
    """

    def __init__(self, n_components: int | None = None):
        """Guarda el numero de componentes (None: todas)."""
        self.n_components = n_components

    def _matriz(self, X: pd.DataFrame) -> np.ndarray:
        """Matriz estandarizada: numericas centradas y reducidas, e indicadores ponderados por la proporcion de cada modalidad."""
        numericas = (X[self.numericas_].to_numpy(dtype=float) - self.medias_) / self.desviaciones_
        indicadores = np.column_stack([(X[col].to_numpy() == modalidad).astype(float)
                                       for col, modalidad in self.modalidades_])
        return np.hstack([numericas, (indicadores - self.proporciones_) / np.sqrt(self.proporciones_)])

    def fit(self, X: pd.DataFrame, y=None) -> 'PCAmix':
        """Aprende medias, desviaciones y proporciones de las modalidades y hace el SVD de la matriz estandarizada."""
        self.numericas_ = list(X.select_dtypes(include='float').columns)
        categoricas = [c for c in X.columns if c not in self.numericas_]
        self.medias_ = X[self.numericas_].mean().to_numpy()
        self.desviaciones_ = X[self.numericas_].std(ddof=0).to_numpy()
        self.modalidades_ = [(col, modalidad) for col in categoricas for modalidad in sorted(X[col].unique())]
        self.proporciones_ = np.array([(X[col] == modalidad).mean() for col, modalidad in self.modalidades_])
        self.nombres_ = self.numericas_ + [f'{col}={modalidad}' for col, modalidad in self.modalidades_]
        _, valores_singulares, vt = np.linalg.svd(self._matriz(X), full_matrices=False)
        autovalores = valores_singulares ** 2 / len(X)
        self.explained_variance_ratio_ = autovalores / autovalores.sum()
        self.components_ = vt[:self.n_components]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Proyecta filas nuevas sobre las componentes aprendidas."""
        return self._matriz(X) @ self.components_.T


def importancia_permutacion(modelo: BaseEstimator, X: pd.DataFrame, y: pd.Series,
                            grupos: dict[str, list[str]], n_repeticiones: int = 5) -> pd.DataFrame:
    """Aumento del RMSE en log al permutar a la vez las columnas de cada grupo (bloque o variable).

    modelo: ya ajustado. X, y: datos de validacion, que el modelo no ha visto.
    grupos: {nombre: [columnas]}; las columnas de un grupo se permutan con la misma permutacion, asi que se
        rompe su relacion con el precio y con el resto, pero no entre ellas (p. ej. latitude y longitude).
    Devuelve una fila por grupo y repeticion con el RMSE base, el permutado y el aumento.
    """
    rng = np.random.default_rng(SEMILLA)
    rmse_base = root_mean_squared_error(y, modelo.predict(X))
    filas = []
    for nombre, columnas in grupos.items():
        for repeticion in range(n_repeticiones):
            orden = rng.permutation(len(X))
            permutado = X.copy()
            for col in columnas:
                # set_axis conserva el dtype (category incluido) y alinea con el indice de X
                permutado[col] = X[col].iloc[orden].set_axis(X.index)
            rmse = root_mean_squared_error(y, modelo.predict(permutado))
            filas.append({'grupo': nombre, 'repeticion': repeticion, 'rmse_base': rmse_base,
                          'rmse_permutado': rmse, 'aumento_rmse': rmse - rmse_base})
    return pd.DataFrame(filas)


def importancia_en_folds(experimento: Experimento, datos: pd.DataFrame, folds: list[Fold],
                         grupos: dict[str, list[str]], n_repeticiones: int = 5) -> pd.DataFrame:
    """importancia_permutacion en la validacion de cada fold, reentrenando un clon en su train.

    Los grupos se recortan a las columnas del experimento y se descartan los que quedan vacios.
    Devuelve las filas de importancia_permutacion con el experimento y el fold.
    """
    grupos = {nombre: [c for c in cols if c in experimento.columnas] for nombre, cols in grupos.items()}
    grupos = {nombre: cols for nombre, cols in grupos.items() if cols}
    X, y = datos[experimento.columnas], datos[TARGET]
    tablas = []
    for f in folds:
        modelo = clone(experimento.modelo).fit(X.iloc[f.train], y.iloc[f.train])
        tabla = importancia_permutacion(modelo, X.iloc[f.validacion], y.iloc[f.validacion], grupos, n_repeticiones)
        tablas.append(tabla.assign(experimento=experimento.nombre, fold=f.fold))
    return pd.concat(tablas, ignore_index=True)


MIN_FOLDS_MEJORA = 12


def comparar_pareado(resultados_con: pd.DataFrame, resultados_sin: pd.DataFrame,
                     metrica: str = 'rmse_log_val') -> dict[str, float | str | bool]:
    """Delta = metrica_sin - metrica_con por fold (positivo: la variable ayuda).

    Entra si Delta > 0 en al menos 12 de 15 folds (80% si hay otro numero) y el Delta medio supera su
    desviacion tipica. El p del t-test corregido de Nadeau y Bengio (2003) es solo informativo.
    resultados_con, resultados_sin: salidas de evaluar con los mismos folds; se emparejan por (repeticion, fold).
    metrica: columna a comparar; debe ser una metrica de error, i.e. menor es mejor.
    """
    clave = ['repeticion', 'fold']
    pares = resultados_con[clave + [metrica]].merge(resultados_sin[clave + [metrica]], on=clave,
                                                    suffixes=('_con', '_sin'))
    delta = (pares[f'{metrica}_sin'] - pares[f'{metrica}_con']).to_numpy()
    k = len(delta)
    media, sd = delta.mean(), delta.std(ddof=1)
    # Los folds comparten train: la varianza se infla con n_validacion / n_train = 1 / (N_FOLDS - 1)
    t = media / np.sqrt((1 / k + 1 / (N_FOLDS - 1)) * sd ** 2)
    n_mejora = int((delta > 0).sum())
    umbral = MIN_FOLDS_MEJORA if k == 15 else int(np.ceil(0.8 * k))
    return {
        'delta_medio': media,
        'delta_sd': sd,
        'folds_mejora': f'{n_mejora}/{k}',
        'p_nadeau_bengio': float(stats.t.sf(t, df=k - 1)),
        'entra': bool(n_mejora >= umbral and media > sd),
    }


def evaluar_intervalo(nombre: str, modelos_por_cuantil: dict[float, BaseEstimator], columnas: list[str],
                      datos: pd.DataFrame, folds: list[Fold]) -> pd.DataFrame:
    """Regresion cuantilica: cobertura y anchura del intervalo entre el cuantil menor y el mayor, y pinball loss.

    nombre: identificador del experimento en la tabla devuelta.
    modelos_por_cuantil: {alfa: modelo sin ajustar}. La cobertura nominal es alfa_max - alfa_min.
    Devuelve una fila por fold; no escribe en el registro.
    """
    X, y = datos[columnas], datos[TARGET]
    bajo, alto = min(modelos_por_cuantil), max(modelos_por_cuantil)
    filas = []
    for f in folds:
        y_val = y.iloc[f.validacion].to_numpy()
        pred = {alfa: clone(modelo).fit(X.iloc[f.train], y.iloc[f.train]).predict(X.iloc[f.validacion])
                for alfa, modelo in modelos_por_cuantil.items()}
        fila = {
            'experimento': nombre, 'repeticion': f.repeticion, 'fold': f.fold,
            'cobertura_nominal': alto - bajo,
            'cobertura': cobertura_intervalo(y_val, pred[bajo], pred[alto]),
            'cuantiles_cruzados': np.mean(pred[bajo] > pred[alto]),
            'anchura_log_mediana': np.median(pred[alto] - pred[bajo]),
            'anchura_usd_mediana': np.median(np.exp(pred[alto]) - np.exp(pred[bajo])),
        }
        for alfa, p in pred.items():
            fila[f'pinball_{alfa}'] = mean_pinball_loss(y_val, p, alpha=alfa)
        filas.append(fila)
    return pd.DataFrame(filas)


# --- Constructores de experimentos --------------------------------------------------------------------------------
# Cada funcion devuelve un Experimento (nombre, modelo sin ajustar, columnas) listo para evaluar_registrado.

COORDENADAS = ['latitude', 'longitude']
TASAS = ['host_response_rate', 'host_acceptance_rate', 'host_response_time']
REDUCTORES_MIXTOS = {'K2 FAMD': ('famd', FAMDPrince), 'K3 PCAmix': ('pcamix', PCAmix)}


def sin(columnas: list[str], quitar: list[str]) -> list[str]:
    """Las columnas sin las indicadas, en el mismo orden."""
    return [c for c in columnas if c not in set(quitar)]


def lineal(nombre: str, columnas: list[str], modelo: BaseEstimator | None = None, splines: bool = False,
           interacciones: bool = False) -> Experimento:
    """Modelo lineal (OLS si no se da otro estimador) con el preprocesador lineal, con interacciones y splines opcionales."""
    modelo = LinearRegression() if modelo is None else modelo
    return Experimento(nombre, make_pipeline(preprocesador_lineal(columnas, splines, interacciones), modelo),
                       list(columnas))


def referencia_ridge(nombre: str, columnas: list[str]) -> Experimento:
    """Ridge con alpha = 1, interacciones y splines: el modelo de referencia de la seleccion de variables."""
    return lineal(f'F4_ridge_{nombre}', columnas, Ridge(alpha=1.0), splines=True, interacciones=True)


def lineal_elegido(nombre: str, columnas: list[str], elegido: pd.Series) -> Experimento:
    """El lineal elegido (Ridge, o ElasticNet si trae l1_ratio) con sus hiperparametros, sobre otras columnas."""
    if pd.isna(elegido.get('l1_ratio', np.nan)):
        estimador = Ridge(alpha=elegido['alpha'])
    else:
        estimador = ElasticNet(alpha=elegido['alpha'], l1_ratio=elegido['l1_ratio'], max_iter=5000)
    return lineal(nombre, columnas, estimador, splines=True, interacciones=True)


def construir_rf(nombre: str, params: dict, columnas: list[str], n_estimators: int = 150) -> Experimento:
    """Random Forest con el preprocesador ordinal; params son los hiperparametros a ajustar."""
    return Experimento(nombre, make_pipeline(preprocesador_ordinal(columnas),
                                             RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1,
                                                                   random_state=SEMILLA, **params)), columnas)


def construir_hgb(nombre: str, params: dict, columnas: list[str]) -> Experimento:
    """HistGradientBoosting con el preprocesador de arboles (barrio con TargetEncoder) y sin parada temprana."""
    return Experimento(nombre, make_pipeline(preprocesador_arboles(columnas),
                                             HistGradientBoostingRegressor(early_stopping=False,
                                                                           random_state=SEMILLA, **params)),
                       columnas)


def construir_svr(nombre: str, gamma: float, C: float, columnas: list[str]) -> Experimento:
    """SVR aproximado: kernel RBF con Nystroem (1.000 componentes) y LinearSVR, tras el preprocesador lineal."""
    return Experimento(nombre, make_pipeline(preprocesador_lineal(columnas),
                                             Nystroem(gamma=gamma, n_components=1000, random_state=SEMILLA),
                                             LinearSVR(C=C, epsilon=0.1, loss='squared_epsilon_insensitive',
                                                       dual=False, max_iter=5000, random_state=SEMILLA)),
                       columnas)


def construir_catboost(nombre: str, params: dict, columnas: list[str], dispositivo: str = 'CPU',
                       fijos: dict = FIJOS_CATBOOST) -> Experimento:
    """CatBoost con las categoricas nativas. `fijos` no se busca; el base pasa fijos={} y usa los valores por defecto."""
    return Experimento(nombre, RegresorCatBoost(task_type=dispositivo, **{**fijos, **params}), columnas)


def knn_mixto(variante: str, k: int, componentes: int, columnas: list[str]) -> Experimento:
    """kNN con pesos por distancia sobre las componentes de FAMD o PCAmix (variante de REDUCTORES_MIXTOS)."""
    sufijo, reductor = REDUCTORES_MIXTOS[variante]
    return Experimento(f"{variante.split()[0]}_knn_{sufijo}{componentes}_k{k}",
                       make_pipeline(PreparadorMixto(columnas), reductor(componentes),
                                     KNeighborsRegressor(k, weights='distance')), columnas)


def fila_knn(variante: str, k: int, componentes: int, experimento: Experimento, resultados: pd.DataFrame) -> dict:
    """Fila resumen (RMSE medio y desviacion entre folds) de un kNN para la tabla comparativa."""
    return {'variante': variante, 'k': k, 'weights': 'distance', 'componentes': componentes,
            'experimento': experimento.nombre, 'rmse_log_val': resultados['rmse_log_val'].mean(),
            'rmse_log_val_sd': resultados['rmse_log_val'].std(), 'rmse_log_train': resultados['rmse_log_train'].mean()}


def filas_registro(experimento: str, ruta: Path) -> pd.DataFrame:
    """Las filas de un experimento del registro en la repeticion 0, para comparaciones pareadas."""
    return pd.read_csv(ruta).query('experimento == @experimento and repeticion == 0')


# --- Seleccion de variables (regla D4) ----------------------------------------------------------------------------

def seleccionar_variables(modelo: str, tabla_pruebas: pd.DataFrame, columnas: list[str],
                          bloques: dict[str, list[str]]) -> tuple[list[str], list[str]]:
    """Aplica la regla D4 a las pruebas A1-A12 de un modelo y devuelve (variables que se quedan, variables fuera).

    tabla_pruebas: una fila por prueba con 'modelo', 'prueba', 'que_se_prueba', 'entra' y 'delta_medio'.
    bloques: los bloques de variables (B1_ubicacion y B3_amenities_comunes se usan por su nombre).
    """
    def prueba(id_prueba, texto):
        fila = tabla_pruebas[(tabla_pruebas['modelo'] == modelo) & (tabla_pruebas['prueba'] == id_prueba)
                             & (tabla_pruebas['que_se_prueba'] == texto)]
        assert len(fila) == 1, (modelo, id_prueba, texto)
        return fila.iloc[0]

    def entra(id_prueba, texto):
        return bool(prueba(id_prueba, texto)['entra'])

    def delta(id_prueba, texto):
        return prueba(id_prueba, texto)['delta_medio']

    def forma_elegida(id_prueba, opciones, conjunta):
        """A4 y A7: las dos formas solo si ganan a cada forma suelta; si no, la suelta que entra con mayor Delta."""
        if all(entra(id_prueba, texto_conjunta) for texto_conjunta in conjunta.values()):
            return [col for cols in opciones.values() for col in cols]
        candidatas = {texto: cols for texto, cols in opciones.items() if entra(id_prueba, texto)}
        if not candidatas:
            return []
        mejor = max(candidatas, key=lambda texto: delta(id_prueba, texto))
        return candidatas[mejor]

    quitar = set()
    simples = {'A1': ('es_estudio', ['es_estudio']),
               'A2': ('property_type_grp', ['property_type_grp']),
               'A6': ('instant_bookable', ['instant_bookable']),
               'A8': ('reputacion_cartera', ['reputacion_cartera']),
               'A9': ('tasas de respuesta y aceptación', TASAS),
               'A12': ('bloque B3 (amenities comunes)', bloques['B3_amenities_comunes'])}
    for id_prueba, (texto, cols) in simples.items():
        if not entra(id_prueba, texto):
            quitar |= set(cols)
    for col in ['host_identity_verified', 'host_is_superhost', 'host_ambito', 'antiguedad_host_anios']:
        if not entra('A10', col):
            quitar.add(col)

    # A3: se sube por la cadena mientras cada escalon entra; las coordenadas, si ganan a district + barrio
    ubicacion = list(COORDENADAS) if entra('A3', 'coordenadas sobre district + barrio') else []
    for texto, col in [('+ dist_centro_km sobre lat/lon', 'dist_centro_km'), ('+ district', 'district'),
                       ('+ barrio', 'neighbourhood')]:
        if not entra('A3', texto):
            break
        ubicacion.append(col)
    if not ubicacion:
        ubicacion = ['district', 'neighbourhood']
    quitar |= set(bloques['B1_ubicacion']) - set(ubicacion)

    estancia = forma_elegida('A4',
                             {'minimum_nights en bruto (frente a fuera)': ['minimum_nights'],
                              'estancia_min_30 (frente a fuera)': ['estancia_min_30']},
                             {'bruto': 'las dos frente a solo en bruto',
                              'min_30': 'las dos frente a solo estancia_min_30'})
    quitar |= {'minimum_nights', 'estancia_min_30'} - set(estancia)
    cartera = forma_elegida('A7',
                            {'n_anuncios_ny (frente a ninguna)': ['n_anuncios_ny'],
                             'host_total_listings_count (frente a ninguna)': ['host_total_listings_count']},
                            {'n': 'las dos frente a solo n_anuncios_ny',
                             'total': 'las dos frente a solo total_listings'})
    quitar |= {'n_anuncios_ny', 'host_total_listings_count'} - set(cartera)
    return [c for c in columnas if c not in quitar], sorted(quitar)


# --- Lectura de coeficientes del lineal ---------------------------------------------------------------------------

def variable_original(columna: str, columnas: list[str]) -> str:
    """Variable de `columnas` de la que sale una columna del preprocesador lineal ('interacciones' si lo es)."""
    nombre = re.sub(r'^\w+?__', '', columna).removeprefix('missingindicator_')
    if nombre.startswith(INTERACCION_DISTRITO) or '_x_' in nombre:
        return 'interacciones'
    return max((c for c in columnas if nombre == c or nombre.startswith(c + '_')), key=len)


def anuladas(coeficientes: pd.Series, columnas: list[str]) -> set[str]:
    """Variables originales cuyas columnas tienen todas el coeficiente a cero (Lasso o ElasticNet las anulan del todo)."""
    usada = (coeficientes != 0).groupby(coeficientes.index.map(lambda c: variable_original(c, columnas))).any()
    return set(usada[~usada].index)
