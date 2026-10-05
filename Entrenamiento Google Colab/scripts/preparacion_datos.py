"""Preparacion del dataset de modelado de Nueva York a partir de la particion de listings.csv.

Hay tres tipos de transformacion:
  - lo que se aprende del train (PreparadorListings.fit): amenities de la ventana de presencia, tipos de
    propiedad frecuentes y medianas de bedrooms. Se guarda en JSON y se reutiliza congelado;
  - lo que se calcula fila a fila o por anfitrion (transform). Como la particion agrupa por anfitrion,
    calcular los agregados por separado en train y test da lo mismo que sobre todo el snapshot;
  - las reglas que usan el precio (limpiar_registros, excluir_precios_bloqueo): solo al entrenar.

Uso desde la raiz del proyecto (antes hay que crear la particion con particion_datos.py):
    python scripts/preparacion_datos.py
"""
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry
from sklearn.base import BaseEstimator, TransformerMixin

CIUDAD = 'New York'
CITY_HALL = (40.7128, -74.0060)  # centro de referencia externo, nunca derivado del precio
# Ultima resena de reviews.csv: listings.csv no trae la fecha de descarga. Solo fija la escala de la antiguedad
FECHA_SNAPSHOT = '2021-02-05'
NOMBRES_NY = ['new york', 'manhattan', 'brooklyn', 'queens', 'bronx', 'staten island']

# Paises frecuentes; pycountry solo como respaldo porque es lento
PAISES_ISO = {
    'united states': 'US', 'usa': 'US', 'us': 'US', 'u.s.': 'US',
    'canada': 'CA', 'mexico': 'MX',
    'costa rica': 'CR', 'el salvador': 'SV', 'guatemala': 'GT',
    'honduras': 'HN', 'nicaragua': 'NI', 'panama': 'PA', 'panamá': 'PA',
    'colombia': 'CO', 'ecuador': 'EC', 'peru': 'PE', 'perú': 'PE',
    'bolivia': 'BO', 'venezuela': 'VE',
    'argentina': 'AR', 'chile': 'CL', 'paraguay': 'PY', 'uruguay': 'UY',
    'brasil': 'BR', 'brazil': 'BR',
    'cuba': 'CU', 'dominican republic': 'DO', 'república dominicana': 'DO',
    'puerto rico': 'PR',
    'united kingdom': 'GB', 'uk': 'GB', 'england': 'GB',
    'france': 'FR', 'spain': 'ES', 'italy': 'IT', 'germany': 'DE',
    'portugal': 'PT', 'netherlands': 'NL', 'belgium': 'BE', 'switzerland': 'CH',
    'australia': 'AU', 'japan': 'JP', 'china': 'CN', 'india': 'IN',
    'singapore': 'SG', 'thailand': 'TH', 'korea': 'KR', 'south korea': 'KR',
    'filipinas': 'PH',
}

# Ventana de presencia de amenities: criterio ciego al precio
MIN_PRESENCIA, MAX_PRESENCIA = 0.035, 0.97
MIN_ANUNCIOS_PROPERTY_TYPE = 100
REGEX_ESTUDIO = r'\b(?:studio|estudio)s?\b'
ORDEN_RESPUESTA = ['within an hour', 'within a few hours', 'within a day', 'a few days or more', 'sin_dato']
COLS_BOOL_ORIGEN = ['host_is_superhost', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
# Sin perfil de anfitrion no hay superhost ni verificacion; instant_bookable nulo es no activado
COLS_BOOL_NULO_FALSO = ['host_is_superhost', 'instant_bookable', 'host_identity_verified']

# Precios de bloqueo: no son tarifas de mercado (umbrales confirmados con el train en Analisis)
PRECIO_BLOQUEO = 9999
MAX_PRECIO_HABITACION = 1000
HABITACIONES = ['Private room', 'Shared room']

COLS_ID = ['listing_id', 'host_id']
COL_TARGET = 'price'
COLS_CATEGORICAS = ['room_type', 'property_type_grp', 'district', 'neighbourhood', 'host_ambito', 'host_response_time']
BOOLEANAS_FIJAS = ['es_estudio', 'estancia_min_30', 'instant_bookable', 'host_is_superhost', 'host_identity_verified']
# Solo pueden tener nulos las columnas en las que el nulo significa algo y lo trata el Pipeline
NULOS_ESPERADOS = {'antiguedad_host_anios', 'host_total_listings_count', 'host_response_rate',
                   'host_acceptance_rate', 'reputacion_cartera'}

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
RUTA_PREPARADOR = DATA_DIR / 'preparador_ny.json'
RUTAS_MODELADO = {
    'train': DATA_DIR / 'listings_ny_modelado_train.parquet',
    'test': DATA_DIR / 'listings_ny_modelado_test.parquet',
}


def haversine_km(lat1, lon1, lat2, lon2):
    """Distancia sobre la esfera terrestre, en kilometros."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin(np.radians(lat2 - lat1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def parsear_amenities(cadena):
    """Lista de amenities de la cadena cruda, en minusculas."""
    if not isinstance(cadena, str):
        return []
    interior = re.sub(r'^[\[{]|[\]}]$', '', cadena)
    return [a.strip().strip('"').lower() for a in interior.split(',') if a.strip()]


def crear_columna_amenity(amenity):
    return 'am_' + re.sub(r'[^a-z0-9]+', '_', amenity).strip('_')


@lru_cache(maxsize=None)
def iso_pais(pais_texto):
    """ISO-2 de un nombre de pais: primero el diccionario y, si no esta, busqueda aproximada en pycountry."""
    if pais_texto.lower() in PAISES_ISO:
        return PAISES_ISO[pais_texto.lower()]
    try:
        return pycountry.countries.search_fuzzy(pais_texto)[0].alpha_2
    except LookupError:
        return None


def ambito_host(host_location):
    """Donde vive el anfitrion. Cualquiera de los cinco distritos cuenta como misma ciudad."""
    misma_ciudad = host_location.fillna('').str.lower().str.contains('|'.join(NOMBRES_NY))
    # El ultimo elemento de host_location es casi siempre el pais
    iso = host_location.map(lambda t: iso_pais(t.split(',')[-1].strip()) if isinstance(t, str) else None)
    return pd.Series(np.select([misma_ciudad, iso == 'US', iso.notna()],
                               ['misma_ciudad', 'mismo_pais', 'extranjero'], default='desconocido'),
                     index=host_location.index)


def reputacion_cartera(listings):
    """Media de review_scores_rating de los OTROS anuncios del anfitrion (leave-one-out)."""
    puntuacion = listings['review_scores_rating']
    por_host = puntuacion.groupby(listings['host_id'])
    suma_otros = por_host.transform('sum', min_count=1).fillna(0) - puntuacion.fillna(0)
    n_otros = por_host.transform('count') - puntuacion.notna().astype(int)
    return pd.Series(np.where(n_otros > 0, suma_otros / n_otros, np.nan), index=listings.index)


def cargar_listings_ny(ruta_listings):
    listings = pd.read_csv(ruta_listings, low_memory=False)
    return listings[listings['city'] == CIUDAD].reset_index(drop=True)


def limpiar_registros(listings):
    """Solo al entrenar: sin capacidad no hay producto y sin precio no hay target."""
    validos = (listings['accommodates'] > 0) & (listings[COL_TARGET] > 0)
    return listings[validos].reset_index(drop=True)


def es_precio_bloqueo(datos):
    return (datos[COL_TARGET] >= PRECIO_BLOQUEO) | (
        datos['room_type'].isin(HABITACIONES) & (datos[COL_TARGET] > MAX_PRECIO_HABITACION))


def excluir_precios_bloqueo(datos):
    """Solo al entrenar. Quita tarifas que no son de mercado y las categorias que se quedan vacias."""
    datos = datos[~es_precio_bloqueo(datos)].reset_index(drop=True)
    for col in datos.select_dtypes('category'):
        datos[col] = datos[col].cat.remove_unused_categories()
    return datos


class PreparadorListings(BaseEstimator, TransformerMixin):
    """Construye las variables candidatas a partir de filas de listings.csv en bruto."""

    def fit(self, X, y=None):
        conjuntos = X['amenities'].map(lambda c: set(parsear_amenities(c)))
        presencia = (pd.Series(Counter(a for s in conjuntos for a in s)) / len(X)).sort_values(ascending=False)
        self.amenities_comunes_ = presencia[presencia.between(MIN_PRESENCIA, MAX_PRESENCIA)].index.tolist()

        frecuencia = X['property_type'].value_counts()
        self.property_types_frecuentes_ = frecuencia[frecuencia >= MIN_ANUNCIOS_PROPERTY_TYPE].index.tolist()

        self.medianas_bedrooms_ = X.groupby(['room_type', 'accommodates'])['bedrooms'].median().dropna()
        # Respaldo para combinaciones de room_type x accommodates que no aparecen al entrenar
        self.medianas_bedrooms_room_type_ = X.groupby('room_type')['bedrooms'].median()
        return self

    @property
    def bloques_(self):
        """Bloques de la ablacion (docs/plan_modelado.md)."""
        return {
            'B0_producto': ['room_type', 'property_type_grp', 'accommodates', 'bedrooms', 'es_estudio'],
            'B1_ubicacion': ['latitude', 'longitude', 'dist_centro_km', 'district', 'neighbourhood'],
            'B2a_condiciones': ['minimum_nights', 'estancia_min_30', 'instant_bookable'],
            'B2b_anfitrion': ['antiguedad_host_anios', 'n_anuncios_ny', 'host_total_listings_count',
                              'host_is_superhost', 'host_identity_verified', 'host_ambito', 'host_response_time',
                              'host_response_rate', 'host_acceptance_rate', 'reputacion_cartera'],
            'B3_amenities_comunes': [crear_columna_amenity(a) for a in self.amenities_comunes_],
            'texto': ['name'],
        }

    def transform(self, X):
        datos = X.copy()

        # Tipos
        datos['host_since'] = pd.to_datetime(datos['host_since'])
        for col in COLS_BOOL_ORIGEN:
            if not pd.api.types.is_bool_dtype(datos[col]):
                datos[col] = datos[col].map({'t': True, 'f': False}).astype('boolean')
        for col in COLS_BOOL_NULO_FALSO:
            datos[col] = datos[col].fillna(False)

        # Producto. es_estudio usa que bedrooms faltaba, asi que se calcula antes de imputar
        faltaba_bedrooms = datos['bedrooms'].isna()
        titulo_estudio = datos['name'].fillna('').str.contains(REGEX_ESTUDIO, case=False, regex=True)
        datos['es_estudio'] = (datos['room_type'] == 'Entire place') & (faltaba_bedrooms | titulo_estudio)
        clave = pd.MultiIndex.from_frame(datos[['room_type', 'accommodates']])
        por_producto = pd.Series(self.medianas_bedrooms_.reindex(clave).to_numpy(), index=datos.index)
        por_room_type = datos['room_type'].map(self.medianas_bedrooms_room_type_)
        datos['bedrooms'] = datos['bedrooms'].fillna(por_producto).fillna(por_room_type).round()
        # property_type mezcla tipo de estancia y de edificio; la cola se agrupa para no crear columnas casi vacias
        datos['property_type_grp'] = datos['property_type'].where(
            datos['property_type'].isin(self.property_types_frecuentes_), 'Otros')
        datos['estancia_min_30'] = datos['minimum_nights'] >= 30

        # Ubicacion
        datos['dist_centro_km'] = haversine_km(datos['latitude'], datos['longitude'], *CITY_HALL)

        # Anfitrion. Los agregados son sobre la cartera que el anfitrion tiene en este snapshot
        datos['antiguedad_host_anios'] = (pd.Timestamp(FECHA_SNAPSHOT) - datos['host_since']).dt.days / 365.25
        datos['n_anuncios_ny'] = datos.groupby('host_id')['listing_id'].transform('size') - 1
        datos['reputacion_cartera'] = reputacion_cartera(datos)
        datos['host_ambito'] = ambito_host(datos['host_location'])
        # El nulo es un anuncio sin actividad reciente y pide mas a igual producto (Analisis, 11.1): categoria propia
        datos['host_response_time'] = pd.Categorical(
            datos['host_response_time'].astype(object).fillna('sin_dato'), categories=ORDEN_RESPUESTA)

        # Amenities binarias
        conjuntos = datos['amenities'].map(lambda c: set(parsear_amenities(c)))
        amenities = pd.DataFrame({crear_columna_amenity(a): conjuntos.map(lambda s, a=a: a in s)
                                  for a in self.amenities_comunes_}, index=datos.index)
        datos = pd.concat([datos, amenities], axis=1)

        # Seleccion de columnas y tipos finales
        bloques = self.bloques_
        columnas = COLS_ID + [COL_TARGET] * (COL_TARGET in datos) + [c for b in bloques.values() for c in b]
        datos = datos[columnas].copy()
        for col in COLS_CATEGORICAS:
            datos[col] = datos[col].astype('category')
        # astype(bool) falla con nulos en un booleano nullable: si aparece alguno, mejor que rompa aqui
        for col in BOOLEANAS_FIJAS + bloques['B3_amenities_comunes']:
            datos[col] = datos[col].astype(bool)
        datos['name'] = datos['name'].fillna('').astype(str)
        return datos

    def guardar(self, ruta):
        """Parametros aprendidos en JSON legible, para reutilizarlos y citarlos en la memoria."""
        parametros = {
            'fecha_referencia': FECHA_SNAPSHOT,
            'amenities_comunes': self.amenities_comunes_,
            'property_types_frecuentes': self.property_types_frecuentes_,
            'medianas_bedrooms': [{'room_type': rt, 'accommodates': int(acc), 'bedrooms': float(med)}
                                  for (rt, acc), med in self.medianas_bedrooms_.items()],
            'medianas_bedrooms_room_type': self.medianas_bedrooms_room_type_.astype(float).to_dict(),
        }
        Path(ruta).write_text(json.dumps(parametros, indent=2, ensure_ascii=False), encoding='utf-8')

    @classmethod
    def cargar(cls, ruta):
        p = json.loads(Path(ruta).read_text(encoding='utf-8'))
        preparador = cls()
        preparador.amenities_comunes_ = p['amenities_comunes']
        preparador.property_types_frecuentes_ = p['property_types_frecuentes']
        preparador.medianas_bedrooms_ = pd.DataFrame(p['medianas_bedrooms']).set_index(
            ['room_type', 'accommodates'])['bedrooms']
        preparador.medianas_bedrooms_room_type_ = pd.Series(p['medianas_bedrooms_room_type'])
        return preparador


def validar_dataset_modelado(datos, bloques):
    """Si alguna comprobacion falla, el dataset no refleja las decisiones tomadas."""
    assert not es_precio_bloqueo(datos).any() and (datos[COL_TARGET] >= 1).all()
    assert datos['listing_id'].is_unique
    assert datos[COLS_ID + [COL_TARGET] + bloques['B1_ubicacion']].notna().all().all()
    cols_con_nulos = set(datos.columns[datos.isna().any()])
    assert cols_con_nulos <= NULOS_ESPERADOS, f'Nulos inesperados en: {cols_con_nulos - NULOS_ESPERADOS}'


def preparar_dataset_modelado(train, test):
    """El preparador aprende solo del train y se aplica igual a los dos conjuntos.

    Los agregados por anfitrion se calculan antes de excluir los precios de bloqueo: son atributos de toda
    su cartera en NY, no solo de los anuncios que entran al modelo.
    """
    assert set(train['host_id']).isdisjoint(test['host_id']), 'La particion debe agrupar por anfitrion'
    preparador = PreparadorListings().fit(limpiar_registros(train))
    conjuntos = {}
    for nombre, listings in [('train', train), ('test', test)]:
        conjuntos[nombre] = excluir_precios_bloqueo(preparador.transform(limpiar_registros(listings)))
        validar_dataset_modelado(conjuntos[nombre], preparador.bloques_)
    return conjuntos, preparador


def entrenar():
    from particion_datos import cargar_conjunto  # importacion local: particion_datos importa este modulo
    conjuntos, preparador = preparar_dataset_modelado(cargar_conjunto('train'), cargar_conjunto('test'))

    for nombre, datos in conjuntos.items():
        datos.to_parquet(RUTAS_MODELADO[nombre], index=False)
        print(f'{nombre}: {datos.shape[0]:,} anuncios x {datos.shape[1]} columnas -> {RUTAS_MODELADO[nombre].name}')
    (DATA_DIR / 'bloques_modelado.json').write_text(
        json.dumps(preparador.bloques_, indent=2, ensure_ascii=False), encoding='utf-8')
    preparador.guardar(RUTA_PREPARADOR)
    print(f'Preparador ajustado con el train -> {RUTA_PREPARADOR.name}')


if __name__ == '__main__':
    entrenar()
