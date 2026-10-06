"""Particion train/test fija del dataset de Nueva York.

Se hace una sola vez, sobre los anuncios validos y antes de mirar el precio. Los ficheros guardados son la
fuente de verdad: la semilla solo sirve para regenerarla, porque el reparto de StratifiedGroupKFold depende
del orden de las filas y de la version de scikit-learn.

- Agrupada por host_id: los anuncios de un anfitrion comparten su perfil y muchos tienen un gemelo en la cartera.
- Estratificada por room_type x district: equilibra las dos variables a la vez.

Guarda tres ficheros en data/particion/: listings_train.csv, listings_test.csv y particion.json (semilla, recuentos
y MD5 de los dos CSV). Los notebooks finales comprueban el MD5 del test contra particion.json antes de abrirlo.

Uso desde la raiz del proyecto:
    python scripts/particion_datos.py [--forzar]
    python scripts/particion_datos.py --metadatos    # recalcula particion.json desde los CSV ya guardados

El estrato se eligio con estudio_estratificacion.py.
"""
import argparse
import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from config import N_PARTES, SEMILLA
from preparacion_datos import DATA_DIR, cargar_listings_ny, limpiar_registros

COL_GRUPO = 'host_id'
COLS_ESTRATO = ['room_type', 'district']

DIR_PARTICION = DATA_DIR / 'particion'
RUTAS = {
    'train': DIR_PARTICION / 'listings_train.csv',
    'test': DIR_PARTICION / 'listings_test.csv',
}
RUTA_METADATOS = DIR_PARTICION / 'particion.json'
FILTRO_PREVIO = 'city == New York, accommodates > 0, price > 0 (limpiar_registros)'
METODO = f'StratifiedGroupKFold(n_splits={N_PARTES}, shuffle=True), primer fold como test'


def estrato(datos, columnas=COLS_ESTRATO):
    """Etiqueta 'room_type | district' de cada anuncio, que sirve para estratificar la particion y los folds."""
    return datos[columnas].astype(str).agg(' | '.join, axis=1)


def mascara_test(listings, y, semilla):
    """Mascara booleana del primer fold de StratifiedGroupKFold."""
    divisor = StratifiedGroupKFold(n_splits=N_PARTES, shuffle=True, random_state=semilla)
    # Cogemos el primer fold como nuestra particion
    _, idx_test = next(divisor.split(listings, y, listings[COL_GRUPO]))
    es_test = np.zeros(len(listings), dtype=bool)
    es_test[idx_test] = True
    return es_test


def crear_particion(listings, forzar=False):
    """Reparte los anuncios en train y test y los guarda. No sobrescribe una particion existente salvo con forzar."""
    if RUTAS['train'].exists() and not forzar:
        raise FileExistsError(f'Ya existe una particion en {DIR_PARTICION}. Usa --forzar solo si hay que sobreescribirla.')
    es_test = mascara_test(listings, estrato(listings), SEMILLA)
    train, test = listings[~es_test], listings[es_test]
    # Confirmamos la agrupacion por host_id
    assert set(train[COL_GRUPO]).isdisjoint(test[COL_GRUPO]), 'Hay anfitriones en train y en test'
    # Confirmamos que un anuncio no aparece dos veces
    assert listings['listing_id'].is_unique

    DIR_PARTICION.mkdir(parents=True, exist_ok=True)
    train.to_csv(RUTAS['train'], index=False)
    test.to_csv(RUTAS['test'], index=False)
    escribir_metadatos(train, test)
    return train, test


def md5_fichero(ruta):
    """Huella MD5 de un fichero, para comprobar que no ha cambiado."""
    return hashlib.md5(ruta.read_bytes()).hexdigest()


def escribir_metadatos(train, test, fecha=None):
    """Escribe particion.json: como se hizo la particion y el MD5 de los dos CSV guardados."""
    import pandas
    import sklearn
    metadatos = {
        'fecha': fecha or datetime.now().isoformat(timespec='seconds'),
        'semilla': SEMILLA,
        'metodo': METODO,
        'grupo': COL_GRUPO,
        'estrato': ' x '.join(COLS_ESTRATO),
        'filtro_previo': FILTRO_PREVIO,
        'version_sklearn': sklearn.__version__,
        'version_pandas': pandas.__version__,
        'n_anuncios': {'train': len(train), 'test': len(test)},
        'n_anfitriones': {'train': int(train[COL_GRUPO].nunique()), 'test': int(test[COL_GRUPO].nunique())},
        'md5': {nombre: md5_fichero(RUTAS[nombre]) for nombre in ('train', 'test')},
    }
    RUTA_METADATOS.write_text(json.dumps(metadatos, indent=2, ensure_ascii=False), encoding='utf-8')


def cargar_conjunto(nombre):
    """Filas en bruto de listings.csv de 'train' o 'test'."""
    return pd.read_csv(RUTAS[nombre], low_memory=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--listings', default=DATA_DIR / 'listings.csv')
    parser.add_argument('--forzar', action='store_true', help='sobrescribe una particion existente')
    parser.add_argument('--metadatos', action='store_true',
                        help='recalcula particion.json a partir de los CSV ya guardados, sin repartir de nuevo')
    args = parser.parse_args()

    if args.metadatos:
        fecha_original = json.loads(RUTA_METADATOS.read_text(encoding='utf-8'))['fecha'] if RUTA_METADATOS.exists() else None
        escribir_metadatos(cargar_conjunto('train'), cargar_conjunto('test'), fecha_original)
        print(f'Metadatos escritos en {RUTA_METADATOS}')
        raise SystemExit

    listings = limpiar_registros(cargar_listings_ny(args.listings))
    train, test = crear_particion(listings, forzar=args.forzar)
    print(f'Train: {len(train):,} anuncios de {train[COL_GRUPO].nunique():,} anfitriones')
    print(f'Test:  {len(test):,} anuncios de {test[COL_GRUPO].nunique():,} anfitriones')
