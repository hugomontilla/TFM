"""Particion train/test fija del dataset de Nueva York.

Se hace una sola vez, sobre los anuncios validos y antes de mirar el precio. Los ficheros guardados son la
fuente de verdad: la semilla solo sirve para regenerarla, porque el reparto de StratifiedGroupKFold depende
del orden de las filas y de la version de scikit-learn.

- Agrupada por host_id: los anuncios de un anfitrion comparten su perfil y muchos tienen un gemelo en la cartera.
- Estratificada por room_type x district: equilibra las dos variables a la vez.

Uso desde la raiz del proyecto:
    python scripts/particion_datos.py [--forzar]

El estrato se eligio con estudio_estratificacion.py.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from preparacion_datos import DATA_DIR, cargar_listings_ny, limpiar_registros

SEMILLA = 42
N_PARTES = 5  # un fold de cinco como test: el 20%
COL_GRUPO = 'host_id'
COLS_ESTRATO = ['room_type', 'district']

DIR_PARTICION = DATA_DIR / 'particion'
RUTAS = {
    'train': DIR_PARTICION / 'listings_train.csv',
    'test': DIR_PARTICION / 'listings_test.csv',
    'manifiesto': DIR_PARTICION / 'particion.csv',
}


def estrato(datos, columnas=COLS_ESTRATO):
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
    listings[['listing_id', COL_GRUPO]].assign(conjunto=np.where(es_test, 'test', 'train')).to_csv(
        RUTAS['manifiesto'], index=False)
    return train, test


def cargar_conjunto(nombre):
    """Filas en bruto de listings.csv de 'train' o 'test'."""
    return pd.read_csv(RUTAS[nombre], low_memory=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--listings', default=DATA_DIR / 'listings.csv')
    parser.add_argument('--forzar', action='store_true', help='sobrescribe una particion existente')
    args = parser.parse_args()

    listings = limpiar_registros(cargar_listings_ny(args.listings))
    train, test = crear_particion(listings, forzar=args.forzar)
    print(f'Train: {len(train):,} anuncios de {train[COL_GRUPO].nunique():,} anfitriones')
    print(f'Test:  {len(test):,} anuncios de {test[COL_GRUPO].nunique():,} anfitriones')
