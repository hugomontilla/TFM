"""Estudio para elegir por que variables estratificar la particion train/test.

Simula muchas particiones agrupadas por host_id con cada forma de estratificar y mide cuanto se desequilibra
el test frente al train. Solo mira variables explicativas, nunca el precio. La particion definitiva se crea
despues, con el estrato elegido, en particion_datos.py.

Uso desde la raiz del proyecto (~30 min):
    python scripts/estudio_estratificacion.py [--semillas 50]
"""
import argparse

import numpy as np
import pandas as pd

from particion_datos import estrato, mascara_test
from preparacion_datos import DATA_DIR, cargar_listings_ny, limpiar_registros

RUTA_SIMULACION = DATA_DIR.parent / 'outputs' / 'simulacion_estratos.csv'


def desviacion_maxima(categoria, es_test):
    """Mayor diferencia, en puntos porcentuales, entre el peso de una categoria en train y en test."""
    en_train = categoria[~es_test].value_counts(normalize=True)
    en_test = categoria[es_test].value_counts(normalize=True)
    return en_train.subtract(en_test, fill_value=0).abs().max() * 100


def simular_estrategias(listings, n_semillas):
    """Desequilibrio del test con cada forma de estratificar."""
    estrategias = {
        'sin estratificar': np.zeros(len(listings), dtype=int),
        'room_type': estrato(listings, ['room_type']),
        'district': estrato(listings, ['district']),
        'room_type x district': estrato(listings, ['room_type', 'district']),
    }

    filas = []
    for nombre, y in estrategias.items():
        for semilla in range(n_semillas):
            es_test = mascara_test(listings, y, semilla)
            test = listings[es_test]
            filas.append({
                'estrategia': nombre,
                'semilla': semilla,
                'pct_test': es_test.mean() * 100,
                'desv_room_type': desviacion_maxima(listings['room_type'], es_test),
                'desv_district': desviacion_maxima(listings['district'], es_test),
                'desv_celda': desviacion_maxima(estrategias['room_type x district'], es_test),
                'hotel_en_test': (test['room_type'] == 'Hotel room').sum(),
                'shared_en_test': (test['room_type'] == 'Shared room').sum(),
                'staten_en_test': (test['district'] == 'Staten Island').sum(),
            })
    return pd.DataFrame(filas)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--listings', default=DATA_DIR / 'listings.csv')
    parser.add_argument('--semillas', type=int, default=50, help='particiones por estrategia')
    args = parser.parse_args()

    listings = limpiar_registros(cargar_listings_ny(args.listings))
    resultados = simular_estrategias(listings, args.semillas)
    RUTA_SIMULACION.parent.mkdir(exist_ok=True)
    resultados.to_csv(RUTA_SIMULACION, index=False)
    print(resultados.drop(columns='semilla').groupby('estrategia').agg(['mean', 'max']).round(2).T)
