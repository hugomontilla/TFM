"""Parametros comunes del experimento, leidos de .env en la raiz del proyecto.

Un solo sitio para los valores que usa todo el trabajo (semilla, folds, cobertura del intervalo...). Los scripts los
importan de aqui y los notebooks los reciben a traves de ellos. Si falta una clave en .env se para con un error claro
en vez de usar un valor por defecto escondido.
"""
from pathlib import Path

from dotenv import dotenv_values

RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
_ENV = dotenv_values(RAIZ_PROYECTO / '.env')


def _leer(clave, tipo):
    """Valor de `clave` en .env convertido con `tipo`; falla si no esta."""
    if _ENV.get(clave) is None:
        raise KeyError(f"Falta {clave} en {RAIZ_PROYECTO / '.env'}")
    return tipo(_ENV[clave])


SEMILLA = _leer('SEMILLA', int)
N_PARTES = _leer('N_PARTES', int)
N_FOLDS = _leer('N_FOLDS', int)
N_REPETICIONES_CV = _leer('N_REPETICIONES_CV', int)
COBERTURA_NOMINAL = _leer('COBERTURA_NOMINAL', float)
N_REPLICAS_BOOTSTRAP = _leer('N_REPLICAS_BOOTSTRAP', int)
