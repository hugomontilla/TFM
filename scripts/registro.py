"""Acceso al registro de experimentos (outputs/experimentos.csv): una fila por experimento, repeticion y fold.

Es el unico modulo que lee y escribe ese CSV. func_modelado lo usa para guardar y reutilizar resultados, y
func_final para recuperar los hiperparametros y las variables del modelo elegido.
"""
from datetime import datetime
from pathlib import Path

import pandas as pd


def registrar(resultados: pd.DataFrame, ruta: Path) -> None:
    """Añade los resultados al registro; si el experimento ya existia, sustituye sus filas para que re-ejecutar no duplique.

    resultados: salida de evaluar. 
    ruta: CSV del registro (outputs/experimentos.csv).
    """
    resultados = resultados.assign(fecha=datetime.now().isoformat(timespec='seconds'))
    if ruta.exists():
        previos = pd.read_csv(ruta)
        previos = previos[~previos['experimento'].isin(resultados['experimento'].unique())]
        resultados = pd.concat([previos, resultados], ignore_index=True)
    # Se escribe en un temporal y se sustituye de una vez: si el proceso muere a mitad, el registro no se trunca
    temporal = ruta.with_suffix('.tmp')
    resultados.to_csv(temporal, index=False)
    temporal.replace(ruta)


def valor_unico(nombre: str, ruta: Path, columna: str) -> str:
    """Valor de `columna` para un experimento del registro; falla si sus filas no coinciden en ese valor."""
    registro = pd.read_csv(ruta, usecols=['experimento', columna])
    valores = registro.loc[registro['experimento'] == nombre, columna].unique()
    assert len(valores) == 1, f"'{nombre}' tiene {len(valores)} valores distintos de '{columna}' en el registro"
    return valores[0]


def leer_experimento(nombre: str, ruta: Path, folds: list) -> pd.DataFrame:
    """Filas de un experimento ya registrado, sin recalcular nada. Falla si falta o no cubre los folds.

    Sirve para leer en otro notebook un experimento ya ejecutado, sin recalcularlo.
    """
    previos = pd.read_csv(ruta) if ruta.exists() else pd.DataFrame(columns=['experimento'])
    filas = previos[previos['experimento'] == nombre]
    esperado = {(f.repeticion, f.fold) for f in folds}
    if set(zip(filas.get('repeticion', []), filas.get('fold', []))) != esperado:
        raise FileNotFoundError(f"Faltan resultados de '{nombre}' en {ruta.name}: ejecuta el experimento antes de leerlo")
    return filas.reset_index(drop=True)
