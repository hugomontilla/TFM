"""Red neuronal (MLP) de la Fase 8: un regresor de PyTorch con la interfaz de scikit-learn.

Va en un Pipeline detras de su preprocesador como cualquier otro modelo, asi que clone, evaluar y el registro
de experimentos funcionan igual que con Ridge o HGB. Entrena en la GPU si la hay (Google Colab) y si no en la CPU.
"""
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from torch import nn

from func_modelado import COL_BARRIO, SEMILLA, preprocesador_lineal

# Barrios con menos anuncios en el train del fold comparten un unico vector de embedding
MIN_ANUNCIOS_BARRIO = 10
ACTIVACIONES = {'relu': nn.ReLU, 'silu': nn.SiLU}
TAMANO_LOTE_PREDICCION = 8192


def preprocesador_mlp(columnas: list[str], barrio: Literal['target', 'embedding'] = 'target') -> ColumnTransformer:
    """preprocesador_lineal sin splines ni interacciones: la red aprende sola la no linealidad.

    barrio: 'target' codifica neighbourhood con TargetEncoder, como para Ridge y SVR. 'embedding' lo deja como
        entero en la ultima columna para la capa de embedding de la red: -1 los barrios no vistos en el train
        del fold y un indice compartido los que tienen menos de MIN_ANUNCIOS_BARRIO anuncios.
    """
    columnas = list(columnas)
    if barrio == 'target' or COL_BARRIO not in columnas:
        return preprocesador_lineal(columnas)
    resto = [c for c in columnas if c != COL_BARRIO]
    return ColumnTransformer([
        ('lineal', preprocesador_lineal(resto), resto),
        ('barrio', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1,
                                  min_frequency=MIN_ANUNCIOS_BARRIO), [COL_BARRIO]),
    ])


class _Red(nn.Module):
    """Capas densas con activacion y dropout; si hay embedding, el vector del barrio se concatena a la entrada."""

    def __init__(self, n_entradas: int, capas: tuple[int, ...], activacion: str, dropout: float,
                 n_barrios: int, dim_embedding: int):
        super().__init__()
        self.embedding = nn.Embedding(n_barrios, dim_embedding) if dim_embedding else None
        ancho = n_entradas + dim_embedding
        bloques = []
        for neuronas in capas:
            bloques += [nn.Linear(ancho, neuronas), ACTIVACIONES[activacion](), nn.Dropout(dropout)]
            ancho = neuronas
        bloques.append(nn.Linear(ancho, 1))
        self.capas = nn.Sequential(*bloques)

    def forward(self, x: torch.Tensor, barrio: torch.Tensor | None = None) -> torch.Tensor:
        if self.embedding is not None:
            x = torch.cat([x, self.embedding(barrio)], dim=1)
        return self.capas(x).squeeze(1)


class RegresorMLP(BaseEstimator, RegressorMixin):
    """Perceptron multicapa con AdamW, perdida MSE y parada temprana.

    Argumentos:
        capas: neuronas de cada capa oculta, p. ej. (256, 128).
        activacion: 'relu' o 'silu'.
        dropout: fraccion de neuronas que se apagan en cada paso de entrenamiento (regularizacion).
        lr: tasa de aprendizaje inicial de AdamW; se divide entre 2 si la perdida de parada se estanca.
        weight_decay: penalizacion de los pesos de AdamW (el equivalente del alpha de Ridge).
        batch_size: anuncios por paso de gradiente.
        dim_embedding: 0 si el barrio llega codificado (target encoding); si no, tamaño de su embedding, y el
            barrio debe llegar como entero en la ultima columna (preprocesador_mlp con barrio='embedding').
        max_epocas, paciencia: se para tras `paciencia` epocas sin mejorar la perdida de parada y se recuperan
            los pesos de la mejor epoca.
        frac_parada: fraccion del train del fold que se aparta, al azar por filas, para la parada temprana. No
            agrupa por anfitrion (el estimador no ve host_id): los anuncios gemelos pueden retrasar algo la parada.
        n_semillas: redes entrenadas con semillas semilla, semilla + 1, ...; se predice con la media de todas.
        semilla: fija la inicializacion, el orden de los lotes y el apartado de parada.
        dispositivo: 'auto' (GPU si la hay), 'cuda' o 'cpu'.
    """

    def __init__(self, capas: tuple[int, ...] = (128, 64), activacion: str = 'relu', dropout: float = 0.1,
                 lr: float = 1e-3, weight_decay: float = 1e-4, batch_size: int = 256, dim_embedding: int = 0,
                 max_epocas: int = 300, paciencia: int = 20, frac_parada: float = 0.1, n_semillas: int = 1,
                 semilla: int = SEMILLA, dispositivo: str = 'auto'):
        self.capas = capas
        self.activacion = activacion
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.dim_embedding = dim_embedding
        self.max_epocas = max_epocas
        self.paciencia = paciencia
        self.frac_parada = frac_parada
        self.n_semillas = n_semillas
        self.semilla = semilla
        self.dispositivo = dispositivo

    def _separar(self, X) -> tuple[np.ndarray, np.ndarray | None]:
        # Con embedding, la ultima columna es el indice del barrio: -1 (no visto) pasa a 0 y el resto a 1..K
        X = np.asarray(X, dtype=np.float32)
        if not self.dim_embedding:
            return X, None
        return X[:, :-1], X[:, -1].astype(np.int64) + 1

    def _tensores(self, x: np.ndarray, barrio: np.ndarray | None, idx: np.ndarray | slice = slice(None)):
        a_tensor = lambda a: torch.as_tensor(np.ascontiguousarray(a[idx]), device=self.dispositivo_)
        return a_tensor(x), None if barrio is None else a_tensor(barrio)

    @torch.no_grad()
    def _predecir_red(self, red: _Red, x: torch.Tensor, barrio: torch.Tensor | None) -> torch.Tensor:
        red.eval()
        trozos = [red(x[i:i + TAMANO_LOTE_PREDICCION],
                      None if barrio is None else barrio[i:i + TAMANO_LOTE_PREDICCION])
                  for i in range(0, len(x), TAMANO_LOTE_PREDICCION)]
        return torch.cat(trozos)

    def _entrenar(self, x: np.ndarray, barrio: np.ndarray | None, y: np.ndarray, semilla: int) -> tuple[_Red, list]:
        torch.manual_seed(semilla)
        orden = np.random.default_rng(semilla).permutation(len(y))
        n_parada = int(round(self.frac_parada * len(y)))
        idx_parada, idx_ajuste = orden[:n_parada], orden[n_parada:]
        x_aj, b_aj = self._tensores(x, barrio, idx_ajuste)
        x_pa, b_pa = self._tensores(x, barrio, idx_parada)
        y_aj = torch.as_tensor(y[idx_ajuste], device=self.dispositivo_)
        y_pa = torch.as_tensor(y[idx_parada], device=self.dispositivo_)

        red = _Red(x.shape[1], tuple(self.capas), self.activacion, self.dropout, self.n_barrios_,
                   self.dim_embedding).to(self.dispositivo_)
        optimizador = torch.optim.AdamW(red.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        planificador = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizador, factor=0.5,
                                                                  patience=max(1, self.paciencia // 4))
        generador = torch.Generator().manual_seed(semilla)

        mejor, mejor_estado, sin_mejora, historial = np.inf, None, 0, []
        for _ in range(self.max_epocas):
            red.train()
            perdidas = []
            for lote in torch.randperm(len(y_aj), generator=generador).to(self.dispositivo_).split(self.batch_size):
                optimizador.zero_grad()
                perdida = nn.functional.mse_loss(red(x_aj[lote], None if b_aj is None else b_aj[lote]), y_aj[lote])
                perdida.backward()
                optimizador.step()
                perdidas.append(perdida.detach())
            perdida_parada = nn.functional.mse_loss(self._predecir_red(red, x_pa, b_pa), y_pa).item()
            historial.append((torch.stack(perdidas).mean().item(), perdida_parada))
            planificador.step(perdida_parada)
            if perdida_parada < mejor - 1e-5:
                mejor, sin_mejora = perdida_parada, 0
                mejor_estado = {k: v.detach().clone() for k, v in red.state_dict().items()}
            else:
                sin_mejora += 1
                if sin_mejora >= self.paciencia:
                    break
        red.load_state_dict(mejor_estado)
        return red.eval(), historial

    def fit(self, X, y) -> 'RegresorMLP':
        if self.dispositivo == 'auto':
            self.dispositivo_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.dispositivo_ = torch.device(self.dispositivo)
        x, barrio = self._separar(X)
        y = np.asarray(y, dtype=np.float32)
        # La red aprende el target estandarizado; predict deshace el cambio
        self.media_y_, self.escala_y_ = float(y.mean()), float(y.std())
        y = (y - self.media_y_) / self.escala_y_
        self.n_barrios_ = int(barrio.max()) + 1 if barrio is not None else 0
        self.redes_, self.historiales_ = [], []
        for i in range(self.n_semillas):
            red, historial = self._entrenar(x, barrio, y, self.semilla + i)
            self.redes_.append(red)
            self.historiales_.append(pd.DataFrame(historial, columns=['mse_ajuste', 'mse_parada']))
        # Epocas hasta la mejor perdida de parada (la parada llega `paciencia` epocas despues)
        self.epocas_ = [int(h['mse_parada'].idxmin()) + 1 for h in self.historiales_]
        return self

    def predict(self, X) -> np.ndarray:
        x, barrio = self._separar(X)
        if barrio is not None:
            barrio = np.clip(barrio, 0, self.n_barrios_ - 1)
        x_t, b_t = self._tensores(x, barrio)
        pred = torch.stack([self._predecir_red(red, x_t, b_t) for red in self.redes_]).mean(dim=0)
        return pred.cpu().numpy().astype(np.float64) * self.escala_y_ + self.media_y_


def incorporar_registro(origen: Path, destino: Path) -> list[str]:
    """Copia al registro principal los experimentos de otro registro (el de Colab), sustituyendo los que ya hubiera.

    A diferencia de registrar, conserva la fecha original de cada fila. Devuelve los experimentos copiados.
    """
    nuevos = pd.read_csv(origen)
    if destino.exists():
        previos = pd.read_csv(destino)
        nuevos = pd.concat([previos[~previos['experimento'].isin(nuevos['experimento'].unique())], nuevos],
                           ignore_index=True)
    nuevos.to_csv(destino, index=False)
    return list(pd.read_csv(origen)['experimento'].unique())
