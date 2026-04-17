from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def coerce_time_series(data: Any) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim == 1:
        return array.reshape(1, array.shape[0])
    if array.ndim != 2:
        raise ValueError("Expected a 1D or 2D array-like time series.")
    return array


def load_state_csv(
    path: str | Path,
    *,
    drop_time_column: bool = True,
    drop_last_timestep: bool = True,
) -> np.ndarray:
    frame = pd.read_csv(path, header=None)
    array = frame.to_numpy().transpose()

    if drop_time_column:
        array = array[1:, :]
    if drop_last_timestep:
        array = array[:, :-1]

    return coerce_time_series(array)
