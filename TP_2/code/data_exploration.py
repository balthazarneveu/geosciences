import json
from pathlib import Path
import numpy as np
from typing import List, Tuple
HERE = Path(__file__).parent


def get_data(path: Path = HERE/'buildups_public.json') -> dict:
    with open(path, 'r') as f:
        buildup_data = json.load(f)
    print(list(buildup_data[0].keys()))
    return buildup_data


def prepare_data(data: dict) -> Tuple[List[np.ndarray], List[int]]:
    # Raw data preparation
    raw_pressure = [p['buildup_pressure'] for p in data]
    labels = [press['buildup_label'] for press in data]
    labels = [0 if label == 'Tight' else 1 for label in labels]
    # Interpolate and normalize the pressure
    pressure = [reinterpret_pressure(p) for p in raw_pressure]
    return pressure, labels


def reinterpret_pressure(
    p: np.ndarray,
    n_points: int = 200,
    normalize_range: List[float] = [0, 1],
    check_integrity: bool = True
) -> np.ndarray:
    if check_integrity:  # check that there are no NaNs
        assert ((np.array(p) == np.NaN).sum()) == 0
    resample = np.interp(np.linspace(0, len(p), n_points),
                         np.arange(len(p)), p)
    normalized = (resample - resample.min())/(resample.max() - resample.min())
    normalized = normalized * \
        (normalize_range[1] - normalize_range[0]) + normalize_range[0]
    return normalized
