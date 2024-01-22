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


def prepare_augmented_dataset(
    pressure: List[np.ndarray],
    labels:  List[int],
    seed: int = 0,
    ratio_noisy_label: float = 0.2,
    ratio_tight: float = 0.4
) -> Tuple[List[np.ndarray], List[int]]:
    np.random.seed(seed)
    labels_noisy = add_noisy_labels_to_dataset(pressure, labels, ratio_noisy_label=ratio_noisy_label)
    pressure_imb, labels_imb = force_dataset_imbalance(pressure, labels_noisy, ratio_tight=ratio_tight)
    return pressure_imb, labels_imb


def add_noisy_labels_to_dataset(pressure: List[np.ndarray], labels:  List[int], ratio_noisy_label: float = 0.2):
    index = np.random.choice(len(pressure), int(ratio_noisy_label*len(pressure)))
    labels_noisy = np.copy(labels)
    labels_noisy[index] = 1-labels_noisy[index]
    return labels_noisy


def get_dataset_balance(labs:  List[int], desc: str = "") -> None:
    total_normal_samples = np.sum(np.array(labs) == 1)
    print(f"{desc} Total normal samples: {total_normal_samples} = {total_normal_samples/len(labs):.1%}")
    print(f"{desc} Total tight samples: {len(labs)-total_normal_samples} = {1-total_normal_samples/len(labs):.1%}")


def force_dataset_imbalance(pressure: List[np.ndarray], labels:  List[int], ratio_tight: float = 0.4) -> Tuple[List[np.ndarray], List[int]]:
    pretests_imbalanced = np.copy(pressure)
    labels_imbalanced = np.copy(labels)

    index_sort = np.argsort(labels_imbalanced)
    labels_imbalanced = labels_imbalanced[index_sort]
    pretests_imbalanced = pretests_imbalanced[index_sort]

    num_tight = np.sum(labels_imbalanced == 0)
    num_normal = np.sum(labels_imbalanced == 1)

    # we would like num_tight/(num_tight+num_normal) = 0.4
    rate = ratio_tight
    num_skipped = num_tight - int(num_normal*rate/(1-rate))

    pretests_imbalanced = pretests_imbalanced[num_skipped:]
    labels_imbalanced = labels_imbalanced[num_skipped:]
    return pretests_imbalanced, labels_imbalanced
