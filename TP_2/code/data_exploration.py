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
    seed: int = None,
    ratio_noisy_label: float = 0.2,
    ratio_tight: float = 0.4
) -> Tuple[List[np.ndarray], List[int]]:
    if seed is not None:
        np.random.seed(seed)
    if ratio_noisy_label == 0:
        labels_noisy = np.copy(labels)
    else:
        labels_noisy = add_noisy_labels_to_dataset(pressure, labels, ratio_noisy_label=ratio_noisy_label)
    get_dataset_balance(labels_noisy, desc="Original")
    if ratio_tight is not None:
        pressure_imb, labels_imb = force_dataset_imbalance(pressure, labels_noisy, ratio_tight=ratio_tight)
        get_dataset_balance(labels_imb, desc="Imbalanced")
    else:
        pressure_imb, labels_imb = pressure, labels_noisy
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

    # we would like num_tight/(num_tight+num_normal) = ratio_tight
    num_tight_final = ratio_tight * num_normal / (1 - ratio_tight)
    num_skipped = num_tight - int(num_tight_final)

    pretests_imbalanced = pretests_imbalanced[num_skipped:]
    labels_imbalanced = labels_imbalanced[num_skipped:]
    return pretests_imbalanced, labels_imbalanced


def prepare_evaluation_indices(labels, eval_set_proportion, class_balance):
    # Convert labels to a NumPy array for easier processing
    labels = np.array(labels)
    unique_classes = np.unique(labels)

    # Number of samples per class in the evaluation set
    samples_per_class = int(len(labels) * eval_set_proportion * class_balance)

    # Collect indices for each class
    indices = []
    for class_label in unique_classes:
        class_indices = np.where(labels == class_label)[0]
        selected_indices = np.random.choice(class_indices, min(samples_per_class, len(class_indices)), replace=False)
        indices.extend(selected_indices)

    return indices


def prepare_whole_dataset(data, labels):
    # Now you can use evaluation_indices to extract the corresponding samples and labels from your dataset
    indexes_testset = prepare_evaluation_indices(labels, 0.2, 0.5)
    indexes_testset = np.array(indexes_testset)
    assert len(np.unique(indexes_testset)) == len(
        indexes_testset), "Same indexes appear multiple times in the evaluation set!"
    test_data = data[indexes_testset]
    test_label = labels[indexes_testset]

    train_indices = np.setdiff1d(np.arange(len(labels)), indexes_testset)
    train_data = data[train_indices]
    train_label = labels[train_indices]
    return train_data, test_data, train_label, test_label