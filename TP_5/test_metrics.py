import numpy as np
import torch
from metrics import compute_metrics
from shared import ACCURACY, PRECISION, RECALL, F1_SCORE, IOU
import pytest


def test_compute_metrics():
    # Test case 1: Perfect prediction
    y_pred = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
    y_true = torch.tensor([[[[1.0, 1.0], [1.0, 1.0]]]])
    expected_metrics = {
        ACCURACY: 1.0,
        PRECISION: 1.0,
        RECALL: 1.0,
        F1_SCORE: 1.0,
        IOU: 1.0
    }
    pred_metrics = compute_metrics(y_pred, y_true)
    for key, metric in pred_metrics.items():
        assert metric == expected_metrics[key]
    # Test case 2: Batch of images
    N, C, H, W = 16, 1, 36, 36
    y_pred = torch.ones((N, C, H, W))

    y_true = torch.ones((N, C, H, W))
    expected_metrics = {
        ACCURACY: 1.0,
        PRECISION: 1.0,
        RECALL: 1.0,
        F1_SCORE: 1.0,
        IOU: 1.0
    }
    pred_metrics = compute_metrics(y_pred, y_true)
    for key, metric in pred_metrics.items():
        assert metric == expected_metrics[key]


def generate_data(N, C, H, W, scenario='perfect'):
    """
    Generate synthetic binary segmentation data.

    Args:
        N, C, H, W (int): Dimensions for the data.
        scenario (str): Type of data to generate - 'perfect', 'misprediction', or 'partial'.

    Returns:
        y_pred, y_true (torch.Tensor): Synthetic predictions and labels.
    """
    y_true = torch.zeros((N, C, H, W))
    y_pred = torch.zeros((N, C, H, W))

    if scenario == 'perfect':
        y_true[:, :, :H//2, :] = 1  # Half of the image is the object
        y_pred = y_true.clone()  # Perfect prediction

    elif scenario == 'misprediction':
        y_true[:, :, :H//2, :] = 1  # Half of the image is the object
        y_pred[:, :, H//2:, :] = 1  # Completely opposite prediction

    elif scenario == 'partial':
        y_true[:, :, :H//2, :] = 1  # Half of the image is the object
        y_pred[:, :, :H//4, :] = 1  # Partial overlap with true labels
        # 1/4 of the image is misclassified
        # -> IoU = 1/2
        # -> Precision = 1, Recall = 0.5
        # -> F1 Score = 2 * (1 * 0.5) / (1 + 0.5) = 2/3
        # 1/2 of the positive area is miscassified

    return y_pred, y_true


@pytest.mark.parametrize("scenario", ['perfect', 'misprediction', 'partial'])
def test_metrics_images(scenario):
    N, C, H, W = 16, 1, 36, 36
    y_pred, y_true = generate_data(N, C, H, W, scenario)
    metrics = compute_metrics(
        y_pred,
        y_true,
        epsilon=1e-12 if scenario == 'misprediction' else 0.
    )

    if scenario == 'perfect':
        assert metrics[ACCURACY] == 1, f"Accuracy should be 1 in perfect scenario, got {metrics[ACCURACY]}"
        assert metrics[PRECISION] == 1, f"Precision should be 1 in perfect scenario, got {metrics[PRECISION]}"
        assert metrics[RECALL] == 1, f"Recall should be 1 in perfect scenario, got {metrics[RECALL]}"
        assert metrics[F1_SCORE] == 1, f"F1 Score should be 1 in perfect scenario, got {metrics[F1_SCORE]}"
        assert metrics[IOU] == 1, f"IoU should be 1 in perfect scenario, got {metrics[IOU]}"

    elif scenario == 'misprediction':
        assert metrics[ACCURACY] == 0, f"Accuracy should be 0 in misprediction scenario, got {metrics[ACCURACY]}"
    elif scenario == 'partial':
        # Here, we expect all metrics to be between 0 and 1, but not exactly known values.
        assert metrics[ACCURACY] == 3./4.
        assert metrics[PRECISION] == 1.
        assert metrics[RECALL] == 0.5
        assert metrics[IOU] == 1./2.
        assert np.isclose(metrics[F1_SCORE].item(), 2./3.)
