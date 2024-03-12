import torch
from shared import ACCURACY, PRECISION, RECALL, F1_SCORE, IOU


def compute_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5):
    y_pred_thresh = (torch.sigmoid(y_pred) > threshold).float()
    correct_predictions = (y_pred_thresh == y_true).float().sum()
    accuracy = correct_predictions / y_true.numel()
    return accuracy.item()


def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5, epsilon=1E-12, reduce="mean") -> dict:
    """Compute metrics for binary segmentation

    Args:
        y_pred (torch.Tensor): [N, C, H, W]
        y_true (torch.Tensor): [N, C, H, W]
        threshold (float, optional): proabibility separation threshold. Defaults to 0.5.

    Returns:
        dict: Dictionary with the following metrics:
        - Accuracy
        - Precision
        - Recall
        - F1 Score = Dice Coefficient
        - Intersection over Union (IoU)
    """
    reduction_dimension = (-1, -2, -3)

    # total_shape = 1
    # for idx in reduction_dimension:
    #     total_shape *= y_pred.shape[idx]

    # Convert predictions to binary (0 or 1) based on the threshold
    y_pred_bin = (torch.sigmoid(y_pred) > threshold).float()
    # True Positives, False Positives, True Negatives, False Negatives
    total_positives = (y_true == 1).float().sum(dim=reduction_dimension)
    true_positive = ((y_pred_bin == 1) & (y_true == 1)).float().sum(dim=reduction_dimension)
    # WARNING! If there are no positives, we set the true positive to 0.1 to fake a single pixel
    true_positive = true_positive * (total_positives != 0) + (total_positives == 0) * \
        torch.ones_like(true_positive) * 1.
    false_positive = ((y_pred_bin == 1) & (y_true == 0)).float().sum(dim=reduction_dimension)
    true_negative = ((y_pred_bin == 0) & (y_true == 0)).float().sum(dim=reduction_dimension)
    false_negative = ((y_pred_bin == 0) & (y_true == 1)).float().sum(dim=reduction_dimension)
    # print(f"{true_positive.item()=:}, {false_positive.item()=:}, {true_negative.item()=:}, {false_negative.item()=:}")

    # Precision
    precision = true_positive / (true_positive + false_positive + epsilon)

    # Recall -> Accuracy of positive areas!
    recall = true_positive / (true_positive + false_negative + epsilon)

    # F1 Score
    # Same as dice coefficient
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)

    # Accuracy
    accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)

    # Intersection over Union (IoU)
    iou = true_positive / (true_positive + false_positive + false_negative + epsilon)
    if reduce == "mean":
        return {
            ACCURACY: accuracy.mean(),
            PRECISION: precision.mean(),
            RECALL: recall.mean(),
            F1_SCORE: f1_score.mean(),
            IOU: iou.mean()
        }
    elif reduce == "sum":
        return {
            ACCURACY: accuracy.sum(),
            PRECISION: precision.sum(),
            RECALL: recall.sum(),
            F1_SCORE: f1_score.sum(),
            IOU: iou.sum()
        }
    elif reduce == "none" or reduce is None:
        return {
            ACCURACY: accuracy,
            PRECISION: precision,
            RECALL: recall,
            F1_SCORE: f1_score,
            IOU: iou
        }
    else:
        raise ValueError(f"Unknown reduction method {reduce}")
