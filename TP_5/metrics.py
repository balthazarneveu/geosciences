import torch
from shared import ACCURACY, PRECISION, RECALL, F1_SCORE


def compute_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5):
    y_pred_thresh = (torch.sigmoid(y_pred) > threshold).float()
    correct_predictions = (y_pred_thresh == y_true).float().sum()
    accuracy = correct_predictions / y_true.numel()
    return accuracy.item()


def compute_metrics(y_pred: torch.Tensor, y_true: torch.Tensor, threshold: float = 0.5):
    # Convert predictions to binary (0 or 1) based on the threshold
    y_pred_bin = (torch.sigmoid(y_pred) > threshold).float()
    # True Positives, False Positives, True Negatives, False Negatives
    true_positive = ((y_pred_bin == 1) & (y_true == 1)).float().sum()
    false_positive = ((y_pred_bin == 1) & (y_true == 0)).float().sum()
    true_negative = ((y_pred_bin == 0) & (y_true == 0)).float().sum()
    false_negative = ((y_pred_bin == 0) & (y_true == 1)).float().sum()

    # Precision
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0.0

    # Recall -> Accuracy of positive areas!
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0.0

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Accuracy
    accuracy = (true_positive + true_negative) / (true_positive + false_positive + true_negative + false_negative)

    return {
        ACCURACY: accuracy,
        PRECISION: precision,
        RECALL: recall,
        F1_SCORE: f1_score,
    }
