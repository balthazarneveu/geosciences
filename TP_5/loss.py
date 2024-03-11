import torch
from typing import Optional
from shared import LOSS_BCE, LOSS_BCE_WEIGHTED, LOSS_DICE
# Interesting article on Kaggle about loss functions for segmentation:
# https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch


def compute_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    mode: Optional[str] = LOSS_BCE,
    binary_labels_flag: bool = True,
) -> torch.Tensor:
    """
    Compute loss based on the predicted and true values.

    Args:
        y_pred (torch.Tensor): [N, C, H, W] predicted values (logits! not probabilities).
        y_true (torch.Tensor): [N, C, H, W] true values.
        mode (Optional[str], optional): mode of loss computation.
        Defaults to LOSS_BCE.

    Returns:
        torch.Tensor: The computed loss.
    """
    assert mode in [LOSS_BCE, LOSS_DICE, LOSS_BCE_WEIGHTED], f"Mode {mode} not supported"
    y_pred_flat = y_pred.view(-1)
    if not binary_labels_flag:
        y_true = torch.sigmoid(y_true)
    if mode == LOSS_BCE:
        y_true_flat = y_true.view(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            y_pred_flat,
            y_true_flat
        )
    elif mode == LOSS_BCE_WEIGHTED:
        y_true_flat = y_true.view(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            y_pred_flat,
            y_true_flat,
            pos_weight=torch.Tensor([2.]).to(y_pred.device)
        )
    elif mode == LOSS_DICE:
        # https://chenriang.me/f1-equal-dice-coefficient.html
        # Equivalence between F1 and dice coefficient
        # Smooth here is used to avoid division by zero
        smooth = 1.E-7
        dimensions = (-1, -2, -3)
        y_pred_proba = torch.sigmoid(y_pred)
        y_true_flat = y_true.sum(dim=dimensions)
        intersection = (y_pred_proba * y_true.float()).sum(dim=dimensions)
        dice = (2.*intersection + smooth)/(y_pred_proba.sum(dim=dimensions) + y_true_flat + smooth)
        loss = 1 - dice
        loss = loss.mean()
    else:
        raise ValueError(f"Mode {mode} not supported")
    return loss
