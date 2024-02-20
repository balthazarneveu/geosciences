import torch
from typing import Tuple, Optional


def augment_wrap_roll(img: torch.Tensor, lab: torch.Tensor, shift: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Roll pixels horizontally to avoid negative index

    Args:
        img (torch.Tensor): [N, 1, H, W] image tensor
        lab (torch.Tensor): [N, 1, H, W] label tensor
        shift (Optional[int], optional): forced shift value. Defaults to None.
        If not provided, a random shift is used

    Returns:
        torch.Tensor, torch.Tensor: rolled image, labels
    """
    if shift is None:
        shift = torch.randint(0, img.shape[-1], (1,)).item()
    rolled_img = torch.roll(img, shift, (-1,))
    rolled_lab = torch.roll(lab, shift, (-1,))
    return rolled_img, rolled_lab


def augment_flip(
    img: torch.Tensor,
    lab: torch.Tensor,
    flip: Optional[Tuple[bool, bool]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Roll pixels horizontally to avoid negative index

    Args:
        img (torch.Tensor): [N, 1, H, W] image tensor
        lab (torch.Tensor): [N, 1, H, W] label tensor
        flip (Optional[bool], optional): forced flip_h, flip_v value. Defaults to None.
        If not provided, a random flip_h, flip_v values are used
    Returns:
        torch.Tensor, torch.Tensor: flipped image, labels

    WARNING
    =======
    The vertical flip may be questionable as the objects we're trying to segment
    may be directed vertically.

    """
    if flip is None:
        flip = torch.randint(0, 2, (2,))
    flipped_img = img
    flipped_lab = lab
    if flip[0] > 0:
        flipped_img = torch.flip(flipped_img, (-1,))
        flipped_lab = torch.flip(flipped_lab, (-1,))
    if flip[1] > 0:
        flipped_img = torch.flip(flipped_img, (-2,))
        flipped_lab = torch.flip(flipped_lab, (-2,))
    return flipped_img, flipped_lab
