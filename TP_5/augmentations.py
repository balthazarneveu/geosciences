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
