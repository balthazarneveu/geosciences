import torch
from typing import Tuple


def get_cross_products(tangents_3d: torch.tensor, num_points=800) -> Tuple[torch.tensor, torch.tensor]:
    random_pairs_indexes = torch.randint(0, tangents_3d.shape[-2], (num_points, 2))
    random_pairs = tangents_3d[:, random_pairs_indexes, :]
    cross_product = torch.cross(random_pairs[:, :, 0, :], random_pairs[:, :, 1, :], dim=-1)
    cross_product_norm = cross_product.norm(dim=-1, keepdim=True)

    cross_product = cross_product/cross_product_norm
    cross_product.shape
    return cross_product, cross_product_norm
