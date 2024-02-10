import torch
from typing import Tuple, Union, List
from constants import DEPTH_STEP
from plane_cylinder_projections import normal_vector_to_angles, image_vector_to_3d_plane_tangent
import numpy as np
import torch

# --------- METHOD 1 Randomized cross products method ---------


def compute_3d_tangent_vectors(
    coords: Union[np.ndarray, torch.Tensor],
    tan_vec_2d: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the 3D tangent vectors based on the input coordinates and 2D tangent vectors.

    Args:
        coords (Union[np.ndarray, torch.Tensor]): The input coordinates.
        tan_vec_2d (np.ndarray): The 2D tangent vectors.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The computed 3D tangent vectors.
    """

    if isinstance(coords, np.ndarray):
        img_coords = torch.from_numpy(coords).float()
    else:
        img_coords = coords

    azi_list = torch.deg2rad(img_coords[:, 1]).unsqueeze(0)
    alt_list = -img_coords[:, 0]*DEPTH_STEP
    tangents_3d = []
    for idx in range(azi_list.shape[1]):
        azi = azi_list[0, idx]
        alt = alt_list[idx]
        delta_azi = tan_vec_2d[idx, 0]
        delta_alt = tan_vec_2d[idx, 1]
        tan3d = image_vector_to_3d_plane_tangent(azi, alt, delta_azi, delta_alt)
        tan3d = tan3d / tan3d.norm()  # normalize the 3D tangent
        tangents_3d.append(tan3d)
    tangents_3d = torch.stack(tangents_3d, dim=0).unsqueeze(0)
    return tangents_3d


def get_cross_products(tangents_3d: torch.tensor, num_points=800) -> Tuple[torch.tensor, torch.tensor]:
    """
    Compute cross products by picking random pairs of 3D tangents.

    Args:
        tangents_3d (torch.tensor): 3D tangents.
        num_points (int, optional): Number of random pairs to generate. Defaults to 800.

    Returns:
        Tuple[torch.tensor, torch.tensor]: Cross products and their norms.
    """

    random_pairs_indexes = torch.randint(0, tangents_3d.shape[-2], (num_points, 2))
    random_pairs = tangents_3d[:, random_pairs_indexes, :]
    cross_product = torch.cross(random_pairs[:, :, 0, :], random_pairs[:, :, 1, :], dim=-1)
    cross_product_norm = cross_product.norm(dim=-1, keepdim=True)

    cross_product = cross_product/cross_product_norm
    cross_product *= torch.sign(cross_product[..., 2].unsqueeze(-1))  # force upwards
    return cross_product, cross_product_norm


def extract_dip_azimuth(
        dip_az_estim: torch.Tensor,
        bins: List[int] = [20, 20]) -> Tuple[float, float, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extracts dip and azimuth from a large tensor of estimated dip and azimuth from randomized cross products.
    Method 1

    Args:
        dip_az_estim (torch.Tensor): Estimated dip and azimuth tensor.
        bins (List[int], optional): Number of bins for 2D histogram calculation. Defaults to [20, 20].

    Returns:
        Tuple[float, float, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: Tuple containing 
        best dip, best azimuth, histogram, bin edges.
    """

    histo, bin_edges = torch.histogramdd(dip_az_estim, bins=bins)
    maxi = torch.max(histo)
    amax_mode = torch.where(histo == maxi)
    # Compute the midpoint of the bins to ge the dip and azimuth
    x_mid = (bin_edges[1][amax_mode[1]+1] + bin_edges[1][amax_mode[1]]) / 2.0
    y_mid = (bin_edges[0][amax_mode[0]+1] + bin_edges[0][amax_mode[0]]) / 2.0
    best_dip = y_mid[0]
    best_azimuth = x_mid[0]
    # print(f"Estimated dip: {np.rad2deg(best_dip):.3f}° and azimuth: {np.rad2deg(best_azimuth):.3f}°")
    return best_dip, best_azimuth, histo, bin_edges

# --------- METHOD 2 Eigen vectors of covariance matrix ---------


def extract_dip_azimuth_by_plane_normal_estimation(tangent3d_tensor: torch.Tensor) -> torch.Tensor:
    """
    Extracts the dip and azimuth angles by finding the estimated plane normal.
    This is solved by finding the eigenvector corresponding to the smallest eigenvalue of the covariance matrix
    Eigen vectors are orthogonal unitary
    Method 2

    Args:
        tangent3d_tensor (torch.Tensor): The tensor containing tangent vectors in 3D.

    Returns:
        torch.Tensor: The tensor containing the dip and azimuth angles.

    """
    cov = torch.bmm(tangent3d_tensor.transpose(1, 2), tangent3d_tensor)
    eig_val, eig_vec = torch.linalg.eigh(cov)  # The eigenvalues are returned in ascending order.
    estimated_normals = eig_vec[:, :, 0]  # Extract eigen vector corresponding to the smallest eigenvalue
    estimated_normals *= torch.sign(estimated_normals[..., -1]).unsqueeze(-1)
    dip_az_est = normal_vector_to_angles(estimated_normals)
    return dip_az_est
