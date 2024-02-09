import torch
from plane_cylinder_projections import intersect_plane_with_cylinder, image_vector_to_3d_plane_tangent
from typing import Tuple
import numpy as np
from constants import DEPTH_STEP
from skimage import filters

DEFAULT_PLANE_ANGLES = torch.tensor(
    [
        [0., 0.,  -0.3],
        [torch.pi/4, torch.pi/4, 0.],
        [0.8*torch.pi/2., torch.pi/4, -1.],
        [0.9*torch.pi/2., torch.pi/3, -0.5],
        [0.1, torch.pi/2, -1.2],
    ]
)


def create_planes_projection(
        plane_angle: torch.Tensor = DEFAULT_PLANE_ANGLES,
        num_points: int = 360) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create planes projection.

    Creates a projection of planes in a cylindrical coordinate system.

    Args:
        plane_angle (torch.Tensor, optional): Angles of the planes in radians. Defaults to DEFAULT_PLANE_ANGLES.
        [N, 3]
        num_points (int, optional): Number of points to sample on the azimuth coordinates. Defaults to 360.
        [L]

    Returns:
        torch.Tensor, torch.Tensor: azimuth [rad], altitude[m] coordinates of the planes projected.
        [N, L], [N, L]
    """
    # Let's sample the azimuth coordinates and compute the altitude of the planes
    azimuth_coordinates_phi = torch.linspace(0, 2*torch.pi, num_points).unsqueeze(0)
    azimuth_coordinates_phi = azimuth_coordinates_phi.repeat(plane_angle.shape[0], 1)
    altitude_z = intersect_plane_with_cylinder(azimuth_coordinates_phi, plane_angle)
    return azimuth_coordinates_phi, altitude_z


def simulated_splat_image(
    azimuth_coordinates_phi: torch.Tensor,
    altitude_z: torch.Tensor,
    noise_amplitude: float = 0.02,
    sigma_blur: float = 2,
    w: int = 365,
    h: int = 600
) -> np.ndarray:
    w = 365
    h = 600
    img = np.zeros((h, w), dtype=np.float32)
    img += np.random.normal(0, noise_amplitude, img.shape)
    for batch_idx in range(altitude_z.shape[0]):
        for idx in range(altitude_z.shape[1]):
            z = altitude_z[batch_idx, idx]
            azi_deg = azimuth_coordinates_phi[batch_idx, idx]
            azi_deg = np.rad2deg(azi_deg)
            row = int(-z/DEPTH_STEP)
            if row < 0 or row >= h:
                continue

            col = int(azi_deg)
            if col < 0 or col >= w:
                continue
            img[row, col] = 1.
    img = filters.gaussian(img, sigma=sigma_blur)
    return img


def get_3d_tangent_estimation(
    azimuth_coordinates_phi: torch.Tensor,
    altitude_z: torch.Tensor,
) -> torch.Tensor:
    """
    Computation of the tangent vectors using the jacobian of `image_vector_to_3d_plane_tangent`
    matches with the finite differences computation.
    Based on the ground truth 3D points.
    """
    batches_index = range(azimuth_coordinates_phi.shape[0])
    estimated_grad_list_all = []
    for batch_idx in batches_index:
        estimated_grad_list = []
        for element_idx in range(azimuth_coordinates_phi.shape[1]-1):
            azi, alt = azimuth_coordinates_phi[batch_idx, element_idx], altitude_z[batch_idx, element_idx]
            delta_azi = azimuth_coordinates_phi[batch_idx, element_idx+1] - \
                azimuth_coordinates_phi[batch_idx, element_idx]
            delta_alt = altitude_z[batch_idx, element_idx+1] - altitude_z[batch_idx, element_idx]
            estimated_grad = image_vector_to_3d_plane_tangent(azi, alt, delta_azi, delta_alt)
            estimated_grad_list.append(estimated_grad)
        estimated_grad_list = torch.stack(estimated_grad_list)
        estimated_grad_list_all.append(estimated_grad_list)
    estimated_grad_list_all = torch.stack(estimated_grad_list_all)
    return estimated_grad_list_all
