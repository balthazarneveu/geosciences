import torch
from plane_cylinder_projections import intersect_plane_with_cyliner
from typing import Tuple
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
    altitude_z = intersect_plane_with_cyliner(azimuth_coordinates_phi, plane_angle)
    return azimuth_coordinates_phi, altitude_z
