import torch

ONE_INCH = 0.0254
DIAMETER = 8.5 * ONE_INCH
RADIUS = DIAMETER / 2


def normal_vector_to_angles(vec_normal: torch.Tensor) -> torch.Tensor:
    """
    Converts a normal vector to (dip, azimuth) angles.

    Args:
        vec_normal (torch.Tensor): A 3D vector representing the normal vector of a plane.

    Returns:
        torch.Tensor: A 2D tensor containing the (dip, azimuth) angles in radians.

    Raises:
        ValueError: If the input tensor does not have a shape of (3,).
    """
    vec = vec_normal / torch.norm(vec_normal, dim=-1, keepdim=True)
    azimuth = torch.atan2(vec[..., 1], vec[..., 0])
    dip = torch.acos(vec[..., 2])
    return torch.stack((dip, azimuth), dim=-1)


def angles_to_normal_vector(angles: torch.Tensor) -> torch.Tensor:
    """
    Converts dip, azimuth angles to a normal vector.

    Args:
        angles (torch.Tensor): A 2D tensor containing the (dip, azimuth) angles in radians.

    Returns:
        torch.Tensor: A 3D tensor representing the normal vector of a plane.
    """
    dip = angles[..., 0]
    azimuth = angles[..., 1]
    x = torch.sin(dip) * torch.cos(azimuth)
    y = torch.sin(dip) * torch.sin(azimuth)
    z = torch.cos(dip)
    return torch.stack((x, y, z), dim=-1)


def intersect_plane_with_cyliner(
    azimuth_coordinate_phi: torch.Tensor,
    plane_angles_dip_azimuth_origin_z0: torch.Tensor,
    borehole_radius: float = RADIUS
) -> torch.Tensor:
    """
    Intersect a plane parameterized by (dip, azimuth, origin) with a cylinder at a given azitmuth.

    Args:
        azimuth_coordinate_phi (torch.Tensor): azimuthal coordinate of the point on the cylinder.
        plane_angles_dip_azimuth_origin_z0 (torch.Tensor): dip, azimuth, and origin coordinates of the plane.
        borehole_radius (float, optional): radius of the cylinder (meters). Defaults to RADIUS.

    Returns:
        torch.Tensor: altitude coordinate z of the projected point on the plane (meters).
    """

    dip, azimuth = plane_angles_dip_azimuth_origin_z0[..., 0], plane_angles_dip_azimuth_origin_z0[..., 1]
    plane_origin_z0 = plane_angles_dip_azimuth_origin_z0[..., 2]
    altitude_z = plane_origin_z0 + borehole_radius*torch.tan(dip)*torch.cos(azimuth_coordinate_phi-azimuth)
    return altitude_z


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    azimuth_coordinates_phi = torch.linspace(0, 2*torch.pi, 360).unsqueeze(-1)
    plane_angle = torch.tensor(
        [
            [torch.pi/4, torch.pi/4, 0.],
            [0.8*torch.pi/2., torch.pi/4, 1.],
        ]
    )
    depth = intersect_plane_with_cyliner(azimuth_coordinates_phi, plane_angle)
    plt.plot(torch.rad2deg(azimuth_coordinates_phi), depth)
    plt.xlabel("Azimuth (degrees)")
    plt.ylabel("Depth (m)")
    plt.grid()
    plt.show()
