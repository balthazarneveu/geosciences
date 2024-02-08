import torch


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
