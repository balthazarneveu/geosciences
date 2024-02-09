import torch
import numpy as np
from constants import DEPTH_STEP, RADIUS


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
        azimuth_coordinate_phi (torch.Tensor): [1 or N, W] azimuthal coordinate of the point on the cylinder.
        plane_angles_dip_azimuth_origin_z0 (torch.Tensor): [N, 3] dip, azimuth, and origin coordinates of the plane.
        borehole_radius (float, optional): radius of the cylinder (meters). Defaults to RADIUS.

    Returns:
        torch.Tensor: altitude coordinate z of the projected point on the plane (meters). [N, W]
    """
    params = plane_angles_dip_azimuth_origin_z0.unsqueeze(-1)
    dip, azimuth = params[..., 0, :], params[..., 1, :]
    plane_origin_z0 = params[..., 2, :]
    sine_wave = borehole_radius*torch.tan(dip)*torch.cos(azimuth_coordinate_phi-azimuth)
    altitude_z = plane_origin_z0 + sine_wave
    return altitude_z


def angle_to_3d_vector(
    azimuth_coordinate_phi: torch.Tensor,
    altitude_z=None,
    borehole_radius: float = RADIUS
) -> torch.Tensor:
    x = borehole_radius * torch.cos(azimuth_coordinate_phi)
    y = borehole_radius * torch.sin(azimuth_coordinate_phi)
    z = torch.zeros_like(x) if altitude_z is None else altitude_z
    return torch.stack((x, y, z), dim=-1)


def image_vector_to_3d_plane_tangent(
    azi: torch.Tensor,
    alt: torch.Tensor,
    delta_azi: torch.Tensor,
    delta_alt: torch.Tensor
) -> torch.Tensor:
    """Take a 2D image vector (phi, alti) and convert it to a 3D plane tangent vector."""
    # There's no obvious way to compute the jacobian per batch element, so ... loop over the elements
    # There's a hacky solution proposed in pytorch forums, but it's not clear if it's correct
    # https://discuss.pytorch.org/t/computing-batch-jacobian-efficiently/80771
    jac = torch.autograd.functional.jacobian(
        angle_to_3d_vector,
        (azi, alt)
    )
    estimated_3d_grad = jac[0]*delta_azi + jac[1]*delta_alt
    return estimated_3d_grad


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    azimuth_coordinates_phi = torch.linspace(0, 2*torch.pi, 360).unsqueeze(0)
    azimuth_coordinates_phi = azimuth_coordinates_phi.repeat(2, 1)
    plane_angle = torch.tensor(
        [
            [torch.pi/4, torch.pi/4, 0.],
            [0.8*torch.pi/2., torch.pi/4, 1.],
        ]
    )
    altitude_z = intersect_plane_with_cyliner(azimuth_coordinates_phi, plane_angle)
    print(altitude_z.shape)
    plt.plot(torch.rad2deg(azimuth_coordinates_phi.T), altitude_z.T)
    plt.xlabel("Azimuth (degrees)")
    plt.ylabel("Depth (m)")
    plt.grid()
    plt.show()


def get_tangent_vec_from_gradients(
    grad_horizontal: np.ndarray,
    grad_vertical: np.ndarray,
    scaling_horizontal_step: float = np.pi/180.,
    scaling_vertical_step: float = DEPTH_STEP,
    normalize: bool = True
) -> np.ndarray:
    """Rotate by 90° to get the tangent vector + rescale

    dI/dx = gray levels / delta pixels
    dI/d(SI unit) = dI/dx / scaling_factor

    Args:
        grad_horizontal (np.ndarray): delta gray levels / delta degrees (dazimuth°)

        grad_vertical (np.ndarray): delta gray levels / delta depth (dz)

        scaling_horizontal_step (float, optional): Scaling to go from image units to SI unit. Defaults to np.pi/180
        pi/180 * pixel step (1 horizontal pixel <-> 1 degrees =image units) = radians (SI units)

        scaling_vertical_step (float, optional): Scaling to go from image units (pixel step) to SI unit (m). Defaults to DEPTH_STEP.
        DEPTH_STEP = 0.00254 * pixel step (1 vertical pixel = 0.00254 m = image units) -> meters (SI units)

    Returns:
        np.ndarray: (N, 2) array of the tangent vectors
    """
    tangents_si_units = np.array([
        -grad_vertical/scaling_vertical_step,
        grad_horizontal/scaling_horizontal_step
    ]
    ).T
    if normalize:
        tangents_si_units = tangents_si_units / np.linalg.norm(tangents_si_units, axis=1)[:, None]
    return tangents_si_units
