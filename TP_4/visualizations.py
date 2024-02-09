from matplotlib import pyplot as plt
import numpy as np
import torch
from constants import DEPTH_STEP
import matplotlib.colors as mcolors
from plane_cylinder_projections import image_vector_to_3d_plane_tangent


def plot_ground_truth_3d(
        azimuth_coordinates_phi,
        altitude_z,
        p3D_gt,
        img_coords=None, p3D_est=None):
    decim = 30
    # Plotting
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(121)
    # ----------------- 2D plot -----------------
    if img_coords is not None:
        ax.plot(
            torch.rad2deg(np.deg2rad(img_coords[:, 1])),
            -img_coords[:, 0]*DEPTH_STEP,
            "+",
            markersize=1, alpha=0.2,
            label="Estimated plane points"
        )
    ax.plot(torch.rad2deg(azimuth_coordinates_phi.T), altitude_z.T)
    ax.set_xlabel("Azimuth (degrees)")
    ax.set_ylabel("Depth (m)")
    ax.grid()
    ax.legend()
    ax.set_title('Plane locations in 2D space')
    # plt.show()

    # ----------------- 3D plot -----------------

    ax = fig.add_subplot(122, projection='3d')

    # Extract x, y, z coordinates for plotting
    for batch_index in range(p3D_gt.shape[0]):
        x_coords, y_coords, z_coords = p3D_gt[batch_index, :, 0], p3D_gt[batch_index, :, 1], p3D_gt[batch_index, :, 2]
        ax.scatter(x_coords, y_coords, z_coords, "o", label=f"Groundtruth Plane {batch_index}")
    if p3D_est is not None:
        for batch_index in range(p3D_est.shape[0]):
            p3D_est_dec = p3D_est[:, ::decim, :]
            x_coords, y_coords, z_coords = p3D_est_dec[batch_index, :,
                                                       0], p3D_est_dec[batch_index, :, 1], p3D_est_dec[batch_index, :, 2]
            ax.scatter(x_coords, y_coords, z_coords, "+", label=f"Estimated Plane points {batch_index}", marker='x')
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Plane locations in 3D space')
    plt.show()

    # ----------------- 3D quiver -----------------
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    # colors = ['r', 'g', 'b', 'y']
    # colors = ["tab:orange", "tab:blue", "tab:green", "tab:red"]
    colors = list(mcolors.TABLEAU_COLORS)
    # Extract x, y, z coordinates for plotting
    t3d = (p3D_gt[..., 1:, :] - p3D_gt[..., :-1, :])/3.
    for batch_index in range(p3D_gt.shape[0]):
        x_coords, y_coords, z_coords = p3D_gt[batch_index, :-1,
                                              0], p3D_gt[batch_index, :-1, 1], p3D_gt[batch_index, :-1, 2]
        tangent3d_x, tangent3d_y, tangent3d_z = t3d[batch_index, :, 0], t3d[batch_index, :, 1], t3d[batch_index, :, 2]
        factor = 50.
        dec = 20
        ax.scatter(
            x_coords[::dec], y_coords[::dec], z_coords[::dec],
            color=colors[batch_index % len(colors)],
            label=f"Groundtruth Plane tangents {batch_index}")
        ax.quiver(x_coords[::dec], y_coords[::dec], z_coords[::dec],
                  factor*tangent3d_x[::dec], factor*tangent3d_y[::dec], factor*tangent3d_z[::dec],
                  color=colors[batch_index % len(colors)],
                  arrow_length_ratio=0.1,
                  #   arrow
                  label=f"Plane {batch_index}")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    ax.set_title('Plane tangents in 3D space')
    plt.show()


def validation_of_3d_tangent_estimation(
    azimuth_coordinates_phi: torch.Tensor,
    altitude_z: torch.Tensor,
    p3D_gt: torch.Tensor = None,
    batches_index: list = None
):
    """
    Computation of the tangent vectors using the jacobian of `image_vector_to_3d_plane_tangent`
    check that the jacobian computation of the 3D tangent
    matches with the finite differences computation.
    """
    if batches_index is None:
        batches_index = range(azimuth_coordinates_phi.shape[0])
    for batch_idx in batches_index:
        estimated_grad_list = []
        if p3D_gt is not None:
            tangent3d = (p3D_gt[..., 1:, :] - p3D_gt[..., :-1, :])
        for element_idx in range(azimuth_coordinates_phi.shape[1]-1):
            azi, alt = azimuth_coordinates_phi[batch_idx, element_idx], altitude_z[batch_idx, element_idx]
            delta_azi = azimuth_coordinates_phi[batch_idx, element_idx+1] - \
                azimuth_coordinates_phi[batch_idx, element_idx]
            delta_alt = altitude_z[batch_idx, element_idx+1] - altitude_z[batch_idx, element_idx]
            estimated_grad = image_vector_to_3d_plane_tangent(azi, alt, delta_azi, delta_alt)
            estimated_grad_list.append(estimated_grad)
        estimated_grad_list = torch.stack(estimated_grad_list)
        for dim_idx, dim_name, dim_color in zip(range(3), "xyz", "rgb"):
            plt.plot(
                torch.rad2deg(azimuth_coordinates_phi[batch_idx, :-1]), estimated_grad_list[:, dim_idx],
                color=dim_color, label=f"tangent {dim_name}"
            )
            if p3D_gt is not None:
                plt.plot(
                    torch.rad2deg(azimuth_coordinates_phi[batch_idx, :-1]), tangent3d[batch_idx, :, dim_idx],
                    "--",
                    color=dim_color,
                    linewidth=4, alpha=0.5, label=f"tangent {dim_name} (gt)"
                )
        plt.xlabel("Azimuth (degrees)")
        plt.ylabel("3D Tangent")
        plt.legend()
        plt.grid()
        plt.title(f"Estimated 3D tangent by finite differences vs computing using the jacobian - slide {batch_idx}")
        plt.show()
