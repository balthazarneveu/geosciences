from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
from constants import DEPTH_STEP
import matplotlib.colors as mcolors
COLORS = list(mcolors.TABLEAU_COLORS)


def plot_ground_truth_3d(
        azimuth_coordinates_phi,
        altitude_z,
        p3D_gt,
        img_coords=None,
        p3D_est=None,
        name="Ground truth"):
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
        ax.scatter(x_coords, y_coords, z_coords, "o", label=f"{name} Plane {batch_index}")
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
    # Extract x, y, z coordinates for plotting
    t3d = (p3D_gt[..., 1:, :] - p3D_gt[..., :-1, :])/3.
    colors = COLORS
    for batch_index in range(p3D_gt.shape[0]):
        x_coords, y_coords, z_coords = p3D_gt[batch_index, :-1,
                                              0], p3D_gt[batch_index, :-1, 1], p3D_gt[batch_index, :-1, 2]
        tangent3d_x, tangent3d_y, tangent3d_z = t3d[batch_index, :, 0], t3d[batch_index, :, 1], t3d[batch_index, :, 2]
        factor = 50.
        dec = 20
        ax.scatter(
            x_coords[::dec], y_coords[::dec], z_coords[::dec],
            color=colors[batch_index % len(colors)],
            label=f"{name} Plane tangents {batch_index}")
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
    estimated_grad_list_all: torch.Tensor,
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
        estimated_grad_list = estimated_grad_list_all[batch_idx, ...]
        if p3D_gt is not None:
            tangent3d = (p3D_gt[..., 1:, :] - p3D_gt[..., :-1, :])
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


def show_borehole_image(img: np.ndarray, title: str = 'Simulated borehole image'):
    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.xlabel('Azimuth (°)')
    plt.ylabel('Depth = Negative altitude')
    plt.imshow(
        img,
        cmap='hot'
    )
    plt.show()


def show_gradients_magnitudes(img_grad: np.ndarray, bins: int = 20):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_grad, cmap='hot')
    plt.title('Gradient magnitudes map')
    plt.subplot(1, 2, 2)
    plt.hist(img_grad.flatten(), bins=bins)
    plt.title('Histogram of image gradient magnitude')
    plt.grid()
    plt.show()


def plot_tangents_and_gradients_field(
    coords: np.ndarray,
    img_grad_h: np.ndarray,
    img_grad_v: np.ndarray,
    img: np.ndarray,
    decimation: int = 100,
    title='2D tangents to the projected 3D points'
):
    gradients_unscaled = np.array([img_grad_h, img_grad_v]).T
    gradients_unscaled = gradients_unscaled / np.linalg.norm(gradients_unscaled, axis=1)[:, None]
    tangents_unscaled = np.array([-img_grad_v, img_grad_h]).T
    tangents_unscaled = tangents_unscaled / np.linalg.norm(tangents_unscaled, axis=1)[:, None]
    plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap='hot')
    for coord_idx in range(coords.shape[0]):
        if coord_idx % decimation != 0:
            continue
        plt.arrow(
            coords[coord_idx, 1], coords[coord_idx, 0],
            20.*gradients_unscaled[coord_idx, 0],
            -20.*gradients_unscaled[coord_idx, 1],
            head_width=2., head_length=3.,
            fc='b',
            ec='tab:orange',
        )
        plt.arrow(
            coords[coord_idx, 1], coords[coord_idx, 0],
            -20.*tangents_unscaled[coord_idx, 0],
            20.*tangents_unscaled[coord_idx, 1],
            head_width=2., head_length=3.,
            fc='b',
            ec='tab:cyan',
            alpha=0.5
        )
    plt.title(title)
    plt.show()


def plot_3d_scatter(
    point_cloud=None,
    sizes=None,
    forced_color=None,
    alpha=1.,
    label="Plane 3D points",
    title='Plane tangents in 3D space',
    label_vector="Normal vector",
    vects: np.ndarray = None
):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    colors = COLORS
    if point_cloud is not None:
        for batch_index in range(point_cloud.shape[0]):
            color = colors[batch_index % len(colors)]
            tangent3d_x = point_cloud[batch_index, :, 0]
            tangent3d_y = point_cloud[batch_index, :, 1]
            tangent3d_z = point_cloud[batch_index, :, 2]
            ax.scatter(
                tangent3d_x,
                tangent3d_y,
                tangent3d_z,
                # s=sizes,
                c=color if forced_color is None else forced_color,
                label=f"{label} {batch_index}",
                alpha=alpha
            )
            # size=cross_product_norm[batch_index, :, 0].numpy()

    if vects is not None:
        for batch_index in range(vects.shape[0]):
            color = colors[batch_index % len(colors)]
            ax.quiver(
                0, 0, 0, vects[batch_index, 0], vects[batch_index, 1], vects[batch_index, 2],
                colors=color,
                label=f"{label_vector} {batch_index}",
            )
    plt.plot(0, 0, 0, "o", color="k", label="origin")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    ax.set_title(title)
    plt.show()


def visualize_accumulator(histo, bin_edges, best_dip, best_azimuth, out_path: Path = None):
    plt.figure(figsize=(5, 5))
    plt.imshow(
        histo,
        extent=[
            np.rad2deg(bin_edges[1][0]), np.rad2deg(bin_edges[1][-1]),
            np.rad2deg(bin_edges[0][-1]), np.rad2deg(bin_edges[0][0]),
        ],
        aspect='auto')
    plt.plot(
        np.rad2deg(best_azimuth),
        np.rad2deg(best_dip),
        marker="x", color="red", markersize=10, label="Mode")
    plt.title(f"Estimated dip: {np.rad2deg(best_dip):.3f}° and azimuth: {np.rad2deg(best_azimuth):.3f}°")
    plt.legend()
    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()
