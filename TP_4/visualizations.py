from matplotlib import pyplot as plt
import numpy as np
import torch
from constants import DEPTH_STEP


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
    colors = ['r', 'g', 'b', 'y']
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
