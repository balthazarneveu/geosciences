import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from visualizations import show_borehole_image
import math
from skimage import filters
from visualizations import show_gradients_magnitudes
from constants import ABSENT_VALUE, DEPTH_STEP
from pathlib import Path
from plane_cylinder_projections import get_tangent_vec_from_gradients, angle_to_3d_vector, normal_vector_to_angles
from plane_extraction import get_cross_products, compute_3d_tangent_vectors, extract_dip_azimuth
import torch
from visualizations import (plot_ground_truth_3d, show_borehole_image,
                            show_gradients_magnitudes, plot_3d_scatter, COLORS,
                            visualize_accumulator)
from simulations import DEFAULT_PLANE_ANGLES, create_planes_projection
from image_processing import extract_2d_gradients_from_image
from tqdm import tqdm
from typing import Tuple
HERE = Path(__file__).parent
DATA = HERE/'data'


def load_data(file_name: str = "5010_5110"):
    image_input = np.load(DATA/f'FMI_STAT_{file_name}.npy')
    num_rows_total, num_columns_total = image_input.shape
    print(image_input.shape)
    tdep = np.load(DATA/f'TDEP_{file_name}.npy')
    mask_absent = (image_input == ABSENT_VALUE)
    image_display = image_input.copy()
    image_display[mask_absent] = np.nan
    return image_input, mask_absent, image_display, tdep


def extract_2d_gradients(img, **kwargs):
    img_grad = filters.sobel(img)
    show_gradients_magnitudes(img_grad, **kwargs)


def quick_check(roi_def=[1600, 2600]):
    image_input, mask_absent, image_display, tdep = load_data()
    roi = image_display[roi_def[0]:roi_def[1]]
    show_borehole_image(roi, title=f'Real borehole image {roi_def[0]/DEPTH_STEP}m - {roi_def[1]/DEPTH_STEP}m')
    extract_2d_gradients(roi)


def process_roi(image_display, roi_def=[500, 1300], debug: bool = False, out_path: Path = None):
    out_name = f'roi_{roi_def[0]:06d}_{roi_def[1]:06d}'
    roi = image_display[roi_def[0]:roi_def[1]]
    if debug:
        show_borehole_image(
            roi,
            title=f'Slice of real borehole image {roi_def[0]*DEPTH_STEP:.3f}m - {roi_def[1]*DEPTH_STEP:.3f}m')
    img_grad_h, img_grad_v, img_grad, coords = extract_2d_gradients_from_image(
        roi,
        threshold_magnitude=5000.,
        thresold_abs_v_grad=3000
    )
    tan_vec_2d = get_tangent_vec_from_gradients(img_grad_h, img_grad_v, normalize=False)
    img_coords = torch.from_numpy(coords).float()
    p3D_est = angle_to_3d_vector(np.deg2rad(img_coords[:, 1]), -img_coords[:, 0]*DEPTH_STEP).unsqueeze(0)
    if debug:
        plot_3d_scatter(p3D_est[:, ::2], title="Estimated 3D locations of high gradients",
                        forced_color="tab:orange", alpha=0.2)
    tangents_3d = compute_3d_tangent_vectors(coords, tan_vec_2d)
    cross_product_estimated, cross_product_norm_estim = get_cross_products(
        tangents_3d,
        num_points=50000
        # num_points=10000
    )
    dip_az_estim = normal_vector_to_angles(cross_product_estimated)
    # dip_az_estim[..., 1] = dip_az_estim[..., 1] % (np.pi)
    # best_dip, best_azimuth, histo, bin_edges = extract_dip_azimuth(dip_az_estim, bins=[20, 20])
    best_dip, best_azimuth, histo, bin_edges = extract_dip_azimuth(dip_az_estim, bins=[20, 20])
    visualize_accumulator(
        histo, bin_edges, best_dip, best_azimuth,
        out_path=None if out_path is None else out_path/(out_name+'_accumulator.png')
    )
    estimated_plane_angle = torch.tensor(
        [
            [best_dip, best_azimuth, 0.],
        ]
    )
    # estimated_normals = angles_to_normal_vector(estimated_plane_angle)
    azimuth_coordinates_phi, altitude_z = create_planes_projection(estimated_plane_angle)
    p3D_est = angle_to_3d_vector(azimuth_coordinates_phi, altitude_z=altitude_z)  # [N, L, 3]
    if debug:
        plot_ground_truth_3d(
            azimuth_coordinates_phi,
            altitude_z,
            p3D_est,
            name="Estimation"
        )
    plt.figure(figsize=(10, 10))
    plt.imshow(roi, cmap='hot')
    for offset in range(50, roi_def[-1]-roi_def[0]-50, 20):
        plt.plot(np.rad2deg(azimuth_coordinates_phi).T, offset-(altitude_z.T/DEPTH_STEP), 'k-', alpha=0.4)
    plt.title(f'Real borehole image: depth [{roi_def[0]*DEPTH_STEP:.3f}m - {roi_def[1]*DEPTH_STEP:.3f}m]' +
              f'\nEstimated plane: dip {np.rad2deg(best_dip):.3f}°, azimuth {np.rad2deg(best_azimuth):.3f}°')
    plt.xlabel('Azimuth (°)')
    plt.ylabel(f'Depth ({DEPTH_STEP:.3f}m/pixel)')
    if out_path is not None:
        plt.savefig(out_path/(out_name+'_dip_picking.png'))
    else:
        plt.show()


def main(debug: bool = False, out_folder: Path = HERE/'results', forced_roi: Tuple[int, int] = None, roi_size: int = 800):
    if out_folder is not None:
        out_folder.mkdir(exist_ok=True, parents=True)
    image_input, mask_absent, image_display, tdep = load_data()

    if forced_roi is not None:
        process_roi(image_display, forced_roi, debug=debug, out_path=out_folder)
        return
    for roi_start in tqdm(range(0, image_display.shape[0]-roi_size, roi_size)):
        roi_def = [roi_start, roi_start+roi_size]
        process_roi(image_display, roi_def, debug=debug, out_path=out_folder)


if __name__ == "__main__":
    # quick_check()
    main(out_folder=HERE/'results_roi_size_200', roi_size=200)
    # main(out_folder=HERE/'results_roi_size_400', roi_size=400)
    # main(forced_roi=[1350, 1700], out_folder=HERE/'results_manual')
