import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from constants import ABSENT_VALUE, DEPTH_STEP
from pathlib import Path
from plane_cylinder_projections import get_tangent_vec_from_gradients, angle_to_3d_vector, normal_vector_to_angles
from plane_extraction import (
    get_cross_products, compute_3d_tangent_vectors, extract_dip_azimuth, extract_dip_azimuth_by_plane_normal_estimation
)
import torch
from visualizations import (plot_ground_truth_3d, show_borehole_image,
                            show_gradients_magnitudes, plot_3d_scatter,
                            visualize_accumulator, plot_tangents_and_gradients_field)
from simulations import create_planes_projection
from image_processing import extract_2d_gradients_from_image
from tqdm import tqdm
from typing import Tuple
HERE = Path(__file__).parent
DATA = HERE/'data'


def load_data(file_name: str = "5010_5110"):
    image_input = np.load(DATA/f'FMI_STAT_{file_name}.npy')
    num_rows_total, num_columns_total = image_input.shape
    tdep = np.load(DATA/f'TDEP_{file_name}.npy')
    mask_absent = (image_input == ABSENT_VALUE)
    image_display = image_input.copy()
    image_display[mask_absent] = np.nan
    return image_input, mask_absent, image_display, tdep


def extract_2d_gradients(img, **kwargs):
    img_grad = filters.sobel(img)
    show_gradients_magnitudes(img_grad, **kwargs)


def process_roi(image_display, roi_def=[500, 1300], debug: bool = False, out_path: Path = None, method: int = 1):
    assert method in [1, 2], "Method must be 1 or 2"
    out_name = f'roi_{roi_def[0]:06d}_{roi_def[1]:06d}_method_{method}'
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
    if debug:
        img_grad_h, img_grad_v, img_grad, coords = extract_2d_gradients_from_image(
            roi,
            threshold_magnitude=5000.,
            thresold_abs_v_grad=3000
        )
        plot_tangents_and_gradients_field(coords, img_grad_h, img_grad_v, roi, decimation=3,
                                          title='Tangents and gradients field')
    tan_vec_2d = get_tangent_vec_from_gradients(img_grad_h, img_grad_v, normalize=False)
    img_coords = torch.from_numpy(coords).float()
    p3D_est = angle_to_3d_vector(np.deg2rad(img_coords[:, 1]), -img_coords[:, 0]*DEPTH_STEP).unsqueeze(0)
    if debug:
        plot_3d_scatter(p3D_est[:, ::2], title="Estimated 3D locations of high gradients",
                        forced_color="tab:orange", alpha=0.2)
    tangents_3d = compute_3d_tangent_vectors(coords, tan_vec_2d)
    if method == 1:
        cross_product_estimated, cross_product_norm_estim = get_cross_products(
            tangents_3d,
            num_points=50000
            # num_points=10000
        )
        dip_az_estim = normal_vector_to_angles(cross_product_estimated)
        weight_cross_product = cross_product_norm_estim.squeeze(-1)
        weight = weight_cross_product
        # IDEALLY WE SHOULD ALSO WEIGHT BY GRADIENT NORM 
        # requires refactoring the function get_cross_products
        # weight_gradient_norms = torch.from_numpy(img_grad/img_grad.sum()).float().unsqueeze(0)
        # weight = weight_cross_product*weight_gradient_norms
        best_dip, best_azimuth, histo, bin_edges = extract_dip_azimuth(
            dip_az_estim,
            bins=[20, 20],
            weights=weight
        )
        visualize_accumulator(
            histo, bin_edges, best_dip, best_azimuth,
            out_path=None if out_path is None else out_path/(out_name+'_accumulator.png')
        )
    elif method == 2:
        dip_az_estim = extract_dip_azimuth_by_plane_normal_estimation(
            tangents_3d,
            weights=torch.from_numpy(img_grad/img_grad.sum()).float().unsqueeze(0)
        )
        best_dip, best_azimuth = dip_az_estim[0, 0], dip_az_estim[0, 1]
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


def main(
    file_name: str = "5010_5110",
    debug: bool = False,
    out_folder: Path = HERE/'results',
    forced_roi: Tuple[int, int] = None,
    roi_size: int = 800,
    method: str = 1
):
    assert method in [1, 2], "Method must be 1 or 2"
    if out_folder is not None:
        out_folder.mkdir(exist_ok=True, parents=True)
    image_input, mask_absent, image_display, tdep = load_data(file_name=file_name)

    if forced_roi is not None:
        process_roi(image_display, forced_roi, debug=debug, out_path=out_folder, method=method)
        return
    for roi_start in tqdm(range(0, image_display.shape[0]-roi_size, roi_size)):
        roi_def = [roi_start, roi_start+roi_size]
        process_roi(image_display, roi_def, debug=debug, out_path=out_folder, method=method)


if __name__ == "__main__":
    main(forced_roi=[1350, 1700], out_folder=HERE/'results_manual', method=1)
    main(out_folder=HERE/'__results_roi_size_200', roi_size=200, method=1)
    # main(out_folder=HERE/'results_roi_size_400', roi_size=400)
    # main(forced_roi=[1350, 1700], out_folder=HERE/'results_manual', method=1)
