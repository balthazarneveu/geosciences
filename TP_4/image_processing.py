from skimage import filters
import numpy as np
from typing import Tuple


def extract_2d_gradients_from_image(
    img: np.ndarray,
    threshold_magnitude: float = 0.02,
    plain: bool = False,
    thresold_abs_v_grad: float = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts and filters the horizontal and vertical gradients from an image.

    Parameters:
        img (np.ndarray): Input image (h, w) - gray level.
        threshold_magnitude (float): Threshold magnitude for gradient filtering.
        plain (bool): return the plain gradients or the filtered ones.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing
        - horizontal gradient,
        - vertical gradient
        - overall gradient magnitude of the image.
        - coordinates of the high amplitude gradients.
    """
    img_grad = filters.sobel(img)
    img_grad_v = filters.sobel_h(img)  # find the horizontal edges  =vertical gradient
    img_grad_h = filters.sobel_v(img)  # find the vertical edges = horizontal gradient
    high_amplitude = (img_grad > threshold_magnitude)
    if thresold_abs_v_grad is not None:
        high_amplitude *= (np.fabs(img_grad_h) < thresold_abs_v_grad)
    img_grad_h[~high_amplitude] = np.NaN
    img_grad_v[~high_amplitude] = np.NaN
    img_grad[~high_amplitude] = np.NaN
    if plain:
        return img_grad_h, img_grad_v, img_grad, None
    coords = np.array(np.where(high_amplitude)).T
    img_grad_h = img_grad_h[high_amplitude]
    img_grad_v = img_grad_v[high_amplitude]
    img_grad = img_grad[high_amplitude]
    return img_grad_h, -img_grad_v, img_grad, coords
