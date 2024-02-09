from skimage import filters
import numpy as np
from typing import Tuple


def extract_2d_gradients_from_image(
    img: np.ndarray, threshold_magnitude: float = 0.02,
    plain: bool = False
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
    img_grad_h[~high_amplitude] = np.NaN
    img_grad_v[~high_amplitude] = np.NaN
    img_grad[~high_amplitude] = np.NaN
    # Make tangents go from legt to right
    img_grad_h *= -np.sign(img_grad_v)
    img_grad_v *= -np.sign(img_grad_v)
    if plain:
        return img_grad_h, img_grad_v, img_grad, None
    coords = np.array(np.where(high_amplitude)).T
    img_grad_h = img_grad_h[high_amplitude]
    img_grad_v = img_grad_v[high_amplitude]
    img_grad = img_grad[high_amplitude]
    # gradients_image_scales = np.array([img_grad_h[high_amplitude], img_grad_v[high_amplitude]]).T
    # gradients_unscaled = gradients_image_scales / np.linalg.norm(gradients_image_scales, axis=1)[:, None]
    # tangents_unscaled = np.array([-img_grad_v[high_amplitude], img_grad_h[high_amplitude]]).T
    # tangents_unscaled = tangents_unscaled / np.linalg.norm(tangents_unscaled, axis=1)[:, None]

    return img_grad_h, img_grad_v, img_grad, coords
