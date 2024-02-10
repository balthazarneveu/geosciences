import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from visualizations import show_borehole_image
import math
from skimage import filters
from visualizations import show_gradients_magnitudes
from constants import ABSENT_VALUE
from pathlib import Path
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


if __name__ == "__main__":
    image_input, mask_absent, image_display, tdep = load_data()
    roi = image_display[1600:2600]
    show_borehole_image(roi, title='Real borehole image')
    extract_2d_gradients(roi)
