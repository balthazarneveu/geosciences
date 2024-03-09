from PIL import Image, ImageDraw, ImageFont
import random
import numpy as np
import torch

from typing import Tuple

# letters = ["MVA", "SLB", "/", "\\", "-", "|", "+", "*"]
# letters = ["MVA", "SLB"]
# import string
# letters = string.ascii_letters  # Get all ASCII letters (uppercase and lowercase)


def generate_annotation(
    letters=["MVA", "SLB", "BN"],
    amplitude_rotation=10,
    amount_of_objects_range=[0, 5],
    font_range=[8, 14],  # [14, 20],
    h=36,
    w=36
) -> Tuple[torch.Tensor, torch.Tensor]:
    H, W = 2*h, 2*w
    background_color = 0  # Black

    # Reinitialize the image and drawing context
    image = Image.new('L', (W, H), color=background_color)
    draw = ImageDraw.Draw(image)

    # Add multiple letters with random gray levels and rotations

    amount_of_objects = np.random.randint(amount_of_objects_range[0],  amount_of_objects_range[1])
    for _ in range(amount_of_objects):

        # font_size = 12
        # font_size = random.randint(12, 20)
        # font_size = random.randint(8, 12)
        font_size = random.randint(font_range[0], font_range[1])
        font = ImageFont.truetype("arial.ttf", font_size)
        random_letter = random.choice(letters)
        text_size = draw.textsize(random_letter, font=font)

        random_gray_level = random.randint(0, 255)

        # Create a separate image for the letter to apply rotation
        letter_image = Image.new('L', (W, H), color=background_color)
        letter_draw = ImageDraw.Draw(letter_image)

        position_offset_x = random.randint(0, W - text_size[0])
        position_offset_y = random.randint(0, H - text_size[1])
        letter_draw.text(
            (position_offset_x, position_offset_y),
            random_letter,
            fill=random_gray_level,
            font=font
        )

        # Sample random rotatio
        rotation_angle = random.randint(-amplitude_rotation, amplitude_rotation)
        # Apply rotation
        if amplitude_rotation > 0:
            rotated_letter_image = letter_image.rotate(rotation_angle, expand=0, fillcolor=background_color)
        else:
            rotated_letter_image = letter_image
        # Combine the rotated letter image with the main image
        image = Image.blend(image, rotated_letter_image, alpha=0.5)

    # Generate the annotation mask for the combined letters
    image_np = np.array(image)
    mask = np.where(image_np != background_color, 1, 0)

    # Convert back to PyTorch tensors
    image_tensor = torch.tensor(image_np).float() / 255.  # Normalize to range [0, 1]

    # Generate circular images
    image_tensor[:, :w] += image_tensor[:, w:] * mask[:, w:]
    mask[:, :w] += mask[:, w:]
    image_tensor, mask = crop_center(image_tensor, mask, h=h, w=w)
    mask_tensor = torch.tensor(mask).float()
    roll_shift = random.randint(0, w-1)
    mask_tensor = torch.roll(mask_tensor, shifts=roll_shift, dims=-1)
    image_tensor = torch.roll(image_tensor, shifts=roll_shift, dims=-1)
    return image_tensor, mask_tensor


def crop_center(img, mask, h=36, w=36):
    return img[h//4:h//4+h, :w], mask[h//4:h//4+h, :w].clip(0, 1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def visualize_annotation(image_tensor, mask_tensor):
        # Visualize the image and the mask
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image_tensor, cmap='gray')
        axs[0].set_title('Image with Random Letters')
        axs[0].axis('off')

        axs[1].imshow(mask_tensor, cmap='gray')
        axs[1].set_title('Annotation Mask')
        axs[1].axis('off')

        plt.show()

    for _ in range(20):
        img, mask = generate_annotation()
        print(img.shape, mask.shape)
        visualize_annotation(img, mask)
