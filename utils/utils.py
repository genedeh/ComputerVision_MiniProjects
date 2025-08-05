import numpy as np
from PIL import Image


def convert_to_pil(image_array):
    return Image.fromarray(image_array if image_array.ndim == 3 else np.stack([image_array]*3, axis=-1))
