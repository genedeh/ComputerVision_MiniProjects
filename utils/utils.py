from io import BytesIO
import numpy as np
from PIL import Image


def convert_to_pil(image_array):
    return Image.fromarray(image_array if image_array.ndim == 3 else np.stack([image_array]*3, axis=-1))


def get_image_download_link(np_image, filename="processed.png"):
    if len(np_image.shape) == 2: 
        img_pil = Image.fromarray(np_image).convert("RGB")
    else:
        img_pil = Image.fromarray(np_image)

    # Convert to bytes
    buf = BytesIO()
    img_pil.save(buf, format="PNG")
    byte_im = buf.getvalue()

    return byte_im
