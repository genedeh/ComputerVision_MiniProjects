import tempfile
import cv2
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


def process_video(input_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # safer for browsers
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    out = cv2.VideoWriter(
        out_path,
        fourcc,
        int(cap.get(cv2.CAP_PROP_FPS)),
        (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Example processing
        cv2.rectangle(frame, (50, 50), (200, 200), (0, 255, 0), 2)
        out.write(frame)

    cap.release()
    out.release()  # very important to fully close before reading

    return out_path
