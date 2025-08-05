import cv2
import numpy as np
from PIL import Image


class ImageFilters:
    def __init__(self, image: Image.Image):
        self.image = np.array(image.convert("RGB"))  # Ensure consistent format
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)

    def otsu_threshold(self):
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(self.gray_image, (5, 5), 0)

        # Apply Otsu's thresholding
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return th

    def canny_edge(self, threshold1=100, threshold2=200):
        edges = cv2.Canny(self.gray_image, threshold1, threshold2)
        return edges

    def gaussian_blur(self, kSize=5):
        return cv2.GaussianBlur(self.image, (kSize, kSize), sigmaX=0)

    def find_contours(self, edge_thresh1=50, edge_thresh2=150, retrieval_mode="RETR_TREE"):

        # Canny edge detection
        edges = self.canny_edge(edge_thresh1, edge_thresh2)

        # Map retrieval mode string to cv2 constant
        mode_map = {
           "RETR_EXTERNAL": cv2.RETR_EXTERNAL,
           "RETR_LIST": cv2.RETR_LIST,
           "RETR_TREE": cv2.RETR_TREE,
           "RETR_CCOMP": cv2.RETR_CCOMP
        }

        contours, _ = cv2.findContours(
            edges, mode_map[retrieval_mode], cv2.CHAIN_APPROX_SIMPLE)

         # Draw contours on a copy of the original image
        contour_image = self.image.copy()
        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

        return contour_image, self.gray_image

    def template_match(self, template_image: np.ndarray, method_name="TM_CCOEFF_NORMED"):
        template_gray = cv2.cvtColor(template_image, cv2.COLOR_RGB2GRAY)

        w, h = template_gray.shape[::-1]
        method = getattr(cv2, method_name)

        res = cv2.matchTemplate(self.gray_image, template_gray, method)
        _, _, min_loc, max_loc = cv2.minMaxLoc(res)

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
          top_left = min_loc
        else:
          top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # Draw rectangle on original RGB image
        output_image = self.image.copy()
        cv2.rectangle(output_image, top_left, bottom_right, (0, 255, 0), 2)

        return output_image


