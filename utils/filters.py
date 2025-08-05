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

    def watershed_segmentation(self):
        _, thresh = cv2.threshold(
          self.gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(
          dist_transform, 0.7 * dist_transform.max(), 255, 0)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1 
        markers[unknown == 255] = 0

        image_bgr = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        markers = cv2.watershed(image_bgr, markers)

        segmented_img = image_bgr.copy()
        segmented_img[markers == -1] = [0, 225, 0]  # green boundary

        return cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)

    def convert_color(self, mode="RGB"):
        if mode == "RGB":
          return self.image  # Already in RGB
        elif mode == "BGR":
          return cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
        elif mode == "Grayscale":
          return self.gray_image
        elif mode == "HSV":
          return cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        else:
          raise ValueError("Unsupported color mode selected")
    
    def resize(self, width=None, height=None):
        h, w = self.image.shape[:2]
        width = width or w
        height = height or h
        return cv2.resize(self.image, (width, height))

    def crop(self, x1=0, y1=0, x2=None, y2=None):
        h, w = self.image.shape[:2]
        x2 = x2 or w
        y2 = y2 or h
        return self.image[y1:y2, x1:x2]

    def apply_mask(self):
        h, w = self.image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        radius = min(w, h) // 4
        cv2.circle(mask, center, radius, 255, -1)

        masked = cv2.bitwise_and(self.image, self.image, mask=mask)
        return masked

    def adjust_brightness_contrast(self, brightness=0, contrast=0):
        img = np.int16(self.image)
        img = img * (contrast / 127 + 1) - contrast + brightness
        img = np.clip(img, 0, 255)
        return np.uint8(img)
