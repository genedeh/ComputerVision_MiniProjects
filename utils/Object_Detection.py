import cv2
import time

human_hog = cv2.HOGDescriptor()
human_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

car_tracker = {}

car_cascade = cv2.CascadeClassifier("cascades/cars.xml")

if car_cascade.empty():
    raise IOError("Failed to load car cascade. Check the file path.")



def detect_humans(image):
    """Detect humans in an image using HOG + SVM."""
    rects, _ = human_hog.detectMultiScale(image,
                                          winStride=(8, 8),
                                          padding=(16, 16),
                                          scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  
    return image


def detect_cars(image):
    """Detect cars in an image using Haar Cascade (HOG for cars is rare in OpenCV)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    for (x, y, w, h) in cars:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  
    return image


def detect_general(image):
    """Detect both humans and cars with different colors."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ---- Humans ----
    rects, weights = human_hog.detectMultiScale(
        image,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05
    )
    for (x, y, w, h), weight in zip(rects, weights):
        label = f"Human {weight * 100:.1f}%"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # ---- Cars ----
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    current_time = time.time()
    for (x, y, w, h) in cars:
        car_id = f"{x}_{y}_{w}_{h}"  # Unique-ish key for each detected car

        # Update tracking time
        if car_id not in car_tracker:
            car_tracker[car_id] = current_time
        elapsed = current_time - car_tracker[car_id]

        label = f"Car {elapsed:.1f}s"
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return image
