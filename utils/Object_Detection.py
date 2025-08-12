import cv2

human_hog = cv2.HOGDescriptor()
human_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

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

    # Humans
    rects, _ = human_hog.detectMultiScale(image,
                                          winStride=(8, 8),
                                          padding=(16, 16),
                                          scale=1.05)
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Cars
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)
    for (x, y, w, h) in cars:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return image
