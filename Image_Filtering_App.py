import streamlit as st
from PIL import Image
import numpy as np
import requests
from utils.filters import ImageFilters
from utils.utils import convert_to_pil

st.title("🖼️ Image Filter Playground")


uploaded_file = st.file_uploader(
    "Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
else:
    image = Image.open(requests.get(
        "https://picsum.photos/400/300", stream=True).raw)

# Instantiate filter class
processor = ImageFilters(image)

# Sidebar for filter selection
filter_option = st.sidebar.selectbox("Choose a filter",  [
    "Original", "Canny Edge Detection", "Otsu Thresholding", "Blur", "Contour Detection", "Template Matching"
])
# Optional dynamic sliders based on filter
if filter_option == "Otsu Thresholding":
    output = processor.otsu_threshold()
elif filter_option == "Canny Edge Detection":
    t1 = st.sidebar.slider("Threshold #1", 50, 300, 100)
    t2 = st.sidebar.slider("Threshold #2", 50, 300, 200)
    output = processor.canny_edge(t1, t2)
elif filter_option == "Blur":
    blurSizeType = st.sidebar.selectbox("Blur Level", [
        "5", "23", "25"
    ], 0)
    kSize = int(blurSizeType)
    output = processor.gaussian_blur(kSize=kSize)
elif filter_option == "Contour Detection":
    st.sidebar.subheader("Contour Settings")
    t1 = st.sidebar.slider("Canny Threshold1", 0, 255, 50)
    t2 = st.sidebar.slider("Canny Threshold2", 0, 255, 150)

    retrieval_mode = st.sidebar.selectbox("Retrieval Mode", [
        "RETR_EXTERNAL", "RETR_LIST", "RETR_TREE", "RETR_CCOMP"
    ])

    if st.sidebar.button("Detect Contours"):
        contours_img, gray = processor.find_contours(
            t1, t2, retrieval_mode)

        tab1, tab2, tab3 = st.tabs([
            "Contours", "Grayscale", "Original"
        ])
        tab1.image(contours_img, caption="Detected Contours",
                   use_column_width=True)
        tab2.image(gray, caption="Grayscale Image", use_column_width=True)
        tab3.image(image, caption="Original Image", use_column_width=True)

    else:
        st.info("Click the button to detect contours.")
elif filter_option == "Template Matching":
    st.sidebar.subheader("Upload Images")
    main_file = st.sidebar.file_uploader(
        "Main Image", type=["png", "jpg", "jpeg"])
    template_file = st.sidebar.file_uploader(
        "Template Image", type=["png", "jpg", "jpeg"])

    if main_file and template_file:
        main_img = Image.open(main_file).convert("RGB")
        template_img = Image.open(template_file).convert("RGB")

        processor = ImageFilters(main_img)  # Main image processor


        result_img = processor.template_match(
            np.array(template_img))

        tab1, tab2 = st.tabs(
            ["Matched Result", "Main Image"])
        tab1.image(result_img, caption="Matched Region", use_column_width=True)
        tab2.image(main_img, caption="Original Main Image",
                   use_column_width=True)
    else:
        st.info("Please upload both the main image and the template image.")
else:
    output = np.array(image)

# Display side-by-side
if filter_option not in ["Template Matching", "Contour Detection"]:
    tab1, tab2 = st.tabs(["Filtered Image", "Original Image"])
    tab1.image(convert_to_pil(output), use_column_width=True)
    tab2.image(image, use_column_width=True)
