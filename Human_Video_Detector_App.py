import streamlit as st
import cv2
import tempfile
import os
import numpy as np
from utils.Object_Detection import detect_humans, detect_cars, detect_general

st.title("🚀 HOG Object Detection App")
st.write("Upload an image or video and detect **Humans**, **Cars**, or **Both**.")

option = st.selectbox("Select detection type:", ["Humans", "Cars", "General"])

uploaded_file = st.file_uploader("Upload Image or Video", type=[
                                 "jpg", "jpeg", "png", "mp4", "avi"])


if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension in ["jpg", "jpeg", "png"]:
        # IMAGE
        file_bytes = np.asarray(
            bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        if option == "Humans":
            output_image = detect_humans(image)
        elif option == "Cars":
            output_image = detect_cars(image)
        else:
            output_image = detect_general(image)

        st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB),
                 caption="Processed Image")

    elif file_extension in ["mp4", "avi"]:
        # VIDEO
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = "output.mp4"
        out = cv2.VideoWriter(out_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                              (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        # Get total number of frames in the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0
        progress_bar = st.progress(0)

        while current_frame < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if option == "Humans":
                frame = detect_humans(frame)
            elif option == "Cars":
                frame = detect_cars(frame)
            else:
                frame = detect_general(frame)

            print(f"Processing frame {current_frame + 1}/{total_frames}")
            out.write(frame)
            current_frame += 1
            progress = int((current_frame / total_frames) * 100)
            progress_bar.progress(progress)


        cap.release()
        out.release()
        st.success("Processing complete!")

        st.video(out_path)
        os.remove(tfile.name)
