# task2.py

import streamlit as st
import cv2
from ultralytics import YOLO

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="YOLOv11 Object Tracking",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("Real-Time Object Detection and Tracking")
st.write("YOLOv11m + Streamlit + OpenCV")

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
model = YOLO("yolo11m.pt")

# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
st.sidebar.header("Settings")

source = st.sidebar.radio(
    "Select Source",
    ["Webcam", "Video File"]
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.4,
    0.1
)

# -----------------------------
# VIDEO UPLOAD
# -----------------------------
video_file = None

if source == "Video File":

    video_file = st.sidebar.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

# -----------------------------
# START BUTTON
# -----------------------------
start = st.button("Start Detection")

# -----------------------------
# FRAME DISPLAY
# -----------------------------
frame_window = st.empty()

# -----------------------------
# START DETECTION
# -----------------------------
if start:

    # Webcam
    if source == "Webcam":

        cap = cv2.VideoCapture(0)

    # Video File
    else:

        if video_file is None:

            st.warning("Please upload a video file.")
            st.stop()

        with open("temp_video.mp4", "wb") as f:

            f.write(video_file.read())

        cap = cv2.VideoCapture("temp_video.mp4")

    # -----------------------------
    # VIDEO SAVER
    # -----------------------------
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = "output_tracked.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (frame_width, frame_height)
    )

    # -----------------------------
    # PROCESS VIDEO
    # -----------------------------
    while cap.isOpened():

        success, frame = cap.read()

        if not success:
            break

        # -----------------------------
        # YOLO TRACKING
        # -----------------------------
        results = model.track(
            frame,
            persist=True,
            conf=confidence,
            verbose=False
        )

        # -----------------------------
        # DRAW DETECTIONS
        # -----------------------------
        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()

            track_ids = results[0].boxes.id.int().cpu().tolist()

            classes = results[0].boxes.cls.int().cpu().tolist()

            confidences = results[0].boxes.conf.cpu().tolist()

            for box, track_id, cls, conf in zip(
                boxes,
                track_ids,
                classes,
                confidences
            ):

                x1, y1, x2, y2 = map(int, box)

                class_name = model.names[cls]

                # Bounding Box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Label
                label = f"{class_name} | ID:{track_id} | {conf:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        # -----------------------------
        # SAVE FRAME
        # -----------------------------
        out.write(frame)

        # -----------------------------
        # DISPLAY FRAME
        # -----------------------------
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        frame_window.image(
            frame_rgb,
            channels="RGB",
            use_container_width=True
        )

    # -----------------------------
    # RELEASE
    # -----------------------------
    cap.release()

    out.release()

    # -----------------------------
    # SUCCESS MESSAGE
    # -----------------------------
    
    st.success("Detection Finished")

    # -----------------------------
    # DOWNLOAD OUTPUT VIDEO
    # -----------------------------
    with open(output_path, "rb") as file:

        st.download_button(
            label="Download Detected Video",
            data=file,
            file_name="tracked_output.mp4",
            mime="video/mp4"
        )
