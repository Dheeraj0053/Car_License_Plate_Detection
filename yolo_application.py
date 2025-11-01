import streamlit as st
from ultralytics import YOLO
import cv2
import os
import tempfile
import datetime
from PIL import Image
import numpy as np
import easyocr

st.set_page_config(page_title="License Plate Detection", page_icon="🚗", layout="wide")

MODEL_PATH = r"C:\Users\dk675\OneDrive\Desktop\License_Plate _Detection\best.pt"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()
ocr_reader = easyocr.Reader(['en'])

st.markdown("""
    <style>
        .main-title {
            font-size: 2rem;
            color: #0a1931;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
        }
        .sub-title {
            text-align: center;
            color: #555;
            font-size: 1rem;
            margin-bottom: 2rem;
        }
        img.display-img {
            max-width: 600px;
            width: 100%;
            border-radius: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }
        [data-testid="stAppViewContainer"] {
            background-color: #f9f9fb;
        }
    </style>
""", unsafe_allow_html=True)


st.markdown('<h1 class="main-title">Car License Plate Detection</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload an image or video — the model will detect and read the license plate 🚘</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📂 Upload Image or Video", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])


def detect_license_plates_image(image_path):
    results = model.predict(source=image_path, conf=0.6, device="cpu", verbose=False)
    img = cv2.imread(image_path)
    detected_text = "Not Detected"
    confidence_value = 0.0

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            confidence_value = confidence * 100

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Confidence text above box
            label = f"{confidence_value:.1f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # OCR
            plate_crop = img[y1:y2, x1:x2]
            if plate_crop.size > 0:
                ocr_result = ocr_reader.readtext(plate_crop)
                if len(ocr_result) > 0:
                    detected_text = " ".join([text[1] for text in ocr_result])

    return img, confidence_value, detected_text


def detect_license_plates_video(video_path):
    cap = cv2.VideoCapture(video_path)
    detected_numbers = set()
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"video_result_{timestamp}.mp4")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    progress_text = st.empty()
    progress_bar = st.progress(0)
    preview = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        results = model.predict(source=frame, conf=0.6, device="cpu", verbose=False)
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0]) * 100

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{conf:.1f}%"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                plate_crop = frame[y1:y2, x1:x2]
                if plate_crop.size > 0:
                    ocr_result = ocr_reader.readtext(plate_crop)
                    if len(ocr_result) > 0:
                        detected_text = " ".join([text[1] for text in ocr_result])
                        detected_numbers.add(detected_text)

        out.write(frame)

        # Update progress 
        if frame_count % 5 == 0:
            progress_bar.progress(frame_count / total_frames)
            progress_text.text(f"Processing frame {frame_count}/{total_frames}")
            frame_resized = cv2.resize(frame, (640, 360))
            preview.image(frame_resized, channels="BGR", use_container_width=True)

    cap.release()
    out.release()
    progress_bar.empty()
    progress_text.empty()
    preview.empty()

    return output_path, detected_numbers



#  Main Application
if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_path = tmp_file.name

    if file_ext in ["jpg", "jpeg", "png"]:
        st.info("🧠 Detecting license plate from image...")
        img, conf, plate_text = detect_license_plates_image(temp_path)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"prediction_{timestamp}.jpg")
        cv2.imwrite(output_path, img)

        st.success("✅ Detection complete!")
        st.image(output_path, caption="Detected License Plate", width=600)
        st.write(f"**🔹 Model Confidence:** {conf:.2f}%")
        st.write(f"**🔹 Detected Plate Number:** {plate_text}")

        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Result Image", f, file_name=f"prediction_{timestamp}.jpg")

    elif file_ext in ["mp4", "mov", "avi"]:
        st.info("🎞 Processing video, please wait...")
        output_path, detected_numbers = detect_license_plates_video(temp_path)

        st.success("✅ Video processing complete!")
        st.video(output_path)
        if detected_numbers:
            st.write("**🔹 Detected License Plate Numbers:**")
            for num in detected_numbers:
                st.write(f"- {num}")
        else:
            st.warning("No license plates detected in the video.")

        with open(output_path, "rb") as f:
            st.download_button("⬇️ Download Processed Video", f, file_name=f"video_result.mp4")

st.markdown("""
<hr style="border: 0.5px solid #ccc;">
<p style='text-align: center; color: gray;'>
Developed with ❤️ using Streamlit, YOLOv8, and EasyOCR | © 2025
</p>
""", unsafe_allow_html=True)
