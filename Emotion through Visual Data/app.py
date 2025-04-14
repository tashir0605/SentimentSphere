import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ------------------- CSS Styling for Contrast & Layout ------------------- #
st.markdown("""
<style>
    body {
        background-color: #000000;
    }

    .stApp {
        background-color: #000000;
        color: #f1f1f1;
        font-family: 'Segoe UI', sans-serif;
    }

    h1 {
        color: #ffffff;
        text-align: center;
        font-size: 2.5rem;
        margin-top: 1rem;
    }

    .stButton > button {
        background-color: #1e40af; /* Deep blue */
        color: #ffffff;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: bold;
        border: none;
        transition: 0.3s ease;
    }

    .stButton > button:hover {
        background-color: #1d4ed8; /* Slightly lighter blue */
    }

    .stImage > img {
        border-radius: 12px;
        box-shadow: 0 8px 16px rgba(255,255,255,0.1);
        margin-top: 1rem;
    }

    .emotion-label {
        font-size: 24px;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
        padding: 1rem;
        border-radius: 12px;
        background: #000000;
        color: #111111; /* Dark text on light background */
        box-shadow: 0 4px 12px rgba(255,255,255,0.2);
    }

    .markdown-text-container {
        color: #ffffff !important;
    }

    .block-container {
        padding: 2rem 1rem;
    }

    .element-container {
        margin-bottom: 1rem;
    }

    .stMarkdown h3 {
        color: #facc15; /* Highlight for section headers */
    }

    .stMarkdown ol li {
        margin: 0.5rem 0;
        color: #ffffff
    }
</style>
""", unsafe_allow_html=True)

# ------------------- Load Model and Cascade ------------------- #
@st.cache_resource
def load_resources():
    try:
        model = load_model('model.h5')
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return model, face_cascade
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None

# Emotion label colors for bounding boxes
emotion_colors = {
    'Angry': (255, 0, 0),        # Red
    'Disgust': (0, 200, 0),      # Green
    'Fear': (128, 0, 128),       # Purple
    'Happy': (255, 193, 7),      # Amber
    'Sad': (30, 144, 255),       # Dodger Blue
    'Surprise': (255, 105, 180), # Hot Pink
    'Neutral': (100, 100, 100)   # Gray
}

# ------------------- Session Setup ------------------- #
if 'model_loaded' not in st.session_state:
    model, face_cascade = load_resources()
    if model and face_cascade:
        st.session_state.model = model
        st.session_state.face_cascade = face_cascade
        st.session_state.model_loaded = True
        st.session_state.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    else:
        st.session_state.model_loaded = False

if 'running' not in st.session_state:
    st.session_state.running = False
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = "Neutral"

# ------------------- App Title ------------------- #
st.title("🎭 Real-time Emotion Recognition")

# ------------------- Button Logic ------------------- #
col_start, col_stop = st.columns([1, 1])
start_btn = col_start.button("▶️ Start Camera")
stop_btn = col_stop.button("⏹️ Stop Camera")

# ------------------- Create Layout Columns ------------------- #
col1, col2 = st.columns([1, 1])
frame_placeholder = col1.empty()
emotion_placeholder = col2.empty()
chart_placeholder = col2.empty()

# ------------------- Main Video Processing Function ------------------- #
def process_video():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = st.session_state.face_cascade.detectMultiScale(gray, 1.1, 5)

        preds = np.zeros(len(st.session_state.emotion_labels))

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            try:
                roi = cv2.resize(roi, (48, 48))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                preds = st.session_state.model.predict(roi, verbose=0)[0]
                emotion_idx = np.argmax(preds)
                emotion_label = st.session_state.emotion_labels[emotion_idx]
                st.session_state.current_emotion = emotion_label

                color = emotion_colors.get(emotion_label, (255, 255, 255))
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                label = f"{emotion_label}: {preds[emotion_idx]:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                label_y = max(y - 10, 0)
                cv2.rectangle(frame,
                              (x, label_y - text_size[1] - 10),
                              (x + text_size[0] + 10, label_y + 10),
                              (0, 0, 0), -1)
                cv2.putText(frame, label,
                            (x + 5, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            except Exception as e:
                st.warning(f"Face error: {e}")

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if len(faces) > 0:
            r, g, b = emotion_colors.get(st.session_state.current_emotion, (255, 255, 255))
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            emotion_placeholder.markdown(
                f'<div class="emotion-label" style="border: 2px solid {hex_color}; color: {hex_color};">'
                f'Detected Emotion: {st.session_state.current_emotion}</div>', unsafe_allow_html=True
            )
            df = pd.DataFrame({'Confidence': preds * 100}, index=st.session_state.emotion_labels)
            chart_placeholder.bar_chart(df)

        time.sleep(0.1)

    cap.release()
    frame_placeholder.empty()
    emotion_placeholder.empty()
    chart_placeholder.empty()

# ------------------- Start/Stop Logic ------------------- #
if start_btn:
    st.session_state.running = True
    process_video()

if stop_btn:
    st.session_state.running = False

# ------------------- Instructions ------------------- #
st.markdown("""
---
<div style='color: #111111; font-size: 1rem;'>
    <h3>📝 Instructions:</h3>
    <ol>
        <li>Click <strong>'Start Camera'</strong> to begin real-time emotion detection.</li>
        <li>Make sure your face is clearly visible and well-lit.</li>
        <li>Click <strong>'Stop Camera'</strong> to turn off the webcam feed.</li>
    </ol>
</div>
""", unsafe_allow_html=True)
