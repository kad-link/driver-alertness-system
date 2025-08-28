import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load trained model
model = load_model("drowsiness_model_colab.h5")

# Parameters
IMG_SIZE = (64, 64)
SEQUENCE_LENGTH = 20

st.set_page_config(page_title="Driver Alertness System", layout="centered")
st.title("Driver Alertness Detection (Real-time)")

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frames = deque(maxlen=SEQUENCE_LENGTH)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        resized = cv2.resize(img, IMG_SIZE)
        resized = resized / 255.0
        self.frames.append(resized)

        label = "WAITING..."
        color = (255, 255, 0)

        if len(self.frames) == SEQUENCE_LENGTH:
            X = np.expand_dims(self.frames, axis=0)
            pred = model.predict(X)[0][0]

            if pred > 0.5:
                label = "DROWSY"
                color = (0, 0, 255)
            else:
                label = "AWAKE"
                color = (0, 255, 0)

       
        cv2.putText(img, label, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        return img

webrtc_streamer(
    key="drowsiness-detection",
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False}
)
