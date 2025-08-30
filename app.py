import streamlit as st
import cv2
import numpy as np
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import time
from collections import deque

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.warning("TensorFlow not available - running with basic detection only")

IMG_SIZE = (32, 32)
SEQUENCE_LENGTH = 10

@st.cache_resource
def load_cascades():
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        return face_cascade, eye_cascade
    except Exception as e:
        st.error(f"Error loading Haar cascades: {e}")
        return None, None

@st.cache_resource
def load_model():
    if not TF_AVAILABLE:
        return None
        
    model_path = "drowsiness_model.h5"
    if not os.path.exists(model_path):
        return None
    
    try:
        tf.keras.utils.disable_interactive_logging()
        
        cnn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu')
        ])

        model = tf.keras.Sequential([
            tf.keras.layers.TimeDistributed(cnn, input_shape=(10, 32, 32, 3)),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.load_weights(model_path)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

class SimpleEyeDetector:
    def __init__(self, face_cascade, eye_cascade):
        self.face_cascade = face_cascade
        self.eye_cascade = eye_cascade
        self.eye_closure_threshold = 0.5
        self.closed_eye_start_time = None
        self.eyes_currently_closed = False
        self.alert_threshold_seconds = 2.0
        
    def detect_eye_closure(self, frame):
        if self.face_cascade is None or self.eye_cascade is None:
            return False, False, "Detection unavailable", 0.0
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) == 0:
                self.closed_eye_start_time = None
                self.eyes_currently_closed = False
                return False, False, "No face detected", 0.0
            
            current_time = time.time()
            (x, y, w, h) = faces[0]
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
            eyes_closed = len(eyes) < 2
            
            if eyes_closed:
                if not self.eyes_currently_closed:
                    self.closed_eye_start_time = current_time
                    self.eyes_currently_closed = True
                    closed_duration = 0.0
                else:
                    closed_duration = current_time - self.closed_eye_start_time
                status = f"Eyes closed ({closed_duration:.1f}s)"
            else:
                self.closed_eye_start_time = None
                self.eyes_currently_closed = False
                closed_duration = 0.0
                status = "Eyes open"
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            return True, eyes_closed, status, closed_duration
            
        except Exception as e:
            return False, False, f"Error: {str(e)}", 0.0

class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self, model, face_cascade, eye_cascade):
        self.model = model
        self.eye_detector = SimpleEyeDetector(face_cascade, eye_cascade)
        self.frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.frame_count = 0
        self.last_status = "Starting..."
        self.last_closed_duration = 0.0
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        self.frame_count += 1
        
        if self.frame_count % 3 == 0:
            try:
                face_detected, eyes_closed, status, closed_duration = self.eye_detector.detect_eye_closure(img)
                
                if face_detected:
                    if eyes_closed and closed_duration >= 2.0:
                        status = f"WAKE UP! Eyes closed {closed_duration:.1f}s"
                        color = (0, 0, 255)
                    elif eyes_closed:
                        color = (0, 165, 255)
                    else:
                        color = (0, 255, 0)
                else:
                    color = (255, 0, 255)
                
                self.last_status = status
                self.last_closed_duration = closed_duration
                
            except Exception:
                color = (255, 255, 255)
                self.last_status = "Processing..."
        
        cv2.putText(img, f"Status: {self.last_status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if self.last_closed_duration > 0:
            timer_text = f"Closed: {self.last_closed_duration:.1f}s"
            cv2.putText(img, timer_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.set_page_config(page_title="Drowsiness Detection", layout="wide")
    
    st.title("Driver Drowsiness Detection System")
    st.markdown("Check your infotainment screen for real-time monitoring")
    
    face_cascade, eye_cascade = load_cascades()
    model = load_model()
    
    if face_cascade is None or eye_cascade is None:
        st.error("Failed to load detection cascades")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Video")
        
        if model:
            st.success("Drive cautiously")
        else:
            st.info("Using basic detection only")
        
        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        webrtc_ctx = webrtc_streamer(
            key="drowsiness-detection",
            video_processor_factory=lambda: DrowsinessProcessor(model, face_cascade, eye_cascade),
            rtc_configuration=rtc_configuration,
            media_stream_constraints={
                "video": {"width": 640, "height": 480},
                "audio": False
            }
        )
    
    with col2:
        st.subheader("Detection Info")
        
        st.info("""
        **Features:**
        - Real-time Face detection
        - Eye closure monitoring
        - 2-second alert threshold
        
        **Instructions:**
        1. Allow camera access
        2. Position face in view
        3. Close eyes for 2+ seconds to test alert
        """)
        
        if webrtc_ctx.state.playing:
            st.success("Camera Active")
        else:
            st.error("Camera Inactive")

if __name__ == "__main__":
    main()