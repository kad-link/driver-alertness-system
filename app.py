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
    st.error("TensorFlow is required for this application!")

IMG_SIZE = (32, 32)
SEQUENCE_LENGTH = 10

@st.cache_resource
def load_model():
    if not TF_AVAILABLE:
        st.error("TensorFlow not available - cannot load LSTM+CNN model")
        return None
        
    model_path = "drowsiness_model.h5"
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found. Please ensure the trained model is available.")
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
        st.success("So far so good...")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

class LSTMDrowsinessDetector:
    def __init__(self, model):
        self.model = model
        self.frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.drowsiness_threshold = 0.5
        self.alert_threshold_seconds = 2.0
        self.drowsy_start_time = None
        self.currently_drowsy = False
        
    def preprocess_frame(self, frame):
        """Preprocess frame for LSTM+CNN model"""
        try:
            resized = cv2.resize(frame, IMG_SIZE)
            normalized = resized.astype(np.float32) / 255.0
            return normalized
        except Exception as e:
            print(f"Frame preprocessing error: {e}")
            return np.zeros((*IMG_SIZE, 3), dtype=np.float32)
    
    def detect_drowsiness(self, frame):
        """Main drowsiness detection using LSTM+CNN model"""
        if self.model is None:
            return False, False, "Model not available", 0.0, 0.0
        
        try:
            current_time = time.time()
            
            processed_frame = self.preprocess_frame(frame)
            self.frame_sequence.append(processed_frame)
            
            if len(self.frame_sequence) < SEQUENCE_LENGTH:
                remaining = SEQUENCE_LENGTH - len(self.frame_sequence)
                return True, False, f"Initializing... ({remaining} frames)", 0.0, 0.0
            
            sequence = np.array(list(self.frame_sequence))
            sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
            
            drowsiness_prob = self.model.predict(sequence, verbose=0)[0][0]
            is_drowsy = drowsiness_prob > self.drowsiness_threshold
            
            
            if is_drowsy:
                if not self.currently_drowsy:
                    self.drowsy_start_time = current_time
                    self.currently_drowsy = True
                    drowsy_duration = 0.0
                else:
                    drowsy_duration = current_time - self.drowsy_start_time
                
                if drowsy_duration >= self.alert_threshold_seconds:
                    status = f"⚠️ WAKE UP! Drowsy for {drowsy_duration:.1f}s"
                else:
                    status = f"😴 Drowsiness detected ({drowsy_duration:.1f}s)"
            else:
                self.drowsy_start_time = None
                self.currently_drowsy = False
                drowsy_duration = 0.0
                status = "👁️ Alert - Stay focused"
            
            return True, is_drowsy, status, drowsy_duration, drowsiness_prob
            
        except Exception as e:
            return False, False, f"Detection error: {str(e)}", 0.0, 0.0

class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self, model):
        self.detector = LSTMDrowsinessDetector(model)
        self.frame_count = 0
        self.last_status = "Starting LSTM+CNN detection..."
        self.last_drowsy_duration = 0.0
        self.last_confidence = 0.0
        self.last_is_drowsy = False
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        self.frame_count += 1
        
        if self.frame_count % 2 == 0:
            try:
                detection_success, is_drowsy, status, drowsy_duration, confidence = self.detector.detect_drowsiness(img)
                
                if detection_success:
                    self.last_status = status
                    self.last_drowsy_duration = drowsy_duration
                    self.last_confidence = confidence
                    self.last_is_drowsy = is_drowsy
                else:
                    self.last_status = status
                    
            except Exception as e:
                self.last_status = f"Processing error: {str(e)}"
        
        if "WAKE UP" in self.last_status:
            text_color = (0, 0, 255)  
        elif "Drowsiness detected" in self.last_status:
            text_color = (0, 165, 255)  
        elif "Alert" in self.last_status:
            text_color = (0, 255, 0)  
        else:
            text_color = (255, 255, 255)  
        
        cv2.putText(img, f"Status: {self.last_status}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        
        if self.last_confidence > 0:
            confidence_text = f"Drowsiness Confidence: {self.last_confidence:.2f}"
            cv2.putText(img, confidence_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        if self.last_drowsy_duration > 0:
            timer_text = f"Drowsy Duration: {self.last_drowsy_duration:.1f}s"
            cv2.putText(img, timer_text, (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
       
        frame_info = f"Frame: {self.frame_count} | Seq: {len(self.detector.frame_sequence)}/{SEQUENCE_LENGTH}"
        cv2.putText(img, frame_info, (10, img.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.set_page_config(page_title="LSTM+CNN Drowsiness Detection", layout="wide")
    
    st.title(" Driver Drowsiness Detection System")
    st.markdown("**Check your infotainment screen for cautions **")
    
    
    model = load_model()
    
    if model is None:
        st.error(" Cannot proceed without the LSTM+CNN model. Please ensure 'drowsiness_model.h5' is available.")
        st.info(" **Requirements:**\n- TensorFlow installed\n- Trained drowsiness_model.h5 file in the current directory")
        return
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Live Cam")
        
        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        webrtc_ctx = webrtc_streamer(
            key="lstm-drowsiness-detection",
            video_processor_factory=lambda: DrowsinessProcessor(model),
            rtc_configuration=rtc_configuration,
            media_stream_constraints={
                "video": {"width": 640, "height": 480, "frameRate": 30},
                "audio": False
            }
        )
    
    with col2:
        
        st.subheader("📋 Instructions")
        st.markdown("""
        1. **Allow camera access** if not given
        2. Face the camera clearly 
        """)
        
        if webrtc_ctx.state.playing:
            st.success("🟢 **Camera Active** - LSTM+CNN model running")
        else:
            st.error("TURN ON YOUR CAM ")
        
        st.subheader("⚠️ Alert Levels")
        st.markdown("""
        - 🟢 **Green**: Alert and focused
        - 🟡 **Yellow**: Drowsiness detected
        - 🔴 **Red**: WAKE UP! Prolonged drowsiness
        """)

if __name__ == "__main__":
    main()