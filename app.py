# import cv2
# import numpy as np
# import tensorflow as tf
# from collections import deque
# import os

# IMG_SIZE = (32, 32)
# SEQUENCE_LENGTH = 10

# def create_model_architecture():
#     cnn = tf.keras.Sequential([
#         tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
#         tf.keras.layers.MaxPooling2D((2,2)),
#         tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#         tf.keras.layers.MaxPooling2D((2,2)),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(128, activation='relu')
#     ])

#     model = tf.keras.Sequential([
#         tf.keras.layers.TimeDistributed(cnn, input_shape=(10, 32, 32, 3)),
#         tf.keras.layers.LSTM(64),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])
    
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def load_model_safely(model_path):
#     print(f"Loading model from {model_path}...")
    
#     if not os.path.exists(model_path):
#         print(f"Model file not found: {model_path}")
#         return None
    
#     try:
#         print("Loading model weights...")
#         model = create_model_architecture()
#         model.load_weights(model_path)
#         print("Model loaded successfully!")
#         return model
#     except Exception as e:
#         print(f"Model loading failed: {e}")
    
#     return None

# class HaarEyeDetector:
    
#     def __init__(self):
#         try:
#             self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#             self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#             print("Haar cascades loaded successfully")
#         except Exception as e:
#             print(f"Error loading Haar cascades: {e}")
#             self.face_cascade = None
#             self.eye_cascade = None
        
#         self.eye_closure_threshold = 0.5  
#         self.closed_eye_start_time = None  
#         self.eyes_currently_closed = False
#         self.alert_threshold_seconds = 2.0  
        
#     def detect_eye_closure(self, frame):
#         import time
        
#         if self.face_cascade is None or self.eye_cascade is None:
#             return False, False, "Haar cascades not available", 0.0
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
#         if len(faces) == 0:
#             self.closed_eye_start_time = None
#             self.eyes_currently_closed = False
#             return False, False, "No face detected", 0.0
        
#         face_detected = True
#         eyes_closed = False
#         current_time = time.time()
#         closed_duration = 0.0
        
        
#         (x, y, w, h) = faces[0]
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]
        
#         eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        
#         if len(eyes) < 2:
#             eyes_closed = True
#             status = f"Eyes possibly closed (detected {len(eyes)} eyes)"
#         else:
#             eye_openness_scores = []
            
#             for (ex, ey, ew, eh) in eyes:
                
#                 eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
                
#                 mean_intensity = np.mean(eye_region)
#                 normalized_intensity = mean_intensity / 255.0
                
#                 eye_openness_scores.append(normalized_intensity)
                
#                 cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
#             avg_eye_openness = np.mean(eye_openness_scores) if eye_openness_scores else 1.0
            
#             if avg_eye_openness < self.eye_closure_threshold:
#                 eyes_closed = True
#                 status = f"Eyes closed (darkness: {avg_eye_openness:.2f})"
#             else:
#                 eyes_closed = False
#                 status = f"Eyes open (brightness: {avg_eye_openness:.2f})"
        
#         if eyes_closed:
#             if not self.eyes_currently_closed:
#                 self.closed_eye_start_time = current_time
#                 self.eyes_currently_closed = True
#                 closed_duration = 0.0
#             else:
#                 closed_duration = current_time - self.closed_eye_start_time
#         else:
#             self.closed_eye_start_time = None
#             self.eyes_currently_closed = False
#             closed_duration = 0.0
        
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
#         return face_detected, eyes_closed, status, closed_duration

# class ImprovedDrowsinessDetector:
#     def __init__(self, model):
#         self.model = model
#         self.eye_detector = HaarEyeDetector()
        
#         self.frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
        
#         self.model_threshold = 0.5
#         self.model_drowsy_count = 0
#         self.no_face_count = 0
        
#         self.model_alert_threshold = 3
#         self.no_face_alert_threshold = 15
        
#     def preprocess_frame(self, frame):
#         resized = cv2.resize(frame, IMG_SIZE)
#         normalized = resized / 255.0
#         return normalized
    
#     def predict_drowsiness(self, frame):
#         face_detected, eyes_closed, eye_status, closed_duration = self.eye_detector.detect_eye_closure(frame)
        
#         if not face_detected:
#             self.no_face_count += 1
#             status = "No Face Detected"
#             confidence = 0.0
            
#             if self.no_face_count > self.no_face_alert_threshold:
#                 status = "DRIVER NOT VISIBLE!"
#                 confidence = 0.9
#         else:
#             self.no_face_count = 0
            
#             model_prediction = 0.0
#             if self.model:
#                 processed_frame = self.preprocess_frame(frame)
#                 self.frame_sequence.append(processed_frame)
                
#                 if len(self.frame_sequence) >= SEQUENCE_LENGTH:
#                     try:
#                         sequence = np.array([list(self.frame_sequence)])
#                         model_prediction = self.model.predict(sequence, verbose=0)[0][0]
#                     except Exception as e:
#                         print(f"Model prediction error: {e}")
            
#             if model_prediction > self.model_threshold:
#                 self.model_drowsy_count += 1
#             else:
#                 self.model_drowsy_count = max(0, self.model_drowsy_count - 1)
            
#             if eyes_closed:
#                 if closed_duration >= self.eye_detector.alert_threshold_seconds:
#                     status = f"EYES CLOSED FOR {closed_duration:.1f}s - WAKE UP!"
#                     confidence = 1.0
#                 else:
#                     status = f"Eyes closed ({closed_duration:.1f}s)"
#                     confidence = closed_duration / self.eye_detector.alert_threshold_seconds
#             else:
#                 model_drowsy = self.model_drowsy_count >= self.model_alert_threshold
#                 if model_drowsy:
#                     status = f"MODEL ALERT - {eye_status}"
#                     confidence = model_prediction
#                 else:
#                     status = eye_status
#                     confidence = model_prediction * 0.3
        
#         return confidence, status, face_detected, closed_duration
    
#     def should_alert(self):
#         import time
#         return (self.eye_detector.eyes_currently_closed and 
#                 self.eye_detector.closed_eye_start_time is not None and
#                 (time.time() - self.eye_detector.closed_eye_start_time) >= self.eye_detector.alert_threshold_seconds)

# def run_improved_drowsiness_detection(model_path):
#     import time
    
#     model = load_model_safely(model_path)
#     if model is None:
#         print("Running without ML model (Haar cascade only)")
    
#     detector = ImprovedDrowsinessDetector(model)
    
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam")
#         return
    
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#     cap.set(cv2.CAP_PROP_FPS, 30)
    
#     print("🎥 Starting improved drowsiness detection...")
#     print("👀 Close your eyes for more than 2 seconds to trigger alert")
#     print("⌨️  Press 'q' to quit")
    
#     frame_count = 0
    
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             frame_count += 1
            
#             if frame_count % 3 == 0:
#                 confidence, status, face_detected, closed_duration = detector.predict_drowsiness(frame)
                
#                 if "WAKE UP" in status:
#                     color = (0, 0, 255)  
#                 elif "closed" in status.lower() and closed_duration > 0:
#                     color = (0, 165, 255) 
#                 elif "No Face" in status or "NOT VISIBLE" in status:
#                     color = (255, 0, 255)  
#                 else:
#                     color = (0, 255, 0)  
                
#                 cv2.putText(frame, f"Status: {status}", (10, 30), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
#                 if closed_duration > 0:
#                     timer_text = f"Eyes closed: {closed_duration:.1f}s / 2.0s"
#                     cv2.putText(frame, timer_text, (10, 60), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
#                 if detector.should_alert():
#                     cv2.putText(frame, "WAKE UP! ", (10, 150), 
#                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
#                     if frame_count % 10 == 0:  
#                         print(f"WAKE UP ALERT! Eyes closed for {closed_duration:.1f} seconds!")
            
#             cv2.imshow('Drowsiness Detection - 2 Second Rule', frame)
            
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
                
#     except KeyboardInterrupt:
#         print("\n  Stopping detection...")
    
#     finally:
#         cap.release()
#         cv2.destroyAllWindows()
#         print("Detection stopped")

# if __name__ == "__main__":
#     MODEL_PATH = "drowsiness_model.h5"
#     run_improved_drowsiness_detection(MODEL_PATH)






















# import streamlit as st
# import cv2
# import numpy as np
# import tensorflow as tf
# from collections import deque
# import os
# from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
# import av
# import threading
# import time

# IMG_SIZE = (32, 32)
# SEQUENCE_LENGTH = 10

# def create_model_architecture():
#     cnn = tf.keras.Sequential([
#         tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
#         tf.keras.layers.MaxPooling2D((2,2)),
#         tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
#         tf.keras.layers.MaxPooling2D((2,2)),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(128, activation='relu')
#     ])

#     model = tf.keras.Sequential([
#         tf.keras.layers.TimeDistributed(cnn, input_shape=(10, 32, 32, 3)),
#         tf.keras.layers.LSTM(64),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(1, activation='sigmoid')
#     ])
    
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     return model

# def load_model_safely(model_path):
#     if not os.path.exists(model_path):
#         return None
    
#     try:
#         model = create_model_architecture()
#         model.load_weights(model_path)
#         return model
#     except Exception as e:
#         st.error(f"Model loading failed: {e}")
#         return None

# class HaarEyeDetector:
#     def __init__(self):
#         try:
#             self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#             self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#         except Exception as e:
#             st.error(f"Error loading Haar cascades: {e}")
#             self.face_cascade = None
#             self.eye_cascade = None
        
#         self.eye_closure_threshold = 0.5  
#         self.closed_eye_start_time = None  
#         self.eyes_currently_closed = False
#         self.alert_threshold_seconds = 2.0  
        
#     def detect_eye_closure(self, frame):
#         if self.face_cascade is None or self.eye_cascade is None:
#             return False, False, "Haar cascades not available", 0.0
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
#         if len(faces) == 0:
#             self.closed_eye_start_time = None
#             self.eyes_currently_closed = False
#             return False, False, "No face detected", 0.0
        
#         face_detected = True
#         eyes_closed = False
#         current_time = time.time()
#         closed_duration = 0.0
        
#         (x, y, w, h) = faces[0]
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]
        
#         eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        
#         if len(eyes) < 2:
#             eyes_closed = True
#             status = f"Eyes possibly closed (detected {len(eyes)} eyes)"
#         else:
#             eye_openness_scores = []
            
#             for (ex, ey, ew, eh) in eyes:
#                 eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
#                 mean_intensity = np.mean(eye_region)
#                 normalized_intensity = mean_intensity / 255.0
#                 eye_openness_scores.append(normalized_intensity)
#                 cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
#             avg_eye_openness = np.mean(eye_openness_scores) if eye_openness_scores else 1.0
            
#             if avg_eye_openness < self.eye_closure_threshold:
#                 eyes_closed = True
#                 status = f"Eyes closed (darkness: {avg_eye_openness:.2f})"
#             else:
#                 eyes_closed = False
#                 status = f"Eyes open (brightness: {avg_eye_openness:.2f})"
        
#         if eyes_closed:
#             if not self.eyes_currently_closed:
#                 self.closed_eye_start_time = current_time
#                 self.eyes_currently_closed = True
#                 closed_duration = 0.0
#             else:
#                 closed_duration = current_time - self.closed_eye_start_time
#         else:
#             self.closed_eye_start_time = None
#             self.eyes_currently_closed = False
#             closed_duration = 0.0
        
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
#         return face_detected, eyes_closed, status, closed_duration

# class DrowsinessVideoTransformer(VideoProcessorBase):
#     def __init__(self):
#         self.model = None
#         self.eye_detector = HaarEyeDetector()
#         self.frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
#         self.model_threshold = 0.5
#         self.model_drowsy_count = 0
#         self.no_face_count = 0
#         self.model_alert_threshold = 3
#         self.no_face_alert_threshold = 15
#         self.alert_placeholder = None
        
#     def set_model(self, model):
#         self.model = model
        
#     def set_alert_placeholder(self, placeholder):
#         self.alert_placeholder = placeholder
    
#     def preprocess_frame(self, frame):
#         resized = cv2.resize(frame, IMG_SIZE)
#         normalized = resized / 255.0
#         return normalized
    
#     def should_alert(self):
#         return (self.eye_detector.eyes_currently_closed and 
#                 self.eye_detector.closed_eye_start_time is not None and
#                 (time.time() - self.eye_detector.closed_eye_start_time) >= self.eye_detector.alert_threshold_seconds)
    
#     def predict_drowsiness(self, frame):
#         face_detected, eyes_closed, eye_status, closed_duration = self.eye_detector.detect_eye_closure(frame)
        
#         if not face_detected:
#             self.no_face_count += 1
#             status = "No Face Detected"
#             confidence = 0.0
            
#             if self.no_face_count > self.no_face_alert_threshold:
#                 status = "DRIVER NOT VISIBLE!"
#                 confidence = 0.9
#         else:
#             self.no_face_count = 0
            
#             model_prediction = 0.0
#             if self.model:
#                 processed_frame = self.preprocess_frame(frame)
#                 self.frame_sequence.append(processed_frame)
                
#                 if len(self.frame_sequence) >= SEQUENCE_LENGTH:
#                     try:
#                         sequence = np.array([list(self.frame_sequence)])
#                         model_prediction = self.model.predict(sequence, verbose=0)[0][0]
#                     except Exception as e:
#                         pass
            
#             if model_prediction > self.model_threshold:
#                 self.model_drowsy_count += 1
#             else:
#                 self.model_drowsy_count = max(0, self.model_drowsy_count - 1)
            
#             if eyes_closed:
#                 if closed_duration >= self.eye_detector.alert_threshold_seconds:
#                     status = f"EYES CLOSED FOR {closed_duration:.1f}s - WAKE UP!"
#                     confidence = 1.0
#                 else:
#                     status = f"Eyes closed ({closed_duration:.1f}s)"
#                     confidence = closed_duration / self.eye_detector.alert_threshold_seconds
#             else:
#                 model_drowsy = self.model_drowsy_count >= self.model_alert_threshold
#                 if model_drowsy:
#                     status = f"MODEL ALERT - {eye_status}"
#                     confidence = model_prediction
#                 else:
#                     status = eye_status
#                     confidence = model_prediction * 0.3
        
#         return confidence, status, face_detected, closed_duration

#     def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
#         img = frame.to_ndarray(format="bgr24")
        
#         confidence, status, face_detected, closed_duration = self.predict_drowsiness(img)
        
#         if "WAKE UP" in status:
#             color = (0, 0, 255)  
#         elif "closed" in status.lower() and closed_duration > 0:
#             color = (0, 165, 255) 
#         elif "No Face" in status or "NOT VISIBLE" in status:
#             color = (255, 0, 255)  
#         else:
#             color = (0, 255, 0)  
        
#         cv2.putText(img, f"Status: {status}", (10, 30), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
#         if closed_duration > 0:
#             timer_text = f"Eyes closed: {closed_duration:.1f}s / 2.0s"
#             cv2.putText(img, timer_text, (10, 60), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
#         if self.should_alert():
#             cv2.putText(img, "WAKE UP! ", (10, 150), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
#             if self.alert_placeholder:
#                 self.alert_placeholder.error("🚨 DROWSINESS ALERT! WAKE UP!")
        
#         return av.VideoFrame.from_ndarray(img, format="bgr24")

# def main():
#     st.set_page_config(page_title="Drowsiness Detection", layout="wide")
    
#     st.title("🚗 Drowsiness Detection System")
#     st.markdown("Real-time drowsiness detection using computer vision and machine learning")
    
#     col1, col2 = st.columns([2, 1])
    
#     with col1:
#         st.subheader("Live Camera Feed")
        
#         MODEL_PATH = "drowsiness_model.h5"
#         model = load_model_safely(MODEL_PATH)
        
#         if model:
#             st.success("✅ ML Model loaded successfully")
#         else:
#             st.warning("⚠️ Running with Haar cascade only (no ML model found)")
        
#         alert_placeholder = st.empty()
        
#         def create_processor():
#             processor = DrowsinessVideoTransformer()
#             if model:
#                 processor.set_model(model)
#             processor.set_alert_placeholder(alert_placeholder)
#             return processor
        
#         rtc_configuration = RTCConfiguration({
#             "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#         })
        
#         webrtc_ctx = webrtc_streamer(
#             key="drowsiness-detection",
#             video_processor_factory=create_processor,
#             rtc_configuration=rtc_configuration,
#             media_stream_constraints={
#                 "video": {"width": 640, "height": 480, "frameRate": 30},
#                 "audio": False
#             },
#             async_processing=True
#         )
    
#     with col2:
#         st.subheader("System Information")
        
#         st.info("""
#         **Detection Features:**
#         - Face detection using Haar cascades
#         - Eye closure monitoring
#         - 2-second alert threshold
#         - ML model prediction (if available)
        
#         **Alert Triggers:**
#         - Eyes closed > 2 seconds
#         - No face detected for extended time
#         - ML model drowsiness prediction
#         """)
        
#         st.warning("""
#         **Instructions:**
#         1. Allow camera access when prompted
#         2. Position your face clearly in view
#         3. System will alert if drowsiness detected
#         4. Close eyes for 2+ seconds to test
#         """)
        
#         if webrtc_ctx.state.playing:
#             st.success("🟢 Camera Active")
#         else:
#             st.error("🔴 Camera Inactive")

# if __name__ == "__main__":
#     main()










import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from collections import deque
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading
import time

IMG_SIZE = (32, 32)
SEQUENCE_LENGTH = 10

def create_model_architecture():
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
    return model

def load_model_safely(model_path):
    if not os.path.exists(model_path):
        return None
    
    try:
        model = create_model_architecture()
        model.load_weights(model_path)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

class HaarEyeDetector:
    def __init__(self):
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        except Exception as e:
            st.error(f"Error loading Haar cascades: {e}")
            self.face_cascade = None
            self.eye_cascade = None
        
        self.eye_closure_threshold = 0.5  
        self.closed_eye_start_time = None  
        self.eyes_currently_closed = False
        self.alert_threshold_seconds = 2.0  
        
    def detect_eye_closure(self, frame):
        if self.face_cascade is None or self.eye_cascade is None:
            return False, False, "Haar cascades not available", 0.0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            self.closed_eye_start_time = None
            self.eyes_currently_closed = False
            return False, False, "No face detected", 0.0
        
        face_detected = True
        eyes_closed = False
        current_time = time.time()
        closed_duration = 0.0
        
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
        
        if len(eyes) < 2:
            eyes_closed = True
            status = f"Eyes possibly closed (detected {len(eyes)} eyes)"
        else:
            eye_openness_scores = []
            
            for (ex, ey, ew, eh) in eyes:
                eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
                mean_intensity = np.mean(eye_region)
                normalized_intensity = mean_intensity / 255.0
                eye_openness_scores.append(normalized_intensity)
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            avg_eye_openness = np.mean(eye_openness_scores) if eye_openness_scores else 1.0
            
            if avg_eye_openness < self.eye_closure_threshold:
                eyes_closed = True
                status = f"Eyes closed (darkness: {avg_eye_openness:.2f})"
            else:
                eyes_closed = False
                status = f"Eyes open (brightness: {avg_eye_openness:.2f})"
        
        if eyes_closed:
            if not self.eyes_currently_closed:
                self.closed_eye_start_time = current_time
                self.eyes_currently_closed = True
                closed_duration = 0.0
            else:
                closed_duration = current_time - self.closed_eye_start_time
        else:
            self.closed_eye_start_time = None
            self.eyes_currently_closed = False
            closed_duration = 0.0
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        return face_detected, eyes_closed, status, closed_duration

class DrowsinessVideoTransformer(VideoProcessorBase):
    def __init__(self):
        self.model = None
        self.eye_detector = HaarEyeDetector()
        self.frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.model_threshold = 0.5
        self.model_drowsy_count = 0
        self.no_face_count = 0
        self.model_alert_threshold = 3
        self.no_face_alert_threshold = 15
        self.alert_placeholder = None
        self.frame_count = 0
        self.process_every_n_frames = 2
        
    def set_model(self, model):
        self.model = model
        
    def set_alert_placeholder(self, placeholder):
        self.alert_placeholder = placeholder
    
    def preprocess_frame(self, frame):
        resized = cv2.resize(frame, IMG_SIZE)
        normalized = resized / 255.0
        return normalized
    
    def should_alert(self):
        return (self.eye_detector.eyes_currently_closed and 
                self.eye_detector.closed_eye_start_time is not None and
                (time.time() - self.eye_detector.closed_eye_start_time) >= self.eye_detector.alert_threshold_seconds)
    
    def predict_drowsiness(self, frame):
        face_detected, eyes_closed, eye_status, closed_duration = self.eye_detector.detect_eye_closure(frame)
        
        if not face_detected:
            self.no_face_count += 1
            status = "No Face Detected"
            confidence = 0.0
            
            if self.no_face_count > self.no_face_alert_threshold:
                status = "DRIVER NOT VISIBLE!"
                confidence = 0.9
        else:
            self.no_face_count = 0
            
            model_prediction = 0.0
            if self.model:
                processed_frame = self.preprocess_frame(frame)
                self.frame_sequence.append(processed_frame)
                
                if len(self.frame_sequence) >= SEQUENCE_LENGTH:
                    try:
                        sequence = np.array([list(self.frame_sequence)])
                        model_prediction = self.model.predict(sequence, verbose=0)[0][0]
                    except Exception as e:
                        pass
            
            if model_prediction > self.model_threshold:
                self.model_drowsy_count += 1
            else:
                self.model_drowsy_count = max(0, self.model_drowsy_count - 1)
            
            if eyes_closed:
                if closed_duration >= self.eye_detector.alert_threshold_seconds:
                    status = f"EYES CLOSED FOR {closed_duration:.1f}s - WAKE UP!"
                    confidence = 1.0
                else:
                    status = f"Eyes closed ({closed_duration:.1f}s)"
                    confidence = closed_duration / self.eye_detector.alert_threshold_seconds
            else:
                model_drowsy = self.model_drowsy_count >= self.model_alert_threshold
                if model_drowsy:
                    status = f"MODEL ALERT - {eye_status}"
                    confidence = model_prediction
                else:
                    status = eye_status
                    confidence = model_prediction * 0.3
        
        return confidence, status, face_detected, closed_duration

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        self.frame_count += 1
        
        if self.frame_count % self.process_every_n_frames == 0:
            try:
                confidence, status, face_detected, closed_duration = self.predict_drowsiness(img)
                
                if "WAKE UP" in status:
                    color = (0, 0, 255)  
                elif "closed" in status.lower() and closed_duration > 0:
                    color = (0, 165, 255) 
                elif "No Face" in status or "NOT VISIBLE" in status:
                    color = (255, 0, 255)  
                else:
                    color = (0, 255, 0)  
                
                cv2.putText(img, f"Status: {status}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if closed_duration > 0:
                    timer_text = f"Eyes closed: {closed_duration:.1f}s / 2.0s"
                    cv2.putText(img, timer_text, (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                if self.should_alert():
                    cv2.putText(img, "WAKE UP! ", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            except Exception as e:
                cv2.putText(img, "Processing...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.set_page_config(page_title="Drowsiness Detection", layout="wide")
    
    st.title("🚗 Drowsiness Detection System")
    st.markdown("Real-time drowsiness detection using computer vision and machine learning")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Live Camera Feed")
        
        MODEL_PATH = "drowsiness_model.h5"
        model = load_model_safely(MODEL_PATH)
        
        if model:
            st.success("✅ ML Model loaded successfully")
        else:
            st.warning("⚠️ Running with Haar cascade only (no ML model found)")
        
        alert_placeholder = st.empty()
        
        def create_processor():
            processor = DrowsinessVideoTransformer()
            if model:
                processor.set_model(model)
            processor.set_alert_placeholder(alert_placeholder)
            return processor
        
        rtc_configuration = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })
        
        webrtc_ctx = webrtc_streamer(
            key="drowsiness-detection",
            video_processor_factory=create_processor,
            rtc_configuration=rtc_configuration,
            media_stream_constraints={
                "video": {"width": 640, "height": 480},
                "audio": False
            }
        )
    
    with col2:
        st.subheader("System Information")
        
        st.info("""
        **Detection Features:**
        - Face detection using Haar cascades
        - Eye closure monitoring
        - 2-second alert threshold
        - ML model prediction (if available)
        
        **Alert Triggers:**
        - Eyes closed > 2 seconds
        - No face detected for extended time
        - ML model drowsiness prediction
        """)
        
        st.warning("""
        **Instructions:**
        1. Allow camera access when prompted
        2. Position your face clearly in view
        3. System will alert if drowsiness detected
        4. Close eyes for 2+ seconds to test
        """)
        
        if webrtc_ctx.state.playing:
            st.success("🟢 Camera Active")
        else:
            st.error("🔴 Camera Inactive")

if __name__ == "__main__":
    main()